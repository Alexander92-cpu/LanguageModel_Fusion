from collections import defaultdict
import math
from typing import Dict, Optional, List, Tuple, Union

import editdistance
import numpy as np
from omegaconf import DictConfig
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .eval_data import DataPool
from .train_ngram.utils import tokenize_str
from .word_language_model.data import tokenize_str as lstm_tokenize_str


RescoreOutput = Dict[str,
                       Union[List[str],
                             Dict[str, float],
                             List[Dict[str, Union[str, List[str], Dict[str, np.array]]]]
                             ]
                    ]

class Rescore:
    def __init__(self, cfg: DictConfig, data_pool: DataPool) -> None:
        self.cfg = cfg
        self.data_pool = data_pool
        self.apply_methods = any(item for item in self.cfg.rescore.methods.values())
        self.methods = {'baseline': self.cfg.rescore.methods.baseline,
                        'sf': self.cfg.rescore.methods.sf, 'lodr': self.cfg.rescore.methods.lodr,
                        'dr': self.cfg.rescore.methods.dr, 'ilme': self.cfg.rescore.methods.ilme}
        self.device = self.cfg.root_params.device
        self.bos_token_id = self.data_pool.ft_lm_model.config.bos_token_id
        self.eos_token_id = self.data_pool.ft_lm_model.config.eos_token_id
        self.rescore_params = self.cfg.rescore.params

    def eval_rnnt(
            self,
            audio_dataloader: DataLoader
        ) -> RescoreOutput:
        info = defaultdict(list)
        for batch in tqdm(audio_dataloader, leave=True, desc="Decoding",
                          total=len(audio_dataloader)):
            input_signal, input_signal_length, references = batch
            input_signal = input_signal.to(self.device)
            input_signal_length = input_signal_length.to(self.device)
            current_hypotheses = self.rnnt_alignments(input_signal, input_signal_length)
            all_hypotheses, all_hypotheses_zero = current_hypotheses
            for idx, nbests in enumerate(tqdm(all_hypotheses, leave=False, desc="Decode nbests")):
                unique_idx = []
                temp = set()
                for i, nbest in enumerate(nbests):
                    if nbest.text not in temp:
                        temp.add(nbest.text)
                        unique_idx.append(i)
                utexts = [nbests[i].text for i in unique_idx]

                scores = {'asr_scores': np.array([nbests[i].score for i in unique_idx])}
                scores['gpt2_scores'], scores['num_tokens'] = self.nll_word_sent_batch(utexts)

                if self.methods['lodr']:
                    scores['lodr_scores'], _ = self.get_ngram_lm_score(utexts)
                if self.methods['dr']:
                    scores['dr_scores'], _ = self.ilstm_inference(utexts)
                if self.methods['ilme']:
                    scores['ilme_scores'] = np.array([all_hypotheses_zero[idx][i].score for i in unique_idx])

                for key, use_method in self.methods.items():
                    rescore = scores['asr_scores'].copy()
                    if key != 'baseline' and use_method:
                        for param in self.rescore_params[key]:
                            rescore += self.rescore_params[key][param] * scores[param]
                        rescore = rescore.tolist()
                        st_rescore = rescore.index(max(rescore))
                        best_hypothesis = nbests[unique_idx[st_rescore]]
                    elif key == 'baseline':
                        best_hypothesis = nbests[0]
                    else:
                        continue
                    info[key].append(best_hypothesis.text)
                info['outputs'].append({'scores': scores,
                                        'utexts': utexts, 'reference': references[idx]})
            info['references'].extend(references)
        if self.cfg.rescore.calculate_wer:
            info['wer'] = {}
            for key, use_method in self.methods.items():
                if use_method:
                    if len(info[key]) != len(info['references']):
                        raise ValueError("Error while computing wer!")
                    info['wer'][key] = self.calculate_wer(info[key], info['references'])
                else:
                    info['wer'][key] = None
        return info

    def rnnt_alignments(
            self,
            input_signal: torch.tensor,
            input_signal_length: torch.tensor
        ) -> Tuple[List[List[Hypothesis]], List[List[Hypothesis]]]:
        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                encoded, encoded_len = self.data_pool.asr_model.forward(input_signal=input_signal,
                                                                        input_signal_length=input_signal_length)
                current_hypotheses = self.data_pool.asr_model.decoding.rnnt_decoder_predictions_tensor(encoded, encoded_len, return_hypotheses=True)
                _, all_hypotheses = current_hypotheses
                all_hypotheses_zero = [[] for _ in range(input_signal_length.size(0))]
                if self.methods['ilme']:
                    zero_encoded = torch.zeros_like(encoded)
                    current_hypotheses_zero = self.data_pool.asr_model.decoding.rnnt_decoder_predictions_tensor(zero_encoded, encoded_len, return_hypotheses=True)
                    _, all_hypotheses_zero = current_hypotheses_zero
        return all_hypotheses, all_hypotheses_zero

    def nll_word_sent_batch(self, text: List[str]) -> Tuple[np.array, np.array]:
        input_ids = []
        num_tokens = []
        for line in text:
            line = [self.bos_token_id] + self.data_pool.asr_tokenizer.tokenizer.Encode(line) + [self.eos_token_id]
            num_tokens.append(len(line))
            input_ids.append(torch.tensor(line, dtype=torch.long).unsqueeze(0).to(self.device))

        neg_log_likelihoods = []
        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                for sentence in input_ids:
                    outputs = self.data_pool.ft_lm_model(input_ids=sentence, labels=sentence)
                    neg_log_likelihoods.append(-sentence.size(1) * outputs.loss.cpu().item())

        return np.array(neg_log_likelihoods), np.array(num_tokens)

    def get_ngram_lm_score(self, text: List[str]) -> Tuple[np.array, np.array]:
        tokenized_text = tokenize_str(text, self.data_pool.asr_tokenizer, self.cfg.kenlm.offset)
        tokens_num = [len(item) for item in tokenized_text]
        scores = []

        for tokenized_sentence in tokenized_text:
            sentence = ' '.join(tokenized_sentence)
            sentence_score = self.data_pool.ngram_lm.score(sentence) * math.log(10)
            scores.append(sentence_score)

        return np.array(scores), np.array(tokens_num)

    def ilstm_inference(self, data_source: List[str]) -> Tuple[np.array, np.array]:
        tokenized_text = []
        ilstm_tokens_num = []
        for line in data_source:
            line = lstm_tokenize_str(self.data_pool.lstm_tokenizer, "<bos> " + line + ' <eos>')
            ilstm_tokens_num.append(len(line))
            tokenized_text.append(torch.unsqueeze(torch.LongTensor(line), dim=0).t().contiguous().to(self.device))
        sentence_probs = []
        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                for src in tokenized_text:
                    hidden = self.data_pool.lstm.init_hidden(1)
                    hidden = [item.to(self.device) for item in hidden]
                    log_probs, hidden = self.data_pool.lstm(src, hidden)
                    log_gen_probs = torch.gather(log_probs[:-1, :], 1, src[1:, :]).squeeze(-1).cpu().numpy()
                    sentence_probs.append(log_gen_probs.sum())
        return np.array(sentence_probs), np.array(ilstm_tokens_num)

    @staticmethod
    def calculate_wer(seqs_hat: List[str], seqs_true: List[str]) -> Optional[float]:
        """Calculate sentence-level WER score.

        :param list seqs_hat: prediction
        :param list seqs_true: reference
        :return: average sentence-level WER score
        :rtype float
        """
        word_eds, word_ref_lens = [], []
        for i, seq_hat_text in enumerate(seqs_hat):
            seq_true_text = seqs_true[i]
            hyp_words = seq_hat_text.split()
            ref_words = seq_true_text.split()
            word_eds.append(editdistance.eval(hyp_words, ref_words))
            word_ref_lens.append(len(ref_words))
        if word_ref_lens:
            wer = 100 * float(sum(word_eds)) / sum(word_ref_lens)
        else:
            wer = None
        return wer
