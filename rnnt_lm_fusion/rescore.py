"""Module containing various functions and classes for rescoring.

We implemented the following rescoring methods:
- Shallow Fusion
- Density Ratio
- LODR
- ILME

Authors:
    Alexandru Mazurenco (2024)

License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import editdistance
import numpy as np
import torch
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .eval_data import DataPool
from .train_ngram.utils import tokenize_str
from .word_language_model.data import tokenize_str as lstm_tokenize_str

RescoreOutput = Dict[
    str,
    Union[
        List[str],
        Dict[str, float],
        List[Dict[str, Union[str, List[str], Dict[str, np.array]]]],
    ],
]


class Rescore:
    """Class for performing rescoring using various methods."""

    def __init__(self, cfg: DictConfig, data_pool: DataPool) -> None:
        """Initialize a Rescore object with the provided configuration settings and data pool.

        Args:
            cfg (DictConfig): A dictionary-like object containing configuration settings.
            data_pool (DataPool): An object containing the necessary data for rescoring.
        """
        self.cfg = cfg
        self.data_pool = data_pool
        self.apply_methods = any(item for item in self.cfg.rescore.methods.values())
        self.methods = {
            "baseline": self.cfg.rescore.methods.baseline,
            "sf": self.cfg.rescore.methods.sf,
            "lodr": self.cfg.rescore.methods.lodr,
            "dr": self.cfg.rescore.methods.dr,
            "ilme": self.cfg.rescore.methods.ilme,
        }
        self.device = self.cfg.root_params.device
        self.bos_token_id = self.data_pool.ft_lm_model.config.bos_token_id
        self.eos_token_id = self.data_pool.ft_lm_model.config.eos_token_id

    def eval_rnnt(self, audio_dataloader: DataLoader) -> RescoreOutput:
        """Evaluate the RNN-T model and perform rescoring using the provided audio dataloader.

        This method performs evaluation of the RNN-T model using the provided audio
        dataloader, returning the rescore output.

        Args:
            audio_dataloader (DataLoader): DataLoader containing audio data for evaluation.

        Returns:
            RescoreOutput: Rescore output containing evaluation results.
        """
        info = defaultdict(list)
        for batch in tqdm(
            audio_dataloader, leave=True, desc="Decoding", total=len(audio_dataloader)
        ):
            self.process_batch(batch, info)
        info["references"] = [item for batch in info["references"] for item in batch]
        if self.cfg.rescore.calculate_wer:
            self.compute_wer(info)
        return info

    def process_batch(
        self, batch: Tuple[torch.tensor, torch.tensor, Tuple[str]], info: RescoreOutput
    ) -> None:
        """Process a batch of data during evaluation.

        Args:
            batch (Tuple[torch.tensor, torch.tensor, Tuple[str]]): The batch of input signals,
                input signal lengths, and corresponding references.
            info (RescoreOutput): The rescore output container.
        """
        input_signal, input_signal_length, references = batch
        input_signal = input_signal.to(self.device)
        input_signal_length = input_signal_length.to(self.device)
        info["references"].append(references)
        current_hypotheses = self.rnnt_alignments(input_signal, input_signal_length)
        all_hypotheses, all_hypotheses_zero = current_hypotheses
        for idx, (nbests, model_zero_scores) in enumerate(
            tqdm(
                zip(all_hypotheses, all_hypotheses_zero),
                leave=False,
                desc="Decode nbests",
            )
        ):
            info["outputs"].append({"reference": info["references"][idx]})
            self.process_nbests(nbests, info, model_zero_scores)

    def process_nbests(
        self,
        nbests: List[Hypothesis],
        info: RescoreOutput,
        all_hypotheses_zero: List[Optional[Hypothesis]],
    ) -> None:
        """Process the n-best hypotheses for a batch during evaluation.

        This method extracts unique hypotheses from the n-best list, calculates scores
        for each hypothesis, and updates the rescore output container with the extracted
        hypotheses, their scores, and other information.

        Args:
            nbests (List[Hypothesis]): The list of n-best hypotheses.
            info (RescoreOutput): The rescore output container.
            all_hypotheses_zero (List[Optional[Hypothesis]]): The list of model "zero" scores for
                                                              the hypotheses.
        """
        unique_idx = []
        temp = set()
        for i, nbest in enumerate(nbests):
            if nbest.text not in temp:
                temp.add(nbest.text)
                unique_idx.append(i)
        utexts = [nbests[i].text for i in unique_idx]
        info["outputs"][-1].update({"utexts": utexts})
        scores = self.calculate_scores(utexts, nbests, unique_idx, all_hypotheses_zero)
        info["outputs"][-1].update({"scores": scores})
        self.rescore_and_store_scores(scores, nbests, unique_idx, info)

    def calculate_scores(
        self,
        utexts: List[str],
        nbests: List[Hypothesis],
        unique_idx: List[int],
        all_hypotheses_zero: List[Optional[Hypothesis]],
    ) -> Dict[str, np.array]:
        """Calculate scores for the unique hypotheses extracted from the n-best list.

        This method calculates various scores for each unique hypothesis, including ASR scores,
        GPT-2 scores, LODR scores, DR scores, and ILME scores (if applicable), and returns them
        in a dictionary.

        Args:
            utexts (List[str]): The list of unique hypothesis texts.
            nbests (List[Hypothesis]): The list of n-best hypotheses.
            unique_idx (List[int]): The indices of the unique hypotheses in the n-best list.
            all_hypotheses_zero (List[Optional[Hypothesis]]): The list of model "zero" scores for
                                                              the hypotheses.

        Returns:
            Dict[str, np.array]: A dictionary containing the calculated scores.
                Keys:
                    - "asr_scores": Array of ASR scores for the unique hypotheses.
                    - "gpt2_scores": Array of GPT-2 scores for the unique hypotheses.
                    - "num_tokens": Number of tokens in the generated sentences.
                    - "lodr_scores" (optional): Array of LODR scores for the unique hypotheses.
                    - "dr_scores" (optional): Array of DR scores for the unique hypotheses.
                    - "ilme_scores" (optional): Array of ILME scores for the unique hypotheses.
        """
        scores = {"asr_scores": np.array([nbests[i].score for i in unique_idx])}
        scores["gpt2_scores"], scores["num_tokens"] = self.nll_word_sent_batch(utexts)

        if self.methods["lodr"]:
            scores["lodr_scores"], _ = self.get_ngram_lm_score(utexts)
        if self.methods["dr"]:
            scores["dr_scores"], _ = self.ilstm_inference(utexts)
        if self.methods["ilme"]:
            scores["ilme_scores"] = np.array(
                [all_hypotheses_zero[i].score for i in unique_idx]
            )

        return scores

    def rescore_and_store_scores(
        self,
        scores: Dict[str, np.array],
        nbests: List[Hypothesis],
        unique_idx: List[int],
        info: RescoreOutput,
    ) -> None:
        """Rescore the hypotheses and store the best hypothesis for each scoring method.

        This method rescores the hypotheses using various methods and stores the best hypothesis
        for each scoring method in the rescore output container.

        Args:
            scores (Dict[str, np.array]): A dictionary containing scores for the hypotheses.
            nbests (List[Hypothesis]): The list of n-best hypotheses.
            unique_idx (List[int]): The indices of the unique hypotheses in the n-best list.
            info (RescoreOutput): The rescore output container.
        """
        for key, use_method in self.methods.items():
            rescore = scores["asr_scores"].copy()
            if key != "baseline" and use_method:
                rescore = self.rescore_with_method(scores, key, rescore)
                st_rescore = rescore.index(max(rescore))
                best_hypothesis = nbests[unique_idx[st_rescore]]
            elif key == "baseline":
                best_hypothesis = nbests[0]
            else:
                continue
            info[key].append(best_hypothesis.text)

    def rescore_with_method(
        self, scores: Dict[str, np.array], key: str, rescore: np.array
    ) -> np.array:
        """Rescore the hypotheses using a specific method.

        This method rescores the hypotheses using the specified method and updates
        the scores accordingly.

        Args:
            scores (Dict[str, np.array]): A dictionary containing scores for the hypotheses.
            key (str): The key representing the rescore method to be applied.
            rescore (np.array): The initial scores to be rescored.

        Returns:
            np.array: The rescored scores.
        """
        for param in self.cfg.rescore.params[key]:
            rescore += self.cfg.rescore.params[key][param] * scores[param]
        return rescore

    def compute_wer(self, info: RescoreOutput) -> None:
        """Compute the Word Error Rate (WER) for different methods based on the
           provided information.

        Args:
            info (dict): A dictionary containing the necessary information for computing WER.
                It should have the following keys:
                - "references": A list of reference sentences.
                - Other keys corresponding to different methods, each containing a list
                  of hypothesis sentences.

        Raises:
            ValueError: If the lengths of hypothesis sentences for any method do not match
                        the length of reference sentences.
        """
        info["wer"] = {}
        for key, use_method in self.methods.items():
            if use_method:
                if len(info[key]) != len(info["references"]):
                    raise ValueError("Error while computing wer!")
                info["wer"][key] = self.calculate_wer(info[key], info["references"])
            else:
                info["wer"][key] = None

    def rnnt_alignments(
        self, input_signal: torch.tensor, input_signal_length: torch.tensor
    ) -> Tuple[List[List[Hypothesis]], List[List[Hypothesis]]]:
        """Compute RNN-T alignments for the given input signal.

        This method computes RNN-T alignments for the given input signal using the ASR model
        associated with the class instance. It returns two lists of hypotheses: one with
        alignments computed using the original input signal and another with alignments
        computed using a zero input signal (if the 'ilme' method is enabled).

        Args:
            input_signal (torch.tensor): The input signal tensor.
            input_signal_length (torch.tensor): The length tensor for the input signal.

        Returns:
            Tuple[List[List[Hypothesis]], List[List[Hypothesis]]]: A tuple containing two lists
            of hypotheses. The first list contains alignments computed using the original input
            signal, and the second list contains alignments computed using a zero input signal.
        """
        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                encoded, encoded_len = self.data_pool.asr_model.forward(
                    input_signal=input_signal, input_signal_length=input_signal_length
                )
                current_hypotheses = (
                    self.data_pool.asr_model.decoding.rnnt_decoder_predictions_tensor(
                        encoded, encoded_len, return_hypotheses=True
                    )
                )
                _, all_hypotheses = current_hypotheses
                all_hypotheses_zero = [[] for _ in range(input_signal_length.size(0))]
                if self.methods["ilme"]:
                    zero_encoded = torch.zeros_like(encoded)
                    decoding_method = (
                        self.data_pool.asr_model.decoding.rnnt_decoder_predictions_tensor
                    )
                    current_hypotheses_zero = decoding_method(
                        zero_encoded, encoded_len, return_hypotheses=True
                    )
                    _, all_hypotheses_zero = current_hypotheses_zero
        return all_hypotheses, all_hypotheses_zero

    def nll_word_sent_batch(self, text: List[str]) -> Tuple[np.array, np.array]:
        """Calculate the negative log-likelihood of word-level sentences in batch.

        This method calculates the negative log-likelihood (NLL) of word-level sentences
        provided in batch. It tokenizes each sentence, computes the NLL using the
        fine-tuned language model, and returns the NLL values along with the number of
        tokens in each sentence.

        Args:
            text (List[str]): A list of strings representing word-level sentences.

        Returns:
            Tuple[np.array, np.array]: A tuple containing two NumPy arrays. The first array
            contains the negative log-likelihood values for each sentence, and the second
            array contains the number of tokens in each sentence.
        """
        input_ids = []
        num_tokens = []
        for line in text:
            line = (
                [self.bos_token_id]
                + self.data_pool.asr_tokenizer.tokenizer.Encode(line)
                + [self.eos_token_id]
            )
            num_tokens.append(len(line))
            input_ids.append(
                torch.tensor(line, dtype=torch.long).unsqueeze(0).to(self.device)
            )

        neg_log_likelihoods = []
        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                for sentence in input_ids:
                    outputs = self.data_pool.ft_lm_model(
                        input_ids=sentence, labels=sentence
                    )
                    neg_log_likelihoods.append(
                        -sentence.size(1) * outputs.loss.cpu().item()
                    )

        return np.array(neg_log_likelihoods), np.array(num_tokens)

    def get_ngram_lm_score(self, text: List[str]) -> Tuple[np.array, np.array]:
        """Calculate n-gram language model scores for the given text.

        This method calculates n-gram language model scores for the given text. It first
        tokenizes the text using the ASR tokenizer and then computes the score for each
        tokenized sentence using the n-gram language model.

        Args:
            text (List[str]): A list of strings representing sentences.

        Returns:
            Tuple[np.array, np.array]: A tuple containing two NumPy arrays. The first array
            contains the n-gram language model scores for each sentence, and the second array
            contains the number of tokens in each sentence.
        """
        tokenized_text = tokenize_str(
            text, self.data_pool.asr_tokenizer, self.cfg.kenlm.offset
        )
        tokens_num = [len(item) for item in tokenized_text]
        scores = []

        for tokenized_sentence in tokenized_text:
            sentence = " ".join(tokenized_sentence)
            sentence_score = self.data_pool.ngram_lm.score(sentence) * math.log(10)
            scores.append(sentence_score)

        return np.array(scores), np.array(tokens_num)

    def ilstm_inference(self, data_source: List[str]) -> Tuple[np.array, np.array]:
        """Calculate LSTM scores for the given text.

        This method calculates LSTM scores on the given
        data source. It tokenizes each input string, computes the log probabilities using
        the LSTM model, and returns the log probabilities for each sentence along with the
        number of tokens in each sentence.

        Args:
            data_source (List[str]): A list of strings representing input data.

        Returns:
            Tuple[np.array, np.array]: A tuple containing two NumPy arrays. The first array
            contains the log probabilities for each sentence, and the second array contains
            the number of tokens in each sentence.
        """
        tokenized_text = []
        ilstm_tokens_num = []
        for line in data_source:
            line = lstm_tokenize_str(
                self.data_pool.lstm_tokenizer, "<bos> " + line + " <eos>"
            )
            ilstm_tokens_num.append(len(line))
            tokenized_text.append(
                torch.unsqueeze(torch.LongTensor(line), dim=0)
                .t()
                .contiguous()
                .to(self.device)
            )
        sentence_probs = []
        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                for src in tokenized_text:
                    hidden = self.data_pool.lstm.init_hidden(1)
                    hidden = [item.to(self.device) for item in hidden]
                    log_probs, hidden = self.data_pool.lstm(src, hidden)
                    log_gen_probs = (
                        torch.gather(log_probs[:-1, :], 1, src[1:, :])
                        .squeeze(-1)
                        .cpu()
                        .numpy()
                    )
                    sentence_probs.append(log_gen_probs.sum())
        return np.array(sentence_probs), np.array(ilstm_tokens_num)

    @staticmethod
    def calculate_wer(seqs_hat: List[str], seqs_true: List[str]) -> Optional[float]:
        """Calculate sentence-level WER score.

        :param list seqs_hat: prediction
        :param list seqs_true: reference
        :return: average sentence-level WER score
        :rtype float

        Note:
        Author: Johns Hopkins University (Shinji Watanabe) (2017)
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
