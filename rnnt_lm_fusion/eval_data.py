import copy
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple, Union

import datasets
import nemo.collections.asr as nemo_asr
import torch
import torchaudio
from omegaconf import DictConfig, open_dict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel

import kenlm

from .word_language_model.model import RNNModel


class DataPool:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = self.cfg.root_params.device
        self.asr_model = None
        self.asr_tokenizer = None
        self.start_tokens = None
        self.ft_lm_model = None
        self.ngram_lm = None
        self.lstm_tokenizer = None
        self.lstm = None

    def get_asr_model(self) -> None:
        asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(
            self.cfg.asr_model.model, map_location=self.device
        )
        asr_model.freeze()
        decoding_config = copy.deepcopy(asr_model.cfg.decoding)
        decoding_config.strategy = self.cfg.asr_model.strategy
        decoding_config.beam.beam_size = self.cfg.asr_model.beam_size
        decoding_config.beam.return_best_hypothesis = False
        with open_dict(decoding_config):
            decoding_config.fused_batch_size = -1
            decoding_config.preserve_alignments = False
        asr_model.change_decoding_strategy(decoding_config)
        asr_model.cfg.validation_ds.batch_size = self.cfg.asr_model.batch_size
        self.asr_model = asr_model

    def get_data(self, data: Dict[str, Dataset]) -> Dict[str, DataLoader]:
        ap = AudioPool(self.asr_model.cfg, self.cfg.root_params.eval_do_lowercase)
        datasets_ = ap.get_datasets(data)
        return ap.get_dataloaders(datasets_)

    def get_asr_tokenizer(self) -> None:
        asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(
            self.cfg.asr_model.model, map_location=self.device
        )
        self.asr_tokenizer = asr_model.tokenizer
        start_token = self.cfg.tokenizer.start_token
        self.start_tokens = {
            i
            for i in range(len(self.asr_tokenizer.tokenizer))
            if start_token in self.asr_tokenizer.tokenizer.id_to_piece([i])[0]
        }

    def get_lms(self) -> None:
        ft_lm_model = GPT2LMHeadModel.from_pretrained(self.cfg.gpt2.dir_model)
        self.ft_lm_model = ft_lm_model.to(self.device)

        self.ngram_lm = kenlm.LanguageModel(self.cfg.kenlm.model)

        with open(self.cfg.lstm.tokenizer_path, "rb") as file:
            self.lstm_tokenizer = pickle.load(file)
        self.lstm = self.get_lstm(self.cfg.lstm)

    @staticmethod
    def get_lstm(cfg: DictConfig) -> RNNModel:
        model = RNNModel(
            cfg.model_type,
            cfg.num_words,
            cfg.emsize,
            cfg.nhid,
            cfg.nlayers,
            cfg.dropout,
            cfg.tied,
        )
        model.load_state_dict(torch.load(cfg.save))
        model.rnn.flatten_parameters()
        model = model.to(cfg.device)
        model.eval()
        return model


class AudioPool:
    def __init__(self, asr_model_config: DictConfig, do_lowercase: bool) -> None:
        self.asr_model_config = asr_model_config
        self.do_lowercase = do_lowercase

    def get_datasets(
        self, data: Dict[str, datasets.arrow_dataset.Dataset]
    ) -> Dict[str, Dict[str, Union[torch.tensor, str]]]:
        return {
            key: self.get_data(dataset["file"], dataset["text"])
            for key, dataset in data.items()
        }

    def get_data(
        self, audio_paths: List[str], references: List[str]
    ) -> Dict[str, Union[torch.tensor, str]]:
        with ThreadPoolExecutor() as executor:
            results = list(
                tqdm(
                    executor.map(self.process_audio, audio_paths, references),
                    total=len(audio_paths),
                    leave=False,
                    desc="Read data",
                )
            )
        data = {"audio": [], "text": []}
        for audio_tensor, reference in results:
            data["audio"].append(audio_tensor)
            data["text"].append(reference)
        return data

    def process_audio(self, audio_path: str, text: str) -> Tuple[torch.tensor, str]:
        audio_path = list(Path(audio_path).parent.rglob(Path(audio_path).name))
        assert len(audio_path) == 1
        audio_tensor = self.read_audio(audio_path[0], self.asr_model_config.sample_rate)
        if self.do_lowercase:
            text = text.lower()
        return audio_tensor, text

    def get_dataloaders(
        self, datasets_: Dict[str, Dict[str, Union[torch.tensor, str]]]
    ) -> Dict[str, DataLoader]:
        return {
            key: self.create_dataloader(data["audio"], data["text"])
            for key, data in datasets_.items()
        }

    @staticmethod
    def read_audio(path: str, sample_rate: int) -> torch.tensor:
        try:
            wav, sr = torchaudio.load(path, normalize=True, channels_first=True)
        except Exception as exc:
            raise OSError(f"Error reading {path}") from exc

        if wav.size()[0] != 1:
            raise ValueError("Consider only mono format for audiofiles!")

        if sr != sample_rate:
            transform = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=sample_rate
            )
            wav = transform(wav)
            sr = sample_rate

        if sr != sample_rate:
            raise ValueError(f"{sr} != {sample_rate}")

        return torch.squeeze(wav, dim=0)

    def create_dataloader(
        self, audio: List[torch.tensor], texts: List[str]
    ) -> DataLoader:
        dataset = AudioDataset(audio, texts)
        dataloader = DataLoader(
            dataset,
            batch_size=self.asr_model_config.validation_ds.batch_size,
            shuffle=False,
            collate_fn=collate,
        )
        return dataloader


class AudioDataset(Dataset):
    def __init__(
        self,
        audio: List[torch.tensor],
        references: List[str],
    ) -> None:
        self.audio = audio
        self.references = references

    def __len__(self) -> int:
        return len(self.audio)

    def __getitem__(self, i: int) -> torch.Tensor:
        audio_tensor = torch.tensor(self.audio[i].size(0), dtype=torch.long)
        return self.audio[i], audio_tensor, self.references[i]


def collate(
    batch: Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    input_signals, input_signal_lengths, labels = zip(*batch)
    return (
        pad_sequence(input_signals).permute(1, 0),
        torch.stack(input_signal_lengths),
        labels,
    )
