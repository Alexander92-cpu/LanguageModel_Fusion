"""
This module contains classes and functions for managing various data
components for ASR and language models evaluation.

Author: Alexandru Mazurenco (2024)
License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""

import copy
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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

from .word_language_model.data import Dictionary
from .word_language_model.model import RNNModel, RNNModelConfig


@dataclass
class LMModels:
    """A data class representing language models and associated components.

    Attributes:
        ft_lm_model (Optional[GPT2LMHeadModel]): Fine-tuned GPT2 language model.
        ngram_lm (Optional[kenlm.LanguageModel]): N-gram language model.
        lstm_tokenizer (Optional[Dictionary]): Tokenizer for LSTM model.
        lstm (Optional[RNNModel]): LSTM language model.
    """

    ft_lm_model: GPT2LMHeadModel = None
    ngram_lm: kenlm.LanguageModel = None
    lstm_tokenizer: Dictionary = None
    lstm: RNNModel = None


class DataPool:
    """
    A class to manage various data components for ASR and language model training/evaluation.

    Attributes:
        cfg (DictConfig): Configuration settings.
        asr_model: ASR model.
        asr_tokenizer: ASR tokenizer.
        start_tokens: Start tokens for tokenization.
        ft_lm_model: Fine-tuned language model.
        ngram_lm: N-gram language model.
        lstm_tokenizer: LSTM tokenizer.
        lstm: LSTM model.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.asr_model = None
        self.asr_tokenizer = None
        self.start_tokens = None
        self.lm_models = LMModels()

    def get_asr_model(self) -> None:
        """
        Retrieves the ASR model and sets up decoding strategy.
        """
        asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(
            self.cfg.asr_model.model, map_location=self.cfg.root_params.device
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
        """
        Retrieves and preprocesses audio data.

        Args:
            data (Dict[str, Dataset]): Dictionary containing audio datasets.

        Returns:
            Dict[str, DataLoader]: Dictionary containing audio dataloaders.
        """
        ap = AudioPool(self.asr_model.cfg, self.cfg.root_params.eval_do_lowercase)
        datasets_ = ap.get_datasets(data)
        return ap.get_dataloaders(datasets_)

    def get_asr_tokenizer(self) -> None:
        """
        Retrieves the ASR tokenizer.
        """
        asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(
            self.cfg.asr_model.model, map_location=self.cfg.root_params.device
        )
        self.asr_tokenizer = asr_model.tokenizer
        start_token = self.cfg.tokenizer.start_token
        self.start_tokens = {
            i
            for i in range(len(self.asr_tokenizer.tokenizer))
            if start_token in self.asr_tokenizer.tokenizer.id_to_piece([i])[0]
        }

    def get_lms(self) -> None:
        """
        Retrieves and initializes GPT-2, N-gram, and LSTM language models.
        """
        ft_lm_model = GPT2LMHeadModel.from_pretrained(self.cfg.gpt2.dir_model)
        self.lm_models.ft_lm_model = ft_lm_model.to(self.cfg.root_params.device)

        self.lm_models.ngram_lm = kenlm.LanguageModel(self.cfg.kenlm.model)

        with open(self.cfg.lstm.tokenizer_path, "rb") as file:
            self.lm_models.lstm_tokenizer = pickle.load(file)
        self.lm_models.lstm = self.get_lstm(self.cfg.lstm)

    @staticmethod
    def get_lstm(cfg: DictConfig) -> RNNModel:
        """
        Initializes and loads the LSTM language model.

        Args:
            cfg (DictConfig): LSTM configuration settings.

        Returns:
            RNNModel: Initialized LSTM model.
        """
        config = RNNModelConfig(
            rnn_type=cfg.model_type,
            ntoken=cfg.num_words,
            ninp=cfg.emsize,
            nhid=cfg.nhid,
            nlayers=cfg.nlayers,
            dropout=cfg.dropout,
            tie_weights=cfg.tied,
        )
        model = RNNModel(config)
        model.load_state_dict(torch.load(cfg.save))
        model.rnn.flatten_parameters()
        model = model.to(cfg.device)
        model.eval()
        return model


class AudioPool:
    """
    A class to manage audio data for ASR model training.

    Attributes:
        asr_model_config (DictConfig): ASR model configuration.
        do_lowercase (bool): Flag indicating whether to convert text to lowercase.
    """

    def __init__(self, asr_model_config: DictConfig, do_lowercase: bool) -> None:
        self.asr_model_config = asr_model_config
        self.do_lowercase = do_lowercase

    def get_datasets(
        self, data: Dict[str, datasets.arrow_dataset.Dataset]
    ) -> Dict[str, Dict[str, Union[torch.tensor, str]]]:
        """
        Retrieves and preprocesses audio datasets.

        Args:
            data (Dict[str, datasets.arrow_dataset.Dataset]): Dictionary containing audio datasets.

        Returns:
            Dict[str, Dict[str, Union[torch.tensor, str]]]: Dictionary containing
            preprocessed audio data.
        """
        return {
            key: self.get_data(dataset["file"], dataset["text"])
            for key, dataset in data.items()
        }

    def get_data(
        self, audio_paths: List[str], references: List[str]
    ) -> Dict[str, Union[torch.tensor, str]]:
        """
        Retrieves and preprocesses audio data.

        Args:
            audio_paths (List[str]): List of audio file paths.
            references (List[str]): List of corresponding text references.

        Returns:
            Dict[str, Union[torch.tensor, str]]: Dictionary containing preprocessed audio data.
        """
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
        """
        Processes individual audio files.

        Args:
            audio_path (str): Path to the audio file.
            text (str): Text reference corresponding to the audio file.

        Returns:
            Tuple[torch.tensor, str]: Tuple containing the audio tensor and text reference.
        """
        audio_path = list(Path(audio_path).parent.rglob(Path(audio_path).name))
        assert len(audio_path) == 1
        audio_tensor = self.read_audio(audio_path[0], self.asr_model_config.sample_rate)
        if self.do_lowercase:
            text = text.lower()
        return audio_tensor, text

    def get_dataloaders(
        self, datasets_: Dict[str, Dict[str, Union[torch.tensor, str]]]
    ) -> Dict[str, DataLoader]:
        """
        Creates dataloaders for audio datasets.

        Args:
            datasets_ (Dict[str, Dict[str, Union[torch.tensor, str]]]):
            Dictionary containing preprocessed audio data.

        Returns:
            Dict[str, DataLoader]: Dictionary containing audio dataloaders.
        """
        return {
            key: self.create_dataloader(data["audio"], data["text"])
            for key, data in datasets_.items()
        }

    @staticmethod
    def read_audio(path: str, sample_rate: int) -> torch.tensor:
        """
        Reads an audio file from the specified path and returns it as a tensor.

        Args:
            path (str): The path to the audio file.
            sample_rate (int): The desired sample rate for the audio.

        Returns:
            torch.tensor: The audio waveform as a tensor.

        Raises:
            OSError: If there is an error reading the audio file.
            ValueError: If the audio is not in mono format or if the sample rate
                        of the audio does not match the specified sample_rate.
        """
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
        """
        Creates a DataLoader for the provided audio samples and their corresponding texts.

        Args:
            audio (List[torch.tensor]): A list of audio waveforms as tensors.
            texts (List[str]): A list of texts corresponding to the audio samples.

        Returns:
            DataLoader: A DataLoader object containing the audio-text pairs.

        Note:
            This function assumes that `self.asr_model_config.validation_ds.batch_size`
            is a valid batch size.

        """
        dataset = AudioDataset(audio, texts)
        dataloader = DataLoader(
            dataset,
            batch_size=self.asr_model_config.validation_ds.batch_size,
            shuffle=False,
            collate_fn=collate,
        )
        return dataloader


class AudioDataset(Dataset):
    """
    Dataset class for handling audio data paired with corresponding references.

    Args:
        audio (List[torch.Tensor]): A list of torch tensors containing audio data.
        references (List[str]): A list of strings containing references corresponding
                                to the audio data.

    Attributes:
        audio (List[torch.Tensor]): A list of torch tensors containing audio data.
        references (List[str]): A list of strings containing references corresponding
                                to the audio data.
    """

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
    """
    Collates a batch of audio samples and their corresponding texts.

    Args:
        batch (Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]): A tuple containing
            lists of input signals, input signal lengths, and labels.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, List[str]]: A tuple containing the padded input signals,
            input signal lengths, and labels.

    """
    input_signals, input_signal_lengths, labels = zip(*batch)
    return (
        pad_sequence(input_signals).permute(1, 0),
        torch.stack(input_signal_lengths),
        labels,
    )
