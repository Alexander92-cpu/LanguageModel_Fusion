"""
This module provides functionalities for language model training and evaluation.

It includes classes and functions for managing various components required
for training/evaluation/testing GPT-2, N-gram and LSTM language models, and
preparing datasets for language model training.

Author: Alexandru Mazurenco (2024)
License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""

import copy
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import nemo.collections.asr as nemo_asr
import torch
from omegaconf import DictConfig
from sentencepiece import SentencePieceProcessor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments

from .train_ngram.train import Ngram
from .word_language_model.main import WLM


class LMPool:
    """
    A class to manage various components required for language model pooling.

    Attributes:
        cfg (DictConfig): Configuration settings.
        asr_tokenizer (SentencePieceProcessor): ASR tokenizer.
        bos_token_id (int): ID of the beginning of sentence token.
        eos_token_id (int): ID of the end of sentence token.
        pad_token_id (int): ID of the padding token.
        fusion_lm_model (GPT2LMHeadModel): Fusion language model.
        training_args (TrainingArguments): Training arguments for GPT-2.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.asr_tokenizer = None
        self.bos_token_id = None
        self.eos_token_id = None
        self.pad_token_id = None
        self.fusion_lm_model = None
        self.training_args = None

    def set_asr_tokenizer(self) -> None:
        """
        Sets the ASR tokenizer and related token IDs.
        """
        device = self.cfg.root_params.device
        asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(
            self.cfg.asr_model.model, map_location=device
        )
        self.asr_tokenizer = asr_model.tokenizer.tokenizer
        self.bos_token_id = len(self.asr_tokenizer)
        self.eos_token_id = len(self.asr_tokenizer) + 1
        self.pad_token_id = len(self.asr_tokenizer) + 2

    def set_gpt2_model(self) -> None:
        """
        Sets up the GPT-2 language model.
        """
        lm_model = GPT2LMHeadModel.from_pretrained(self.cfg.gpt2.model_name)

        lm_model_config = lm_model.config
        lm_model_config.bos_token_id = self.bos_token_id
        lm_model_config.eos_token_id = self.eos_token_id
        lm_model_config.pad_token_id = self.pad_token_id
        lm_model_config.vocab_size = len(self.asr_tokenizer) + 3

        fusion_lm_model = GPT2LMHeadModel(lm_model_config)

        sd_lm = lm_model.state_dict()
        sd_flm: Dict[str, torch.tensor] = dict(fusion_lm_model.state_dict())
        filtered_sd_lm = {}
        for k, v in sd_lm.items():
            if k in sd_flm and v.size() == sd_flm[k].size():
                filtered_sd_lm[k] = v
            else:
                logging.warning("%s is absent!", k)
        sd_flm.update(filtered_sd_lm)
        fusion_lm_model.load_state_dict(sd_flm)
        self.fusion_lm_model = fusion_lm_model

    def set_gtp2_arguments(self) -> None:
        """
        Sets up the training arguments for GPT-2.
        """
        self.training_args = TrainingArguments(
            output_dir=self.cfg.gpt2.output_dir,
            per_device_train_batch_size=self.cfg.gpt2.per_device_train_batch_size,
            per_device_eval_batch_size=self.cfg.gpt2.per_device_eval_batch_size,
            evaluation_strategy=self.cfg.gpt2.evaluation_strategy,
            save_strategy=self.cfg.gpt2.save_strategy,
            save_total_limit=self.cfg.gpt2.save_total_limit,
            logging_steps=self.cfg.gpt2.logging_steps,
            gradient_accumulation_steps=self.cfg.gpt2.gradient_accumulation_steps,
            num_train_epochs=self.cfg.gpt2.num_train_epochs,
            weight_decay=self.cfg.gpt2.weight_decay,
            warmup_steps=self.cfg.gpt2.warmup_steps,
            lr_scheduler_type=self.cfg.gpt2.lr_scheduler_type,
            learning_rate=self.cfg.gpt2.learning_rate,
            fp16=self.cfg.gpt2.fp16,
            logging_dir=self.cfg.gpt2.logging_dir,
            load_best_model_at_end=self.cfg.gpt2.load_best_model_at_end,
            report_to=self.cfg.gpt2.report_to,
        )

    def set_gpt2_trainer(
        self, train_dataset: Dataset, eval_dataset: Dataset
    ) -> Trainer:
        """
        Sets up the trainer for GPT-2 fine-tuning.

        Args:
            train_dataset (Dataset): Training dataset.
            eval_dataset (Dataset): Evaluation dataset.

        Returns:
            Trainer: GPT-2 trainer.
        """
        trainer = Trainer(
            model=self.fusion_lm_model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=lambda x: collate(
                x, self.fusion_lm_model.config.pad_token_id
            ),
        )
        return trainer

    @staticmethod
    def test_gpt2(trainer: Trainer) -> Dict[str, float]:
        """
        Tests the performance of the fine-tuned GPT-2 model.

        Args:
            trainer (Trainer): GPT-2 trainer.

        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics.
        """
        baseline = {}
        baseline["eval_loss"] = trainer.evaluate()["eval_loss"]
        baseline["perplexity"] = torch.exp(torch.tensor(baseline["eval_loss"])).item()
        return baseline

    def fine_tuning_gpt2(
        self, trainer: Trainer, checkpoint_path: Optional[Path] = None
    ) -> None:
        """
        Fine-tunes the GPT-2 model.

        Args:
            trainer (Trainer): GPT-2 trainer.
            checkpoint_path (Optional[Path], optional): Path to checkpoint.
        """
        trainer.train(checkpoint_path)
        trainer.save_model(self.cfg.gpt2.dir_model)

    def load_ft_gpt2(self, model_dir_save: Path) -> None:
        """
        Loads a fine-tuned GPT-2 model.

        Args:
            model_dir_save (Path): Directory containing the saved model.
        """
        self.fusion_lm_model = GPT2LMHeadModel.from_pretrained(model_dir_save)

    def train_ngram(self) -> None:
        """
        Trains the N-gram language model.
        """
        trainer = Ngram(self.cfg)
        trainer.train()

    def train_lstm(self) -> None:
        """
        Trains the LSTM language model.
        """
        wlm = WLM(self.cfg)
        wlm.run()


@dataclass
class TextDatasetConfig:
    """
    Configuration settings for the TextDataset.

    Attributes:
        tokenizer (SentencePieceProcessor): Tokenizer for text encoding.
        bos_id (int): ID of the beginning of sentence token.
        eos_id (int): ID of the end of sentence token.
        file_paths (List[str]): List of file paths containing text data.
        block_size (int): Size of the block for text encoding.
    """

    tokenizer: SentencePieceProcessor
    bos_id: int
    eos_id: int
    file_paths: List[str]
    block_size: int


class TextDataset(Dataset):
    """
    Dataset class for text data.

    Attributes:
        examples (List[torch.Tensor]): List of encoded text examples.
        raw_text (List[str]): List of raw text data.
    """

    def __init__(self, config: TextDatasetConfig) -> None:
        self.examples = []
        self.raw_text = []
        for file_path in config.file_paths:
            if os.path.isfile(file_path) is False:
                raise ValueError(f"Input file path {file_path} not found")

            text = [
                word.strip()
                for word in Path(file_path).read_text(encoding="utf-8").splitlines()
                if word.strip()
            ]
            self.raw_text.extend(text)

            tokenized_text = []
            for line in tqdm(text, desc="Encode text", leave=False):
                # s = tokenizer.Encode(line, out_type=str)
                line = [config.bos_id] + config.tokenizer.Encode(line) + [config.eos_id]
                tokenized_text += line

            for i in range(
                0, len(tokenized_text) - config.block_size + 1, config.block_size
            ):
                self.examples.append(tokenized_text[i : i + config.block_size])
            # if i + block_size < len(tokenized_text):
            #     self.examples.append(tokenized_text[i + block_size:])
        self.examples = [torch.tensor(e, dtype=torch.long) for e in self.examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.examples[i]


def collate(input_ids: List[torch.Tensor], pad_id: int) -> Dict[str, torch.Tensor]:
    """
    Collates a batch of input tensors for training a language model.

    This function takes a list of input tensors, pads them to the same length,
    creates attention masks, and prepares labels for the input tensors.

    Args:
        input_ids (List[torch.Tensor]): A list of input tensors containing token IDs.
        pad_id (int): The ID of the padding token.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the following keys:
            - 'input_ids': A tensor of padded input token IDs.
            - 'attention_mask': A tensor of attention masks.
            - 'labels': A tensor of padded labels for the input tokens.
    """
    attention_mask = [
        torch.tensor([1] * (item.size(0)), dtype=torch.long) for item in input_ids
    ]
    attention_mask = pad_sequence(attention_mask, padding_value=0).permute(1, 0)
    labels = copy.deepcopy(input_ids)
    labels = pad_sequence(input_ids, padding_value=-100).permute(1, 0)
    input_ids = pad_sequence(input_ids, padding_value=pad_id).permute(1, 0)
    data = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    return data
