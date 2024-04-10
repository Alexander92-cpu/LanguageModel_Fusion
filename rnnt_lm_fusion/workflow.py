"""Module containing various functions and classes for workflow management.

This module provides functionality for workflow management, including data
processing, model training, hyperparameter optimization, and evaluation.

Author: Alexandru Mazurenco (2024)
License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from datasets import load_dataset
from git import Repo
from omegaconf import DictConfig
from torch.utils.data import Dataset

from .eval_data import DataPool
from .lm_train import LMPool, TextDataset, TextDatasetConfig
from .optimize import Optimizator
from .rescore import Rescore, RescoreOutput


def set_seed(seed: int):
    """Set the random seed for reproducibility.

    This function sets the random seed to the specified value, allowing for
    reproducible results in random processes.

    Args:
        seed (int): An integer value representing the seed to be set.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class Workflow:
    """Class representing a workflow for data processing and model train/evaluation."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize a Workflow object with the provided configuration settings.

        Args:
            cfg (DictConfig): A dictionary-like object containing configuration settings.
        """
        self.cfg = cfg
        self.datasets = {}
        self.eval_dataloaders = None
        self.eval_data_pool = None
        self.lm_pool = None

    def download_data(self) -> None:
        """Download necessary for experiments data published on HG hub."""
        repo_url = self.cfg.root_params.data_url
        clone_dir = self.cfg.root_params.data_dir
        if not os.listdir(clone_dir):
            Path(clone_dir).mkdir(exist_ok=True, parents=True)
            Repo.clone_from(url=repo_url, to_path=clone_dir, branch="main")
        else:
            logging.info("%s already exists!", clone_dir)

    def get_data(self) -> None:
        """Prepare and retrieve data for experiments.

        This method initializes an LM pool with the configuration settings,
        sets an ASR tokenizer, and retrieves data using the LibriSpeech ASR dataset
        """
        self.lm_pool = LMPool(self.cfg)
        self.lm_pool.set_asr_tokenizer()

        if self.cfg.root_params.dataset == "librispeech":
            self.get_data_librispeech_asr()
        else:
            raise ValueError(f"Dataset {self.cfg.root_params.dataset} is not supported")

    def train_gpt2(self) -> None:
        """Fine-tune the GPT-2 model.

        This method sets up the GPT-2 model and its training parameters using the
        LM pool, then performs fine-tuning of the GPT-2 model using the provided
        training and validation datasets.
        """
        self.lm_pool.set_gpt2_model()
        self.lm_pool.set_gtp2_arguments()
        trainer = self.lm_pool.set_gpt2_trainer(
            self.datasets["train"], self.datasets["validation"]
        )
        self.lm_pool.fine_tuning_gpt2(trainer)

    def evaluate_gpt2(self) -> Dict[str, float]:
        """Perform the evaluation of the GPT-2 model.

        This method evaluates the performance of the GPT-2 model using the specified
        evaluation dataset (test parts of LibriSpeech ASR dataset) and metrics (perplexity).
        """
        self.lm_pool.load_ft_gpt2(self.cfg.gpt2.dir_model)

        trainer = self.lm_pool.set_gpt2_trainer(None, self.datasets["test"])
        return self.lm_pool.test_gpt2(trainer)

    def get_data_librispeech_asr(self) -> None:
        """Load the LibriSpeech ASR datasets."""

        def build_data(lm_dir_data: str, suffix: str) -> Dict[str, str]:
            target_dataset = load_dataset("librispeech_asr", suffix)
            paths_to_data = {}
            Path(Path(lm_dir_data)).mkdir(exist_ok=True, parents=True)
            for key in target_dataset:
                paths_to_data[key] = f"{lm_dir_data}/{suffix}_{key}.txt"
                if not Path(paths_to_data[key]).exists():
                    with open(paths_to_data[key], "wt", encoding="utf-8") as fo:
                        for line in target_dataset[key]["text"]:
                            fo.write(line.lower() + "\n")
            return paths_to_data

        paths_to_data = {}
        datasets = dict(self.cfg.librispeech)
        for dtypes in datasets.values():
            for suffix in dtypes:
                if suffix not in paths_to_data:
                    paths_to_data[suffix] = build_data(self.cfg.lm.dir_data, suffix)

        tpaths_to_data = defaultdict(list)
        for dtype, dnames in datasets.items():
            for suffix, dname in dnames.items():
                tpaths_to_data[dtype].append(paths_to_data[suffix][dname])

        for key, value in tpaths_to_data.items():
            config = TextDatasetConfig(
                tokenizer=self.lm_pool.asr_tokenizer,
                bos_id=self.lm_pool.bos_token_id,
                eos_id=self.lm_pool.eos_token_id,
                file_paths=value,
                block_size=self.cfg.lm.block_size,
            )
            self.datasets[key] = TextDataset(config)

    def train_lstm(self) -> None:
        """Train the LSTM model using specified data types.

        This method prepares data for training by combining raw text data from
        different sources specified in the configuration. It then trains the LSTM
        model using the prepared data.
        """
        Path(Path(self.cfg.lstm.dir_lstm_data)).mkdir(parents=True, exist_ok=True)
        for key, text_dataset in self.datasets.items():
            data_path = Path(self.cfg.lstm.dir_lstm_data) / (key + ".txt")
            if not data_path.exists():
                with open(data_path, "wt", encoding="utf-8") as fo:
                    for line in text_dataset.raw_text:
                        fo.write(line + "\n")
        self.lm_pool.train_lstm()

    def train_ngram(self) -> None:
        """Train an N-gram language model.

        This method combines the raw text from the training and validation datasets,
        generates a training file if it does not already exist, and then trains an
        N-gram language model using the LM pool.
        """
        raw_text = (
            self.datasets["train"].raw_text + self.datasets["validation"].raw_text
        )
        if not os.path.isfile(self.cfg.kenlm.train_file):
            Path(Path(self.cfg.kenlm.train_file).parent).mkdir(parents=True, exist_ok=True)
            with open(self.cfg.kenlm.train_file, "wt", encoding="utf-8") as fo:
                for line in raw_text:
                    fo.write(line + "\n")
        self.lm_pool.train_ngram()

    def get_eval_data_pool(self, data: Dict[str, Dataset]):
        """Get evaluation data pool for the given datasets.

        This method initializes a data pool for evaluation, including acquiring
        necessary components such as ASR tokenizer, language models, and ASR model,
        and then retrieves evaluation dataloaders from the data pool for the provided
        datasets.

        Args:
            data (Dict[str, Dataset]): A dictionary containing datasets for evaluation,
                where keys are identifiers for each dataset and values are corresponding
                Dataset objects.
        """
        eval_data_pool = DataPool(self.cfg)
        eval_data_pool.get_asr_tokenizer()
        eval_data_pool.get_lms()
        eval_data_pool.get_asr_model()
        self.eval_dataloaders = eval_data_pool.get_data(data)
        self.eval_data_pool = eval_data_pool

    def optimize_hyperparams(self, data: Dict[str, RescoreOutput]) -> None:
        """Optimize hyperparameters using the provided data.

        This method initializes an optimizer with the configuration settings and
        provided data, then proceeds to optimize hyperparameters.

        Args:
            data (Dict[str, RescoreOutput]): Data used for optimization.
        """
        optimizator = Optimizator(self.cfg, data)
        optimizator.optimize()

    def rescore(self) -> Dict[str, RescoreOutput]:
        """Rescore evaluation data using the Rescore module.

        This method initializes a Rescore object with the provided configuration
        settings and evaluation data pool, then evaluates the RNN-T model on each
        dataset in the evaluation dataloaders. It logs the Word Error Rate (WER) for
        each dataset and returns the results.

        Returns:
            Dict[str, RescoreOutput]: A dictionary mapping dataset names to RescoreOutput
            objects containing the evaluation results.
        """
        rescorer = Rescore(self.cfg, self.eval_data_pool)
        rescorer_results = {}
        for key, data in self.eval_dataloaders.items():
            rescorer_results[key] = rescorer.eval_rnnt(data)
            logging.info("%s: %s", key, rescorer_results[key]["wer"])
        return rescorer_results
