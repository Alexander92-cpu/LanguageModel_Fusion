"""
This module provides project's usage examples.

Author: Alexandru Mazurenco (2024)
License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import pickle
from pathlib import Path

import hydra
import torch
from datasets import load_dataset
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from rnnt_lm_fusion import Workflow, set_seed

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def train(cfg: DictConfig) -> None:
    """Train models using the specified configuration.

    This function initializes a workflow object with the provided configuration
    settings, downloads necessary data, trains N-gram, LSTM, and GPT-2 models,
    and evaluates the GPT-2 model.

    Args:
        cfg (DictConfig): A dictionary-like object containing configuration settings.
    """
    wf = Workflow(cfg)

    wf.download_data()
    wf.get_data()

    wf.train_ngram()

    wf.train_lstm()

    wf.train_gpt2()
    baseline = wf.evaluate_gpt2()
    logging.info("Fine_tuning: %s", baseline)


def optimize(cfg: DictConfig) -> None:
    """Find the best parameters for rescoring on validation datasets.

    This function initializes a workflow object with the provided configuration
    settings, downloads necessary data, loads optimization data, rescoring, and
    optimizes hyperparameters based on the obtained information.

    Args:
        cfg (DictConfig): A dictionary-like object containing configuration settings.
    """
    wf = Workflow(cfg)

    wf.download_data()
    data_file = Path(cfg.optimize.optimize_data_file)

    if not data_file.exists():
        data = {
            "validation_other": load_dataset(
                "librispeech_asr", "other", split="validation"
            ),
            "validation_clean": load_dataset(
                "librispeech_asr", "clean", split="validation"
            ),
        }
        wf.get_eval_data_pool(data)
        info = wf.rescore()
        Path(data_file.parent).mkdir(parents=True, exist_ok=True)
        with open(data_file, "wb") as file:
            pickle.dump(info, file)
    with open(data_file, "rb") as file:
        info = pickle.load(file)
    wf.optimize_hyperparams(info)


def test(cfg: DictConfig) -> None:
    """Test different rescoring techniques using trained models.

    This function initializes a workflow object with the provided configuration
    settings, downloads necessary data, loads evaluation data, and performs
    rescoring using trained models.

    Args:
        cfg (DictConfig): A dictionary-like object containing configuration settings.
    """
    wf = Workflow(cfg)

    wf.download_data()

    data = {
        "validation_other": load_dataset(
            "librispeech_asr", "other", split="validation"
        ),
        "validation_clean": load_dataset(
            "librispeech_asr", "clean", split="validation"
        ),
        "test_other": load_dataset("librispeech_asr", "other", split="test"),
        "test_clean": load_dataset("librispeech_asr", "clean", split="test"),
    }

    wf.get_eval_data_pool(data)

    wf.rescore()


def main(config_path: str, config_name: str) -> None:
    """This function contains basic calls for training, optimization, and testing.

    Args:
        config_path (str): Path to the directory containing configuration files.
        config_name (str): Name of the configuration file.
    """
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name, return_hydra_config=True)
        HydraConfig().cfg = cfg
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        cfg.root_params.log_dir = hydra_cfg.run.dir
        OmegaConf.resolve(cfg)

    set_seed(cfg.root_params.seed)

    train(cfg)
    optimize(cfg)
    test(cfg)


if __name__ == "__main__":
    main(config_path="conf", config_name="config")
