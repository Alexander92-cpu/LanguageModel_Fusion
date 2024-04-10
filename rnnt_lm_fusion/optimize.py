"""
Module for hyperparameter optimization in rescoring.

This module contains the `Optimizator` class, which is responsible for optimizing
hyperparameters used in the rescoring process. It uses Optuna for hyperparameter optimization.

Author: Alexandru Mazurenco (2024)
License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""

from pathlib import Path
from typing import Dict, List

import optuna
from omegaconf import DictConfig
from optuna.trial import Trial

from .rescore import Rescore, RescoreOutput


class Optimizator:
    """
    Class for optimizing hyperparameters used in rescoring.

    Args:
        cfg (DictConfig): Configuration containing optimization settings.
        data_pool (RescoreOutput): Rescore output data.

    Attributes:
        cfg (DictConfig): Configuration containing optimization settings.
        data (List[dict]): List of dictionaries containing rescore data.
    """

    def __init__(self, cfg: DictConfig, data_pool: RescoreOutput) -> None:
        self.cfg = cfg
        self.prepare_data(data_pool)

    def prepare_data(self, data_pool: RescoreOutput) -> None:
        """
        Prepares rescore data for optimization.

        Args:
            data_pool (RescoreOutput): Rescore output data.
        """
        self.data = []
        for _, info in data_pool.items():
            for batch in info["outputs"]:
                dict_batch = {i: {} for i in range(len(batch["utexts"]))}
                for score_param, score_values in batch["scores"].items():
                    for idx, score_value in enumerate(score_values):
                        dict_batch[idx][score_param] = score_value
                if isinstance(batch["reference"], str):
                    batch["reference"] = [batch["reference"]] * len(batch["utexts"])
                for idx, (tokens_num_value, reference) in enumerate(zip(batch["utexts"], batch["reference"])):
                    dict_batch[idx]["transcription"] = tokens_num_value
                    dict_batch[idx]["reference"] = reference
                self.data.append(list(dict_batch.values()))

    def optimize(self) -> None:
        """
        Optimizes hyperparameters for rescoring.
        """
        sampler = optuna.samplers.TPESampler(multivariate=True, group=True)

        for study_name, values in self.cfg.rescore.params.items():
            if values:
                db_dir = Path(self.cfg.optimize.db_exp).parent
                db_dir.mkdir(parents=True, exist_ok=True)
                storage_path = f"sqlite:///{self.cfg.optimize.db_exp}"
                sampler = optuna.samplers.TPESampler(multivariate=True, group=True)
                study = optuna.create_study(
                    storage=storage_path,
                    study_name=study_name,
                    sampler=sampler,
                    direction="minimize",
                    load_if_exists=self.cfg.optimize.load_if_exists,
                    pruner=optuna.pruners.MedianPruner(
                        n_startup_trials=2, n_warmup_steps=5, interval_steps=3
                    ),
                )
                study.optimize(
                    lambda trial, study_name=study_name, values_keys=list(
                        values.keys()
                    ): get_best_hyperparams(
                        trial,
                        self.data,
                        values_keys,
                        self.cfg.optimize.bounds[study_name],
                        self.cfg.optimize.step,
                    ),
                    gc_after_trial=True,
                    n_trials=self.cfg.optimize.n_trials,
                    n_jobs=self.cfg.optimize.n_jobs,
                )


def get_best_hyperparams(
    trial: Trial,
    data: List[dict],
    params: List[str],
    bounds: Dict[str, List[int]],
    step: float,
) -> float:
    """
    Function to obtain the best hyperparameters.

    Args:
        trial (Trial): Optuna trial.
        data (List[dict]): List of dictionaries containing rescore data.
        params (List[str]): List of parameter names.
        bounds (Dict[str, List[int]]): Dictionary of parameter bounds.
        step (float): Step size for parameter suggestion.

    Returns:
        float: Word error rate (WER) after applying the suggested hyperparameters.
    """
    generated_params = {}
    for param in params:
        generated_params[param] = trial.suggest_float(
            param, bounds[param][0], bounds[param][1], step=step
        )
    transcriptions = []
    references = []
    for nbests in data:
        scores = []
        for sample in nbests:
            rescore = sample["asr_scores"]
            for param in params:
                rescore += generated_params[param] * sample[param]
            scores.append(rescore)
        st_rescore = scores.index(max(scores))
        transcriptions.append(nbests[st_rescore]["transcription"])
        references.append(nbests[st_rescore]["reference"])
    wer = Rescore.calculate_wer(transcriptions, references)
    if wer is None:
        wer = 100.0

    return round(wer, 4)
