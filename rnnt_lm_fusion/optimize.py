from typing import Dict, List

from omegaconf import DictConfig
import optuna

from .rescore import Rescore, RescoreOutput


class Optimizator:
    def __init__(
            self,
            cfg: DictConfig,
            data_pool: RescoreOutput
        ) -> None:
        self.cfg = cfg
        self.prepare_data(data_pool)

    def prepare_data(self, data_pool: RescoreOutput) -> None:
        self.data = []
        for _, info in data_pool.items():
            for batch in info['outputs']:
                dict_batch = {i: {} for i in range(len(batch['utexts']))}
                for score_param, score_values in batch['scores'].items():
                    for idx, score_value in enumerate(score_values):
                        dict_batch[idx][score_param] = score_value
                for idx, tokens_num_value in enumerate(batch['utexts']):
                    dict_batch[idx]['transcription'] = tokens_num_value
                    dict_batch[idx]['reference'] = batch['reference']
                self.data.append(list(dict_batch.values()))

    def optimize(self) -> None:
        sampler = optuna.samplers.TPESampler(multivariate=True, group=True)

        for study_name, values in self.cfg.rescore.params.items():
            if values:
                storage_path = f"sqlite:///{self.cfg.optimize.db_exp}"
                sampler = optuna.samplers.TPESampler(multivariate=True, group=True)
                study = optuna.create_study(storage=storage_path,
                                            study_name=study_name,
                                            sampler=sampler,
                                            direction='minimize',
                                            load_if_exists=self.cfg.optimize.load_if_exists,
                                            pruner=optuna.pruners.MedianPruner(
                                                n_startup_trials=2, n_warmup_steps=5,
                                                interval_steps=3
                                            ))
                study.optimize(lambda trial, study_name=study_name, values_keys=list(values.keys()):
                               get_best_hyperparams(trial,
                                                    self.data,
                                                    values_keys,
                                                    self.cfg.optimize.bounds[study_name],
                                                    self.cfg.optimize.step),
                               gc_after_trial=True,
                               n_trials=self.cfg.optimize.n_trials,
                               n_jobs=self.cfg.optimize.n_jobs)


def get_best_hyperparams(
        trial: optuna.trial._trial.Trial,
        data: List[dict],
        params: List[str],
        bounds: Dict[str, List[int]],
        step: float
    ) -> float:
    generated_params = {}
    for param in params:
        generated_params[param] = trial.suggest_float(param, bounds[param][0],
                                                      bounds[param][1], step=step)
    transcriptions = []
    references = []
    for nbests in data:
        scores = []
        for sample in nbests:
            rescore = sample['asr_scores']
            for param in params:
                rescore += generated_params[param] * sample[param]
            scores.append(rescore)
        st_rescore = scores.index(max(scores))
        transcriptions.append(nbests[st_rescore]['transcription'])
        references.append(nbests[0]['reference'])
    wer = Rescore.calculate_wer(transcriptions, references)
    if wer is None:
        wer = 100.0

    return round(wer, 4)
