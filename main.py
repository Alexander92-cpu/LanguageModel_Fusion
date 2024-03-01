import logging
import os
from pathlib import Path
import pickle
import random
from typing import Dict

from datasets import load_dataset
from git import Repo
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

from rnnt_lm_fusion import (
    DataPool,
    LMPool,
    Optimizator,
    Rescore,
    RescoreOutput,
    TextDataset
)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def set_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class Workflow:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.datasets = {}
        self.eval_dataloaders = None
        self.eval_data_pool = None
        self.lm_pool = None

    def download_data(self) -> None:
        repo_url = self.cfg.root_params.data_url
        clone_dir = self.cfg.root_params.data_dir
        if not Path(clone_dir).exists():
            Path(clone_dir).mkdir(exist_ok=True, parents=True)
            Repo.clone_from(url=repo_url, to_path=clone_dir, branch='main')

    def get_data(self) -> None:
        self.lm_pool = LMPool(self.cfg)
        self.lm_pool.set_asr_tokenizer()
        self.get_data_librispeech_asr()

    def train_gpt2(self) -> None:
        self.lm_pool.set_gpt2_model()
        self.lm_pool.set_gtp2_arguments()
        trainer = self.lm_pool.set_gpt2_trainer(self.datasets['train'], self.datasets['validation'])
        self.lm_pool.fine_tuning_gpt2(trainer)

    def evaluate_gpt2(self) -> None:
        self.lm_pool.load_ft_gpt2(self.cfg.gpt2.dir_model)

        trainer = self.lm_pool.set_gpt2_trainer(None, self.datasets['test_other'])
        baseline = self.lm_pool.test_gpt2(trainer)
        logging.info('Fine_tuning_other: %s', baseline)

        trainer = self.lm_pool.set_gpt2_trainer(None, self.datasets['test_clean'])
        baseline = self.lm_pool.test_gpt2(trainer)
        logging.info('Fine_tuning_clean: %s', baseline)

    def get_data_librispeech_asr(self) -> None:
        def build_data(lm_dir_data: str, suffix: str) -> Dict[str, str]:
            target_dataset = load_dataset("librispeech_asr", suffix)
            paths_to_data = {}
            for key in target_dataset:
                paths_to_data[key] = f'{lm_dir_data}/{suffix}_{key}.txt'
                if not Path(paths_to_data[key]).exists():
                    Path(paths_to_data[key]).mkdir(exist_ok=True, parents=True)
                    with open(paths_to_data[key], 'wt', encoding='utf-8') as fo:
                        for line in target_dataset[key]['text']:
                            fo.write(line.lower() + '\n')
            return paths_to_data

        paths_to_data = {}
        for suffix in ['other', 'clean']:
            paths_to_data[suffix] = build_data(self.cfg.lm.dir_data, suffix)

        tpaths_to_data = {}
        tpaths_to_data['train'] = [paths_to_data['other']['train.500'],
                                   paths_to_data['clean']['train.360']]
        tpaths_to_data['validation'] = [paths_to_data['other']['validation'],
                                        paths_to_data['clean']['validation']]
        tpaths_to_data['test_other'] = [paths_to_data['other']['test']]
        tpaths_to_data['test_clean'] = [paths_to_data['clean']['test']]

        for key, value in tpaths_to_data.items():
            self.datasets[key] = TextDataset(self.lm_pool.asr_tokenizer,
                                             self.lm_pool.bos_token_id,
                                             self.lm_pool.eos_token_id,
                                             value, self.cfg.lm.block_size)

    def train_lstm(self) -> None:
        data_types = {'train': ['train'], 'validation': ['validation'],
                      'test': ['test_other', 'test_clean']}
        for key, value in data_types.items():
            data_path = Path(self.cfg.lstm.dir_lstm_data) / (key + '.txt')
            if not data_path.exists():
                raw_text = [text for v in value for text in self.datasets[v].raw_text]
                with open(data_path, 'wt', encoding='utf-8') as fo:
                    for line in raw_text:
                        fo.write(line + '\n')
        self.lm_pool.train_lstm()

    def train_ngram(self) -> None:
        raw_text = self.datasets['train'].raw_text + self.datasets['validation'].raw_text
        if not os.path.isfile(self.cfg.kenlm.train_file):
            with open(self.cfg.kenlm.train_file, 'wt', encoding='utf-8') as fo:
                for line in raw_text:
                    fo.write(line + '\n')
        self.lm_pool.train_ngram()

    def load_eval_data(self) -> None:
        data = {'validation_other': load_dataset("librispeech_asr", "other", split='validation'),
                'validation_clean': load_dataset("librispeech_asr", "clean", split='validation'),
                "test_other": load_dataset("librispeech_asr", "other", split='test'),
                "test_clean": load_dataset("librispeech_asr", "clean", split='test')}
        eval_data_pool = DataPool(self.cfg)
        eval_data_pool.get_asr_tokenizer()
        eval_data_pool.get_lms()
        eval_data_pool.get_asr_model()
        self.eval_dataloaders = eval_data_pool.get_data(data)
        self.eval_data_pool = eval_data_pool

    def load_optimization_data(self) -> None:
        data = {'validation_other': load_dataset("librispeech_asr", "other", split='validation'),
                'validation_clean': load_dataset("librispeech_asr", "clean", split='validation')}
        eval_data_pool = DataPool(self.cfg)
        eval_data_pool.get_asr_tokenizer()
        eval_data_pool.get_lms()
        eval_data_pool.get_asr_model()
        self.eval_dataloaders = eval_data_pool.get_data(data)
        self.eval_data_pool = eval_data_pool

    def optimize_hyperparams(self, data) -> None:
        optimizator = Optimizator(self.cfg, data)
        optimizator.optimize()

    def rescore(self) -> Dict[str, RescoreOutput]:
        rescorer = Rescore(self.cfg, self.eval_data_pool)
        rescorer_results = {}
        for key, data in self.eval_dataloaders.items():
            rescorer_results[key] = rescorer.eval_rnnt(data)
            logging.info("%s: %s", key, rescorer_results[key]['wer'])
        return rescorer_results


def train(cfg: DictConfig) -> None:
    wf = Workflow(cfg)

    wf.download_data()
    wf.get_data()

    wf.train_ngram()

    wf.train_lstm()

    wf.train_gpt2()
    wf.evaluate_gpt2()


def optimize(cfg: DictConfig) -> None:
    wf = Workflow(cfg)

    wf.download_data()
    data_file = Path(cfg.optimize.optimize_data_file)

    if not data_file.exists():
        wf.load_optimization_data()
        info = wf.rescore()
        Path(data_file.parent).mkdir(parents=True, exist_ok=True)
        with open(data_file, 'wb') as file:
            pickle.dump(info, file)
    with open(data_file, 'rb') as file:
        info = pickle.load(file)
    wf.optimize_hyperparams(info)


def test(cfg: DictConfig) -> None:
    wf = Workflow(cfg)

    wf.download_data()
    wf.load_eval_data()
    wf.rescore()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    cfg.root_params.log_dir = hydra_cfg.run.dir
    OmegaConf.resolve(cfg)

    set_seed(cfg.root_params.seed)

    train(cfg)
    optimize(cfg)
    test(cfg)


if __name__ == '__main__':
    main()
