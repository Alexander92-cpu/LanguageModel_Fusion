"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.



This script would train an N-gram language model with KenLM library
(https://github.com/kpu/kenlm) which can be used
with the beam search decoders on top of the ASR models. This script supports
both character level and BPE level
encodings and models which is detected automatically from the type of the model.
After the N-gram model is trained, and stored in the binary format, you may use
'scripts/ngram_lm/eval_beamsearch_ngram.py' to evaluate it on an ASR model.
The code:
https://github.com/NVIDIA/NeMo/tree/main/scripts/asr_language_modeling/ngram_lm
You need to install the KenLM library and also the beam search decoders to use this feature.
Please refer to 'scripts/ngram_lm/install_beamsearch_decoders.sh' on how to install them.

USAGE: python train_kenlm.py --nemo_model_file <path to the .nemo file of the model> \
                             --train_file <path to the training text or JSON manifest file \
                             --kenlm_bin_path <path to the bin folder of KenLM library> \
                             --kenlm_model_file <path to store the binary KenLM model> \
                             --ngram_length <order of N-gram model>
"""

import logging
import os
import subprocess
import sys

import nemo.collections.asr as nemo_asr
import torch
from omegaconf import DictConfig

from .utils import TokenizeConfig, read_train_file, tokenize_text

TOKEN_OFFSET = 100

CHUNK_SIZE = 8192
CHUNK_BUFFER_SIZE = 512


class Ngram:
    """
    NeMo's beam search decoders only support char-level encodings.
    In order to make it work with BPE-level encodings, we
    use a trick to encode the sub-word tokens of the training data as unicode characters
    and train a char-level KenLM.
    TOKEN_OFFSET is the offset in the unicode table to be used to encode the BPE sub-words.
    This encoding scheme reduces
    the required memory significantly, and the LM and its binary blob format require less
    storage space.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg.kenlm
        self.train_file = cfg.kenlm.train_file
        self.nemo_model_file = cfg.asr_model.model
        self.kenlm_model_file = cfg.kenlm.model
        self.ngram_length = cfg.kenlm.ngram
        self.kenlm_bin_path = cfg.kenlm.kenlm_bin_path

        logging.info("Loading nemo model %s ...", self.nemo_model_file)
        if self.nemo_model_file.endswith(".nemo"):
            self.model = nemo_asr.models.ASRModel.restore_from(
                self.nemo_model_file, map_location=torch.device("cpu")
            )
        else:
            logging.warning(
                """nemo_model_file does not end with .nemo, therefore trying to load
                a pretrained model with this name."""
            )
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                self.nemo_model_file, map_location=torch.device("cpu")
            )

    def train(self):
        """DATASET SETUP"""
        logging.info("Encoding the train file %s ...", self.train_file)
        dataset = read_train_file(self.train_file, lowercase=self.cfg.do_lowercase)
        encoded_train_file = f"{self.kenlm_model_file}.tmp.txt"

        tokenizer_config = TokenizeConfig(
            chunk_size=CHUNK_SIZE,
            buffer_size=CHUNK_BUFFER_SIZE,
            token_offset=TOKEN_OFFSET,
        )
        tokenize_text(
            dataset,
            self.model.tokenizer,
            path=encoded_train_file,
            config=tokenizer_config,
        )
        # --discount_fallback is needed for training KenLM for BPE-based models
        discount_arg = "--discount_fallback"

        arpa_file = f"{self.kenlm_model_file}.tmp.arpa"
        kenlm_args = [
            os.path.join(self.kenlm_bin_path, "lmplz"),
            "-o",
            f"{self.ngram_length}",
            "--text",
            encoded_train_file,
            "--arpa",
            arpa_file,
            discount_arg,
        ]

        ret = subprocess.run(
            kenlm_args,
            capture_output=False,
            text=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=True,
        )
        if ret.returncode != 0:
            raise RuntimeError("Training KenLM was not successful!")
        logging.info("Running binary_build command \n\n{%s}\n\n", " ".join(kenlm_args))
        kenlm_args = [
            os.path.join(self.kenlm_bin_path, "build_binary"),
            "trie",
            arpa_file,
            self.kenlm_model_file,
        ]
        ret = subprocess.run(
            kenlm_args,
            capture_output=False,
            text=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=True,
        )

        if ret.returncode != 0:
            raise RuntimeError("Training KenLM was not successful!")

        if self.cfg.remove_temp_files:
            self.remove_temp_files(arpa_file, encoded_train_file)

    def remove_temp_files(self, arpa_file: str, encoded_train_file: str):
        """
        Delete all temp files created during training process

        Args:
            arpa_file (str): temp arpa model
            encoded_train_file (str): tokenized train text file
        """
        os.remove(encoded_train_file)
        logging.info(
            "Deleted the temporary encoded training file %s.", encoded_train_file
        )
        os.remove(arpa_file)
        logging.info("Deleted the arpa file %s.", arpa_file)
