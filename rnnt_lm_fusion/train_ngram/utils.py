# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Utility methods to be used for training N-gram LM with KenLM in train_kenlm.py
"""

import json
import logging
import os
from typing import List

from joblib import Parallel, delayed
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
from tqdm.auto import tqdm


def read_train_file(path: str, lowercase: bool = False):
    lines_read = 0
    text_dataset = []

    with open(path, 'rt', encoding='utf-8') as f:
        reader = tqdm(iter(f.readline, ''), desc="Read 0 lines", unit=' lines')
        for line in reader:
            if path.endswith('.json'):
                line = json.loads(line)['text']

            line = line.replace("\n", "").strip()
            if lowercase:
                line = line.lower()

            if line:
                text_dataset.append(line)

                lines_read += 1
                if lines_read % 100000 == 0:
                    reader.set_description(f"Read {lines_read} lines")

    return text_dataset


def tokenize_str(texts: List[str], tokenizer: SentencePieceTokenizer, offset: int):
    tokenized_text = []
    for text in texts:
        tok_text = tokenizer.text_to_ids(text)
        tok_text = [chr(token + offset) for token in tok_text]
        tokenized_text.append(tok_text)
    return tokenized_text


def tokenize_text(
        data: List[str],
        tokenizer: SentencePieceTokenizer,
        path: str,
        chunk_size: int = 8192,
        buffer_size: int = 32,
        token_offset: int = 100
    ):
    dataset_len = len(data)
    logging.info("Chunking %i rows into %.4f tasks (each chunk contains %i elements)",
                 dataset_len, dataset_len / float(chunk_size), chunk_size)

    current_step = 0
    if os.path.exists(path):
        logging.info("Deleting previous file : %s", path)
        os.remove(path)

    with Parallel(n_jobs=-2, verbose=10) as parallel:
        while True:
            start = current_step * chunk_size
            end = min((current_step + buffer_size) * chunk_size, dataset_len)

            tokenized_data = parallel(
                delayed(tokenize_str)(data[start : start + chunk_size], tokenizer, token_offset)
                for start in range(start, end, chunk_size)
            )

            # Write dataset
            write_dataset(tokenized_data, path)
            current_step += len(tokenized_data)
            logging.info("Finished writing %i chunks to %s. Current chunk index = %i",
                         len(tokenized_data), path, current_step)
            del tokenized_data
            if end >= dataset_len:
                break


def write_dataset(chunks: List[List[str]], path: str):
    basedir = os.path.dirname(path)

    if not os.path.exists(basedir):
        os.makedirs(basedir, exist_ok=True)

    with open(path, 'at+', encoding='utf-8') as f:
        for chunk_idx in tqdm(range(len(chunks)), desc='Chunk ', total=len(chunks), unit=' chunks'):
            for text in chunks[chunk_idx]:
                line = ' '.join(text)
                f.write(f"{line}\n")
