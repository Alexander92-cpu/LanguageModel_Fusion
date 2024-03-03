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
from dataclasses import dataclass
from typing import List

from joblib import Parallel, delayed
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import (
    SentencePieceTokenizer,
)
from tqdm.auto import tqdm


@dataclass
class TokenizeConfig:
    """
    Configuration class for tokenization.

    Attributes:
        chunk_size (int): The size of each chunk for parallel processing.
        buffer_size (int): The number of chunks to process in parallel.
        token_offset (int): The offset value to add to each token.
    """

    chunk_size: int = 8192
    buffer_size: int = 32
    token_offset: int = 100


def read_train_file(path: str, lowercase: bool = False):
    """
    Read and process a training file.

    Args:
        path (str): The path to the training file.
        lowercase (bool, optional): Whether to convert the text to lowercase. Defaults to False.

    Returns:
        List[str]: A list of processed text data from the file.
    """
    lines_read = 0
    text_dataset = []

    with open(path, "rt", encoding="utf-8") as f:
        reader = tqdm(iter(f.readline, ""), desc="Read 0 lines", unit=" lines")
        for line in reader:
            if path.endswith(".json"):
                line = json.loads(line)["text"]

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
    """
    Tokenize a list of strings.

    Args:
        texts (List[str]): The list of strings to tokenize.
        tokenizer (SentencePieceTokenizer): The tokenizer to use.
        offset (int): The offset value to add to each token.

    Returns:
        List[List[str]]: A list of tokenized strings.
    """
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
    config: TokenizeConfig = TokenizeConfig(),
):
    """
    Tokenize a list of texts and write the tokenized data to a file.

    Args:
        data (List[str]): The list of texts to tokenize.
        tokenizer (SentencePieceTokenizer): The tokenizer to use.
        path (str): The path to write the tokenized data.
        config (TokenizeConfig, optional): Configuration for tokenization.
    """
    dataset_len = len(data)
    logging.info(
        "Chunking %i rows into %.4f tasks (each chunk contains %i elements)",
        dataset_len,
        dataset_len / float(config.chunk_size),
        config.chunk_size,
    )

    current_step = 0
    if os.path.exists(path):
        logging.info("Deleting previous file : %s", path)
        os.remove(path)

    with Parallel(n_jobs=-2, verbose=10) as parallel:
        while True:
            start = current_step * config.chunk_size
            end = min(
                (current_step + config.buffer_size) * config.chunk_size, dataset_len
            )

            tokenized_data = parallel(
                delayed(tokenize_str)(
                    data[start : start + config.chunk_size],
                    tokenizer,
                    config.token_offset,
                )
                for start in range(start, end, config.chunk_size)
            )

            # Write dataset
            write_dataset(tokenized_data, path)
            current_step += len(tokenized_data)
            logging.info(
                "Finished writing %i chunks to %s. Current chunk index = %i",
                len(tokenized_data),
                path,
                current_step,
            )
            del tokenized_data
            if end >= dataset_len:
                break


def write_dataset(chunks: List[List[str]], path: str):
    """
    Write tokenized dataset chunks to a file.

    Args:
        chunks (List[List[str]]): The list of tokenized dataset chunks.
        path (str): The path to write the tokenized dataset.
    """
    basedir = os.path.dirname(path)

    if not os.path.exists(basedir):
        os.makedirs(basedir, exist_ok=True)

    with open(path, "at+", encoding="utf-8") as f:
        for chunk_idx in tqdm(
            range(len(chunks)), desc="Chunk ", total=len(chunks), unit=" chunks"
        ):
            for text in chunks[chunk_idx]:
                line = " ".join(text)
                f.write(f"{line}\n")
