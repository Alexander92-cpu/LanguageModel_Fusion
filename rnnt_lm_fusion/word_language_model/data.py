"""
This module provides classes for handling text corpora and dictionaries.
https://github.com/pytorch/examples/blob/main/word_language_model/data.py
"""

import os
from io import open
from typing import List

import torch


class Dictionary:
    """
    A class for managing word-to-index mappings and word statistics.

    Args:
        words_limit (int): The maximum number of words to include in the dictionary.

    Attributes:
        word2idx (dict): A dictionary mapping words to their corresponding indices.
        idx2word (list): A list containing words indexed by their corresponding indices.
        statistics (dict): A dictionary containing word frequency statistics.
        words_limit (int): The maximum number of words to include in the dictionary.
    """

    def __init__(self, words_limit: int) -> None:
        self.word2idx = {}
        self.idx2word = []
        self.statistics = {}
        self.words_limit = words_limit

    def add_word(self, word: str) -> int:
        """
        Adds a word to the dictionary if it does not exist already.

        Args:
            word (str): The word to add to the dictionary.

        Returns:
            int: The index assigned to the word in the dictionary.
        """
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def collect(self, word: str) -> None:
        """
        Collects word frequency statistics.

        Args:
            word (str): The word to collect statistics for.
        """
        if word not in self.statistics:
            self.statistics[word] = 1
        else:
            self.statistics[word] += 1

    def limit(self) -> None:
        """
        Limits the dictionary size based on word frequency statistics.
        """
        self.statistics = dict(
            sorted(self.statistics.items(), key=lambda item: item[1], reverse=True)
        )
        for idx, key in enumerate(self.statistics):
            if self.words_limit == -1 or idx < self.words_limit:
                self.add_word(key)
            else:
                break

    def __len__(self) -> int:
        return len(self.idx2word)


class Corpus:
    """
    A class for processing text corpora and creating tokenized datasets.

    Args:
        path (str): The path to the directory containing the text files.
        words_limit (int): The maximum number of words to include in the dictionary.

    Attributes:
        dictionary (Dictionary): An instance of the Dictionary class for managing word mappings.
        train (torch.Tensor): A tokenized dataset for training.
        valid (torch.Tensor): A tokenized dataset for validation.
        test (torch.Tensor): A tokenized dataset for testing.
    """

    def __init__(self, path: str, words_limit: int) -> None:
        self.dictionary = Dictionary(words_limit)
        self.dictionary.add_word("<unk>")
        self.dictionary.add_word("<bos>")
        self.dictionary.add_word("<eos>")
        self.collect_words(os.path.join(path, "train.txt"))
        self.collect_words(os.path.join(path, "validation.txt"))
        self.collect_words(os.path.join(path, "test.txt"))
        self.dictionary.limit()
        self.train = self.tokenize(os.path.join(path, "train.txt"))
        self.valid = self.tokenize(os.path.join(path, "validation.txt"))
        self.test = self.tokenize(os.path.join(path, "test.txt"))

    def collect_words(self, path: str) -> None:
        """
        Collects words from a text file and updates word frequency statistics in the dictionary.

        Args:
            path (str): The path to the text file.
        """
        assert os.path.exists(path)
        with open(path, "rt", encoding="utf8") as f:
            for line in f:
                for word in line.split():
                    self.dictionary.collect(word)

    def tokenize(self, path: str) -> torch.tensor:
        """
        Tokenizes a text file and returns a tensor representing the tokenized content.

        Args:
            path (str): The path to the text file.

        Returns:
            torch.Tensor: A tensor containing token indices representing the tokenized content.
        """
        assert os.path.exists(path)

        # Tokenize file content
        with open(path, "rt", encoding="utf-8") as f:
            idss = []
            for line in f:
                words = ["<bos>"] + line.split() + ["<eos>"]
                ids = []
                for word in words:
                    if word in self.dictionary.word2idx:
                        ids.append(self.dictionary.word2idx[word])
                    else:
                        ids.append(self.dictionary.word2idx["<unk>"])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids


def tokenize_str(tokenizer: Dictionary, sentence: str) -> List[int]:
    """
    Tokenizes a string using a given Dictionary.

    Args:
        tokenizer (Dictionary): The Dictionary used for tokenization.
        sentence (str): The input string to tokenize.

    Returns:
        List[int]: A list of token indices representing the tokenized string.
    """
    ids = []
    for word in sentence.split():
        if word in tokenizer.word2idx:
            ids.append(tokenizer.word2idx[word])
        else:
            ids.append(tokenizer.word2idx["<unk>"])
    return ids
