import os
from io import open
from typing import List

import torch


class Dictionary:
    def __init__(self, words_limit: int) -> None:
        self.word2idx = {}
        self.idx2word = []
        self.statistics = {}
        self.words_limit = words_limit

    def add_word(self, word: str) -> int:
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def collect(self, word: str) -> None:
        if word not in self.statistics:
            self.statistics[word] = 1
        else:
            self.statistics[word] += 1

    def limit(self) -> None:
        self.statistics = dict(sorted(self.statistics.items(),
                                      key=lambda item: item[1], reverse=True))
        for idx, key in enumerate(self.statistics):
            if self.words_limit == -1 or idx < self.words_limit:
                self.add_word(key)
            else:
                break

    def __len__(self) -> int:
        return len(self.idx2word)


class Corpus:
    def __init__(self, path: str, words_limit: int) -> None:
        self.dictionary = Dictionary(words_limit)
        self.dictionary.add_word("<unk>")
        self.dictionary.add_word("<bos>")
        self.dictionary.add_word("<eos>")
        self.collect_words(os.path.join(path, 'train.txt'))
        self.collect_words(os.path.join(path, 'validation.txt'))
        self.collect_words(os.path.join(path, 'test.txt'))
        self.dictionary.limit()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'validation.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def collect_words(self, path: str) -> None:
        assert os.path.exists(path)
        with open(path, 'rt', encoding="utf8") as f:
            for line in f:
                for word in line.split():
                    self.dictionary.collect(word)

    def tokenize(self, path: str) -> torch.tensor:
        """Tokenizes a text file."""
        assert os.path.exists(path)

        # Tokenize file content
        with open(path, 'rt', encoding="utf8") as f:
            idss = []
            for line in f:
                words = ['<bos>'] + line.split() + ['<eos>']
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
    ids = []
    for word in sentence.split():
        if word in tokenizer.word2idx:
            ids.append(tokenizer.word2idx[word])
        else:
            ids.append(tokenizer.word2idx["<unk>"])
    return ids
