"""
Description: This module provides a PyTorch implementation of an RNN-based language model.
https://github.com/pytorch/examples/blob/main/word_language_model/model.py
"""

from dataclasses import dataclass
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class RNNModelConfig:
    """
    Configuration class for RNNModel.

    Args:
        rnn_type (str): Type of RNN architecture. Options: 'LSTM', 'GRU', 'RNN_TANH', or 'RNN_RELU'.
        ntoken (int): Number of tokens in the vocabulary.
        ninp (int): Size of each input vector.
        nhid (int): Number of hidden units per layer.
        nlayers (int): Number of layers.
        dropout (float, optional): Dropout probability. Defaults to 0.5.
        tie_weights (bool, optional): Whether to tie input and output embeddings.
    """

    rnn_type: str
    ntoken: int
    ninp: int
    nhid: int
    nlayers: int
    dropout: float = 0.5
    tie_weights: bool = False


class RNNModel(nn.Module):
    """
    Container module with an encoder, a recurrent module, and a decoder.

    Args:
        rnn_type (str): Type of RNN architecture. Options: 'LSTM',
                                                  'GRU', 'RNN_TANH', or 'RNN_RELU'.
        ntoken (int): Number of tokens in the vocabulary.
        ninp (int): Size of each input vector.
        nhid (int): Number of hidden units per layer.
        nlayers (int): Number of layers.
        dropout (float, optional): Dropout probability. Defaults to 0.5.
        tie_weights (bool, optional): Whether to tie input and output embeddings.
    """

    def __init__(self, config: "RNNModelConfig") -> None:
        super().__init__()
        self.config = config
        self.drop = nn.Dropout(config.dropout)
        self.encoder = nn.Embedding(config.ntoken, config.ninp)
        if config.rnn_type in ["LSTM", "GRU"]:
            self.rnn = getattr(nn, config.rnn_type)(
                config.ninp, config.nhid, config.nlayers, dropout=config.dropout
            )
        else:
            try:
                nonlinearity = {"RNN_TANH": "tanh", "RNN_RELU": "relu"}[config.rnn_type]
            except KeyError as e:
                raise ValueError(
                    """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']"""
                ) from e
            self.rnn = nn.RNN(
                config.ninp,
                config.nhid,
                config.nlayers,
                nonlinearity=nonlinearity,
                dropout=config.dropout,
            )
        self.decoder = nn.Linear(config.nhid, config.ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling"
        # (Inan et al. 2016) https://arxiv.org/abs/1611.01462
        if config.tie_weights:
            if config.nhid != config.ninp:
                raise ValueError(
                    "When using the tied flag, nhid must be equal to emsize"
                )
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.nhid = config.nhid
        self.nlayers = config.nlayers

    def init_weights(self) -> None:
        """
        Initialize weights for the encoder and decoder.
        """
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(
        self, tokens_ids: torch.tensor, hidden: Tuple[torch.tensor, torch.tensor]
    ) -> Tuple[torch.tensor, Tuple[torch.tensor, torch.tensor]]:
        """
        Forward pass of the model.

        Args:
            tokens_ids (torch.tensor): Input token IDs.
            hidden (Tuple[torch.tensor, torch.tensor]): Initial hidden state.

        Returns:
            Tuple[torch.tensor, Tuple[torch.tensor, torch.tensor]]:
            Log probabilities and hidden state.
        """
        emb = self.drop(self.encoder(tokens_ids))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.config.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(
        self, bsz: int
    ) -> Union[Tuple[torch.tensor, torch.tensor], torch.tensor]:
        """
        Initialize the hidden state.

        Args:
            bsz (int): Batch size.

        Returns:
            Union[Tuple[torch.tensor, torch.tensor], torch.tensor]: Initial hidden state.
        """
        weight = next(self.parameters())
        if self.config.rnn_type == "LSTM":
            return (
                weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid),
            )
        return weight.new_zeros(self.nlayers, bsz, self.nhid)
