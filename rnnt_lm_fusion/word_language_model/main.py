# coding: utf-8
import logging
import math
from pathlib import Path
import pickle
import time
from typing import Tuple, Union

from omegaconf import DictConfig
import torch
import torch.onnx
from torch import nn

from .data import Corpus
from .model import RNNModel


class WLM:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg.lstm
        self.corpus = None
        self.criterion = None
        self.model = None

    def run(self):
        # Set the random seed manually for reproducibility.
        torch.manual_seed(self.cfg.seed)
        if torch.cuda.is_available():
            if not self.cfg.cuda:
                print("WARNING: You have a CUDA device, so you should probably run with cuda.")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            if not self.cfg.mps:
                print("WARNING: You have mps device, to enable macOS GPU run with --mps.")

        use_mps = self.cfg.mps and torch.backends.mps.is_available()
        if self.cfg.cuda:
            device = torch.device("cuda")
        elif use_mps:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        ###############################################################################
        # Load data
        ###############################################################################

        self.corpus = Corpus(self.cfg.dir_lstm_data, self.cfg.words_limit)

        if not Path(self.cfg.tokenizer_path).exists():
            with open(self.cfg.tokenizer_path, 'wb') as file:
                pickle.dump(self.corpus.dictionary, file)

        train_data = self.batchify(self.corpus.train, self.cfg.batch_size, device)
        val_data = self.batchify(self.corpus.valid, self.cfg.eval_batch_size, device)
        test_data = self.batchify(self.corpus.test, self.cfg.eval_batch_size, device)

        ###############################################################################
        # Build the model
        ###############################################################################

        ntokens = len(self.corpus.dictionary)
        self.model = RNNModel(self.cfg.model_type, ntokens, self.cfg.emsize, self.cfg.nhid,
                                self.cfg.nlayers, self.cfg.dropout, self.cfg.tied).to(device)

        self.criterion = nn.NLLLoss()

        ###############################################################################
        # Training code
        ###############################################################################

        # Loop over epochs.
        lr = self.cfg.lr
        best_val_loss = None

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            for epoch in range(1, self.cfg.epochs+1):
                epoch_start_time = time.time()
                self.train(epoch, train_data, lr)
                val_loss = self.evaluate(val_data)
                logging.info('-' * 89)
                logging.info('| end of epoch %i | time: %.2fs | valid loss %.2f | '
                             'valid ppl %.2f', epoch, (time.time() - epoch_start_time),
                             val_loss, math.exp(val_loss))
                logging.info('-' * 89)
                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss or val_loss < best_val_loss:
                    torch.save(self.model.state_dict(), self.cfg.save)
                    best_val_loss = val_loss
                else:
                    # Anneal the learning rate if no improvement has been seen in the
                    # validation dataset.
                    lr /= 4.0
        except KeyboardInterrupt:
            logging.info('-' * 89)
            logging.info('Exiting from training early')

        # # Load the best saved model.
        self.model.load_state_dict(torch.load(self.cfg.save))
        self.model = self.model.to(device)
        if self.cfg.model_type in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
            self.model.rnn.flatten_parameters()

        # Run on test data.
        test_loss = self.evaluate(test_data)
        logging.info('=' * 89)
        logging.info('| End of training | test loss %.2f | test ppl %.2f',
                     test_loss, math.exp(test_loss))
        logging.info('=' * 89)

    @staticmethod
    def batchify(data: torch.tensor, bsz: int, device: torch.device) -> torch.tensor:
        # Starting from sequential data, batchify arranges the dataset into columns.
        # For instance, with the alphabet as the sequence and batch size 4, we'd get
        # ┌ a g m s ┐
        # │ b h n t │
        # │ c i o u │
        # │ d j p v │
        # │ e k q w │
        # └ f l r x ┘.
        # These columns are treated as independent by the model, which means that the
        # dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
        # batch processing.
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(device)

    @staticmethod
    def repackage_hidden(
        h: Union[Tuple[torch.tensor, torch.tensor],
                 torch.tensor]
        ) -> Union[Tuple[torch.tensor, torch.tensor],
                   torch.tensor]:
        """Wraps hidden states in new Tensors, to detach them from their history."""

        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(WLM.repackage_hidden(v) for v in h)

    @staticmethod
    def get_batch(source: torch.tensor, i: int, bptt: int) -> Tuple[torch.tensor, torch.tensor]:
        # get_batch subdivides the source data into chunks of length args.bptt.
        # If source is equal to the example output of the batchify function, with
        # a bptt-limit of 2, we'd get the following two Variables for i = 0:
        # ┌ a g m s ┐ ┌ b h n t ┐
        # └ b h n t ┘ └ c i o u ┘
        # Note that despite the name of the function, the subdivison of data is not
        # done along the batch dimension (i.e. dimension 1), since that was handled
        # by the batchify function. The chunks are along dimension 0, corresponding
        # to the seq_len dimension in the LSTM.
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target

    def evaluate(self, data_source: torch.tensor) -> float:
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        total_loss = 0.
        hidden = self.model.init_hidden(self.cfg.eval_batch_size)
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, self.cfg.bptt):
                data, targets = self.get_batch(data_source, i, self.cfg.bptt)
                output, hidden = self.model(data, hidden)
                hidden = self.repackage_hidden(hidden)
                total_loss += len(data) * self.criterion(output, targets).item()
        return total_loss / (len(data_source) - 1)

    def train(self, epoch: int, train_data: torch.tensor, lr: float):
        # Turn on training mode which enables dropout.
        self.model.train()
        total_loss = 0.
        start_time = time.time()
        hidden = self.model.init_hidden(self.cfg.batch_size)
        for batch, i in enumerate(range(0, train_data.size(0) - 1, self.cfg.bptt)):
            data, targets = self.get_batch(train_data, i, self.cfg.bptt)
            # Starting each batch, we detach the hidden state from how it was previously
            # produced. If we didn't, the model would try backpropagating all the way to start
            # of the dataset.
            self.model.zero_grad()
            hidden = self.repackage_hidden(hidden)
            output, hidden = self.model(data, hidden)
            loss = self.criterion(output, targets)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip)
            for p in self.model.parameters():
                p.data.add_(p.grad, alpha=-lr)

            total_loss += loss.item()

            if batch % self.cfg.log_interval == 0 and batch > 0:
                cur_loss = total_loss / self.cfg.log_interval
                elapsed = time.time() - start_time
                logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | '
                             'ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // self.cfg.bptt, lr,
                    elapsed * 1000 / self.cfg.log_interval, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
            if self.cfg.dry_run:
                break
