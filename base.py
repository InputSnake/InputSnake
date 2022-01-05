import math
from argparse import ArgumentParser, Namespace
from typing import Any, Union
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.utilities.data import to_categorical
from pytorch_lightning.utilities.argparse import add_argparse_args, from_argparse_args

from transformers import AdamW


class DiagBase(LightningModule):

    def training_step(self, batch, batch_index):
        x, y = batch
        preds = self(x)
        loss = F.binary_cross_entropy(preds, y)
        if hasattr(self, "fs"):
            reg = torch.mean(self.fs.regularizer((self.fs.mu + 0.5) / 1.0))
            loss = loss + 0.1 * reg
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        self.val_hamming(preds, y.int())
        self.log("score", self.val_hamming, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        self.test_hamming(preds, y.int())
        self.log("HAMMING", self.test_hamming, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = AdamW(params, lr=self.lr)
        return optimizer

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        """Extends existing argparse by default `LightningDataModule` attributes."""
        return add_argparse_args(cls, parent_parser, **kwargs)

    @classmethod
    def from_argparse_args(cls, args: Union[Namespace, ArgumentParser], **kwargs):
        """Create an instance from CLI arguments.

        Args:
            args: The parser or namespace to take arguments from. Only known arguments will be
                parsed and passed to the :class:`~pytorch_lightning.core.datamodule.LightningDataModule`.
            **kwargs: Additional keyword arguments that may override ones in the parser or namespace.
                These must be valid DataModule arguments.

        Example::

            parser = ArgumentParser(add_help=False)
            parser = LightningDataModule.add_argparse_args(parser)
            module = LightningDataModule.from_argparse_args(args)
        """
        return from_argparse_args(cls, args, **kwargs)


class LosBase(DiagBase):

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = to_categorical(y)
        preds = self(x)
        self.val_kappa(preds, y)
        self.log("score", self.val_kappa, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = to_categorical(y)
        preds = self(x)
        self.test_kappa(preds, y)
        self.log("KAPPA", self.test_kappa, on_step=False, on_epoch=True)


class MortBase(DiagBase):

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        self.val_auroc(preds, y.int())
        self.log("score", self.val_auroc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        self.test_auroc(preds, y.int())
        self.test_aupr(preds, y)
        self.log("AUROC", self.test_auroc, on_step=False, on_epoch=True)
        self.log("AUPR", self.test_aupr, on_step=False, on_epoch=True)


class Iteration1Base(LosBase):

    def training_step(self, batch, batch_index):
        x, y = batch
        preds = self(x, batch_index)
        loss = F.binary_cross_entropy(preds, y[1])
        if hasattr(self, "fs"):
            reg = torch.mean(self.fs.regularizer((self.fs.mu + 0.5) / 1.0))
            loss = loss + 0.1 * reg
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = to_categorical(y[1])
        preds = self(x, batch_idx)
        self.val_kappa(preds, y)
        self.log("score", self.val_kappa, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = to_categorical(y[1])
        preds = self(x, batch_idx)
        self.test_kappa(preds, y)
        self.log("KAPPA", self.test_kappa, on_step=False, on_epoch=True)


class IterationBase(DiagBase):

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x, batch_idx)
        diag_loss = F.binary_cross_entropy(preds[0], y[0])
        los_loss = F.binary_cross_entropy(preds[1], y[1])
        los_reg = torch.mean(self.los.fs.regularizer((self.los.fs.mu + 0.5) / 1.0))
        los_loss = los_loss + 0.1 * los_reg
        mort_loss = F.binary_cross_entropy(preds[2], y[2])
        mort_reg = torch.mean(self.mort.fs.regularizer((self.mort.fs.mu + 0.5) / 1.0))
        mort_loss = mort_loss + 0.1 * mort_reg
        loss = diag_loss + los_loss + mort_loss
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x, batch_idx)
        self.diag.val_hamming(preds[0], y[0].int())
        self.log("diag_score", self.diag.val_hamming, on_step=False, on_epoch=True)
        self.los.val_kappa(preds[1], to_categorical(y[1]))
        self.log("los_score", self.los.val_kappa, on_step=False, on_epoch=True)
        self.mort.val_auroc(preds[2], y[2].int())
        self.log("score", self.mort.val_auroc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x, batch_idx)
        self.diag.test_hamming(preds[0], y[0].int())
        self.log("HAMMING", self.diag.test_hamming, on_step=False, on_epoch=True)
        self.los.test_kappa(preds[1], to_categorical(y[1]))
        self.log("KAPPA", self.los.test_kappa, on_step=False, on_epoch=True)
        self.mort.test_auroc(preds[2], y[2].int())
        self.log("AUROC", self.mort.test_auroc, on_step=False, on_epoch=True)


class FeatureSelector(LightningModule):
    def __init__(self, input_dim, sigma):
        super(FeatureSelector, self).__init__()
        self.mu = torch.nn.Parameter(0.01 * torch.randn(input_dim, ), requires_grad=True)
        self.noise = torch.randn(self.mu.size())
        self.sigma = sigma

    def forward(self, prev_x):
        z = self.mu + self.sigma * self.noise.normal_() * self.training
        stochastic_gate = self.hard_sigmoid(z)
        new_x = prev_x * stochastic_gate
        return new_x

    @staticmethod
    def hard_sigmoid(x):
        return torch.clamp(x + 0.5, 0.0, 1.0)

    @staticmethod
    def regularizer(x):
        """ Gaussian CDF. """
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

    def _apply(self, fn):
        super(FeatureSelector, self)._apply(fn)
        self.noise = fn(self.noise)
        return self


class Cnn(nn.Module):
    """
    Multilayer CNN with 1D convolutions
    Input: (batch_size, length, input_size)
    Output: (batch_size, n_neurons)
    """

    def __init__(self,
                 input_size: int = 7508,
                 length: int = 48,
                 depth: int = 2,
                 filter_size: int = 3,
                 n_filters: int = 64,
                 n_neurons: int = 64,
                 dropout: float = 0.2,
                 activation: str = "relu") -> None:
        super().__init__()
        self.depth = depth
        padding = int(np.floor(filter_size / 2))
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu
        if depth == 1:
            self.conv1 = nn.Conv1d(input_size, n_filters, filter_size, padding=padding)
            self.pool1 = nn.MaxPool1d(2, 2)
            self.fc1 = nn.Linear(int(length * n_filters / 2), n_neurons)
            self.fc1_drop = nn.Dropout(dropout)

        elif depth == 2:
            self.conv1 = nn.Conv1d(input_size, n_filters, filter_size, padding=padding)
            self.pool1 = nn.MaxPool1d(2, 2)
            self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size, padding=padding)
            self.pool2 = nn.MaxPool1d(2, 2)
            self.fc1 = nn.Linear(int(length * n_filters / 4), n_neurons)
            self.fc1_drop = nn.Dropout(dropout)

        elif depth == 3:
            self.conv1 = nn.Conv1d(input_size, n_filters, filter_size, padding=padding)
            self.pool1 = nn.MaxPool1d(2, 2)
            self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size, padding=padding)
            self.pool2 = nn.MaxPool1d(2, 2)
            self.conv3 = nn.Conv1d(n_filters, n_filters, filter_size, padding=padding)
            self.pool3 = nn.MaxPool1d(2, 2)
            self.fc1 = nn.Linear(int(length * n_filters / 8), n_neurons)
            self.fc1_drop = nn.Dropout(dropout)

    def forward(self, x):

        x = x.transpose(1, 2)

        x = self.pool1(self.activation(self.conv1(x)))
        if self.depth == 2 or self.depth == 3:
            x = self.pool2(self.activation(self.conv2(x)))
        if self.depth == 3:
            x = self.pool3(self.activation(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1_drop(self.fc1(x)))
        return x
