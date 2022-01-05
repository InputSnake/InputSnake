import random
from typing import Any

import torch
from torch import nn
import torchmetrics
import metrics
import base


class DiagCnn(base.DiagBase):
    def __init__(self,
                 input_size: int = 7508,
                 length: int = 48,
                 depth: int = 2,
                 filter_size: int = 3,
                 n_filters: int = 64,
                 n_neurons: int = 128,
                 output_size: int = 25,
                 dropout: float = 0.0,
                 lr: float = 1e-4,
                 activation: str = "relu",
                 feature_selector: bool = False,
                 **kwargs: Any) -> None:
        super(DiagCnn, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        if feature_selector:
            self.fs = base.FeatureSelector(input_size, 1.0)
        self.enc = base.Cnn(input_size=input_size, length=length, depth=depth, filter_size=filter_size,
                            n_filters=n_filters, n_neurons=n_neurons, dropout=dropout, activation=activation)
        self.pred = nn.Linear(n_neurons, output_size)

        self.val_hamming = torchmetrics.HammingDistance(compute_on_step=False)
        self.test_hamming = torchmetrics.HammingDistance(compute_on_step=False)

    def forward(self, x):
        if hasattr(self, "fs"):
            x = self.fs(x)
        output = self.enc(x)
        return torch.sigmoid(self.pred(output)).squeeze(1)


class LosCnn(base.LosBase):
    def __init__(self,
                 input_size: int = 7508,
                 length: int = 48,
                 depth: int = 2,
                 filter_size: int = 4,
                 n_filters: int = 128,
                 n_neurons: int = 32,
                 output_size: int = 9,
                 dropout: float = 0.4,
                 lr: float = 1e-4,
                 activation: str = "relu",
                 feature_selector: bool = False,
                 **kwargs: Any) -> None:
        super(LosCnn, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        if feature_selector:
            self.fs = base.FeatureSelector(input_size, 1.0)
        self.enc = base.Cnn(input_size=input_size, length=length, depth=depth, filter_size=filter_size,
                            n_filters=n_filters, n_neurons=n_neurons, dropout=dropout, activation=activation)
        self.pred = nn.Linear(n_neurons, output_size)

        self.val_kappa = torchmetrics.CohenKappa(num_classes=9, weights="linear", compute_on_step=False)
        self.test_kappa = torchmetrics.CohenKappa(num_classes=9, weights="linear", compute_on_step=False)

    def forward(self, x):
        if hasattr(self, "fs"):
            x = self.fs(x)
        output = self.enc(x)
        return torch.softmax(self.pred(output), dim=-1)


class MortCnn(base.MortBase):
    def __init__(self,
                 input_size: int = 7508,
                 length: int = 48,
                 depth: int = 1,
                 filter_size: int = 3,
                 n_filters: int = 64,
                 n_neurons: int = 64,
                 output_size: int = 1,
                 dropout: float = 0.2,
                 lr: float = 1e-4,
                 activation: str = "relu",
                 feature_selector: bool = False,
                 **kwargs: Any) -> None:
        super(MortCnn, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        if feature_selector:
            self.fs = base.FeatureSelector(input_size, 1.0)
        self.enc = base.Cnn(input_size=input_size, length=length, depth=depth, filter_size=filter_size,
                            n_filters=n_filters, n_neurons=n_neurons, dropout=dropout, activation=activation)
        self.pred = nn.Linear(n_neurons, output_size)

        self.val_auroc = torchmetrics.AUROC(pos_label=1)
        self.test_auroc = torchmetrics.AUROC(pos_label=1)
        self.test_aupr = metrics.AUPR()

    def forward(self, x):
        if hasattr(self, "fs"):
            x = self.fs(x)
        output = self.enc(x)
        return torch.sigmoid(self.pred(output)).squeeze(1)


class LosInputSnake(base.LosBase):
    def __init__(self,
                 input_size: int = 7508,
                 length: int = 48,
                 depth: int = 2,
                 filter_size: int = 4,
                 n_filters: int = 128,
                 n_neurons: int = 32,
                 output_size: int = 9,
                 dropout: float = 0.4,
                 lr: float = 1e-4,
                 activation: str = "relu",
                 feature_selector: bool = False,
                 diag_ckpt: str = "/path/to/the/diag_ckpt",
                 **kwargs: Any) -> None:
        super(LosInputSnake, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        if feature_selector:
            self.fs = base.FeatureSelector(input_size + 25, 1.0)
        self.diag = DiagCnn.load_from_checkpoint(diag_ckpt)
        self.diag.freeze()
        self.enc = base.Cnn(input_size=input_size + 25, length=length, depth=depth, filter_size=filter_size,
                            n_filters=n_filters, n_neurons=n_neurons, dropout=dropout, activation=activation)
        self.pred = nn.Linear(n_neurons, output_size)

        self.val_kappa = torchmetrics.CohenKappa(num_classes=9, weights="linear", compute_on_step=False)
        self.test_kappa = torchmetrics.CohenKappa(num_classes=9, weights="linear", compute_on_step=False)

    def forward(self, x):
        diag_pred = self.diag(x)
        diag_addon = diag_pred.unsqueeze(1).repeat(1, x.size(1), 1)
        x = torch.cat([diag_addon, x], dim=-1)
        if hasattr(self, "fs"):
            x = self.fs(x)
        output = self.enc(x)
        return torch.softmax(self.pred(output), dim=-1)


class MortInputSnake(base.MortBase):
    def __init__(self,
                 input_size: int = 7508,
                 length: int = 48,
                 depth: int = 1,
                 filter_size: int = 3,
                 n_filters: int = 64,
                 n_neurons: int = 64,
                 output_size: int = 1,
                 dropout: float = 0.2,
                 lr: float = 1e-4,
                 activation: str = "relu",
                 feature_selector: bool = False,
                 diag_ckpt: str = "/path/to/the/diag_ckpt",
                 los_ckpt: str = "/path/to/the/los_ckpt",
                 **kwargs: Any) -> None:
        super(MortInputSnake, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        if feature_selector:
            self.fs = base.FeatureSelector(input_size + 34, 1.0)
        self.diag = DiagCnn.load_from_checkpoint(diag_ckpt)
        self.los = LosInputSnake.load_from_checkpoint(los_ckpt)
        self.diag.freeze()
        self.los.freeze()
        self.enc = base.Cnn(input_size=input_size + 34, length=length, depth=depth, filter_size=filter_size,
                            n_filters=n_filters, n_neurons=n_neurons, dropout=dropout, activation=activation)
        self.pred = nn.Linear(n_neurons, output_size)

        self.val_auroc = torchmetrics.AUROC(pos_label=1)
        self.test_auroc = torchmetrics.AUROC(pos_label=1)
        self.test_aupr = metrics.AUPR()

    def forward(self, x):
        diag_pred = self.diag(x)
        diag_addon = diag_pred.unsqueeze(1).repeat(1, x.size(1), 1)
        los_pred = self.los(x)
        los_addon = los_pred.unsqueeze(1).repeat(1, x.size(1), 1)
        x = torch.cat([diag_addon, los_addon, x], dim=-1)
        if hasattr(self, "fs"):
            x = self.fs(x)
        output = self.enc(x)
        return torch.sigmoid(self.pred(output)).squeeze(1)


class OtherStrategies(base.IterationBase):

    def __init__(self,
                 input_size: int = 7508,
                 length: int = 48,
                 diag_depth: int = 2,
                 diag_filter_size: int = 3,
                 diag_n_filters: int = 64,
                 diag_n_neurons: int = 128,
                 diag_dropout: float = 0.0,
                 diag_activation: str = "relu",
                 los_depth: int = 2,
                 los_filter_size: int = 4,
                 los_n_filters: int = 128,
                 los_n_neurons: int = 32,
                 los_dropout: float = 0.4,
                 los_activation: str = "relu",
                 mort_depth: int = 1,
                 mort_filter_size: int = 3,
                 mort_n_filters: int = 64,
                 mort_n_neurons: int = 64,
                 mort_dropout: float = 0.2,
                 mort_activation: str = "relu",
                 slope: float = 0.001,
                 lr: float = 1e-4,
                 feature_selector: bool = True,
                 diag_output_size: int = 25,
                 los_output_size: int = 9,
                 mort_output_size: int = 1,
                 **kwargs: Any) -> None:
        super(OtherStrategies, self).__init__()
        self.lr = lr
        self.slope = slope
        self.diag = DiagCnn(input_size=input_size, length=length, depth=diag_depth, filter_size=diag_filter_size,
                            n_filters=diag_n_filters, n_neurons=diag_n_neurons, output_size=diag_output_size,
                            dropout=diag_dropout, activation=diag_activation, feature_selector=False)
        self.los = LosCnn(input_size=input_size + diag_output_size, length=length, depth=los_depth,
                          filter_size=los_filter_size,
                          n_filters=los_n_filters, n_neurons=los_n_neurons, output_size=los_output_size,
                          dropout=los_dropout, activation=los_activation, feature_selector=feature_selector)
        self.mort = MortCnn(input_size=input_size + diag_output_size + los_output_size, length=length, depth=mort_depth,
                            filter_size=mort_filter_size, n_filters=mort_n_filters, n_neurons=mort_n_neurons,
                            output_size=mort_output_size, dropout=mort_dropout, activation=mort_activation,
                            feature_selector=feature_selector)

    def forward(self, inputs, batch_idx):
        x, y = inputs
        diag_pred = self.diag(x)
        if self.training:
            p = max(0.01, 1.0 - self.slope * batch_idx)
            use_true_diag = random.choices([0, 1], weights=[1 - p, p], k=1)[0]
            if use_true_diag:
                diag_addon = y[0].unsqueeze(1).repeat(1, x.size(1), 1)
            else:
                diag_addon = diag_pred.unsqueeze(1).repeat(1, x.size(1), 1)
            los_inp = torch.cat([diag_addon, x], dim=-1)
            los_pred = self.los(los_inp)

            use_true_los = random.choices([0, 1], weights=[1 - p, p], k=1)[0]
            if use_true_los:
                los_addon = y[1].unsqueeze(1).repeat(1, x.size(1), 1)
            else:
                los_addon = los_pred.unsqueeze(1).repeat(1, x.size(1), 1)
            mort_inp = torch.cat([diag_addon, los_addon, x], dim=-1)
            mort_pred = self.mort(mort_inp)
        else:
            diag_inp = diag_pred.unsqueeze(1).repeat(1, x.size(1), 1)
            los_inp = torch.cat([diag_inp, x], dim=-1)
            los_pred = self.los(los_inp)
            los_inp = los_pred.unsqueeze(1).repeat(1, x.size(1), 1)
            mort_inp = torch.cat([diag_inp, los_inp, x], dim=-1)
            mort_pred = self.mort(mort_inp)
        return diag_pred, los_pred, mort_pred