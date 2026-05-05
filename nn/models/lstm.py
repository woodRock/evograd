"""LSTM sequence model for classification or regression."""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn

from ...core.tensor import Tensor
from ..layers import Layer


class _LSTMNetModule(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 num_classes: int, bidirectional: bool, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        dirs = 2 if bidirectional else 1
        self.head = nn.Linear(hidden_size * dirs, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_size)  →  use last hidden state for classification
        out, _ = self.lstm(x)
        return self.head(self.dropout(out[:, -1, :]))


class LSTMNet(Layer):
    """
    Stacked LSTM followed by a linear classification head.

    Parameters
    ----------
    input_size   : feature dimension per time step
    hidden_size  : LSTM hidden units
    num_layers   : number of stacked LSTM layers
    num_classes  : output classes (or 1 for regression)
    bidirectional: use bidirectional LSTM
    dropout      : dropout on LSTM outputs (requires num_layers > 1)
    device       : torch device string

    Input:  (B, T, input_size)
    Output: (B, num_classes) logits
    """
    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, num_classes: int = 10,
                 bidirectional: bool = False, dropout: float = 0.3,
                 device: str = "cpu"):
        self._net = _LSTMNetModule(
            input_size, hidden_size, num_layers,
            num_classes, bidirectional, dropout,
        ).to(device)

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(self._net(x.data))

    def parameters(self):
        return list(self._net.parameters())

    def train(self) -> "LSTMNet":
        self._net.train(); return self

    def eval(self) -> "LSTMNet":
        self._net.eval(); return self

    @property
    def model(self) -> nn.Module:
        return self._net
