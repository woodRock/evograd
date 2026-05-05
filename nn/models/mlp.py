"""Multi-Layer Perceptron."""
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn

from ...core.tensor import Tensor
from ..layers import Layer


class _MLPModule(nn.Module):
    def __init__(self, dims: List[int], activation: str = "relu",
                 dropout: float = 0.0, batch_norm: bool = False):
        super().__init__()
        _act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh,
                "sigmoid": nn.Sigmoid, "silu": nn.SiLU, "leaky_relu": nn.LeakyReLU}
        act_cls = _act.get(activation, nn.ReLU)
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(act_cls())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLP(Layer):
    """
    Fully-connected multi-layer perceptron.

    Parameters
    ----------
    dims        : list of layer widths, e.g. [784, 256, 128, 10]
    activation  : hidden-layer activation (relu|gelu|tanh|sigmoid|silu|leaky_relu)
    dropout     : dropout probability on hidden layers (0 = disabled)
    batch_norm  : insert BatchNorm1d after each hidden linear
    device      : torch device string
    """
    def __init__(self, dims: List[int], activation: str = "relu",
                 dropout: float = 0.0, batch_norm: bool = False,
                 device: str = "cpu"):
        self._net = _MLPModule(dims, activation, dropout, batch_norm).to(device)

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(self._net(x.data))

    def parameters(self):
        return list(self._net.parameters())

    def train(self) -> "MLP":
        self._net.train(); return self

    def eval(self) -> "MLP":
        self._net.eval(); return self

    @property
    def model(self) -> nn.Module:
        return self._net
