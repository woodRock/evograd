"""
Stateful layer containers — the bridge between pure functions and
trainable parameters.

Each Layer holds nn.Module parameters internally but exposes a purely
functional `forward(Tensor) -> Tensor` interface.  The Layer itself is
mutated only by the optimiser (external, explicit); the forward pass
never modifies any tensor in place.

Layers compose with >> just like plain functions:
    net = Linear(784, 256) >> ReLU() >> Linear(256, 10)
    output = net(x)
"""
from __future__ import annotations
import math
from typing import Optional, List

import torch
import torch.nn as nn

from ..core.tensor import Tensor
from . import functional as F


class Layer:
    """Base class. Subclass and implement forward()."""

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __rshift__(self, other: "Layer") -> "Sequential":
        return Sequential(self, other)

    def parameters(self) -> List[torch.nn.Parameter]:
        return []

    def train(self) -> "Layer":
        return self

    def eval(self) -> "Layer":
        return self


class Sequential(Layer):
    def __init__(self, *layers: Layer):
        self._layers: List[Layer] = []
        for l in layers:
            if isinstance(l, Sequential):
                self._layers.extend(l._layers)
            else:
                self._layers.append(l)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self._layers:
            x = layer(x)
        return x

    def __rshift__(self, other: Layer) -> "Sequential":
        return Sequential(*self._layers, other)

    def parameters(self) -> List[torch.nn.Parameter]:
        params = []
        for l in self._layers:
            params.extend(l.parameters())
        return params

    def train(self) -> "Sequential":
        for l in self._layers:
            l.train()
        return self

    def eval(self) -> "Sequential":
        for l in self._layers:
            l.eval()
        return self


# ------------------------------------------------------------------ #
# Core layers
# ------------------------------------------------------------------ #

class Linear(Layer):
    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, device: str = "cpu"):
        self._mod = nn.Linear(in_features, out_features, bias=bias)
        self._mod.to(device)
        self._device = device

    def forward(self, x: Tensor) -> Tensor:
        w = Tensor(self._mod.weight)
        b = Tensor(self._mod.bias) if self._mod.bias is not None \
            else Tensor(torch.zeros(self._mod.out_features,
                                     device=self._device))
        return F.linear(x, w, b)

    def parameters(self):
        return list(self._mod.parameters())


class Conv2d(Layer):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 0, device: str = "cpu"):
        self._mod = nn.Conv2d(in_ch, out_ch, kernel, stride, padding)
        self._mod.to(device)

    def forward(self, x: Tensor) -> Tensor:
        w = Tensor(self._mod.weight)
        b = Tensor(self._mod.bias)
        return F.conv2d(x, w, b, self._mod.stride[0], self._mod.padding[0])

    def parameters(self):
        return list(self._mod.parameters())


class Embedding(Layer):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 padding_idx: Optional[int] = None, device: str = "cpu"):
        self._mod = nn.Embedding(num_embeddings, embedding_dim,
                                  padding_idx=padding_idx)
        self._mod.to(device)
        self._padding_idx = padding_idx

    def forward(self, x: Tensor) -> Tensor:
        return F.embedding(x, Tensor(self._mod.weight), self._padding_idx)

    def parameters(self):
        return list(self._mod.parameters())


# ------------------------------------------------------------------ #
# Activations (stateless — singleton-friendly)
# ------------------------------------------------------------------ #

class ReLU(Layer):
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)

class GELU(Layer):
    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x)

class SiLU(Layer):
    def forward(self, x: Tensor) -> Tensor:
        return F.silu(x)

class Sigmoid(Layer):
    def forward(self, x: Tensor) -> Tensor:
        return F.sigmoid(x)

class Tanh(Layer):
    def forward(self, x: Tensor) -> Tensor:
        return F.tanh(x)

class Softmax(Layer):
    def __init__(self, dim: int = -1):
        self._dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return F.softmax(x, self._dim)


# ------------------------------------------------------------------ #
# Normalisation
# ------------------------------------------------------------------ #

class LayerNorm(Layer):
    def __init__(self, normalized_shape, eps: float = 1e-5,
                 device: str = "cpu"):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self._shape = tuple(normalized_shape)
        self._eps = eps
        self._mod = nn.LayerNorm(normalized_shape, eps=eps)
        self._mod.to(device)

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, self._shape,
                             Tensor(self._mod.weight),
                             Tensor(self._mod.bias), self._eps)

    def parameters(self):
        return list(self._mod.parameters())


class BatchNorm1d(Layer):
    def __init__(self, num_features: int, eps: float = 1e-5,
                 device: str = "cpu"):
        self._mod = nn.BatchNorm1d(num_features, eps=eps)
        self._mod.to(device)
        self._training = True

    def train(self):
        self._mod.train()
        self._training = True
        return self

    def eval(self):
        self._mod.eval()
        self._training = False
        return self

    def forward(self, x: Tensor) -> Tensor:
        return F.batch_norm(
            x,
            Tensor(self._mod.running_mean),
            Tensor(self._mod.running_var),
            Tensor(self._mod.weight),
            Tensor(self._mod.bias),
            training=self._training,
            eps=self._mod.eps,
        )

    def parameters(self):
        return list(self._mod.parameters())


class RMSNorm(Layer):
    def __init__(self, dim: int, eps: float = 1e-6, device: str = "cpu"):
        self._weight = nn.Parameter(torch.ones(dim, device=device))
        self._eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, Tensor(self._weight), self._eps)

    def parameters(self):
        return [self._weight]


# ------------------------------------------------------------------ #
# Regularisation
# ------------------------------------------------------------------ #

class Dropout(Layer):
    def __init__(self, p: float = 0.5):
        self._p = p
        self._training = True

    def train(self):
        self._training = True
        return self

    def eval(self):
        self._training = False
        return self

    def forward(self, x: Tensor) -> Tensor:
        return F.dropout(x, self._p, self._training)


# ------------------------------------------------------------------ #
# Pooling
# ------------------------------------------------------------------ #

class MaxPool2d(Layer):
    def __init__(self, kernel_size: int, stride: int = 1):
        self._k = kernel_size
        self._s = stride

    def forward(self, x: Tensor) -> Tensor:
        return F.max_pool2d(x, self._k, self._s)


class AdaptiveAvgPool2d(Layer):
    def __init__(self, output_size: tuple = (1, 1)):
        self._out = output_size

    def forward(self, x: Tensor) -> Tensor:
        return F.adaptive_avg_pool2d(x, self._out)


# ------------------------------------------------------------------ #
# Flatten
# ------------------------------------------------------------------ #

class Flatten(Layer):
    def __init__(self, start_dim: int = 1):
        self._start = start_dim

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(x.data.flatten(self._start))


# ------------------------------------------------------------------ #
# Residual wrapper
# ------------------------------------------------------------------ #

class Residual(Layer):
    """Skip-connection: output = sublayer(x) + x."""
    def __init__(self, sublayer: Layer):
        self._sub = sublayer

    def forward(self, x: Tensor) -> Tensor:
        return self._sub(x) + x

    def parameters(self):
        return self._sub.parameters()

    def train(self):
        self._sub.train()
        return self

    def eval(self):
        self._sub.eval()
        return self
