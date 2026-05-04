"""
Pure functional neural network operations.

Every function here is referentially transparent:
  - No global state
  - No in-place mutations
  - Same inputs always produce the same output

All functions accept EvoGrad Tensors and return EvoGrad Tensors.
The grad_fn hooks live on the underlying torch data so that PyTorch's
autograd engine can still differentiate through the whole stack.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F

from ..core.tensor import Tensor


# ------------------------------------------------------------------ #
# Activations
# ------------------------------------------------------------------ #

def relu(x: Tensor) -> Tensor:
    return Tensor(F.relu(x.data))

def leaky_relu(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    return Tensor(F.leaky_relu(x.data, negative_slope))

def gelu(x: Tensor) -> Tensor:
    return Tensor(F.gelu(x.data))

def silu(x: Tensor) -> Tensor:                    # swish
    return Tensor(F.silu(x.data))

def sigmoid(x: Tensor) -> Tensor:
    return Tensor(torch.sigmoid(x.data))

def tanh(x: Tensor) -> Tensor:
    return Tensor(torch.tanh(x.data))

def softmax(x: Tensor, dim: int = -1) -> Tensor:
    return Tensor(F.softmax(x.data, dim=dim))

def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    return Tensor(F.log_softmax(x.data, dim=dim))

def mish(x: Tensor) -> Tensor:
    return Tensor(x.data * torch.tanh(F.softplus(x.data)))


_ACTIVATIONS = {
    "relu": relu, "leaky_relu": leaky_relu, "gelu": gelu,
    "silu": silu, "swish": silu, "sigmoid": sigmoid,
    "tanh": tanh, "softmax": softmax, "mish": mish,
}

def activate(x: Tensor, fn: str) -> Tensor:
    if fn not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation: {fn!r}. "
                         f"Choose from {list(_ACTIVATIONS)}")
    return _ACTIVATIONS[fn](x)


# ------------------------------------------------------------------ #
# Linear algebra
# ------------------------------------------------------------------ #

def linear(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """y = x @ W^T + b  (standard fully-connected layer)."""
    out = x.data @ weight.data.T + bias.data
    return Tensor(out)

def bilinear(x1: Tensor, x2: Tensor, weight: Tensor,
             bias: Tensor) -> Tensor:
    return Tensor(F.bilinear(x1.data, x2.data, weight.data, bias.data))


# ------------------------------------------------------------------ #
# Normalisation
# ------------------------------------------------------------------ #

def layer_norm(x: Tensor, normalized_shape: tuple,
               weight: Tensor, bias: Tensor, eps: float = 1e-5) -> Tensor:
    return Tensor(F.layer_norm(
        x.data, normalized_shape, weight.data, bias.data, eps
    ))

def batch_norm(x: Tensor, running_mean: Tensor, running_var: Tensor,
               weight: Tensor, bias: Tensor,
               training: bool = True, eps: float = 1e-5) -> Tensor:
    return Tensor(F.batch_norm(
        x.data, running_mean.data, running_var.data,
        weight.data, bias.data, training, 0.1, eps
    ))

def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """RMSNorm — cheaper than LayerNorm, no mean subtraction."""
    rms = (x.data.pow(2).mean(-1, keepdim=True) + eps).sqrt()
    return Tensor(x.data / rms * weight.data)


# ------------------------------------------------------------------ #
# Convolution
# ------------------------------------------------------------------ #

def conv2d(x: Tensor, weight: Tensor, bias: Tensor,
           stride: int = 1, padding: int = 0) -> Tensor:
    return Tensor(F.conv2d(x.data, weight.data, bias.data, stride, padding))

def max_pool2d(x: Tensor, kernel_size: int, stride: int = 1) -> Tensor:
    return Tensor(F.max_pool2d(x.data, kernel_size, stride))

def avg_pool2d(x: Tensor, kernel_size: int, stride: int = 1) -> Tensor:
    return Tensor(F.avg_pool2d(x.data, kernel_size, stride))

def adaptive_avg_pool2d(x: Tensor, output_size: tuple) -> Tensor:
    return Tensor(F.adaptive_avg_pool2d(x.data, output_size))


# ------------------------------------------------------------------ #
# Attention
# ------------------------------------------------------------------ #

def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor,
                                  mask: Tensor | None = None,
                                  dropout_p: float = 0.0) -> Tensor:
    """
    Multi-head scaled dot-product attention.
    q, k, v: (B, heads, seq, head_dim)
    """
    scale = math.sqrt(q.shape[-1])
    scores = (q.data @ k.data.transpose(-2, -1)) / scale
    if mask is not None:
        scores = scores + mask.data
    attn = F.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    return Tensor(attn @ v.data)


# ------------------------------------------------------------------ #
# Dropout / regularisation (training-time)
# ------------------------------------------------------------------ #

def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    return Tensor(F.dropout(x.data, p=p, training=training))

def alpha_dropout(x: Tensor, p: float = 0.5,
                  training: bool = True) -> Tensor:
    return Tensor(F.alpha_dropout(x.data, p=p, training=training))


# ------------------------------------------------------------------ #
# Embedding
# ------------------------------------------------------------------ #

def embedding(idx: Tensor, weight: Tensor,
              padding_idx: int | None = None) -> Tensor:
    return Tensor(F.embedding(idx.data, weight.data,
                               padding_idx=padding_idx))


# ------------------------------------------------------------------ #
# Loss functions
# ------------------------------------------------------------------ #

def cross_entropy(logits: Tensor, targets: Tensor,
                  reduction: str = "mean") -> Tensor:
    return Tensor(F.cross_entropy(logits.data, targets.data.long(),
                                   reduction=reduction))

def binary_cross_entropy_with_logits(logits: Tensor, targets: Tensor,
                                      reduction: str = "mean") -> Tensor:
    return Tensor(F.binary_cross_entropy_with_logits(
        logits.data, targets.data, reduction=reduction
    ))

def mse_loss(pred: Tensor, target: Tensor,
             reduction: str = "mean") -> Tensor:
    return Tensor(F.mse_loss(pred.data, target.data, reduction=reduction))

def huber_loss(pred: Tensor, target: Tensor, delta: float = 1.0,
               reduction: str = "mean") -> Tensor:
    return Tensor(F.huber_loss(pred.data, target.data, delta=delta,
                                reduction=reduction))

def kl_divergence(log_p: Tensor, q: Tensor,
                  reduction: str = "batchmean") -> Tensor:
    return Tensor(F.kl_div(log_p.data, q.data, reduction=reduction))


# ------------------------------------------------------------------ #
# Composition helper: pipeline operator (>>)
# ------------------------------------------------------------------ #

class Compose:
    """
    Chains pure functions left-to-right:
        f = Compose(linear_fn, relu, dropout_fn)
        y = f(x)
    """
    def __init__(self, *fns):
        self.fns = fns

    def __call__(self, x: Tensor) -> Tensor:
        for fn in self.fns:
            x = fn(x)
        return x

    def __rshift__(self, other):
        if isinstance(other, Compose):
            return Compose(*self.fns, *other.fns)
        return Compose(*self.fns, other)
