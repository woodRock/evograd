"""
Immutable, typed tensors — the fundamental value in EvoGrad.

EvoGrad tensors are intentionally read-only: every op returns a new
tensor rather than mutating its operands.  This makes computation
graphs trivially serializable and enables aggressive compiler fusion.

The actual numeric data lives in a PyTorch tensor; we wrap it and
block in-place writes at the Python level.
"""
from __future__ import annotations
from typing import Optional, Sequence, Union
import torch

from .types import TensorType, DType, Float32, ShapeTuple, ShapeError


_DTYPE_MAP = {
    Float32: torch.float32,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "int32": torch.int32,
    "bool": torch.bool,
}
_RDTYPE_MAP = {v: k for k, v in _DTYPE_MAP.items() if isinstance(k, str)}


class Tensor:
    """
    An immutable, type-tagged tensor.

    Construction:
        t = Tensor.from_torch(torch_tensor)
        t = Tensor.zeros((32, 784))
        t = Tensor.randn((32, 784))

    Arithmetic returns new Tensors:
        y = t.matmul(w)   # y is a new Tensor
        z = y + b         # z is a new Tensor

    The underlying torch tensor is accessible via `.data` but any
    attempt to write `.data = ...` raises AttributeError.
    """
    __slots__ = ("_data", "_type")

    def __init__(self, data: torch.Tensor, ttype: Optional[TensorType] = None):
        object.__setattr__(self, "_data", data)
        if ttype is None:
            shape = tuple(s if s != -1 else None for s in data.shape)
            ttype = TensorType(shape)
        object.__setattr__(self, "_type", ttype)

    # ------------------------------------------------------------------ #
    # Immutability guard
    # ------------------------------------------------------------------ #
    def __setattr__(self, name, value):
        raise AttributeError("EvoGrad tensors are immutable")

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_torch(cls, t: torch.Tensor) -> "Tensor":
        return cls(t)

    @classmethod
    def zeros(cls, shape: ShapeTuple, device: str = "cpu") -> "Tensor":
        concrete = tuple(1 if d is None else d for d in shape)
        return cls(torch.zeros(concrete, device=device))

    @classmethod
    def ones(cls, shape: ShapeTuple, device: str = "cpu") -> "Tensor":
        concrete = tuple(1 if d is None else d for d in shape)
        return cls(torch.ones(concrete, device=device))

    @classmethod
    def randn(cls, shape: ShapeTuple, device: str = "cpu") -> "Tensor":
        concrete = tuple(1 if d is None else d for d in shape)
        return cls(torch.randn(concrete, device=device))

    @classmethod
    def arange(cls, n: int, device: str = "cpu") -> "Tensor":
        return cls(torch.arange(n, dtype=torch.int64, device=device))

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def data(self) -> torch.Tensor:
        return self._data

    @property
    def type(self) -> TensorType:
        return self._type

    @property
    def shape(self) -> tuple:
        return tuple(self._data.shape)

    @property
    def dtype(self) -> torch.dtype:
        return self._data.dtype

    @property
    def device(self) -> torch.device:
        return self._data.device

    @property
    def ndim(self) -> int:
        return self._data.ndim

    # ------------------------------------------------------------------ #
    # Core ops (each returns a new Tensor)
    # ------------------------------------------------------------------ #
    def matmul(self, other: "Tensor") -> "Tensor":
        return Tensor(self._data @ other._data)

    def __add__(self, other: Union["Tensor", float, int]) -> "Tensor":
        o = other._data if isinstance(other, Tensor) else other
        return Tensor(self._data + o)

    def __radd__(self, other):
        return Tensor(other + self._data)

    def __sub__(self, other: Union["Tensor", float, int]) -> "Tensor":
        o = other._data if isinstance(other, Tensor) else other
        return Tensor(self._data - o)

    def __mul__(self, other: Union["Tensor", float, int]) -> "Tensor":
        o = other._data if isinstance(other, Tensor) else other
        return Tensor(self._data * o)

    def __rmul__(self, other):
        return Tensor(other * self._data)

    def __truediv__(self, other: Union["Tensor", float, int]) -> "Tensor":
        o = other._data if isinstance(other, Tensor) else other
        return Tensor(self._data / o)

    def __neg__(self) -> "Tensor":
        return Tensor(-self._data)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return self.matmul(other)

    def __pow__(self, exp: float) -> "Tensor":
        return Tensor(self._data ** exp)

    def transpose(self, dim0: int = -2, dim1: int = -1) -> "Tensor":
        return Tensor(self._data.transpose(dim0, dim1))

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def reshape(self, *shape) -> "Tensor":
        return Tensor(self._data.reshape(*shape))

    def view(self, *shape) -> "Tensor":
        return Tensor(self._data.view(*shape))

    def squeeze(self, dim: Optional[int] = None) -> "Tensor":
        return Tensor(self._data.squeeze(dim) if dim is not None else self._data.squeeze())

    def unsqueeze(self, dim: int) -> "Tensor":
        return Tensor(self._data.unsqueeze(dim))

    def expand(self, *shape) -> "Tensor":
        return Tensor(self._data.expand(*shape))

    def sum(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        if dim is None:
            return Tensor(self._data.sum())
        return Tensor(self._data.sum(dim=dim, keepdim=keepdim))

    def mean(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        if dim is None:
            return Tensor(self._data.mean())
        return Tensor(self._data.mean(dim=dim, keepdim=keepdim))

    def max(self, dim: Optional[int] = None) -> "Tensor":
        if dim is None:
            return Tensor(self._data.max())
        return Tensor(self._data.max(dim=dim).values)

    def argmax(self, dim: int) -> "Tensor":
        return Tensor(self._data.argmax(dim=dim))

    def clip(self, min_val: float, max_val: float) -> "Tensor":
        return Tensor(self._data.clamp(min_val, max_val))

    def to(self, device: str) -> "Tensor":
        return Tensor(self._data.to(device))

    def float(self) -> "Tensor":
        return Tensor(self._data.float())

    def cat(self, other: "Tensor", dim: int = 0) -> "Tensor":
        return Tensor(torch.cat([self._data, other._data], dim=dim))

    def __repr__(self) -> str:
        return f"Tensor({self.shape}, dtype={self._data.dtype}, device={self._data.device})"

    def item(self):
        return self._data.item()

    def numpy(self):
        return self._data.cpu().numpy()

    # ------------------------------------------------------------------ #
    # Gradient tracking (opt-in)
    # ------------------------------------------------------------------ #
    def requires_grad_(self) -> torch.Tensor:
        """Return a torch tensor with gradient tracking for optimizer use."""
        return self._data.clone().requires_grad_(True)
