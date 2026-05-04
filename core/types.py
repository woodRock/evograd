"""
EvoGrad type system: statically-checked tensor shapes and dtypes.

Shapes are tuples of ints (static) or None (dynamic). The type
Shape[batch, 784] is written as (None, 784) — None means "any size."
Mismatches are caught at graph-construction time, not at runtime.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
import re


Dim = Optional[int]
ShapeTuple = Tuple[Dim, ...]


@dataclass(frozen=True)
class DType:
    name: str
    bits: int

    def __repr__(self) -> str:
        return self.name


Float32 = DType("float32", 32)
Float16 = DType("float16", 16)
BFloat16 = DType("bfloat16", 16)
Int64 = DType("int64", 64)
Int32 = DType("int32", 32)
Bool = DType("bool", 1)


@dataclass(frozen=True)
class TensorType:
    """Compile-time type for a tensor: shape + dtype."""
    shape: ShapeTuple
    dtype: DType = Float32

    def rank(self) -> int:
        return len(self.shape)

    def is_compatible_with(self, other: "TensorType") -> bool:
        if self.dtype != other.dtype:
            return False
        if len(self.shape) != len(other.shape):
            return False
        return all(
            a is None or b is None or a == b
            for a, b in zip(self.shape, other.shape)
        )

    def unify(self, other: "TensorType") -> "TensorType":
        """Unify two compatible types; raises ShapeError on mismatch."""
        if not self.is_compatible_with(other):
            raise ShapeError(
                f"Cannot unify tensor types: {self} vs {other}"
            )
        unified = tuple(
            a if a is not None else b
            for a, b in zip(self.shape, other.shape)
        )
        return TensorType(unified, self.dtype)

    def __repr__(self) -> str:
        dims = ", ".join("?" if d is None else str(d) for d in self.shape)
        return f"Tensor[{dims}; {self.dtype}]"

    @classmethod
    def parse(cls, spec: str) -> "TensorType":
        """Parse 'Float[batch, 784]' or 'Float[?, 256, 256]' syntax."""
        m = re.match(r"(\w+)\[([^\]]+)\]", spec.strip())
        if not m:
            raise SyntaxError(f"Invalid type spec: {spec!r}")
        dtype_name, dims_str = m.group(1).lower(), m.group(2)
        dtype_map = {
            "float": Float32, "float32": Float32,
            "float16": Float16, "bfloat16": BFloat16,
            "int": Int64, "int64": Int64, "int32": Int32,
            "bool": Bool,
        }
        dtype = dtype_map.get(dtype_name, Float32)
        dims: list[Dim] = []
        for part in dims_str.split(","):
            p = part.strip()
            if p in ("?", "_", "batch", "b", "n"):
                dims.append(None)
            else:
                try:
                    dims.append(int(p))
                except ValueError:
                    dims.append(None)
        return cls(tuple(dims), dtype)


class ShapeError(TypeError):
    """Raised when tensor shapes are incompatible at compile time."""
    pass


def broadcast_shapes(a: ShapeTuple, b: ShapeTuple) -> ShapeTuple:
    """NumPy-style broadcast shape inference."""
    if len(a) != len(b):
        # Pad shorter shape with leading 1s
        pad = len(a) - len(b)
        if pad > 0:
            b = (1,) * pad + b
        else:
            a = (1,) * (-pad) + a
    result: list[Dim] = []
    for da, db in zip(a, b):
        if da is None or db is None:
            result.append(None)
        elif da == 1:
            result.append(db)
        elif db == 1:
            result.append(da)
        elif da == db:
            result.append(da)
        else:
            raise ShapeError(f"Cannot broadcast shapes {a} and {b}")
    return tuple(result)


def matmul_shape(a: ShapeTuple, b: ShapeTuple) -> ShapeTuple:
    """Infer output shape for matrix multiply."""
    if len(a) < 2 or len(b) < 2:
        raise ShapeError(f"matmul requires rank >= 2, got {a} and {b}")
    m, k1 = a[-2], a[-1]
    k2, n = b[-2], b[-1]
    if k1 is not None and k2 is not None and k1 != k2:
        raise ShapeError(
            f"matmul inner dims mismatch: {k1} != {k2} in {a} @ {b}"
        )
    batch_a, batch_b = a[:-2], b[:-2]
    if batch_a and batch_b:
        batch = broadcast_shapes(batch_a, batch_b)
    else:
        batch = batch_a or batch_b
    return batch + (m, n)
