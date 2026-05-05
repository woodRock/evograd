"""
Data loading and batching utilities for EvoGrad.

These utilities wrap EvoGrad Tensors in a PyTorch-like API while
maintaining the library's immutability and device-safety guarantees.
"""
from __future__ import annotations
import math
import torch
from typing import Sequence, Iterator, Tuple, Optional

from .tensor import Tensor


class TensorDataset:
    """
    Dataset wrapping one or more EvoGrad Tensors.

    Each tensor must have the same size in the first dimension (N).
    """

    def __init__(self, *tensors: Tensor):
        assert len(tensors) > 0, "Dataset must contain at least one tensor"
        self._tensors = tensors
        self._n = tensors[0].shape[0]
        for t in tensors:
            assert t.shape[0] == self._n, (
                f"Tensor size mismatch: expected {self._n}, got {t.shape[0]}"
            )

    def __getitem__(self, index: int) -> Tuple[Tensor, ...]:
        return tuple(Tensor(t.data[index]) for t in self._tensors)

    def __len__(self) -> int:
        return self._n

    @property
    def tensors(self) -> Tuple[Tensor, ...]:
        return self._tensors


class DataLoader:
    """
    Batch iterator for TensorDataset.

    Parameters
    ----------
    dataset    : TensorDataset to iterate over
    batch_size : number of samples per batch
    shuffle    : whether to shuffle the indices every epoch
    drop_last  : whether to discard the last incomplete batch
    """

    def __init__(self, dataset: TensorDataset,
                 batch_size: int = 32,
                 shuffle: bool = False,
                 drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._n = len(dataset)

    def __iter__(self) -> Iterator[Tuple[Tensor, ...]]:
        indices = torch.arange(self._n)
        if self.shuffle:
            indices = torch.randperm(self._n)

        for i in range(0, self._n, self.batch_size):
            if i + self.batch_size > self._n and self.drop_last:
                break
            
            batch_idx = indices[i:i + self.batch_size]
            
            # Efficiently slice the underlying torch data and re-wrap as Tensors
            yield tuple(Tensor(t.data[batch_idx]) for t in self.dataset.tensors)

    def __len__(self) -> int:
        n = self._n / self.batch_size
        return math.floor(n) if self.drop_last else math.ceil(n)
