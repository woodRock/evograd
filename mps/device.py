"""
MPS device utilities for Apple Silicon (M1/M2/M3).

Provides:
  auto_device()       — pick the best available device
  to_device(t, dev)   — move a Tensor to a device
  mps_safe_*()        — wrappers for ops not yet on MPS that transparently
                        execute on CPU and return the result on the
                        original device
"""
from __future__ import annotations
import os
import torch


def auto_device() -> str:
    """Return 'mps', 'cuda', or 'cpu' in priority order."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def is_mps(t: torch.Tensor) -> bool:
    return t.device.type == "mps"


# ------------------------------------------------------------------ #
# Ops that are not yet implemented on MPS — run on CPU, return on
# original device.  No silent fallback env-var required.
# ------------------------------------------------------------------ #

def mps_safe_lstsq(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    torch.linalg.lstsq is missing on MPS (as of PyTorch 2.x).
    Solve on CPU and ship the result back to the original device.
    """
    device = A.device
    if device.type == "mps":
        result = torch.linalg.lstsq(A.cpu().double(), b.cpu().double()).solution
        return result.to(device=device, dtype=A.dtype)
    return torch.linalg.lstsq(A.double(), b.double()).solution.to(dtype=A.dtype)


def mps_safe_mode(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    torch.mode is missing on MPS.
    Fall back to CPU for the reduction, return result on original device.
    """
    device = t.device
    if device.type == "mps":
        return torch.mode(t.cpu(), dim=dim).values.to(device)
    return torch.mode(t, dim=dim).values


def mps_safe_sort(t: torch.Tensor, dim: int = -1,
                   descending: bool = False) -> torch.Tensor:
    """torch.sort — works on MPS but wrap uniformly for caller convenience."""
    return torch.sort(t, dim=dim, descending=descending)


# ------------------------------------------------------------------ #
# Benchmark helper
# ------------------------------------------------------------------ #

def warmup_mps(n: int = 3):
    """
    Run a few no-op matmuls to warm up the MPS command queue.
    Avoids the first-call latency spike when timing benchmarks.
    """
    if not torch.backends.mps.is_available():
        return
    x = torch.randn(128, 128, device="mps")
    for _ in range(n):
        _ = x @ x
    torch.mps.synchronize()


def sync():
    """Block until all pending MPS commands have completed."""
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
