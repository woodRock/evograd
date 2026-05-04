"""
EvoGrad MPS operations.

Two-tier design:
  Tier 1 (always available): pure PyTorch MPS — every standard tensor op
    runs natively via Apple's Metal Performance Shaders.  No extra
    dependencies.

  Tier 2 (optional, faster): metalcompute custom kernels — the Metal
    shaders in kernels.metal compiled and invoked via the metalcompute
    package.  Removes per-op kernel-launch overhead by fusing operations
    and using an on-device PCG32 random number generator.

Install tier 2:
    pip install metalcompute

At import time we probe for metalcompute.  The `_MC` flag records which
tier is active; all public functions dispatch accordingly.
"""
from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Optional
import numpy as np
import torch

from .device import is_mps

# ------------------------------------------------------------------ #
# Tier-2 metalcompute probe
# ------------------------------------------------------------------ #

_MC: Optional[object] = None   # metalcompute device, or None
_MC_FNS: dict = {}             # compiled kernel functions

try:
    import metalcompute as mc                    # type: ignore[import]
    _MC_DEV = mc.Device()
    _METAL_SRC = (Path(__file__).parent / "kernels.metal").read_text()
    _MC_PROG = _MC_DEV.kernel(_METAL_SRC)
    _MC_FNS = {
        "tournament":  _MC_PROG.function("tournament_select"),
        "crossover":   _MC_PROG.function("uniform_crossover"),
        "sbx":         _MC_PROG.function("sbx_crossover"),
        "mutate":      _MC_PROG.function("gaussian_mutate"),
        "rastrigin":   _MC_PROG.function("rastrigin_fitness"),
        "ackley":      _MC_PROG.function("ackley_fitness"),
    }
    _MC = _MC_DEV
    print("[EvoGrad] metalcompute found — using custom Metal kernels (tier 2)")
except Exception:
    pass     # tier 1 only, no warning needed

_TIER = 2 if _MC is not None else 1


def active_tier() -> int:
    return _TIER


# ------------------------------------------------------------------ #
# Helpers for metalcompute buffer passing
# ------------------------------------------------------------------ #

def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """MPS → CPU → numpy.  Required for metalcompute buffer hand-off."""
    return t.cpu().numpy() if t.device.type == "mps" else t.numpy()


def _from_numpy(arr: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(arr).to(device)


def _pack_uint4(a: int, b: int, c: int, d: int = 0) -> np.ndarray:
    return np.array([a, b, c, d], dtype=np.uint32)


# ------------------------------------------------------------------ #
# Tournament selection
# ------------------------------------------------------------------ #

def tournament_select(fitness: torch.Tensor, k: int = 7,
                       seed: int | None = None) -> torch.Tensor:
    """
    Batched tournament selection.
    Returns a (P,) index tensor of winner indices.
    """
    P = fitness.shape[0]
    device = fitness.device
    if seed is None:
        seed = int(time.time_ns() & 0xFFFF_FFFF)

    if _TIER == 2:
        fit_np = _to_numpy(fitness.float())
        out_np = np.zeros(P, dtype=np.uint32)
        params = _pack_uint4(P, k, seed)
        _MC_FNS["tournament"](P, fit_np, out_np, params)
        return _from_numpy(out_np.astype(np.int64), str(device))

    # Tier 1: pure PyTorch MPS — vectorised gather
    candidates = torch.randint(0, P, (P, k), device=device)       # (P, k)
    cand_fit = fitness[candidates]                                  # (P, k)
    winners = cand_fit.argmax(dim=1)                               # (P,)
    return candidates[torch.arange(P, device=device), winners]


# ------------------------------------------------------------------ #
# Uniform crossover
# ------------------------------------------------------------------ #

def uniform_crossover(parents: torch.Tensor, rate: float = 0.5,
                       seed: int | None = None) -> torch.Tensor:
    """Parents (P, G) → offspring (P, G). P must be even."""
    P, G = parents.shape
    device = parents.device
    if seed is None:
        seed = int(time.time_ns() & 0xFFFF_FFFF)

    if _TIER == 2:
        par_np = _to_numpy(parents.float())
        off_np = np.empty_like(par_np)
        params = _pack_uint4(P, G, seed)
        rate_buf = np.array([rate], dtype=np.float32)
        _MC_FNS["crossover"](P // 2 * G, par_np, off_np, params, rate_buf)
        return _from_numpy(off_np, str(device))

    # Tier 1
    g = parents.clone()
    p1, p2 = g[0::2], g[1::2]
    mask = torch.rand(P // 2, G, device=device) < rate
    child1 = torch.where(mask, p1, p2)
    child2 = torch.where(mask, p2, p1)
    offspring = torch.empty_like(g)
    offspring[0::2] = child1
    offspring[1::2] = child2
    return offspring


# ------------------------------------------------------------------ #
# SBX crossover
# ------------------------------------------------------------------ #

def sbx_crossover(parents: torch.Tensor, eta: float = 20.0,
                   seed: int | None = None) -> torch.Tensor:
    P, G = parents.shape
    device = parents.device
    if seed is None:
        seed = int(time.time_ns() & 0xFFFF_FFFF)

    if _TIER == 2:
        par_np = _to_numpy(parents.float())
        off_np = np.empty_like(par_np)
        params = _pack_uint4(P, G, seed)
        eta_buf = np.array([eta], dtype=np.float32)
        _MC_FNS["sbx"](P // 2 * G, par_np, off_np, params, eta_buf)
        return _from_numpy(off_np, str(device))

    # Tier 1
    g = parents
    p1, p2 = g[0::2].clone(), g[1::2].clone()
    u = torch.rand(P // 2, G, device=device).clamp(1e-9, 1.0 - 1e-9)
    exp = 1.0 / (eta + 1.0)
    beta = torch.where(u <= 0.5,
                       (2.0 * u).pow(exp),
                       (1.0 / (2.0 * (1.0 - u))).pow(exp))
    child1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    child2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    offspring = torch.empty_like(g)
    offspring[0::2] = child1
    offspring[1::2] = child2
    return offspring


# ------------------------------------------------------------------ #
# Gaussian mutation
# ------------------------------------------------------------------ #

def gaussian_mutate(genomes: torch.Tensor, sigma: float = 0.01,
                     rate: float = 1.0,
                     seed: int | None = None) -> torch.Tensor:
    P, G = genomes.shape
    device = genomes.device
    if seed is None:
        seed = int(time.time_ns() & 0xFFFF_FFFF)

    if _TIER == 2:
        gen_np = _to_numpy(genomes.float())
        out_np = np.empty_like(gen_np)
        params = _pack_uint4(P, G, seed)
        hyper = np.array([sigma, rate], dtype=np.float32)
        _MC_FNS["mutate"](P * G, gen_np, out_np, params, hyper)
        return _from_numpy(out_np, str(device))

    # Tier 1
    g = genomes.clone()
    if rate < 1.0:
        mask = torch.rand_like(g) < rate
        noise = torch.randn_like(g) * sigma
        return torch.where(mask, g + noise, g)
    return g + torch.randn_like(g) * sigma


# ------------------------------------------------------------------ #
# Fitness functions (Metal-accelerated)
# ------------------------------------------------------------------ #

def rastrigin_fitness(genomes: torch.Tensor, A: float = 10.0) -> torch.Tensor:
    P, G = genomes.shape
    device = genomes.device

    if _TIER == 2:
        gen_np = _to_numpy(genomes.float())
        fit_np = np.zeros(P, dtype=np.float32)
        params = _pack_uint4(P, G, 0)
        A_buf = np.array([A], dtype=np.float32)
        _MC_FNS["rastrigin"](P, gen_np, fit_np, params, A_buf)
        return _from_numpy(fit_np, str(device))

    import math
    x = genomes
    cost = A * G + (x ** 2 - A * torch.cos(2 * math.pi * x)).sum(dim=1)
    return -cost


def ackley_fitness(genomes: torch.Tensor) -> torch.Tensor:
    P, G = genomes.shape
    device = genomes.device

    if _TIER == 2:
        gen_np = _to_numpy(genomes.float())
        fit_np = np.zeros(P, dtype=np.float32)
        params = _pack_uint4(P, G, 0)
        _MC_FNS["ackley"](P, gen_np, fit_np, params)
        return _from_numpy(fit_np, str(device))

    import math
    x = genomes
    s1 = (-0.2 * (x ** 2).mean(dim=1).sqrt()).exp()
    s2 = (2 * math.pi * x).cos().mean(dim=1).exp()
    cost = -20.0 * s1 - s2 + 20.0 + math.e
    return -cost
