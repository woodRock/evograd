"""
Vectorized (μ, λ)-Evolution Strategy with self-adaptive step sizes.

Each individual carries its own σ vector (one per dimension), updated
via the log-normal rule before Gaussian mutation.  The entire batch of
λ offspring is generated and evaluated in two tensor ops.

Mutation:
    σ' = σ · exp(τ'·N(0,1) + τ·N_i(0,1))   (per-gene log-normal)
    x' = x + σ' · N(0,1)

Selection:  keep the μ best of λ offspring (comma strategy).
"""
from __future__ import annotations
import math
from typing import Callable, List
import time
import torch

from ..core.tensor import Tensor


Fitness = Callable[[Tensor], Tensor]


class ES:
    """
    (μ, λ)-ES with self-adaptive per-dimension step sizes.

    Parameters
    ----------
    mu        : number of parents selected each generation
    lam       : number of offspring generated (lam >= mu)
    genome_length
    sigma0    : initial step size for all dimensions
    """

    def __init__(self,
                 mu: int,
                 lam: int,
                 genome_length: int,
                 sigma0: float = 0.3,
                 device: str = "cpu"):
        assert lam >= mu, "lam must be >= mu"
        self.mu = mu
        self.lam = lam
        self.n = genome_length
        self._device = device
        self.generation = 0

        # learning rates for self-adaptation
        self.tau       = 1.0 / math.sqrt(2 * genome_length)
        self.tau_prime = 1.0 / math.sqrt(2 * math.sqrt(genome_length))

        # initialise population: (mu, n) positions + (mu, n) step-sizes
        self.pop_x  = torch.randn(mu, genome_length, device=device)
        self.pop_s  = torch.full((mu, genome_length), sigma0, device=device)
        self.best_x: torch.Tensor = self.pop_x[0].clone()
        self.best_f: float = -1e18

    # ------------------------------------------------------------------ #

    def step(self, fitness_fn: Fitness) -> float:
        mu, lam, n = self.mu, self.lam, self.n
        dev = self._device

        # ── generate λ offspring from μ parents (cyclic) ──────────────
        parent_idx = torch.arange(lam, device=dev) % mu
        parent_x   = self.pop_x[parent_idx]         # (λ, n)
        parent_s   = self.pop_s[parent_idx]         # (λ, n)

        # self-adapt step sizes
        global_noise = torch.randn(lam, 1, device=dev)
        local_noise  = torch.randn(lam, n, device=dev)
        child_s = parent_s * torch.exp(
            self.tau_prime * global_noise + self.tau * local_noise
        ).clamp(min=1e-5)

        # mutate positions
        child_x = parent_x + child_s * torch.randn(lam, n, device=dev)

        # ── evaluate ──────────────────────────────────────────────────
        child_fit = fitness_fn(Tensor(child_x)).data  # (λ,)

        # ── (μ, λ)-selection: keep top μ ──────────────────────────────
        top_idx = child_fit.topk(mu).indices
        self.pop_x = child_x[top_idx]
        self.pop_s = child_s[top_idx]

        best_idx = child_fit.argmax()
        best_val = child_fit[best_idx].item()
        if best_val > self.best_f:
            self.best_f = best_val
            self.best_x = child_x[best_idx].clone()

        self.generation += 1
        return self.best_f


def run_es(fitness_fn: Fitness,
           mu: int = 15,
           lam: int = 100,
           genome_length: int = 20,
           generations: int = 100,
           device: str = "cpu",
           verbose: bool = False) -> tuple[float, list]:
    es = ES(mu, lam, genome_length, device=device)
    log = []
    t0 = time.perf_counter()
    for g in range(generations):
        best = es.step(fitness_fn)
        log.append(best)
        if verbose and g % (generations // 5) == 0:
            print(f"  ES gen {g:4d} | best={best:.4f} | "
                  f"{time.perf_counter()-t0:.2f}s")
    return best, log
