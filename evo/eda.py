"""
Vectorized EDA — Univariate Marginal Distribution Algorithm (UMDA).

UMDA models the marginal distribution of each gene independently as a
Gaussian.  Each generation:
  1. Evaluate the population.
  2. Select the top-k individuals (truncation selection).
  3. Estimate μ and σ per gene from the selected set.
  4. Sample a new population from N(μ, σ²).

Because step 3 is just `.mean(0)` and `.std(0)` over the selected
sub-tensor and step 4 is `torch.randn * σ + μ`, the entire generation
runs in three tensor calls — O(1) kernel launches regardless of P.
"""
from __future__ import annotations
import math
from typing import Callable, List
import time
import torch

from ..core.tensor import Tensor
from .population import Population


Fitness = Callable[[Tensor], Tensor]


class UMDA:
    """
    Univariate Marginal Distribution Algorithm (Mühlenbein & Paaß 1996).

    Parameters
    ----------
    pop_size        : total population size
    selection_ratio : fraction of top individuals used to estimate the model
    min_sigma       : lower bound on per-gene σ (prevents premature convergence)
    """

    def __init__(self,
                 pop_size: int,
                 genome_length: int,
                 selection_ratio: float = 0.5,
                 min_sigma: float = 1e-3,
                 device: str = "cpu"):
        self.pop_size = pop_size
        self.genome_length = genome_length
        self.n_select = max(2, int(pop_size * selection_ratio))
        self.min_sigma = min_sigma
        self._device = device
        self.generation = 0

        # initial population
        self._pop = torch.randn(pop_size, genome_length, device=device)
        self.best_x: torch.Tensor = self._pop[0].clone()
        self.best_f: float = -1e18

    # ------------------------------------------------------------------ #

    def step(self, fitness_fn: Fitness) -> float:
        dev = self._device

        # ── evaluate ──────────────────────────────────────────────────
        fit = fitness_fn(Tensor(self._pop)).data        # (P,)

        # ── track best ────────────────────────────────────────────────
        best_idx = fit.argmax()
        best_val = fit[best_idx].item()
        if best_val > self.best_f:
            self.best_f = best_val
            self.best_x = self._pop[best_idx].clone()

        # ── truncation selection → model estimation ────────────────────
        top_idx = fit.topk(self.n_select).indices
        selected = self._pop[top_idx]                   # (k, G)
        mu    = selected.mean(0)                        # (G,)
        sigma = selected.std(0).clamp(min=self.min_sigma)

        # ── sample new population ─────────────────────────────────────
        self._pop = torch.randn(
            self.pop_size, self.genome_length, device=dev
        ) * sigma + mu

        self.generation += 1
        return self.best_f


def run_eda(fitness_fn: Fitness,
            pop_size: int = 128,
            genome_length: int = 20,
            generations: int = 100,
            selection_ratio: float = 0.5,
            device: str = "cpu",
            verbose: bool = False) -> tuple[float, list]:
    eda = UMDA(pop_size, genome_length,
               selection_ratio=selection_ratio, device=device)
    log = []
    t0 = time.perf_counter()
    for g in range(generations):
        best = eda.step(fitness_fn)
        log.append(best)
        if verbose and g % (generations // 5) == 0:
            print(f"  EDA gen {g:4d} | best={best:.4f} | "
                  f"{time.perf_counter()-t0:.2f}s")
    return best, log
