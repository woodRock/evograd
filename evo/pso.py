"""
Vectorized Particle Swarm Optimization (PSO).

The entire swarm lives as two (P, G) tensors — positions and
velocities — so every velocity/position update is a single fused
broadcast op rather than P individual Python calls.

Standard constriction-coefficient PSO (Clerc & Kennedy 2002):
    v' = χ [ v + c1·r1·(pbest - x) + c2·r2·(gbest - x) ]
    x' = x + v'

where χ = 2 / |2 - φ - √(φ²-4φ)|,  φ = c1 + c2 > 4.
"""
from __future__ import annotations
import math
from typing import Callable, List, Optional
import time
import torch

from ..core.tensor import Tensor
from ..mps.device import sync


Fitness = Callable[[Tensor], Tensor]


class PSO:
    """
    Vectorized PSO.  Call step() each generation; all arithmetic
    stays on-device with no per-particle Python overhead.
    """

    def __init__(self,
                 pop_size: int,
                 genome_length: int,
                 omega: float = 0.729,      # inertia weight (constriction χ)
                 c1: float = 1.494,         # cognitive coefficient
                 c2: float = 1.494,         # social coefficient
                 v_max: float = 2.0,
                 device: str = "cpu"):
        self.pop_size = pop_size
        self.genome_length = genome_length
        self.omega = omega
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self._device = device
        self.generation = 0

        self.positions  = torch.randn(pop_size, genome_length, device=device)
        self.velocities = torch.zeros(pop_size, genome_length, device=device)

        self.pbest_pos = self.positions.clone()
        self.pbest_fit = torch.full((pop_size,), -1e18, device=device)

        self.gbest_pos: Optional[torch.Tensor] = None
        self.gbest_fit: float = -1e18

    # ------------------------------------------------------------------ #

    def step(self, fitness_fn: Fitness) -> float:
        fit = fitness_fn(Tensor(self.positions)).data   # (P,)

        # ── update personal bests ─────────────────────────────────────
        improved = fit > self.pbest_fit
        self.pbest_pos = torch.where(
            improved.unsqueeze(1).expand_as(self.pbest_pos),
            self.positions, self.pbest_pos
        )
        self.pbest_fit = torch.where(improved, fit, self.pbest_fit)

        # ── update global best ────────────────────────────────────────
        best_idx = self.pbest_fit.argmax()
        best_val = self.pbest_fit[best_idx].item()
        if best_val > self.gbest_fit:
            self.gbest_fit = best_val
            self.gbest_pos = self.pbest_pos[best_idx].clone()

        # ── velocity + position update (fully vectorised) ─────────────
        gbest = self.gbest_pos.unsqueeze(0)             # (1, G)
        r1 = torch.rand_like(self.velocities)
        r2 = torch.rand_like(self.velocities)

        self.velocities = (
            self.omega * self.velocities
            + self.c1 * r1 * (self.pbest_pos - self.positions)
            + self.c2 * r2 * (gbest - self.positions)
        ).clamp(-self.v_max, self.v_max)

        self.positions = self.positions + self.velocities
        self.generation += 1
        return self.gbest_fit


def run_pso(fitness_fn: Fitness,
            pop_size: int = 128,
            genome_length: int = 20,
            generations: int = 100,
            device: str = "cpu",
            verbose: bool = False) -> tuple[float, list]:
    swarm = PSO(pop_size, genome_length, device=device)
    log = []
    t0 = time.perf_counter()
    for g in range(generations):
        best = swarm.step(fitness_fn)
        log.append(best)
        if verbose and g % (generations // 5) == 0:
            print(f"  PSO gen {g:4d} | best={best:.4f} | "
                  f"{time.perf_counter()-t0:.2f}s")
    return best, log
