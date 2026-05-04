"""
EvoEngine — the main evolutionary loop.

Keeps the entire evolutionary cycle (evaluate → select → crossover →
mutate → replace) on-device with no host/device round-trips between
steps.  A callback hook fires at the end of each generation so the
caller can log, checkpoint, or stop early without coupling to loop
internals.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, List
import time

from ..core.tensor import Tensor
from .population import Population
from . import operators as ops


Fitness = Callable[[Tensor], Tensor]


@dataclass
class EvoConfig:
    """Hyperparameters for one evolutionary run."""
    generations: int = 100
    elite_size: int = 2

    # selection
    selection: str = "tournament"          # tournament | roulette | rank | sus
    tournament_k: int = 7

    # crossover
    crossover: str = "uniform"             # uniform | one_point | two_point | sbx | blend
    crossover_rate: float = 0.5
    sbx_eta: float = 20.0

    # mutation
    mutation: str = "gaussian"             # gaussian | uniform | polynomial | bitflip
    mutation_sigma: float = 0.01
    mutation_rate: float = 1.0             # fraction of genes mutated

    # bounds (for polynomial / uniform mutation)
    lower_bound: float = -1.0
    upper_bound: float = 1.0

    verbose: bool = True
    log_every: int = 10


@dataclass
class GenerationLog:
    generation: int
    best_fitness: float
    mean_fitness: float
    elapsed_s: float


class EvoEngine:
    """
    Stateless engine: call run() with a Population and a fitness function.

    Example
    -------
    engine = EvoEngine(EvoConfig(generations=200, mutation_sigma=0.05))
    pop, logs = engine.run(pop, fitness_fn=rastrigin)
    print(pop.best())
    """

    def __init__(self, config: EvoConfig = None):
        self.config = config or EvoConfig()

    def run(
        self,
        population: Population,
        fitness_fn: Fitness,
        callback: Optional[Callable[[Population, List[GenerationLog]], None]] = None,
    ) -> tuple[Population, List[GenerationLog]]:
        cfg = self.config
        logs: List[GenerationLog] = []
        pop = population

        t0 = time.perf_counter()

        for gen in range(cfg.generations):
            # ── Evaluate ──────────────────────────────────────────────
            pop = pop.evaluate(fitness_fn)

            # ── Log ───────────────────────────────────────────────────
            elapsed = time.perf_counter() - t0
            log = GenerationLog(gen, pop.best_fitness(),
                                pop.mean_fitness(), elapsed)
            logs.append(log)

            if cfg.verbose and (gen % cfg.log_every == 0 or
                                 gen == cfg.generations - 1):
                print(f"  gen {gen:4d} | best={log.best_fitness:.5f} "
                      f"mean={log.mean_fitness:.5f} | {elapsed:.2f}s")

            if callback is not None:
                callback(pop, logs)

            # ── Elitism: preserve top individuals ─────────────────────
            elite_pop = pop.elite(cfg.elite_size)

            # ── Selection ─────────────────────────────────────────────
            selected = self._select(pop)

            # ── Crossover ─────────────────────────────────────────────
            offspring = self._crossover(selected)

            # ── Mutation ──────────────────────────────────────────────
            mutated = self._mutate(offspring)

            # ── Replace (keep elite) ──────────────────────────────────
            import torch
            new_genomes = torch.cat([
                elite_pop.genomes.data,
                mutated.data[cfg.elite_size:],
            ], dim=0)
            pop = pop.replace(Tensor(new_genomes))

        # Final evaluation
        pop = pop.evaluate(fitness_fn)
        return pop, logs

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _select(self, pop: Population) -> Tensor:
        cfg = self.config
        if cfg.selection == "tournament":
            return ops.tournament_select(pop, k=cfg.tournament_k)
        elif cfg.selection == "roulette":
            return ops.roulette_select(pop)
        elif cfg.selection == "rank":
            return ops.rank_select(pop)
        elif cfg.selection == "sus":
            return ops.sus_select(pop)
        else:
            raise ValueError(f"Unknown selection: {cfg.selection!r}")

    def _crossover(self, selected: Tensor) -> Tensor:
        cfg = self.config
        if cfg.crossover == "uniform":
            return ops.uniform_crossover(selected, rate=cfg.crossover_rate)
        elif cfg.crossover == "one_point":
            return ops.one_point_crossover(selected)
        elif cfg.crossover == "two_point":
            return ops.two_point_crossover(selected)
        elif cfg.crossover == "sbx":
            return ops.sbx_crossover(selected, eta=cfg.sbx_eta)
        elif cfg.crossover == "blend":
            return ops.blend_crossover(selected, alpha=cfg.crossover_rate)
        else:
            raise ValueError(f"Unknown crossover: {cfg.crossover!r}")

    def _mutate(self, offspring: Tensor) -> Tensor:
        cfg = self.config
        if cfg.mutation == "gaussian":
            return ops.gaussian_mutate(offspring, sigma=cfg.mutation_sigma,
                                        rate=cfg.mutation_rate)
        elif cfg.mutation == "uniform":
            return ops.uniform_mutate(offspring, rate=cfg.mutation_rate,
                                       low=cfg.lower_bound,
                                       high=cfg.upper_bound)
        elif cfg.mutation == "polynomial":
            return ops.polynomial_mutate(offspring, rate=cfg.mutation_rate,
                                          low=cfg.lower_bound,
                                          high=cfg.upper_bound)
        elif cfg.mutation == "bitflip":
            return ops.bitflip_mutate(offspring, rate=cfg.mutation_rate)
        else:
            raise ValueError(f"Unknown mutation: {cfg.mutation!r}")
