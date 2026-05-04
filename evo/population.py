"""
Population — the central abstraction for GPU-accelerated evolutionary
computation in EvoGrad.

The entire population is stored as a single 2-D tensor:
    shape = (pop_size, genome_length)

Every evolutionary operator (selection, crossover, mutation, fitness)
is expressed as a batched tensor operation — no Python loops, no
CPU/GPU round-trips between individuals.

Supported genome types:
  - real-valued  (float32)   — for weight-space evolution
  - binary       (bool)      — for NAS / feature selection
  - integer      (int64)     — for combinatorial problems
"""
from __future__ import annotations
from typing import Callable, Optional, List
import torch

from ..core.tensor import Tensor


class Population:
    """
    A generation of candidate solutions stored as a contiguous tensor.

    Attributes
    ----------
    genomes : Tensor  shape (P, G)
        P = population size, G = genome length.
    fitness : Tensor | None  shape (P,)
        Filled after evaluate() is called.
    generation : int
        Generation counter (incremented by evolve()).
    """

    def __init__(self, genomes: Tensor, fitness: Optional[Tensor] = None):
        assert genomes.ndim == 2, "genomes must be 2-D: (pop_size, genome_len)"
        self._genomes = genomes
        self._fitness = fitness
        self.generation = 0

    # ------------------------------------------------------------------ #
    # Factory methods
    # ------------------------------------------------------------------ #

    @classmethod
    def random(cls, pop_size: int, genome_length: int,
               low: float = -1.0, high: float = 1.0,
               device: str = "cpu") -> "Population":
        g = torch.rand(pop_size, genome_length, device=device) * (high - low) + low
        return cls(Tensor(g))

    @classmethod
    def gaussian(cls, pop_size: int, genome_length: int,
                 mean: float = 0.0, std: float = 1.0,
                 device: str = "cpu") -> "Population":
        g = torch.randn(pop_size, genome_length, device=device) * std + mean
        return cls(Tensor(g))

    @classmethod
    def binary(cls, pop_size: int, genome_length: int,
               device: str = "cpu") -> "Population":
        g = torch.randint(0, 2, (pop_size, genome_length),
                          dtype=torch.float32, device=device)
        return cls(Tensor(g))

    @classmethod
    def from_network_weights(cls, params: List[torch.nn.Parameter],
                              pop_size: int, noise_std: float = 0.01,
                              device: str = "cpu") -> "Population":
        """
        Seed a population from an existing model's weights.
        Each individual is the flat weight vector + Gaussian noise.
        Useful for warm-starting NAS or ensemble search.
        """
        flat = torch.cat([p.data.flatten() for p in params])
        genomes = flat.unsqueeze(0).expand(pop_size, -1).clone()
        genomes += torch.randn_like(genomes) * noise_std
        return cls(Tensor(genomes.to(device)))

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def genomes(self) -> Tensor:
        return self._genomes

    @property
    def fitness(self) -> Optional[Tensor]:
        return self._fitness

    @property
    def size(self) -> int:
        return self._genomes.shape[0]

    @property
    def genome_length(self) -> int:
        return self._genomes.shape[1]

    @property
    def device(self) -> str:
        return str(self._genomes.device)

    def best(self) -> Tensor:
        """Return the genome with highest fitness."""
        if self._fitness is None:
            raise RuntimeError("Call evaluate() before accessing best()")
        idx = self._fitness.data.argmax()
        return Tensor(self._genomes.data[idx])

    def best_fitness(self) -> float:
        if self._fitness is None:
            raise RuntimeError("Call evaluate() before accessing best_fitness()")
        return self._fitness.data.max().item()

    def mean_fitness(self) -> float:
        if self._fitness is None:
            raise RuntimeError("Call evaluate() before accessing mean_fitness()")
        return self._fitness.data.mean().item()

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def evaluate(self, fitness_fn: Callable[[Tensor], Tensor]) -> "Population":
        """
        Compute fitness for every individual in one vectorised call.

        fitness_fn receives the entire genome matrix (P, G) and must
        return a fitness vector (P,).  This keeps all computation on
        the same device without per-individual Python calls.
        """
        fit = fitness_fn(self._genomes)
        assert fit.shape == (self.size,), (
            f"fitness_fn must return shape ({self.size},), got {fit.shape}"
        )
        return Population(self._genomes, fit)

    # ------------------------------------------------------------------ #
    # Elitism
    # ------------------------------------------------------------------ #

    def elite(self, n: int) -> "Population":
        """Return the top-n individuals (sorted by fitness)."""
        if self._fitness is None:
            raise RuntimeError("evaluate() first")
        idx = self._fitness.data.topk(n).indices
        return Population(
            Tensor(self._genomes.data[idx]),
            Tensor(self._fitness.data[idx]),
        )

    # ------------------------------------------------------------------ #
    # Replacement
    # ------------------------------------------------------------------ #

    def replace(self, new_genomes: Tensor) -> "Population":
        """Return a new Population with updated genomes, fitness reset."""
        p = Population(new_genomes)
        p.generation = self.generation + 1
        return p

    def __repr__(self) -> str:
        fit_str = (
            f"best={self.best_fitness():.4f}, mean={self.mean_fitness():.4f}"
            if self._fitness is not None else "unevaluated"
        )
        return (f"Population(size={self.size}, genome_len={self.genome_length}, "
                f"gen={self.generation}, fitness=[{fit_str}])")
