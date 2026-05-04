"""
Fitness functions for evolutionary computation.

Each fitness function takes the genome matrix (P, G) and returns a
fitness vector (P,) — the entire population is scored in one call,
keeping all tensors on the same device.

Built-in functions
------------------
  rastrigin, ackley, rosenbrock, sphere, schwefel  — benchmark landscapes
  network_fitness    — evaluate a population of weight vectors on a dataset
  ensemble_fitness   — combine multiple decoded networks, vote on accuracy
"""
from __future__ import annotations
from typing import Callable, List, Optional
import math
import torch

from ..core.tensor import Tensor
from ..nn.layers import Layer


# ------------------------------------------------------------------ #
# Synthetic benchmark landscapes (maximisation: return negative cost)
# ------------------------------------------------------------------ #

def rastrigin(genomes: Tensor, A: float = 10.0) -> Tensor:
    """
    Rastrigin: highly multi-modal, widely used for EA benchmarking.
    Global minimum at origin.  We return -f so higher = better.
    """
    x = genomes.data
    n = x.shape[1]
    cost = A * n + (x ** 2 - A * torch.cos(2 * math.pi * x)).sum(dim=1)
    return Tensor(-cost)


def ackley(genomes: Tensor) -> Tensor:
    x = genomes.data
    n = x.shape[1]
    a, b, c = 20.0, 0.2, 2 * math.pi
    s1 = (-b * (x ** 2).mean(dim=1).sqrt()).exp()
    s2 = (c * x).cos().mean(dim=1).exp()
    cost = -a * s1 - s2 + a + math.e
    return Tensor(-cost)


def rosenbrock(genomes: Tensor) -> Tensor:
    """Banana function — narrow curved valley toward global min."""
    x = genomes.data
    xi = x[:, :-1]
    xj = x[:, 1:]
    cost = (100 * (xj - xi ** 2) ** 2 + (1 - xi) ** 2).sum(dim=1)
    return Tensor(-cost)


def sphere(genomes: Tensor) -> Tensor:
    cost = (genomes.data ** 2).sum(dim=1)
    return Tensor(-cost)


def schwefel(genomes: Tensor) -> Tensor:
    x = genomes.data
    cost = 418.9829 * x.shape[1] - (x * torch.sin(x.abs().sqrt())).sum(dim=1)
    return Tensor(-cost)


# ------------------------------------------------------------------ #
# Neural-network fitness: decode genome → weights → accuracy
# ------------------------------------------------------------------ #

class NetworkFitness:
    """
    Scores each individual by loading its genome as the weights of a
    neural network and measuring accuracy (or any scalar metric) on a
    fixed dataset batch.

    Usage:
        fn = NetworkFitness(model_factory, X_batch, y_batch)
        pop = pop.evaluate(fn)

    model_factory() must return a fresh Layer instance with the same
    parameter count as genome_length every time it is called.
    The factory is called once at construction to count params.
    """

    def __init__(self,
                 model_factory: Callable[[], Layer],
                 X: Tensor,
                 y: Tensor,
                 metric: str = "accuracy"):
        self._factory = model_factory
        self._X = X
        self._y = y
        self._metric = metric
        # count parameters
        proto = model_factory()
        self._param_sizes = [p.numel() for p in proto.parameters()]
        self._total_params = sum(self._param_sizes)

    def __call__(self, genomes: Tensor) -> Tensor:
        P = genomes.shape[0]
        device = genomes.device
        scores = torch.zeros(P, device=device)

        # evaluate each individual (vectorised where possible)
        for i in range(P):
            genome = genomes.data[i]
            model = self._factory()
            # load genome into model parameters
            offset = 0
            for param in model.parameters():
                n = param.numel()
                param.data.copy_(
                    genome[offset:offset + n].reshape(param.shape)
                )
                offset += n
            with torch.no_grad():
                out = model(self._X)
                if self._metric == "accuracy":
                    pred = out.data.argmax(dim=1)
                    scores[i] = (pred == self._y.data).float().mean()
                elif self._metric == "neg_loss":
                    from ..nn.functional import cross_entropy
                    loss = cross_entropy(out, self._y)
                    scores[i] = -loss.data
                else:
                    raise ValueError(f"Unknown metric: {self._metric!r}")
        return Tensor(scores)


class EnsembleFitness:
    """
    Fitness for ensemble search: the genome encodes which models to
    include and their weights.  Fitness = weighted vote accuracy.

    genome layout: [w_0, w_1, ..., w_k]  — inclusion weights
    """

    def __init__(self,
                 models: List[Layer],
                 X: Tensor,
                 y: Tensor):
        self._models = models
        self._X = X
        self._y = y
        # precompute logits for all base models
        with torch.no_grad():
            self._logits = torch.stack(
                [m(X).data for m in models], dim=1
            )  # (N, k, C)

    def __call__(self, genomes: Tensor) -> Tensor:
        P = genomes.shape[0]
        # genomes: (P, k) — weights for each model
        w = torch.softmax(genomes.data, dim=1)     # (P, k) normalised
        # weighted vote: (P, N, C)
        logits = self._logits.unsqueeze(0)         # (1, N, k, C)
        w4d = w.unsqueeze(1).unsqueeze(-1)         # (P, 1, k, 1)
        ensemble_logits = (logits * w4d).sum(dim=2) # (P, N, C)
        pred = ensemble_logits.argmax(dim=-1)      # (P, N)
        correct = (pred == self._y.data.unsqueeze(0)).float()  # (P, N)
        return Tensor(correct.mean(dim=1))         # (P,)
