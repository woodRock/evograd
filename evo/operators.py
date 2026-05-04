"""
GPU-accelerated genetic operators.

All operators work on the (P, G) genome tensor directly — no Python
loops over individuals.  Where possible, operations fuse into a single
CUDA kernel call via PyTorch's JIT / inductor backend.

Operators
---------
Selection:
  tournament_select   — O(P · k) parallel tournament comparisons
  roulette_select     — stochastic fitness-proportional, O(P)
  rank_select         — linear rank-based probabilities, O(P log P)
  sus_select          — stochastic universal sampling, O(P)

Crossover:
  uniform_crossover   — each gene inherited from random parent, O(P·G)
  one_point_crossover — single random cut point per individual, O(P·G)
  two_point_crossover — two cut points, O(P·G)
  blend_crossover     — BLX-α for real-coded genomes, O(P·G)
  sbx_crossover       — simulated binary crossover, O(P·G)

Mutation:
  gaussian_mutate     — add N(0,σ) noise per gene, O(P·G)
  uniform_mutate      — random replacement within bounds, O(P·G)
  polynomial_mutate   — polynomial mutation (NAS/real-coded), O(P·G)
  bitflip_mutate      — flip bits at rate p (binary genomes), O(P·G)
  creep_mutate        — small bounded perturbation, O(P·G)
"""
from __future__ import annotations
import torch
import math

from ..core.tensor import Tensor
from .population import Population


# ------------------------------------------------------------------ #
# Selection
# ------------------------------------------------------------------ #

def tournament_select(pop: Population, k: int = 7) -> Tensor:
    """
    Batched tournament selection.

    For each of the P slots in the output, sample k candidates
    uniformly at random and return the one with highest fitness.
    All P tournaments happen in parallel via gather+max on the GPU.

    Returns a (P, G) genome tensor of selected individuals.
    """
    if pop.fitness is None:
        raise RuntimeError("evaluate() the population before selection")
    P, G = pop.size, pop.genome_length
    device = pop.genomes.device
    fitness = pop.fitness.data             # (P,)
    genomes = pop.genomes.data             # (P, G)

    # candidate indices: (P, k)
    candidates = torch.randint(0, P, (P, k), device=device)
    cand_fit = fitness[candidates]         # (P, k)
    winners = cand_fit.argmax(dim=1)       # (P,)
    selected_idx = candidates[torch.arange(P, device=device), winners]
    return Tensor(genomes[selected_idx])


def roulette_select(pop: Population) -> Tensor:
    """
    Fitness-proportional (roulette wheel) selection.
    Negative fitnesses are shifted so the minimum equals a small ε.
    """
    if pop.fitness is None:
        raise RuntimeError("evaluate() first")
    P = pop.size
    device = pop.genomes.device
    fitness = pop.fitness.data.clone()
    fitness -= fitness.min() - 1e-8        # shift to positive
    probs = fitness / fitness.sum()
    idx = torch.multinomial(probs, P, replacement=True)
    return Tensor(pop.genomes.data[idx])


def rank_select(pop: Population) -> Tensor:
    """Linear rank-based selection (rank 1 = best)."""
    if pop.fitness is None:
        raise RuntimeError("evaluate() first")
    P = pop.size
    device = pop.genomes.device
    # argsort descending → rank positions
    order = pop.fitness.data.argsort(descending=True)
    ranks = torch.zeros(P, device=device)
    ranks[order] = torch.arange(P, dtype=torch.float32, device=device)
    # linear rank probability: P-rank / sum
    probs = (P - ranks) / (P * (P + 1) / 2)
    idx = torch.multinomial(probs, P, replacement=True)
    return Tensor(pop.genomes.data[idx])


def sus_select(pop: Population) -> Tensor:
    """
    Stochastic universal sampling — lower variance than roulette.
    Useful when fitness values span many orders of magnitude.
    """
    if pop.fitness is None:
        raise RuntimeError("evaluate() first")
    P = pop.size
    device = pop.genomes.device
    fitness = pop.fitness.data.clone()
    fitness -= fitness.min() - 1e-8
    cumsum = fitness.cumsum(0)
    total = cumsum[-1]
    step = total / P
    start = torch.rand(1, device=device) * step
    pointers = start + step * torch.arange(P, device=device, dtype=torch.float32)
    # vectorised search: for each pointer find its bin in cumsum
    idx = torch.searchsorted(cumsum, pointers)
    idx = idx.clamp(0, P - 1)
    return Tensor(pop.genomes.data[idx])


# ------------------------------------------------------------------ #
# Crossover
# ------------------------------------------------------------------ #

def uniform_crossover(parents: Tensor, rate: float = 0.5) -> Tensor:
    """
    Uniform crossover: pair consecutive individuals and swap each gene
    independently with probability `rate`.

    Input:  (P, G)  —  P must be even
    Output: (P, G)  offspring
    """
    P, G = parents.shape
    assert P % 2 == 0, "Population size must be even for crossover"
    device = parents.device
    g = parents.data.clone()
    p1 = g[0::2]                           # (P/2, G)
    p2 = g[1::2]
    mask = torch.rand(P // 2, G, device=device) < rate
    child1 = torch.where(mask, p1, p2)
    child2 = torch.where(mask, p2, p1)
    offspring = torch.empty_like(g)
    offspring[0::2] = child1
    offspring[1::2] = child2
    return Tensor(offspring)


def one_point_crossover(parents: Tensor) -> Tensor:
    P, G = parents.shape
    assert P % 2 == 0
    device = parents.device
    g = parents.data.clone()
    p1, p2 = g[0::2], g[1::2]
    cuts = torch.randint(1, G, (P // 2,), device=device)   # (P/2,)
    idx = torch.arange(G, device=device).unsqueeze(0)      # (1, G)
    mask = idx < cuts.unsqueeze(1)                          # (P/2, G)
    child1 = torch.where(mask, p1, p2)
    child2 = torch.where(mask, p2, p1)
    offspring = torch.empty_like(g)
    offspring[0::2] = child1
    offspring[1::2] = child2
    return Tensor(offspring)


def two_point_crossover(parents: Tensor) -> Tensor:
    P, G = parents.shape
    assert P % 2 == 0
    device = parents.device
    g = parents.data.clone()
    p1, p2 = g[0::2], g[1::2]
    cuts = torch.sort(
        torch.randint(0, G, (P // 2, 2), device=device), dim=1
    ).values                                                 # (P/2, 2)
    idx = torch.arange(G, device=device).unsqueeze(0)
    mask = (idx >= cuts[:, 0:1]) & (idx < cuts[:, 1:2])
    child1 = torch.where(mask, p2, p1)
    child2 = torch.where(mask, p1, p2)
    offspring = torch.empty_like(g)
    offspring[0::2] = child1
    offspring[1::2] = child2
    return Tensor(offspring)


def sbx_crossover(parents: Tensor, eta: float = 20.0) -> Tensor:
    """
    Simulated Binary Crossover (SBX) — mimics one-point crossover
    in real-coded parameter space.  `eta` controls offspring spread
    (higher = closer to parents).
    """
    P, G = parents.shape
    assert P % 2 == 0
    device = parents.device
    g = parents.data
    p1, p2 = g[0::2].clone(), g[1::2].clone()
    u = torch.rand(P // 2, G, device=device).clamp(1e-9, 1.0 - 1e-9)
    beta = torch.where(
        u <= 0.5,
        (2.0 * u).pow(1.0 / (eta + 1)),
        (1.0 / (2.0 * (1.0 - u))).pow(1.0 / (eta + 1)),
    )
    child1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    child2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    offspring = torch.empty_like(g)
    offspring[0::2] = child1
    offspring[1::2] = child2
    return Tensor(offspring)


def blend_crossover(parents: Tensor, alpha: float = 0.5) -> Tensor:
    """BLX-α: offspring sampled from an expanded interval around parents."""
    P, G = parents.shape
    assert P % 2 == 0
    device = parents.device
    g = parents.data
    p1, p2 = g[0::2], g[1::2]
    d = (p2 - p1).abs()
    lo = torch.min(p1, p2) - alpha * d
    hi = torch.max(p1, p2) + alpha * d
    child1 = lo + torch.rand(P // 2, G, device=device) * (hi - lo)
    child2 = lo + torch.rand(P // 2, G, device=device) * (hi - lo)
    offspring = torch.empty_like(g)
    offspring[0::2] = child1
    offspring[1::2] = child2
    return Tensor(offspring)


# ------------------------------------------------------------------ #
# Mutation
# ------------------------------------------------------------------ #

def gaussian_mutate(offspring: Tensor, sigma: float = 0.01,
                    rate: float = 1.0) -> Tensor:
    """
    Add N(0, σ) to each gene independently.
    `rate` < 1 applies mutation to only that fraction of genes.
    """
    g = offspring.data.clone()
    if rate < 1.0:
        mask = torch.rand_like(g) < rate
        noise = torch.randn_like(g) * sigma
        g = torch.where(mask, g + noise, g)
    else:
        g = g + torch.randn_like(g) * sigma
    return Tensor(g)


def uniform_mutate(offspring: Tensor, rate: float = 0.05,
                   low: float = -1.0, high: float = 1.0) -> Tensor:
    """Replace each gene uniformly at random with probability `rate`."""
    g = offspring.data.clone()
    mask = torch.rand_like(g) < rate
    replacement = torch.rand_like(g) * (high - low) + low
    return Tensor(torch.where(mask, replacement, g))


def polynomial_mutate(offspring: Tensor, eta_m: float = 20.0,
                      rate: float = 0.1,
                      low: float = -1.0, high: float = 1.0) -> Tensor:
    """
    Polynomial mutation — standard in NSGA-II / MOEA/D.
    Produces offspring close to parent with high probability.
    """
    g = offspring.data.clone()
    P, G = g.shape
    device = g.device
    mask = torch.rand(P, G, device=device) < rate
    u = torch.rand(P, G, device=device)
    delta_l = (g - low) / (high - low)
    delta_r = (high - g) / (high - low)
    delta = torch.where(
        u <= 0.5,
        (2.0 * u + (1 - 2 * u) * (1 - delta_l).pow(eta_m + 1)
         ).pow(1 / (eta_m + 1)) - 1,
        1 - (2.0 * (1 - u) + 2 * (u - 0.5)
             * (1 - delta_r).pow(eta_m + 1)
             ).pow(1 / (eta_m + 1)),
    ) * (high - low)
    g_mut = (g + delta).clamp(low, high)
    return Tensor(torch.where(mask, g_mut, g))


def bitflip_mutate(offspring: Tensor, rate: float = 0.01) -> Tensor:
    """Flip each bit (0↔1) with probability `rate`. For binary genomes."""
    g = offspring.data.clone()
    flip = torch.rand_like(g) < rate
    return Tensor(torch.where(flip, 1.0 - g, g))


def creep_mutate(offspring: Tensor, step: float = 0.05,
                 rate: float = 0.1) -> Tensor:
    """Small bounded additive perturbation — useful for integer genomes."""
    g = offspring.data.clone()
    mask = torch.rand_like(g) < rate
    delta = (torch.rand_like(g) * 2 - 1) * step
    return Tensor(torch.where(mask, g + delta, g))
