"""
Serialization utilities for models and populations.

Provides standard ways to save/load weights and candidate sets to disk
leveraging torch.save while maintaining EvoGrad metadata.
"""
from __future__ import annotations
import torch
from typing import Union

from .tensor import Tensor
from ..evo.population import Population
from ..nn.layers import Layer


def save_model(model: Layer, path: str):
    """Save model weights (state dict)."""
    # Extract state dict from underlying modules
    state = {}
    params = model.parameters()
    # Since EvoGrad layers are wrappers, we can't easily use model.state_dict()
    # if it hasn't been implemented to be recursive. 
    # For now, we save the flat list of parameters.
    torch.save([p.data for p in params], path)


def load_model(model: Layer, path: str):
    """Load model weights."""
    weights = torch.load(path)
    params = model.parameters()
    for p, w in zip(params, weights):
        p.data.copy_(w)


def save_population(pop: Population, path: str):
    """Save the entire population (genomes, fitness, generation)."""
    state = {
        "genomes": pop.genomes.data,
        "fitness": pop.fitness.data if pop.fitness is not None else None,
        "generation": pop.generation,
    }
    torch.save(state, path)


def load_population(path: str, device: str = "cpu") -> Population:
    """Load a population from disk."""
    state = torch.load(path, map_location=device)
    genomes = Tensor(state["genomes"].to(device))
    fitness = Tensor(state["fitness"].to(device)) if state["fitness"] is not None else None
    pop = Population(genomes, fitness)
    pop.generation = state["generation"]
    return pop
