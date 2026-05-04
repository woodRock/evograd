"""
Attempt to load the compiled CUDA extension.
Falls back silently to the pure-PyTorch implementations.
"""
import importlib

_cuda_ops = None

try:
    _cuda_ops = importlib.import_module("evograd_cuda")
except (ModuleNotFoundError, ImportError):
    pass


def is_available() -> bool:
    return _cuda_ops is not None


def tournament_select(fitness, k: int, seed: int = 0):
    if _cuda_ops is not None:
        import torch
        return _cuda_ops.tournament_select(fitness, k, seed)
    raise RuntimeError("CUDA extension not compiled. Run: python setup.py build_ext --inplace")


def uniform_crossover(parents, rate: float = 0.5, seed: int = 0):
    if _cuda_ops is not None:
        return _cuda_ops.uniform_crossover(parents, rate, seed)
    raise RuntimeError("CUDA extension not compiled.")


def gaussian_mutate(genomes, sigma: float = 0.01, rate: float = 1.0,
                    seed: int = 0):
    if _cuda_ops is not None:
        return _cuda_ops.gaussian_mutate(genomes, sigma, rate, seed)
    raise RuntimeError("CUDA extension not compiled.")


def rastrigin_fitness(genomes, A: float = 10.0):
    if _cuda_ops is not None:
        return _cuda_ops.rastrigin_fitness(genomes, A)
    raise RuntimeError("CUDA extension not compiled.")
