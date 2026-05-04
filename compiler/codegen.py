"""
IRGraph → executable EvoGrad model.

The code generator walks the IR in topological order and emits:
  - A `torch.nn.Module` subclass for network blocks
  - An `EvoEngine` instance for population blocks
  - A `HybridTrainer` for train blocks

The generated module is returned as a live Python object, not a string
of source — this avoids the eval() security problem and lets PyTorch's
JIT / Inductor compile it further.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import torch
import torch.nn as nn

from ..core.ir import IRGraph, IRNode, NodeKind
from ..core.tensor import Tensor
from ..nn import layers as L
from ..evo.engine import EvoEngine, EvoConfig
from .parser import Program, NetworkBlock, PopulationBlock, EvolveSpec


# ------------------------------------------------------------------ #
# Network codegen
# ------------------------------------------------------------------ #

def _build_network(block: NetworkBlock, device: str = "cpu") -> nn.Module:
    """
    Materialise a NetworkBlock AST into a standard nn.Sequential.

    We emit pure torch.nn modules (not EvoGrad wrappers) so that
    autograd works without extra plumbing.  The EvoGrad functional
    ops are used for inference via the Tensor API; for gradient
    training the PyTorch graph is used directly.
    """
    torch_layers: Dict[str, List[nn.Module]] = {}

    for stmt in block.layers:
        mods: List[nn.Module] = []
        for stage in stmt.pipeline:
            mods.extend(_build_torch_layer(stage, device))
        torch_layers[stmt.var_name] = mods

    ordered: List[nn.Module] = []
    chain = block.forward if block.forward else list(torch_layers.keys())
    for name in chain:
        if name in torch_layers:
            ordered.extend(torch_layers[name])

    return nn.Sequential(*ordered).to(device)


def _build_torch_layer(expr, device: str) -> List[nn.Module]:
    """Return a list of nn.Module objects for one DSL layer expression."""
    name = expr.name
    args = expr.args
    kwargs = expr.kwargs

    if name in ("Dense", "Linear"):
        in_f, out_f = int(args[0]), int(args[1])
        return [nn.Linear(in_f, out_f)]
    elif name == "Conv2d":
        in_ch = int(args[0])
        out_ch = int(args[1])
        k = int(args[2]) if len(args) > 2 else int(kwargs.get("kernel", 3))
        return [nn.Conv2d(in_ch, out_ch, k)]
    elif name == "ReLU":
        return [nn.ReLU()]
    elif name == "GELU":
        return [nn.GELU()]
    elif name == "SiLU":
        return [nn.SiLU()]
    elif name == "Sigmoid":
        return [nn.Sigmoid()]
    elif name == "Tanh":
        return [nn.Tanh()]
    elif name in ("Softmax", "softmax"):
        dim = int(kwargs.get("dim", -1))
        return [nn.Softmax(dim=dim)]
    elif name == "LayerNorm":
        size = int(args[0]) if args else int(kwargs.get("size", 256))
        return [nn.LayerNorm(size)]
    elif name == "Dropout":
        p = float(args[0]) if args else float(kwargs.get("p", 0.5))
        return [nn.Dropout(p=p)]
    elif name == "Flatten":
        return [nn.Flatten()]
    elif name == "MaxPool2d":
        k = int(args[0]) if args else int(kwargs.get("kernel_size", 2))
        return [nn.MaxPool2d(k)]
    else:
        raise ValueError(f"Unknown layer type: {name!r}")


# ------------------------------------------------------------------ #
# Population / EA codegen
# ------------------------------------------------------------------ #

def _build_evo_engine(block: PopulationBlock) -> EvoConfig:
    cfg = EvoConfig()
    if block.evolve:
        e = block.evolve
        if e.select:
            cfg.selection = e.select.name.lower()
            cfg.tournament_k = int(e.select.kwargs.get("k", 7))
        if e.crossover:
            cfg.crossover = e.crossover.name.lower()
            cfg.crossover_rate = float(e.crossover.kwargs.get("rate", 0.5))
        if e.mutate:
            cfg.mutation = e.mutate.name.lower()
            cfg.mutation_sigma = float(e.mutate.kwargs.get("sigma", 0.01))
    return cfg


# ------------------------------------------------------------------ #
# Top-level compilation
# ------------------------------------------------------------------ #

class CompiledProgram:
    """
    The output of compile().

    Attributes:
        networks   — dict of name → nn.Module
        evo_configs — dict of name → EvoConfig
        train_mode — string from the train block ('hybrid', 'gradient', …)
    """
    def __init__(self):
        self.networks: Dict[str, nn.Module] = {}
        self.evo_configs: Dict[str, EvoConfig] = {}
        self.train_mode: Optional[str] = None


def compile_program(program: Program, device: str = "cpu") -> CompiledProgram:
    """Lower a parsed + type-checked Program to live Python objects."""
    out = CompiledProgram()

    for net in program.networks:
        out.networks[net.name] = _build_network(net, device)

    for pop in program.populations:
        out.evo_configs[pop.name] = _build_evo_engine(pop)

    if program.train and program.train.mode:
        out.train_mode = program.train.mode.name.lower()

    return out
