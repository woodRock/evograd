"""
EvoGrad static type checker — catches shape mismatches before any
tensor is allocated.

Pass: parse(src) → Program → TypeChecker().check(program) → IRGraph

The checker:
  1. Resolves named layer definitions to known layer kinds.
  2. Propagates tensor shapes through every forward-pass connection.
  3. Emits a ShapeError with the exact line/node where the mismatch
     occurs so the user can fix it without running the program.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import re

from ..core.types import TensorType, Float32, ShapeError
from ..core.ir import IRGraph, IRNode
from .parser import (
    Program, NetworkBlock, PopulationBlock, TrainBlock,
    TypeExpr, LayerExpr, LayerStmt, EvolveSpec,
)


# ------------------------------------------------------------------ #
# Layer catalogue: maps DSL names → (kind, shape_fn)
# ------------------------------------------------------------------ #

def _infer_dense(in_type: TensorType, args, kwargs) -> TensorType:
    out_features = int(args[1]) if len(args) >= 2 else int(kwargs["out_features"])
    batch = in_type.shape[:-1]
    return TensorType(batch + (out_features,), in_type.dtype)


def _infer_conv2d(in_type: TensorType, args, kwargs) -> TensorType:
    out_ch = int(args[1]) if len(args) >= 2 else int(kwargs.get("out_channels", 1))
    kernel = int(args[2]) if len(args) >= 3 else int(kwargs.get("kernel_size", 3))
    stride = int(kwargs.get("stride", 1))
    padding = int(kwargs.get("padding", 0))
    B, C, H, W = in_type.shape
    H_out = None if H is None else (H + 2 * padding - kernel) // stride + 1
    W_out = None if W is None else (W + 2 * padding - kernel) // stride + 1
    return TensorType((B, out_ch, H_out, W_out), in_type.dtype)


_IDENTITY_LAYERS = {
    "ReLU", "GELU", "SiLU", "Sigmoid", "Tanh", "Softmax",
    "LayerNorm", "BatchNorm", "Dropout", "Flatten", "Residual",
}

_SHAPE_RULES: Dict[str, callable] = {
    "Dense": _infer_dense,
    "Linear": _infer_dense,
    "Conv2d": _infer_conv2d,
}


def _apply_layer(in_type: TensorType, layer: LayerExpr) -> TensorType:
    if layer.name in _IDENTITY_LAYERS:
        return in_type
    if layer.name in _SHAPE_RULES:
        return _SHAPE_RULES[layer.name](in_type, layer.args, layer.kwargs)
    # Unknown layer — pass shape through with a warning
    return in_type


# ------------------------------------------------------------------ #
# TypeChecker
# ------------------------------------------------------------------ #

class TypeChecker:
    """
    Raises ShapeError on any detectable shape mismatch.
    Returns a populated IRGraph on success.
    """

    def check(self, program: Program) -> Dict[str, IRGraph]:
        graphs: Dict[str, IRGraph] = {}

        for net in program.networks:
            graphs[net.name] = self._check_network(net)

        for pop in program.populations:
            graphs[pop.name] = self._check_population(pop, program)

        return graphs

    def _check_network(self, net: NetworkBlock) -> IRGraph:
        graph = IRGraph(net.name)

        # Determine input type
        if net.input_type:
            in_type = self._parse_tensor_type(net.input_type)
        else:
            in_type = TensorType((None, None), Float32)

        inp = graph.input("x", in_type)

        # Resolve layer variable types
        var_types: Dict[str, TensorType] = {"x": in_type}
        var_nodes: Dict[str, IRNode] = {"x": inp}

        # Resolve each layer statement, threading output type through
        # each pipeline stage.  The input to the first stage of a
        # statement is determined by the forward chain order (the
        # previous named layer), not always "x".
        prev_name = "x"
        for stmt in net.layers:
            current_type = var_types[prev_name]
            current_node = var_nodes[prev_name]

            for stage in stmt.pipeline:
                new_type = _apply_layer(current_type, stage)

                if stage.name in ("Dense", "Linear"):
                    in_f = int(stage.args[0]) if stage.args else 1
                    out_f = int(stage.args[1]) if len(stage.args) > 1 else 1
                    if (current_type.shape[-1] is not None and
                            current_type.shape[-1] != in_f):
                        raise ShapeError(
                            f"Layer '{stmt.var_name}': "
                            f"input features {current_type.shape[-1]} "
                            f"!= declared {in_f}"
                        )
                    current_node = graph.linear(
                        stmt.var_name, current_node, in_f, out_f
                    )
                else:
                    current_node = graph.activation(
                        f"{stmt.var_name}_{stage.name}",
                        current_node, stage.name.lower()
                    )

                current_type = new_type

            var_types[stmt.var_name] = current_type
            var_nodes[stmt.var_name] = current_node
            prev_name = stmt.var_name

        # Validate forward chain
        for i, name in enumerate(net.forward):
            if name not in var_types:
                raise ShapeError(
                    f"Forward chain references undefined layer: {name!r}"
                )

        # Validate output type if declared
        if net.forward and net.output_type:
            last_name = net.forward[-1]
            actual = var_types[last_name]
            declared = self._parse_tensor_type(net.output_type)
            try:
                actual.unify(declared)
            except ShapeError:
                raise ShapeError(
                    f"Network '{net.name}': output shape {actual} "
                    f"does not match declared {declared}"
                )

        # Emit output node
        if net.forward:
            last = var_nodes.get(net.forward[-1], inp)
        else:
            last = inp
        graph.output("output", last)
        graph.type_check()
        return graph

    def _check_population(self, pop: PopulationBlock,
                           program: Program) -> IRGraph:
        graph = IRGraph(pop.name)
        pop_size = genome_len = None
        if pop.genome_type:
            tt = self._parse_tensor_type(pop.genome_type)
            if len(tt.shape) == 2:
                pop_size, genome_len = tt.shape
        in_type = TensorType((pop_size, genome_len), Float32)
        pop_node = graph.input("population", in_type)
        fit_node = graph.input("fitness", TensorType((pop_size,), Float32))

        if pop.evolve:
            e = pop.evolve
            sel_method = e.select.name.lower() if e.select else "tournament"
            sel_k = e.select.kwargs.get("k", 7) if e.select else 7
            selected = graph.evo_select("selection", pop_node, fit_node,
                                         method=sel_method, k=int(sel_k))

            xo_method = e.crossover.name.lower() if e.crossover else "uniform"
            xo_rate = e.crossover.kwargs.get("rate", 0.5) if e.crossover else 0.5
            crossed = graph.evo_crossover("crossover", selected,
                                           method=xo_method, rate=float(xo_rate))

            mut_method = e.mutate.name.lower() if e.mutate else "gaussian"
            mut_sigma = e.mutate.kwargs.get("sigma", 0.01) if e.mutate else 0.01
            mutated = graph.evo_mutate("mutation", crossed,
                                        method=mut_method, sigma=float(mut_sigma))

            graph.output("offspring", mutated)
        else:
            graph.output("offspring", pop_node)

        graph.type_check()
        return graph

    @staticmethod
    def _parse_tensor_type(expr: TypeExpr) -> TensorType:
        dims = []
        for d in expr.dims:
            try:
                dims.append(int(d))
            except ValueError:
                dims.append(None)
        return TensorType(tuple(dims), Float32)


# ------------------------------------------------------------------ #
# Public entry point
# ------------------------------------------------------------------ #

def type_check(program: Program) -> Dict[str, IRGraph]:
    """Check a parsed Program and return its IR graphs, or raise ShapeError."""
    return TypeChecker().check(program)
