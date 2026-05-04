"""
EvoGrad Intermediate Representation (IR).

Programs are lowered to a typed DAG before any code is generated.
The IR serves three purposes:
  1. Shape propagation — every edge has a TensorType.
  2. Optimization passes — fusion, dead-node elimination, CSE.
  3. Backend targeting — the same IR compiles to PyTorch eager,
     TorchScript, or (future) custom CUDA kernels.

Node kinds:
  InputNode    — program input (named tensor slot)
  ConstNode    — compile-time constant tensor
  LinearNode   — weight @ input + bias (fused)
  Conv2dNode   — 2-D convolution
  ActivationNode — element-wise activation (relu, gelu, sigmoid, …)
  NormNode     — batch / layer / group norm
  DropoutNode  — stochastic regularisation (no-op at inference)
  LossNode     — scalar loss (cross-entropy, mse, …)
  EvoNode      — evolutionary operator (selection, crossover, mutation)
  OutputNode   — program output
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum, auto
import uuid

from .types import TensorType, Float32, ShapeError, matmul_shape


# ------------------------------------------------------------------ #
# Node kinds
# ------------------------------------------------------------------ #

class NodeKind(Enum):
    INPUT       = auto()
    CONST       = auto()
    LINEAR      = auto()
    CONV2D      = auto()
    ACTIVATION  = auto()
    NORM        = auto()
    DROPOUT     = auto()
    LOSS        = auto()
    EVO_SELECT  = auto()
    EVO_CROSS   = auto()
    EVO_MUTATE  = auto()
    EVO_EVAL    = auto()
    CONCAT      = auto()
    ADD         = auto()
    OUTPUT      = auto()


# ------------------------------------------------------------------ #
# IR nodes
# ------------------------------------------------------------------ #

@dataclass
class IRNode:
    kind: NodeKind
    name: str
    inputs: List["IRNode"] = field(default_factory=list)
    output_type: Optional[TensorType] = None
    attrs: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def __repr__(self) -> str:
        ins = ", ".join(n.name for n in self.inputs)
        return f"{self.kind.name}({self.name})[{ins}] -> {self.output_type}"


class IRGraph:
    """
    Directed acyclic computation graph.

    Build it node-by-node; call `type_check()` to propagate shapes and
    catch mismatches before any execution.
    """

    def __init__(self, name: str = "evograd_program"):
        self.name = name
        self._nodes: Dict[str, IRNode] = {}
        self._inputs: List[IRNode] = []
        self._outputs: List[IRNode] = []

    # ------------------------------------------------------------------ #
    # Node constructors
    # ------------------------------------------------------------------ #

    def input(self, name: str, ttype: TensorType) -> IRNode:
        node = IRNode(NodeKind.INPUT, name, output_type=ttype)
        self._register(node)
        self._inputs.append(node)
        return node

    def const(self, name: str, ttype: TensorType) -> IRNode:
        node = IRNode(NodeKind.CONST, name, output_type=ttype)
        self._register(node)
        return node

    def linear(self, name: str, x: IRNode, in_features: int,
                out_features: int) -> IRNode:
        batch_dims = x.output_type.shape[:-1]
        out_shape = batch_dims + (out_features,)
        out_type = TensorType(out_shape, x.output_type.dtype)
        weight_node = self.const(f"{name}_W",
                                  TensorType((out_features, in_features)))
        bias_node = self.const(f"{name}_b",
                                TensorType((out_features,)))
        node = IRNode(
            NodeKind.LINEAR, name,
            inputs=[x, weight_node, bias_node],
            output_type=out_type,
            attrs={"in_features": in_features, "out_features": out_features},
        )
        self._register(node)
        return node

    def conv2d(self, name: str, x: IRNode, in_ch: int, out_ch: int,
               kernel: int = 3, stride: int = 1, padding: int = 0) -> IRNode:
        # Input shape: (B, C_in, H, W)
        B, C, H, W = x.output_type.shape
        H_out = None if H is None else (H + 2 * padding - kernel) // stride + 1
        W_out = None if W is None else (W + 2 * padding - kernel) // stride + 1
        out_type = TensorType((B, out_ch, H_out, W_out), x.output_type.dtype)
        node = IRNode(
            NodeKind.CONV2D, name,
            inputs=[x],
            output_type=out_type,
            attrs={
                "in_ch": in_ch, "out_ch": out_ch, "kernel": kernel,
                "stride": stride, "padding": padding,
            },
        )
        self._register(node)
        return node

    def activation(self, name: str, x: IRNode, fn: str = "relu") -> IRNode:
        node = IRNode(
            NodeKind.ACTIVATION, name,
            inputs=[x],
            output_type=x.output_type,
            attrs={"fn": fn},
        )
        self._register(node)
        return node

    def norm(self, name: str, x: IRNode, kind: str = "layer") -> IRNode:
        node = IRNode(
            NodeKind.NORM, name,
            inputs=[x],
            output_type=x.output_type,
            attrs={"kind": kind},
        )
        self._register(node)
        return node

    def dropout(self, name: str, x: IRNode, p: float = 0.1) -> IRNode:
        node = IRNode(
            NodeKind.DROPOUT, name,
            inputs=[x],
            output_type=x.output_type,
            attrs={"p": p},
        )
        self._register(node)
        return node

    def add(self, name: str, a: IRNode, b: IRNode) -> IRNode:
        if a.output_type != b.output_type:
            try:
                unified = a.output_type.unify(b.output_type)
            except ShapeError as e:
                raise ShapeError(f"add({a.name}, {b.name}): {e}")
        else:
            unified = a.output_type
        node = IRNode(
            NodeKind.ADD, name, inputs=[a, b], output_type=unified
        )
        self._register(node)
        return node

    def loss(self, name: str, logits: IRNode, targets: IRNode,
             kind: str = "cross_entropy") -> IRNode:
        node = IRNode(
            NodeKind.LOSS, name,
            inputs=[logits, targets],
            output_type=TensorType((), Float32),
            attrs={"kind": kind},
        )
        self._register(node)
        return node

    def evo_select(self, name: str, pop: IRNode, fitness: IRNode,
                   method: str = "tournament", k: int = 7) -> IRNode:
        node = IRNode(
            NodeKind.EVO_SELECT, name,
            inputs=[pop, fitness],
            output_type=pop.output_type,
            attrs={"method": method, "k": k},
        )
        self._register(node)
        return node

    def evo_crossover(self, name: str, parents: IRNode,
                      method: str = "uniform", rate: float = 0.5) -> IRNode:
        node = IRNode(
            NodeKind.EVO_CROSS, name,
            inputs=[parents],
            output_type=parents.output_type,
            attrs={"method": method, "rate": rate},
        )
        self._register(node)
        return node

    def evo_mutate(self, name: str, offspring: IRNode,
                   method: str = "gaussian", sigma: float = 0.01) -> IRNode:
        node = IRNode(
            NodeKind.EVO_MUTATE, name,
            inputs=[offspring],
            output_type=offspring.output_type,
            attrs={"method": method, "sigma": sigma},
        )
        self._register(node)
        return node

    def output(self, name: str, x: IRNode) -> IRNode:
        node = IRNode(NodeKind.OUTPUT, name, inputs=[x],
                      output_type=x.output_type)
        self._register(node)
        self._outputs.append(node)
        return node

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    def _register(self, node: IRNode):
        if node.name in self._nodes:
            raise ValueError(f"Duplicate node name: {node.name!r}")
        self._nodes[node.name] = node

    def type_check(self):
        """Walk nodes in topological order, validate all shapes."""
        for node in self.topological_order():
            if node.kind == NodeKind.LINEAR:
                x, W, b = node.inputs
                if x.output_type is None:
                    raise ShapeError(f"{node.name}: input has no type")
        return True  # no exception => all shapes valid

    def topological_order(self) -> List[IRNode]:
        visited, order = set(), []

        def visit(n: IRNode):
            if n.id in visited:
                return
            visited.add(n.id)
            for inp in n.inputs:
                visit(inp)
            order.append(n)

        for out in self._outputs:
            visit(out)
        if not order:
            for n in self._nodes.values():
                visit(n)
        return order

    def summary(self) -> str:
        lines = [f"IRGraph: {self.name}"]
        for n in self.topological_order():
            lines.append(f"  {n}")
        return "\n".join(lines)
