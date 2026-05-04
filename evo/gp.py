"""
Vectorized Linear Genetic Programming (LGP).

Individuals are fixed-length instruction sequences stored as integer
tensors.  The same instruction stream is executed over a batch of input
points in a single loop — no Python calls per individual.

Instruction set (opcode → operation):
    0  PUSH_X      push current x onto stack
    1  PUSH_CONST  push a constant (encoded in operand nibble)
    2  ADD         pop a, b; push a+b
    3  SUB         pop a, b; push a-b
    4  MUL         pop a, b; push a*b
    5  DIV         pop a, b; push a / (|b|+ε)   (protected)
    6  SIN         pop a; push sin(a)
    7  COS         pop a; push cos(a)
    8  NEG         pop a; push -a
    9  ABS         pop a; push |a|

Each individual: (PROG_LEN,) int64 tensor, values in [0, N_OPS).

The evaluator runs all P programs on all N input points simultaneously
using a Python loop over instruction depth (PROG_LEN steps), but the
innermost arithmetic operates on (P, N) tensors — so the loop depth is
fixed at PROG_LEN regardless of P.

Fitness: negative MSE on regression targets (higher = better).
"""
from __future__ import annotations
import math
from typing import Callable, List, Optional
import time
import torch

from ..core.tensor import Tensor


N_OPS = 10
_STACK_DEPTH = 8    # max operands held simultaneously
_PROG_LEN = 16      # program length (number of instructions)
_SAFE_EPS = 1e-6


def _eval_programs(programs: torch.Tensor,
                   X: torch.Tensor) -> torch.Tensor:
    """
    Execute P programs on N inputs.

    Parameters
    ----------
    programs : (P, PROG_LEN) int64
    X        : (N,)  input values

    Returns
    -------
    outputs  : (P, N)  program outputs
    """
    P, L = programs.shape
    N = X.shape[0]
    device = programs.device

    # Stack: (P, STACK_DEPTH, N) — each individual has STACK_DEPTH slots
    stack = torch.zeros(P, _STACK_DEPTH, N, device=device)
    sp = torch.zeros(P, dtype=torch.long, device=device)  # stack pointers

    # Broadcast X once
    X_bc = X.unsqueeze(0).expand(P, -1)   # (P, N)

    for step in range(L):
        op = programs[:, step]              # (P,)

        # peek/pop helpers (operate on the top of each individual's stack)
        def peek(offset: int = 0) -> torch.Tensor:
            # Returns top-offset element for each individual: (P, N)
            idx = (sp - 1 - offset).clamp(0, _STACK_DEPTH - 1)
            return stack[torch.arange(P, device=device), idx]

        def pop() -> torch.Tensor:
            val = peek(0)
            sp.sub_(1).clamp_(0)
            return val

        def push(val: torch.Tensor):
            idx = sp.clamp(0, _STACK_DEPTH - 1)
            stack[torch.arange(P, device=device), idx] = val
            sp.add_(1).clamp_(max=_STACK_DEPTH)

        # ----- dispatch each opcode -----
        # We can't branch per-individual in a vectorised manner
        # without conditionals, so we compute all results and
        # use torch.where to select based on opcode.

        top0 = peek(0)   # (P, N) — top of stack
        top1 = peek(1)   # (P, N) — second element

        results = {
            0: X_bc,                                    # PUSH_X
            1: (op.float() * 0.5 - 2.5).unsqueeze(1).expand(P, N),  # PUSH_CONST
            2: top1 + top0,                             # ADD
            3: top1 - top0,                             # SUB
            4: top1 * top0,                             # MUL
            5: top1 / (top0.abs() + _SAFE_EPS),         # DIV
            6: torch.sin(top0),                         # SIN
            7: torch.cos(top0),                         # COS
            8: -top0,                                   # NEG
            9: top0.abs(),                              # ABS
        }

        # Stack effect: ops 0,1 push; ops 2-5 pop2+push1 (net -1);
        # ops 6-9 pop1+push1 (net 0).
        is_push   = (op == 0) | (op == 1)      # +1
        is_binary = (op >= 2) & (op <= 5)      # -1
        # unary ops 6-9: net 0, just replace top

        # Compute the value to push/write for every individual
        val = torch.zeros(P, N, device=device)
        for code, res in results.items():
            mask = (op == code).unsqueeze(1).expand(P, N)
            val = torch.where(mask, res, val)

        # Update stack: for binary ops, pop twice then push val
        # For push ops, just push val; for unary, overwrite top
        top_idx = (sp - 1).clamp(0, _STACK_DEPTH - 1)
        row_idx = torch.arange(P, device=device)

        # unary and binary: write result to top-1 position (after adjusting sp)
        sp_after = sp.clone()
        sp_after[is_push] += 1
        sp_after[is_binary] -= 1
        sp_after = sp_after.clamp(0, _STACK_DEPTH)

        write_idx = (sp_after - 1).clamp(0, _STACK_DEPTH - 1)
        stack[row_idx, write_idx] = val
        sp.copy_(sp_after)

    # Return top of stack for each individual
    out_idx = (sp - 1).clamp(0, _STACK_DEPTH - 1)
    result = stack[torch.arange(P, device=device), out_idx]   # (P, N)
    return result.clamp(-1e6, 1e6)


class GP:
    """
    Linear GP for symbolic regression.

    fitness_fn: given outputs (P, N) and targets (N,), return (P,) fitness
    Default fitness: negative MSE.
    """

    def __init__(self,
                 pop_size: int,
                 prog_len: int = _PROG_LEN,
                 n_ops: int = N_OPS,
                 device: str = "cpu"):
        self.pop_size = pop_size
        self.prog_len = prog_len
        self.n_ops = n_ops
        self._device = device
        self.generation = 0

        # (P, L) integer programs
        self.programs = torch.randint(
            0, n_ops, (pop_size, prog_len),
            dtype=torch.long, device=device
        )
        self.fitness = torch.full((pop_size,), -1e18, device=device)
        self.best_prog: torch.Tensor = self.programs[0].clone()
        self.best_f: float = -1e18

    def step(self,
             X: torch.Tensor,
             y: torch.Tensor,
             mutation_rate: float = 0.05,
             tournament_k: int = 5) -> float:
        dev = self._device
        P, L = self.programs.shape

        # ── evaluate ──────────────────────────────────────────────────
        outputs = _eval_programs(self.programs, X)          # (P, N)
        mse = ((outputs - y.unsqueeze(0)) ** 2).mean(dim=1) # (P,)
        self.fitness = -mse                                  # higher = better

        best_idx = self.fitness.argmax()
        if self.fitness[best_idx].item() > self.best_f:
            self.best_f = self.fitness[best_idx].item()
            self.best_prog = self.programs[best_idx].clone()

        # ── tournament selection (vectorised) ─────────────────────────
        cands = torch.randint(0, P, (P, tournament_k), device=dev)
        cand_fit = self.fitness[cands]
        winners = cands[torch.arange(P, device=dev),
                        cand_fit.argmax(dim=1)]
        selected = self.programs[winners].clone()            # (P, L)

        # ── one-point crossover ───────────────────────────────────────
        cuts = torch.randint(1, L, (P // 2,), device=dev)
        idx = torch.arange(L, device=dev).unsqueeze(0)
        mask = (idx < cuts.unsqueeze(1)).unsqueeze(1)        # (P/2,1,L) → broadcast
        p1 = selected[0::2]
        p2 = selected[1::2]
        child1 = torch.where(mask.squeeze(1), p1, p2)
        child2 = torch.where(mask.squeeze(1), p2, p1)
        offspring = torch.empty_like(selected)
        offspring[0::2] = child1
        offspring[1::2] = child2

        # ── point mutation ────────────────────────────────────────────
        flip = torch.rand(P, L, device=dev) < mutation_rate
        random_ops = torch.randint(0, self.n_ops, (P, L), device=dev)
        self.programs = torch.where(flip, random_ops, offspring)

        self.generation += 1
        return self.best_f


def run_gp(X: torch.Tensor, y: torch.Tensor,
           pop_size: int = 128,
           prog_len: int = 16,
           generations: int = 100,
           device: str = "cpu",
           verbose: bool = False) -> tuple[float, list]:
    gp = GP(pop_size, prog_len=prog_len, device=device)
    log = []
    t0 = time.perf_counter()
    for g in range(generations):
        best = gp.step(X.to(device), y.to(device))
        log.append(best)
        if verbose and g % (generations // 5) == 0:
            print(f"  GP gen {g:4d} | best_fitness={best:.5f} | "
                  f"{time.perf_counter()-t0:.2f}s")
    return best, log
