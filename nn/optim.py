"""
Optimizers and learning rate schedulers.

These wrap PyTorch optimizers to work seamlessly with EvoGrad Layers.
Since Layer.parameters() returns standard torch Parameters, we can
leverage PyTorch's mature optimization library.
"""
from __future__ import annotations
import torch
from typing import List, Iterable, Union, Optional

from .layers import Layer


class Optimizer:
    """Base optimizer wrapper."""
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float):
        self._opt: Optional[torch.optim.Optimizer] = None

    def step(self):
        self._opt.step()

    def zero_grad(self):
        self._opt.zero_grad()

    @property
    def param_groups(self):
        return self._opt.param_groups


class SGD(Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float,
                 momentum: float = 0.0, weight_decay: float = 0.0):
        self._opt = torch.optim.SGD(params, lr=lr, momentum=momentum,
                                     weight_decay=weight_decay)


class Adam(Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-3,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        self._opt = torch.optim.Adam(params, lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay)


# ------------------------------------------------------------------ #
# Schedulers
# ------------------------------------------------------------------ #

class Scheduler:
    """Base scheduler wrapper."""
    def __init__(self, optimizer: Optimizer):
        self._sched: Optional[torch.optim.lr_scheduler._LRScheduler] = None

    def step(self):
        self._sched.step()


class StepLR(Scheduler):
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1):
        self._sched = torch.optim.lr_scheduler.StepLR(
            optimizer._opt, step_size=step_size, gamma=gamma
        )


class CosineAnnealingLR(Scheduler):
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0):
        self._sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer._opt, T_max=T_max, eta_min=eta_min
        )
