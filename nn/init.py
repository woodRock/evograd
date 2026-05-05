"""
Initialization schemes for Layer parameters.

These functions operate on a Layer or a list of torch Parameters.
They leverage torch.nn.init for standard schemes.
"""
from __future__ import annotations
import torch
import torch.nn.init as init
from typing import List, Iterable, Union

from .layers import Layer


def xavier_uniform(params: Union[Layer, Iterable[torch.nn.Parameter]],
                   gain: float = 1.0):
    if isinstance(params, Layer):
        params = params.parameters()
    for p in params:
        if p.dim() >= 2:
            init.xavier_uniform_(p, gain=gain)


def xavier_normal(params: Union[Layer, Iterable[torch.nn.Parameter]],
                  gain: float = 1.0):
    if isinstance(params, Layer):
        params = params.parameters()
    for p in params:
        if p.dim() >= 2:
            init.xavier_normal_(p, gain=gain)


def kaiming_uniform(params: Union[Layer, Iterable[torch.nn.Parameter]],
                    a: float = 0, mode: str = 'fan_in',
                    nonlinearity: str = 'leaky_relu'):
    if isinstance(params, Layer):
        params = params.parameters()
    for p in params:
        if p.dim() >= 2:
            init.kaiming_uniform_(p, a=a, mode=mode,
                                  nonlinearity=nonlinearity)


def kaiming_normal(params: Union[Layer, Iterable[torch.nn.Parameter]],
                   a: float = 0, mode: str = 'fan_in',
                   nonlinearity: str = 'leaky_relu'):
    if isinstance(params, Layer):
        params = params.parameters()
    for p in params:
        if p.dim() >= 2:
            init.kaiming_normal_(p, a=a, mode=mode,
                                 nonlinearity=nonlinearity)


def constant(params: Union[Layer, Iterable[torch.nn.Parameter]],
             val: float = 0.0):
    if isinstance(params, Layer):
        params = params.parameters()
    for p in params:
        init.constant_(p, val)


def normal(params: Union[Layer, Iterable[torch.nn.Parameter]],
           mean: float = 0.0, std: float = 1.0):
    if isinstance(params, Layer):
        params = params.parameters()
    for p in params:
        init.normal_(p, mean=mean, std=std)


def uniform(params: Union[Layer, Iterable[torch.nn.Parameter]],
            low: float = 0.0, high: float = 1.0):
    if isinstance(params, Layer):
        params = params.parameters()
    for p in params:
        init.uniform_(p, a=low, b=high)
