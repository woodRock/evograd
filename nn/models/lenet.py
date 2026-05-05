"""LeNet-5 (LeCun et al. 1998) — classic conv-net for 32×32 images."""
from __future__ import annotations
import torch
import torch.nn as nn

from ...core.tensor import Tensor
from ..layers import Layer


class _LeNet5Module(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.Tanh(),
            nn.Linear(120, 84),         nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class LeNet5(Layer):
    """
    LeNet-5.

    Input:  (B, in_channels, 32, 32)
    Output: (B, num_classes) logits
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10,
                 device: str = "cpu"):
        self._net = _LeNet5Module(in_channels, num_classes).to(device)

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(self._net(x.data))

    def parameters(self):
        return list(self._net.parameters())

    def train(self) -> "LeNet5":
        self._net.train(); return self

    def eval(self) -> "LeNet5":
        self._net.eval(); return self

    @property
    def model(self) -> nn.Module:
        return self._net
