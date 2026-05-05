"""AlexNet (Krizhevsky et al. 2012) — 224×224 image classifier."""
from __future__ import annotations
import torch
import torch.nn as nn

from ...core.tensor import Tensor
from ..layers import Layer


class _AlexNetModule(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),         nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.avgpool(self.features(x)))


class AlexNet(Layer):
    """
    AlexNet.

    Input:  (B, 3, 224, 224)
    Output: (B, num_classes) logits
    """
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5,
                 device: str = "cpu"):
        self._net = _AlexNetModule(num_classes, dropout).to(device)

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(self._net(x.data))

    def parameters(self):
        return list(self._net.parameters())

    def train(self) -> "AlexNet":
        self._net.train(); return self

    def eval(self) -> "AlexNet":
        self._net.eval(); return self

    @property
    def model(self) -> nn.Module:
        return self._net
