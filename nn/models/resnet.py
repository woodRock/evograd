"""ResNet family (He et al. 2015)."""
from __future__ import annotations
from typing import List, Type, Union
import torch
import torch.nn as nn

from ...core.tensor import Tensor
from ..layers import Layer


class _BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if stride != 1 or in_ch != out_ch
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class _Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch * 4, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_ch * 4)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch * 4, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch * 4),
            )
            if stride != 1 or in_ch != out_ch * 4
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


_BlockT = Union[Type[_BasicBlock], Type[_Bottleneck]]


class _ResNetModule(nn.Module):
    def __init__(self, block: _BlockT, layer_cfg: List[int],
                 num_classes: int = 1000, in_channels: int = 3):
        super().__init__()
        self._in_ch = 64
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.layer1 = self._make_layer(block, 64,  layer_cfg[0])
        self.layer2 = self._make_layer(block, 128, layer_cfg[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layer_cfg[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layer_cfg[3], stride=2)
        self.pool   = nn.AdaptiveAvgPool2d((1, 1))
        self.fc     = nn.Linear(512 * block.expansion, num_classes)
        self._init_weights()

    def _make_layer(self, block: _BlockT, out_ch: int, n: int,
                    stride: int = 1) -> nn.Sequential:
        layers = [block(self._in_ch, out_ch, stride)]
        self._in_ch = out_ch * block.expansion
        for _ in range(1, n):
            layers.append(block(self._in_ch, out_ch))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.fc(torch.flatten(self.pool(x), 1))


_CONFIGS = {
    "resnet18":  (_BasicBlock,    [2, 2, 2, 2]),
    "resnet34":  (_BasicBlock,    [3, 4, 6, 3]),
    "resnet50":  (_Bottleneck,    [3, 4, 6, 3]),
    "resnet101": (_Bottleneck,    [3, 4, 23, 3]),
    "resnet152": (_Bottleneck,    [3, 8, 36, 3]),
}


class ResNet(Layer):
    """
    ResNet image classifier.

    Parameters
    ----------
    variant     : 'resnet18' | 'resnet34' | 'resnet50' | 'resnet101' | 'resnet152'
    num_classes : output dimension
    in_channels : input channels (3 for RGB, 1 for grayscale)
    device      : torch device string

    Input:  (B, in_channels, H, W)  — H=W=224 for standard ImageNet
    Output: (B, num_classes) logits
    """
    def __init__(self, variant: str = "resnet18", num_classes: int = 1000,
                 in_channels: int = 3, device: str = "cpu"):
        if variant not in _CONFIGS:
            raise ValueError(f"Unknown variant {variant!r}. "
                             f"Choose from {list(_CONFIGS)}")
        block, cfg = _CONFIGS[variant]
        self._net = _ResNetModule(block, cfg, num_classes, in_channels).to(device)

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(self._net(x.data))

    def parameters(self):
        return list(self._net.parameters())

    def train(self) -> "ResNet":
        self._net.train(); return self

    def eval(self) -> "ResNet":
        self._net.eval(); return self

    @property
    def model(self) -> nn.Module:
        return self._net


def resnet18(num_classes: int = 1000, in_channels: int = 3,
             device: str = "cpu") -> ResNet:
    return ResNet("resnet18", num_classes, in_channels, device)

def resnet34(num_classes: int = 1000, in_channels: int = 3,
             device: str = "cpu") -> ResNet:
    return ResNet("resnet34", num_classes, in_channels, device)

def resnet50(num_classes: int = 1000, in_channels: int = 3,
             device: str = "cpu") -> ResNet:
    return ResNet("resnet50", num_classes, in_channels, device)

def resnet101(num_classes: int = 1000, in_channels: int = 3,
              device: str = "cpu") -> ResNet:
    return ResNet("resnet101", num_classes, in_channels, device)

def resnet152(num_classes: int = 1000, in_channels: int = 3,
              device: str = "cpu") -> ResNet:
    return ResNet("resnet152", num_classes, in_channels, device)
