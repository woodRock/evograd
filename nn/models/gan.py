"""Generative Adversarial Network — MLP generator + discriminator."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.tensor import Tensor
from ..layers import Layer


def _mlp(dims, act=nn.LeakyReLU, last_act=None, dropout=0.0):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act(0.2, inplace=True) if act is nn.LeakyReLU else act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    if last_act is not None:
        layers.append(last_act())
    return nn.Sequential(*layers)


class _GeneratorModule(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = _mlp(
            [latent_dim, hidden_dim, hidden_dim * 2, hidden_dim * 4, output_dim],
            act=nn.LeakyReLU, last_act=nn.Tanh,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _DiscriminatorModule(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = _mlp(
            [input_dim, hidden_dim * 2, hidden_dim, 1],
            act=nn.LeakyReLU, dropout=0.3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Generator(Layer):
    """MLP generator: latent noise → synthetic sample."""
    def __init__(self, latent_dim: int = 100, hidden_dim: int = 256,
                 output_dim: int = 784, device: str = "cpu"):
        self._net = _GeneratorModule(latent_dim, hidden_dim, output_dim).to(device)
        self._latent_dim = latent_dim
        self._device = device

    def forward(self, z: Tensor) -> Tensor:
        return Tensor(self._net(z.data))

    def sample(self, n: int) -> Tensor:
        z = torch.randn(n, self._latent_dim, device=self._device)
        return Tensor(self._net(z))

    def parameters(self):
        return list(self._net.parameters())

    def train(self) -> "Generator":
        self._net.train(); return self

    def eval(self) -> "Generator":
        self._net.eval(); return self

    @property
    def model(self) -> nn.Module:
        return self._net


class Discriminator(Layer):
    """MLP discriminator: sample → real/fake logit."""
    def __init__(self, input_dim: int = 784, hidden_dim: int = 256,
                 device: str = "cpu"):
        self._net = _DiscriminatorModule(input_dim, hidden_dim).to(device)

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(self._net(x.data))

    def parameters(self):
        return list(self._net.parameters())

    def train(self) -> "Discriminator":
        self._net.train(); return self

    def eval(self) -> "Discriminator":
        self._net.eval(); return self

    @property
    def model(self) -> nn.Module:
        return self._net


class GAN(Layer):
    """
    Combines Generator and Discriminator with loss helpers.

    Training loop (non-saturating GAN):
        # Discriminator step
        loss_d = gan.disc_loss(real_batch)
        # Generator step
        loss_g = gan.gen_loss(n=batch_size)
    """
    def __init__(self, latent_dim: int = 100, hidden_dim: int = 256,
                 data_dim: int = 784, device: str = "cpu"):
        self.generator     = Generator(latent_dim, hidden_dim, data_dim, device)
        self.discriminator = Discriminator(data_dim, hidden_dim, device)
        self._latent_dim = latent_dim
        self._device = device

    def forward(self, z: Tensor) -> Tensor:
        return self.generator(z)

    def gen_loss(self, n: int) -> Tensor:
        z = torch.randn(n, self._latent_dim, device=self._device)
        fake = self.generator._net(z)
        logits = self.discriminator._net(fake)
        target = torch.ones_like(logits)
        return Tensor(F.binary_cross_entropy_with_logits(logits, target))

    def disc_loss(self, real: Tensor) -> Tensor:
        n = real.data.size(0)
        z = torch.randn(n, self._latent_dim, device=self._device)
        fake = self.generator._net(z).detach()
        real_logits = self.discriminator._net(real.data)
        fake_logits = self.discriminator._net(fake)
        loss_real = F.binary_cross_entropy_with_logits(
            real_logits, torch.ones_like(real_logits))
        loss_fake = F.binary_cross_entropy_with_logits(
            fake_logits, torch.zeros_like(fake_logits))
        return Tensor(0.5 * (loss_real + loss_fake))

    def parameters(self):
        return self.generator.parameters() + self.discriminator.parameters()

    def train(self) -> "GAN":
        self.generator.train(); self.discriminator.train(); return self

    def eval(self) -> "GAN":
        self.generator.eval(); self.discriminator.eval(); return self
