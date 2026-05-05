"""Variational Autoencoder (Kingma & Welling 2013)."""
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.tensor import Tensor
from ..layers import Layer


class _VAEModule(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu_head     = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),  nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @staticmethod
    def elbo_loss(recon: torch.Tensor, x: torch.Tensor,
                  mu: torch.Tensor, logvar: torch.Tensor,
                  beta: float = 1.0) -> torch.Tensor:
        recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
        return recon_loss + beta * kl


class VAE(Layer):
    """
    Variational Autoencoder with MLP encoder and decoder.

    forward() returns (recon, mu, logvar) as EvoGrad Tensors.
    Use .elbo_loss(recon, x, mu, logvar) to compute the training objective.

    Input:  (B, input_dim)   flat vector (e.g. flattened image)
    Output: (recon (B, input_dim), mu (B, latent_dim), logvar (B, latent_dim))
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 latent_dim: int = 32, device: str = "cpu"):
        self._net = _VAEModule(input_dim, hidden_dim, latent_dim).to(device)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        recon, mu, logvar = self._net(x.data)
        return Tensor(recon), Tensor(mu), Tensor(logvar)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        mu, logvar = self._net.encode(x.data)
        return Tensor(mu), Tensor(logvar)

    def decode(self, z: Tensor) -> Tensor:
        return Tensor(self._net.decode(z.data))

    def sample(self, n: int, device: str = "cpu") -> Tensor:
        latent_dim = self._net.mu_head.out_features
        z = torch.randn(n, latent_dim, device=device)
        return Tensor(self._net.decode(z))

    def elbo_loss(self, recon: Tensor, x: Tensor,
                  mu: Tensor, logvar: Tensor, beta: float = 1.0) -> Tensor:
        return Tensor(_VAEModule.elbo_loss(
            recon.data, x.data, mu.data, logvar.data, beta
        ))

    def parameters(self):
        return list(self._net.parameters())

    def train(self) -> "VAE":
        self._net.train(); return self

    def eval(self) -> "VAE":
        self._net.eval(); return self

    @property
    def model(self) -> nn.Module:
        return self._net
