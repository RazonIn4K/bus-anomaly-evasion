"""Tiny MLP autoencoder: d → 64 → 16 → 64 → d, ReLU, MSE reconstruction."""

from __future__ import annotations

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """Untied-weight MLP autoencoder as locked in DESIGN.md §7."""

    def __init__(self, input_dim: int, hidden: int = 64, latent: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def reconstruction_error(self, x: torch.Tensor, reduction: str = "none") -> torch.Tensor:
        """Per-sample MSE (reduction='none') or scalar mean."""
        recon = self.forward(x)
        mse = (recon - x).pow(2).mean(dim=-1)
        if reduction == "mean":
            return mse.mean()
        if reduction == "sum":
            return mse.sum()
        return mse


def build_model(input_dim: int, seed: int = 0) -> Autoencoder:
    torch.manual_seed(seed)
    model = Autoencoder(input_dim=input_dim)
    return model
