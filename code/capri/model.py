"""Detector architecture (Deliverable 3, Step 4).

MLP: input -> 256 -> 128 -> 64 -> C+1, ReLU, dropout 0.2, layer normalisation.
Layer normalisation is used instead of batch normalisation because batch
statistics differ sharply across non-IID clients.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d_in: int, n_classes: int, hidden=(256, 128, 64), dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """64-d penultimate embedding (used by CAPRI-Audit and the failure analysis)."""
        return self.body(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


class CNN1D(nn.Module):
    """1D-CNN detector (Deliverable 3, Step 4 sensitivity analysis): the standardised feature
    vector is treated as a one-channel sequence. Same 64-d penultimate embedding as the MLP."""

    def __init__(self, d_in: int, n_classes: int, channels=(32, 64), dropout: float = 0.2):
        super().__init__()
        c1, c2 = channels
        self.conv = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=3, padding=1), nn.GroupNorm(1, c1), nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=3, padding=1), nn.GroupNorm(1, c2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4))
        self.fc = nn.Sequential(nn.Linear(c2 * 4, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(64, n_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x.unsqueeze(1)).flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


def make_model(arch: str, d_in: int, n_classes: int) -> nn.Module:
    return CNN1D(d_in, n_classes) if arch == "cnn" else MLP(d_in, n_classes)


def get_flat(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def set_flat(model: nn.Module, flat: torch.Tensor) -> None:
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(flat[i:i + n].view_as(p))
            i += n


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, batch: int = 65536) -> torch.Tensor:
    model.eval()
    out = [model(X[i:i + batch]).argmax(1) for i in range(0, len(X), batch)]
    return torch.cat(out) if out else torch.empty(0, dtype=torch.long, device=X.device)


@torch.no_grad()
def predict_logits(model: nn.Module, X: torch.Tensor, batch: int = 65536) -> torch.Tensor:
    model.eval()
    out = [model(X[i:i + batch]) for i in range(0, len(X), batch)]
    return torch.cat(out)


@torch.no_grad()
def embeddings(model: nn.Module, X: torch.Tensor, batch: int = 65536) -> torch.Tensor:
    model.eval()
    return torch.cat([model.embed(X[i:i + batch]) for i in range(0, len(X), batch)])
