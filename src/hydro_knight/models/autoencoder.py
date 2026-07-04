"""
Rung 2: a simple pose autoencoder.

An autoencoder squeezes each 34-dim normalized pose through a narrow bottleneck
and reconstructs it. Trained only on NORMAL swimming poses, it learns to rebuild
normal body shapes accurately; an unusual pose reconstructs poorly. The per-pose
reconstruction error (MSE) is the anomaly score — high error = "this doesn't look
like normal swimming".

Per-frame (no temporal context yet); windowed/LSTM version is Rung 3.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class PoseAutoencoder(nn.Module):
    """34 -> 24 -> bottleneck -> 24 -> 34. The bottleneck forces the net to learn
    a compact code for 'normal pose', so it can't just memorise everything."""

    def __init__(self, dim: int = 34, bottleneck: int = 12):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 24),
            nn.ReLU(),
            nn.Linear(24, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 24),
            nn.ReLU(),
            nn.Linear(24, dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(
    features: np.ndarray,
    epochs: int = 80,
    lr: float = 1e-3,
    bottleneck: int = 12,
    seed: int = 0,
):
    """
    Train on normal poses. Returns (model, scaler) where scaler = (mean, std)
    used to standardize features (zero-mean/unit-variance) — standardization
    makes training stable and is reapplied at scoring time.
    """
    torch.manual_seed(seed)
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-6
    X = torch.tensor((features - mean) / std, dtype=torch.float32)

    model = PoseAutoencoder(dim=features.shape[1], bottleneck=bottleneck)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        recon = model(X)
        loss = loss_fn(recon, X)
        loss.backward()
        opt.step()
    return model, (mean, std)


def reconstruction_error(model, features: np.ndarray, scaler) -> np.ndarray:
    """Per-pose reconstruction MSE (the anomaly score). Higher = more anomalous."""
    mean, std = scaler
    X = torch.tensor((features - mean) / std, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        recon = model(X)
        err = ((recon - X) ** 2).mean(dim=1).numpy()
    return err
