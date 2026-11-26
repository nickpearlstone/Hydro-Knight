"""Rung 3: a temporal-convolution autoencoder over pose-sequence windows."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class TCNAutoencoder(nn.Module):
    """Temporal-conv autoencoder. Encoder downsamples time by 4x (two stride-2
    convs) into a compact code; decoder upsamples back. Channels are the 70
    """

    def __init__(self, n_feat: int = 70):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_feat, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, n_feat, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def _standardize_fit(windows: np.ndarray):
    flat = windows.reshape(-1, windows.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-6
    return mean, std


def _to_tensor(windows: np.ndarray, scaler):
    mean, std = scaler
    x = (windows - mean) / std
    x = np.transpose(x, (0, 2, 1))
    return torch.tensor(x, dtype=torch.float32)


def train_tcn(windows: np.ndarray, epochs: int = 120, lr: float = 1e-3, seed: int = 0):
    """Train the TCN autoencoder on normal-only windows; returns the fitted model and scaler."""
    torch.manual_seed(seed)
    scaler = _standardize_fit(windows)
    X = _to_tensor(windows, scaler)

    model = TCNAutoencoder(n_feat=windows.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    pbar = tqdm(range(epochs), "training")

    history: list[float] = []
    model.train()
    for _ in pbar:
        opt.zero_grad()
        recon = model(X)
        loss = loss_fn(recon, X)
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
        pbar.set_postfix(loss=f"{history[-1]:.4f}")
    model.loss_history_ = history
    return model, scaler


def reconstruction_error(model, windows: np.ndarray, scaler) -> np.ndarray:
    """Per-window reconstruction MSE (mean over time and features) = the anomaly score."""
    X = _to_tensor(windows, scaler)
    model.eval()
    with torch.no_grad():
        recon = model(X)
        err = ((recon - X) ** 2).mean(dim=(1, 2)).numpy()
    return err
