"""
Rung 3: a temporal-convolution autoencoder over pose-sequence windows.

Rather than reconstructing a single pose (the original Rung 2 MLP, since
removed), this reconstructs a whole
window of frames using 1-D convolutions across time. It therefore learns
*motion* — normal swimming dynamics — and flags windows whose temporal pattern
doesn't reconstruct well (erratic flailing, frozen/limp motion, etc.).

Input window: (window, 34). Conv1d works on (batch, channels=34, time=window),
so we transpose features<->time inside. Reconstruction MSE over the window is
the anomaly score. (TCN baseline; STG-NF is the SOTA upgrade target.)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class TCNAutoencoder(nn.Module):
    """
    Temporal-conv autoencoder. Encoder downsamples time by 4x (two stride-2
    convs) into a compact code; decoder upsamples back. Channels are the 34
    pose features. Designed for window lengths divisible by 4 (e.g. 32 -> 16 -> 8).
    """

    def __init__(self, n_feat: int = 34):
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

    def forward(self, x):  # x: (B, n_feat, time)
        return self.decoder(self.encoder(x))


def _standardize_fit(windows: np.ndarray):
    """Per-feature mean/std over all frames in all training windows."""
    flat = windows.reshape(-1, windows.shape[-1])  # (N*window, 34)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-6
    return mean, std


def _to_tensor(windows: np.ndarray, scaler):
    mean, std = scaler
    x = (windows - mean) / std  # (N, window, 34)
    x = np.transpose(x, (0, 2, 1))  # -> (N, 34, window) for Conv1d
    return torch.tensor(x, dtype=torch.float32)


def train_tcn(windows: np.ndarray, epochs: int = 120, lr: float = 1e-3, seed: int = 0):
    """
    Train on normal windows. Returns (model, scaler).

    The per-epoch training loss is recorded on the model as
    `model.loss_history_` (sklearn-style trailing underscore = "learned during
    fit"), so the eval report can plot convergence without changing this
    function's return signature.
    """
    torch.manual_seed(seed)
    scaler = _standardize_fit(windows)
    X = _to_tensor(windows, scaler)

    model = TCNAutoencoder(n_feat=windows.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history: list[float] = []
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        recon = model(X)
        loss = loss_fn(recon, X)
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
    model.loss_history_ = history
    return model, scaler


def reconstruction_error(model, windows: np.ndarray, scaler) -> np.ndarray:
    """Per-window reconstruction MSE (mean over time and features) = anomaly score."""
    X = _to_tensor(windows, scaler)
    model.eval()
    with torch.no_grad():
        recon = model(X)
        err = ((recon - X) ** 2).mean(dim=(1, 2)).numpy()
    return err
