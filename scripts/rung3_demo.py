"""
Rung 3 sanity demo: does the TCN autoencoder learn temporal structure?

Windows the poses in a keypoint Parquet, trains the temporal-conv autoencoder
on normal motion windows, then reports mean reconstruction error on:
  - held-out REAL windows         (should be lowest)
  - time-shuffled windows         (frame order scrambled -> tests if it learned MOTION)
  - vertically-flipped windows    (wrong body orientation)
  - randomized windows            (nonsense)

If 'time-shuffled' scores clearly above 'held-out REAL', the model is using
temporal order, not just per-frame pose — the whole point of Rung 3.

Run: PYTHONPATH=src .venv/bin/python scripts/rung3_demo.py <keypoints.parquet>
"""

import sys

import numpy as np
import pandas as pd

from hydro_knight.features.windows import make_windows
from hydro_knight.models.tcn_autoencoder import reconstruction_error, train_tcn

WINDOW = 32


def main(parquet_path: str):
    """Train the TCN on a clip's windows, then print recon-error on real vs corrupted test sets."""
    np.random.seed(0)
    W, _ = make_windows(pd.read_parquet(parquet_path), window=WINDOW, stride=8)
    if len(W) < 20:
        raise SystemExit(f"too few windows ({len(W)}) — need a longer clip")
    print(f"windows: {W.shape}")

    idx = np.random.permutation(len(W))
    split = int(0.8 * len(W))
    train, test = W[idx[:split]], W[idx[split:]]
    model, scaler = train_tcn(train, epochs=150)

    shuffled = test.copy()
    for i in range(len(shuffled)):
        shuffled[i] = shuffled[i][np.random.permutation(WINDOW)]  # scramble time order
    flipped = test.copy().reshape(len(test), WINDOW, 17, 2)
    flipped[..., 1] *= -1
    flipped = flipped.reshape(len(test), WINDOW, 34)
    rand = (np.random.randn(*test.shape) * test.std()).astype(np.float32)

    for name, X in [
        ("held-out REAL", test),
        ("time-shuffled", shuffled),
        ("vertically-flipped", flipped),
        ("randomized", rand),
    ]:
        e = reconstruction_error(model, X, scaler)
        print(f"  {name:20s}: mean recon-error = {e.mean():.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "raw_local/long.parquet")
