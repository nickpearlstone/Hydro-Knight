"""
Rung 3 sanity demo: does the TCN autoencoder learn temporal structure?
"""

import argparse

import numpy as np
import pandas as pd

from hydro_knight.features.windows import make_windows
from hydro_knight.models.tcn_autoencoder import reconstruction_error, train_tcn

WINDOW = 32


def main() -> None:
    """Train the TCN on a clip's windows, then print recon-error on real vs corrupted test sets."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("parquet", nargs="?", default="raw_local/long.parquet")
    parquet_path = ap.parse_args().parquet
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
        shuffled[i] = shuffled[i][np.random.permutation(WINDOW)]
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
    main()
