"""
Rung 2 sanity demo: does the pose autoencoder flag abnormal poses?

Trains the autoencoder on the poses in a keypoint Parquet, then reports the
mean reconstruction error on (a) held-out real poses, (b) the same poses
flipped vertically (wrong body orientation), (c) randomized poses. If the
autoencoder learned the normal-pose manifold, error should rise sharply from
(a) to (b) to (c).

This validates the *mechanism*. Real anomaly evaluation comes later, training
on a proper normal set (GPU-extracted) and scoring on the labeled distress
clips.

Run: PYTHONPATH=src .venv/bin/python scripts/rung2_demo.py <keypoints.parquet>
"""

import sys

import numpy as np
import pandas as pd

from hydro_knight.features.normalize import features_from_dataframe
from hydro_knight.models.autoencoder import reconstruction_error, train_autoencoder


def main(parquet_path: str):
    np.random.seed(0)
    feats, _ = features_from_dataframe(pd.read_parquet(parquet_path))
    if len(feats) < 30:
        raise SystemExit(f"too few usable poses ({len(feats)}) to demo")

    idx = np.random.permutation(len(feats))
    split = int(0.8 * len(feats))
    train, test = feats[idx[:split]], feats[idx[split:]]
    print(f"{len(feats)} poses -> train {len(train)}, test {len(test)}")

    model, scaler = train_autoencoder(train, epochs=120)

    flipped = test.copy().reshape(-1, 17, 2)
    flipped[:, :, 1] *= -1
    flipped = flipped.reshape(-1, 34)
    rand = (np.random.randn(*test.shape) * test.std()).astype(np.float32)

    for name, X in [
        ("held-out REAL", test),
        ("vertically-flipped", flipped),
        ("randomized", rand),
    ]:
        e = reconstruction_error(model, X, scaler)
        print(f"  {name:20s}: mean recon-error = {e.mean():.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "raw_local/test_bytetrack.parquet")
