"""Hold out whole clips for testing, so no test window shares footage with training."""

from __future__ import annotations

import numpy as np


def split_clips(
    clip_ids: list[str], test_frac: float = 0.2, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Seeded split of clip ids into (train, test); the same ids always split the same way.

    At least one clip always stays in train.
    """
    ids = sorted(set(clip_ids))
    order = np.random.default_rng(seed).permutation(len(ids))
    n_test = min(round(test_frac * len(ids)), len(ids) - 1)
    test = sorted(ids[i] for i in order[:n_test])
    train = sorted(ids[i] for i in order[n_test:])
    return train, test
