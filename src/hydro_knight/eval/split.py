"""Hold out whole clips for testing, so no test window shares footage with training."""

from __future__ import annotations

import hashlib


def _unit(clip_id: str, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}:{clip_id}".encode()).hexdigest()
    return int(digest, 16) / 16**40


def split_clips(
    clip_ids: list[str], test_frac: float = 0.2, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Split clip ids into (train, test); test holds about test_frac of them."""
    ids = sorted(set(clip_ids))
    test = [c for c in ids if _unit(c, seed) < test_frac]
    if ids and len(test) == len(ids):
        test.remove(max(test, key=lambda c: _unit(c, seed)))
    train = [c for c in ids if c not in test]
    return train, test


def assign_folds(clip_ids: list[str], k: int = 5, seed: int = 0) -> dict[str, int]:
    """Give every clip one fold in 0..k-1 for grouped K-fold (each clip tested exactly once)."""
    ids = sorted(set(clip_ids))
    if k < 2 or len(ids) < k:
        raise ValueError(f"need k >= 2 and at least k clips (k={k}, {len(ids)} clips)")
    fold = {c: min(int(_unit(c, seed) * k), k - 1) for c in ids}
    for f in range(k):
        sizes = [list(fold.values()).count(g) for g in range(k)]
        if sizes[f]:
            continue
        centre = (f + 0.5) / k
        donor = min(
            (c for c in ids if sizes[fold[c]] > 1),
            key=lambda c: (abs(_unit(c, seed) - centre), c),
        )
        fold[donor] = f
    return fold
