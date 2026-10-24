"""Hold out whole clips for testing, so no test window shares footage with training.

Each clip's side depends only on its own id and the seed (a hash), never on which
other clips exist. Labels arrive a few clips at a time, and a clip must not move
between train and test as more are labeled, or results from different stages
aren't comparable.
"""

from __future__ import annotations

import hashlib


def _unit(clip_id: str, seed: int) -> float:
    """A stable pseudo-random number in [0, 1) for (seed, clip)."""
    digest = hashlib.sha1(f"{seed}:{clip_id}".encode()).hexdigest()
    return int(digest, 16) / 16**40


def split_clips(
    clip_ids: list[str], test_frac: float = 0.2, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Split clip ids into (train, test); test holds about test_frac of them.

    At least one clip always stays in train (the one whose hash is highest moves back).
    """
    ids = sorted(set(clip_ids))
    test = [c for c in ids if _unit(c, seed) < test_frac]
    if ids and len(test) == len(ids):
        test.remove(max(test, key=lambda c: _unit(c, seed)))
    train = [c for c in ids if c not in test]
    return train, test


def assign_folds(clip_ids: list[str], k: int = 5, seed: int = 0) -> dict[str, int]:
    """Give every clip one fold in 0..k-1 for grouped K-fold (each clip tested exactly once).

    A clip's fold is floor(k * hash), the same hash split_clips uses, so with k = 5 fold 0
    is exactly split_clips' 20% test set for that seed, and adding clips moves no one.
    Folds are uneven. If a fold would be empty, the clip whose hash lies closest to that
    fold's range moves in from a fold with more than one clip; only then can adding clips
    change an assignment.

    Raises:
        ValueError: fewer clips than folds, or k < 2.
    """
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
