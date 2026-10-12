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
