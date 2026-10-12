"""
Victim separation: do the victim's windows score as more unusual than everyone else's?

Two numbers, each next to the same number for random scores (the baseline that showed
the old 0.539 meant nothing):

- pooled AUC: chance that a random victim window (onset -> contact) outscores a random
  normal window from the held-out clips. 0.5 = no separation.
- per-clip rank: in each held-out clip with victim windows, the share of that clip's
  normal windows scoring below the victim's highest window. The random baseline is not
  0.5 here: the max of several random victim windows ranks high by chance, which is
  exactly why it is always reported alongside.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def _per_clip_ranks(normal, normal_clips, victim, victim_clips) -> dict[str, float]:
    out = {}
    for cid in np.unique(victim_clips):
        n = normal[normal_clips == cid]
        v = victim[victim_clips == cid]
        if len(n) and len(v):
            out[str(cid)] = float(np.mean(n < v.max()))
    return out


def _auc(normal, victim) -> float | None:
    if not len(normal) or not len(victim):
        return None
    y = np.r_[np.zeros(len(normal)), np.ones(len(victim))]
    return float(roc_auc_score(y, np.r_[normal, victim]))


def victim_separation(
    normal: np.ndarray,
    normal_clips: np.ndarray,
    victim: np.ndarray,
    victim_clips: np.ndarray,
    n_random: int = 20,
    seed: int = 0,
) -> dict:
    """Separation of victim windows from normal windows, with a random baseline.

    Args:
        normal: anomaly scores of normal (non-victim) test windows.
        normal_clips: clip id of each normal window.
        victim: anomaly scores of victim windows inside onset -> contact.
        victim_clips: clip id of each victim window.
        n_random: random score draws averaged for the baseline.
    Returns:
        Dict: auc, auc_random, clip_rank_median, clip_rank_median_random, per_clip
        ranks, and the window/clip counts behind them.
    """
    ranks = _per_clip_ranks(normal, normal_clips, victim, victim_clips)
    rng = np.random.default_rng(seed)
    rand_auc, rand_rank = [], []
    for _ in range(n_random):
        rn, rv = rng.random(len(normal)), rng.random(len(victim))
        a = _auc(rn, rv)
        if a is not None:
            rand_auc.append(a)
        rr = _per_clip_ranks(rn, normal_clips, rv, victim_clips)
        if rr:
            rand_rank.append(np.median(list(rr.values())))
    return {
        "auc": _auc(normal, victim),
        "auc_random": float(np.mean(rand_auc)) if rand_auc else None,
        "clip_rank_median": float(np.median(list(ranks.values()))) if ranks else None,
        "clip_rank_median_random": float(np.mean(rand_rank)) if rand_rank else None,
        "per_clip_rank": ranks,
        "n_normal_windows": int(len(normal)),
        "n_victim_windows": int(len(victim)),
        "n_clips_scored": len(ranks),
    }
