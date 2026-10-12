"""Victim separation score and its random baseline."""

from __future__ import annotations

import numpy as np

from hydro_knight.eval.separation import victim_separation


def test_perfect_separation_beats_the_random_baseline():
    rng = np.random.default_rng(0)
    normal = rng.random(200)
    clips = np.array(["a", "b"] * 100)
    victim = np.array([5.0, 6.0, 7.0])
    vclips = np.array(["a", "a", "b"])
    s = victim_separation(normal, clips, victim, vclips)
    assert s["auc"] == 1.0 and abs(s["auc_random"] - 0.5) < 0.15
    assert s["clip_rank_median"] == 1.0 and s["n_clips_scored"] == 2


def test_victim_that_looks_normal_scores_low():
    normal = np.linspace(0, 1, 100)
    s = victim_separation(
        normal, np.array(["a"] * 100), np.array([0.05]), np.array(["a"])
    )
    assert s["auc"] < 0.1 and s["clip_rank_median"] < 0.1


def test_no_victim_windows_gives_none_not_a_crash():
    s = victim_separation(np.ones(10), np.array(["a"] * 10), np.empty(0), np.empty(0))
    assert s["auc"] is None and s["n_clips_scored"] == 0
