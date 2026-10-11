"""Whole-clip holdout: disjoint, complete, deterministic."""

from __future__ import annotations

import pytest

from hydro_knight.eval.split import split_clips

IDS = [f"clip{i:02d}" for i in range(57)]


def test_split_is_disjoint_complete_and_sized():
    train, test = split_clips(IDS, test_frac=0.2, seed=0)
    assert set(train).isdisjoint(test) and set(train) | set(test) == set(IDS)
    assert len(test) == round(0.2 * 57)


def test_split_is_deterministic_and_order_free():
    assert split_clips(IDS, seed=3) == split_clips(list(reversed(IDS)), seed=3)
    assert split_clips(IDS, seed=3) != split_clips(IDS, seed=4)


@pytest.mark.parametrize("ids", [["a"], ["a", "b"]])
def test_tiny_sets_keep_at_least_one_training_clip(ids):
    train, test = split_clips(ids, test_frac=0.5)
    assert len(train) >= 1 and len(train) + len(test) == len(ids)
