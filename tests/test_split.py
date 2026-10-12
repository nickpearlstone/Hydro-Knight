"""Whole-clip holdout: disjoint, complete, deterministic."""

from __future__ import annotations

import pytest

from hydro_knight.eval.split import split_clips

IDS = [f"clip{i:02d}" for i in range(57)]


def test_split_is_disjoint_complete_and_sized():
    train, test = split_clips(IDS, test_frac=0.2, seed=0)
    assert set(train).isdisjoint(test) and set(train) | set(test) == set(IDS)
    assert (
        0.1 * 57 <= len(test) <= 0.3 * 57
    )  # about 20%: each clip decides by its own hash


def test_split_is_deterministic_and_order_free():
    assert split_clips(IDS, seed=3) == split_clips(list(reversed(IDS)), seed=3)
    assert split_clips(IDS, seed=3) != split_clips(IDS, seed=4)


@pytest.mark.parametrize("ids", [["a"], ["a", "b"]])
def test_tiny_sets_keep_at_least_one_training_clip(ids):
    train, test = split_clips(ids, test_frac=0.5)
    assert len(train) >= 1 and len(train) + len(test) == len(ids)


def test_labeling_more_clips_never_moves_existing_ones():
    # Labels arrive a few clips at a time; earlier clips must keep their side,
    # or results from different labeling stages aren't comparable.
    train_a, test_a = split_clips(IDS[:20], seed=0)
    train_b, test_b = split_clips(IDS, seed=0)
    assert set(test_a) <= set(test_b) and set(train_a) <= set(train_b)
