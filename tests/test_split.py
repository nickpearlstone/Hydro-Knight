"""Whole-clip holdout: disjoint, complete, deterministic."""

from __future__ import annotations

import pytest

from hydro_knight.eval.split import assign_folds, split_clips

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


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_clip_gets_exactly_one_fold_and_no_fold_is_empty(seed):
    folds = assign_folds(IDS, k=5, seed=seed)
    assert sorted(folds) == sorted(IDS)
    assert set(folds.values()) == set(range(5))


def test_folds_are_deterministic_and_order_free():
    assert assign_folds(IDS, seed=3) == assign_folds(list(reversed(IDS)), seed=3)
    assert assign_folds(IDS, seed=3) != assign_folds(IDS, seed=4)


def test_fold_zero_is_the_old_twenty_percent_holdout():
    folds = assign_folds(IDS, k=5, seed=0)
    _, test = split_clips(IDS, test_frac=0.2, seed=0)
    assert sorted(c for c, f in folds.items() if f == 0) == test


def test_labeling_more_clips_never_moves_a_clip_between_folds():
    small, full = assign_folds(IDS[:30], seed=0), assign_folds(IDS, seed=0)
    assert set(small.values()) == set(range(5))  # no fallback move in the smaller set
    assert all(full[c] == f for c, f in small.items())


def test_an_empty_fold_is_filled_from_a_bigger_one():
    # 5 clips in 5 folds: the hash alone leaves some folds empty for most seeds.
    ids = IDS[:5]
    for seed in range(20):
        folds = assign_folds(ids, k=5, seed=seed)
        assert sorted(folds.values()) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("k,n", [(5, 4), (1, 10)])
def test_too_few_clips_or_folds_is_an_error(k, n):
    with pytest.raises(ValueError):
        assign_folds(IDS[:n], k=k)
