"""Grouped K-fold: pooling held-out scores, and the train + evaluate path end to end on fake clips."""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from v2_fakes import track_rows, write_keypoints

from hydro_knight.eval import kfold
from hydro_knight.features.dataset_v2 import DatasetReport

CLIPS = [f"clip{i}" for i in range(6)]
FULL = {
    "start": 4.0,
    "end": 9.0,
    "label": "distress",
    "contact": 8.0,
    "victim": [
        {"t": 4.0, "x": 120.0, "y": 140.0, "kind": "onset"},
        {"t": 7.0, "x": 120.0, "y": 140.0, "kind": "last_seen"},
    ],
}


def test_pool_adds_the_fold_and_refuses_a_clip_in_two_folds():
    a = pd.DataFrame({"clip_id": ["x", "x"], "score": [1.0, 2.0]})
    b = pd.DataFrame({"clip_id": ["y"], "score": [3.0]})
    pooled = kfold.pool({1: b, 0: a})
    assert list(pooled["fold"]) == [0, 0, 1] and list(pooled["clip_id"]) == [
        "x",
        "x",
        "y",
    ]
    with pytest.raises(ValueError, match="clip x"):
        kfold.pool({0: a, 1: a})


def test_scale_reference_is_the_training_median_and_must_be_positive():
    assert kfold.scale_reference(np.array([1.0, 2.0, 9.0, np.nan])) == 2.0
    with pytest.raises(ValueError):
        kfold.scale_reference(np.zeros(3))


def test_pooled_report_tests_every_folded_clip_in_order():
    base = DatasetReport(victim_tracks={"b": [1], "a": [2]})
    rep = kfold.pooled_report(base, {"b": 1, "a": 0})
    assert rep.used_test == ["b", "a"] and rep.used_train == []
    assert rep.folds == {"b": 1, "a": 0} and base.folds == {}


def _manifest_line(cid):
    return json.dumps(
        {
            "clip_id": cid,
            "source_url": f"u/{cid}",
            "platform": "youtube",
            "start_sec": 0.0,
            "end_sec": -1.0,
            "camera_view": "elevated",
            "setting": "outdoor",
            "time_of_day": "day",
            "weather": "clear",
            "label": "distress",
            "fps": 30.0,
            "events": [FULL],
        }
    )


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Six labeled fake clips (victim track 7, two normal swimmers) under data/ in a temp dir."""
    monkeypatch.chdir(tmp_path)
    kp = tmp_path / "data" / "keypoints_tiled"
    kp.mkdir(parents=True)
    for i, cid in enumerate(CLIPS):
        df = pd.concat(
            [
                track_rows(7, range(0, 330), box=(100, 100, 140, 180)),
                track_rows(
                    9, range(0, 330), box=(400, 100, 440, 180), dx_per_frame=i * 0.1
                ),
                track_rows(11, range(0, 330), box=(700, 300, 740, 380)),
            ]
        )
        write_keypoints(kp / f"{cid}.parquet", df, fps=30.0)
    man = tmp_path / "data" / "manifests" / "pool_footage.jsonl"
    man.parent.mkdir(parents=True)
    man.write_text("\n".join(_manifest_line(c) for c in CLIPS) + "\n")
    return tmp_path


def _run(module, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", [module.__name__, *argv])
    module.main()


def test_kfold_tests_each_clip_once_by_a_model_that_never_saw_it(
    workspace, monkeypatch
):
    from hydro_knight.cli import eval_protocol, train_v2

    _run(
        train_v2,
        monkeypatch,
        "--sizes", "small", "--seeds", "0", "--folds", "3", "--steps", "3",
        "--batch", "8", "--detection", "--out", "runs/kf",
    )  # fmt: skip
    run = workspace / "runs" / "kf"
    summary = json.loads((run / "summary.json").read_text())
    folds = summary["reports"]["0"]["folds"]
    assert sorted(folds) == CLIPS and set(folds.values()) == {0, 1, 2}
    assert sorted(summary["reports"]["0"]["used_test"]) == CLIPS

    # Each fold's checkpoint trained on exactly the clips outside its fold.
    for f in range(3):
        meta = torch.load(run / f"model_small_s0_f{f}.pt", weights_only=False)["meta"]
        assert sorted(meta["test_clips"]) == sorted(
            c for c, g in folds.items() if g == f
        )
        assert set(meta["train_clips"]) == set(CLIPS) - set(meta["test_clips"])
        assert meta["score_scale"] > 0

    # The pooled table holds every clip, each from its own fold's model only.
    alarms = pd.read_parquet(run / "alarms_small_s0.parquet")
    by_clip = alarms.groupby("clip_id")["fold"].unique()
    assert sorted(by_clip.index) == CLIPS
    assert all(len(f) == 1 and f[0] == folds[c] for c, f in by_clip.items())
    ratio = alarms["raw_score"] / alarms["score"]
    scale = {
        f: torch.load(run / f"model_small_s0_f{f}.pt", weights_only=False)["meta"][
            "score_scale"
        ]
        for f in range(3)
    }
    assert np.allclose(ratio, alarms["fold"].map(scale))

    rows = summary["results"]
    assert sorted(r["fold"] for r in rows if "fold" in r) == [0, 1, 2]
    pooled = next(r for r in rows if "fold" not in r)
    assert pooled["folds"] == 3 and pooled["test_windows"] == len(
        alarms[alarms["track_id"] != 7]
    )
    assert len(pooled["detection_crossfit"]) == len(pooled["detection"])
    assert pooled["auc"] is not None and (alarms["track_id"] == 7).any()

    # Re-scoring the saved fold checkpoints reproduces the pooled table exactly.
    _run(
        train_v2,
        monkeypatch,
        "--sizes", "small", "--seeds", "0", "--folds", "3", "--detection",
        "--load-dir", "runs/kf", "--out", "runs/kf_rescored",
    )  # fmt: skip
    again = pd.read_parquet(
        workspace / "runs" / "kf_rescored" / "alarms_small_s0.parquet"
    )
    pd.testing.assert_frame_equal(again, alarms)

    _run(eval_protocol, monkeypatch, str(run), "--n-boot", "20", "--n-random", "2")
    report = (run / "protocol" / "report_small_s0.md").read_text()
    assert f"{len(CLIPS)} rescue clips" in report
    assert "from the other folds" in report and "Per-fold spread" in report
    spread = pd.read_csv(run / "protocol" / "tier1_fold_spread_small_s0.csv")
    assert list(spread["fold"]) == [0, 1, 2] and spread["clips"].sum() == len(CLIPS)


def test_holdout_flag_keeps_the_old_single_split(workspace, monkeypatch):
    from hydro_knight.cli import train_v2

    _run(
        train_v2,
        monkeypatch,
        "--sizes", "small", "--seeds", "0", "--holdout", "--test-frac", "0.5",
        "--steps", "3", "--batch", "8", "--out", "runs/ho",
    )  # fmt: skip
    run = workspace / "runs" / "ho"
    summary = json.loads((run / "summary.json").read_text())
    rep = summary["reports"]["0"]
    assert rep["folds"] == {} and rep["used_train"] and rep["used_test"]
    assert (run / "model_small_s0.pt").exists()
    assert all("fold" not in r for r in summary["results"])
