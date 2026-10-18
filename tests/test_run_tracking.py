"""Tests for MLflow run metadata (eval/run_meta.py), the run figures (eval/figures.py) and,
when the optional `tracking` extra is installed, the MLflow wrapper (eval/tracking.py)."""

import math

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from hydro_knight.eval import figures as F
from hydro_knight.eval import protocol as P
from hydro_knight.eval import run_meta

FAST = P.ProtocolConfig(n_boot=20, n_random=3, n_thresholds=30)


def scores(clips=("c1", "c2"), n_tracks=6, victim_score=1.0, seed=0):
    """Swimmers scored every 0.5 s with noise in [0, 0.5); track 1 is the victim from 15 s on."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in clips:
        for tid in range(1, n_tracks + 1):
            for t in np.arange(0, 40, 0.5):
                s = victim_score if tid == 1 and t >= 15 else rng.random() * 0.5
                rows.append((c, tid, float(t), float(s)))
    return pd.DataFrame(rows, columns=P.SCORE_COLUMNS)


def truths(clips=("c1", "c2")):
    return {c: P.EventTruth(c, 0.0, 20.0, 30.0, frozenset({1})) for c in clips}


# ---------------------------------------------------------------- run_meta


def test_file_hash_and_data_version_track_label_changes(tmp_path):
    m, r = tmp_path / "m.jsonl", tmp_path / "r.json"
    m.write_text("a\n")
    r.write_text("{}")
    assert run_meta.file_sha256(tmp_path / "nope") == "missing"
    assert len(run_meta.file_sha256(m)) == 12
    v1 = run_meta.data_version(m, (r,))
    r.write_text('{"x": 1}')  # a label review changed, the manifest did not
    assert run_meta.data_version(m, (r,)) != v1


def test_standard_tags_drop_none_and_stringify():
    tags = run_meta.standard_tags("train_v2", seed=3, load_dir=None)
    assert tags["command"] == "train_v2" and tags["seed"] == "3"
    assert "load_dir" not in tags
    assert {"data_version", "manifest_sha256", "git_commit"} <= set(tags)


def test_numeric_items_keeps_only_finite_numbers():
    row = {"a": 1, "b": 2.5, "c": True, "d": None, "e": math.nan, "f": "x", "g": [1]}
    assert run_meta.numeric_items(row, "p.") == {"p.a": 1.0, "p.b": 2.5}


def test_mean_sd_over_seeds_skips_missing_values():
    out = run_meta.mean_sd_metrics([{"auc": 0.6, "x": 1}, {"auc": 0.8}])
    assert out["auc.mean"] == pytest.approx(0.7)
    assert out["auc.sd"] == pytest.approx(0.1)
    assert out["x.mean"] == 1 and out["x.sd"] == 0


def test_split_table_counts_windows_and_is_order_free():
    a = run_meta.split_table(["b", "a"], ["a", "a", "b"])
    b = run_meta.split_table(["a", "b"], ["b", "a", "a"])
    pd.testing.assert_frame_equal(a, b)
    assert a.to_dict("list") == {"clip_id": ["a", "b"], "n_windows": [2, 1]}
    assert run_meta.split_table(["z"])["n_windows"].tolist() == [0]


def test_flatten_search_runs_falls_back_to_the_old_folder_param():
    raw = pd.DataFrame(
        {
            "metrics.headline.within_clip_auc": [0.6, 0.7, math.nan],
            "tags.run_folder": ["new", None, None],
            "params.run_folder": [None, "runs/v2_ckpt_after", math.nan],
        }
    )
    out = run_meta.flatten_search_runs(raw)
    assert out["run_folder"].tolist()[:2] == ["new", "v2_ckpt_after"]
    assert out["run_folder"].isna().tolist()[2]
    assert "headline.within_clip_auc" in out
    assert run_meta.flatten_search_runs(pd.DataFrame())["run_folder"].empty


# ---------------------------------------------------------------- figures


def test_loss_curves_one_panel_per_size_one_line_pair_per_seed():
    fig = F.loss_curves(
        {"small": {0: [1.0, 0.5, 0.2], 1: [1.2, 0.6, 0.3]}, "large": {0: [2, 1]}}
    )
    assert isinstance(fig, Figure)
    axes = [a for a in fig.axes if a.get_visible()]
    assert len(axes) == 2
    assert len(axes[0].lines) == 4  # raw + smoothed per seed
    # re-scored checkpoints have no real history: nothing to draw
    assert F.loss_curves({"small": {0: [math.nan]}}) is None


def test_victim_timelines_one_panel_per_clip_with_onset_and_contact():
    t1 = P.tier1(scores(clips=("c1", "c2", "c3")), truths(("c1", "c2", "c3")), FAST)
    fig = F.victim_timelines(
        scores(clips=("c1", "c2", "c3")),
        truths(("c1", "c2", "c3")),
        FAST,
        t1["per_clip"],
        ncols=2,
    )
    visible = [a for a in fig.axes if a.get_visible()]
    assert len(visible) == 3  # 2x2 grid, the spare panel hidden
    ax = visible[0]
    xs = {round(float(line.get_xdata()[0]), 3) for line in ax.lines[2:]}
    assert xs == {0.0, 10.0}  # onset at 0 s, contact 10 s after onset
    assert "AUC" in ax.get_title(loc="left")
    blue = ax.lines[1].get_ydata()
    assert np.nanmin(blue) == 1  # the victim outscores everyone once it is flagged
    assert F.victim_timelines(scores(), {}, FAST) is None


def test_catch_vs_false_alarms_draws_model_and_both_baselines():
    t1 = P.tier1(scores(), truths(), FAST)
    fig = F.catch_vs_false_alarms(
        t1["proxy"], t1["proxy_random_caught"], t1["proxy_shuffled_caught"]
    )
    ax = fig.axes[0]
    assert len(ax.lines) == 3
    labels = {t.get_text() for t in ax.texts}
    assert labels == {"model", "random scores", "track-shuffled"}
    model = ax.lines[0].get_ydata()
    assert model.max() == 1.0  # a perfect victim score catches both rescues
    assert F.catch_vs_false_alarms({"points": []}, {}, {}) is None


def test_per_clip_results_sorts_by_auc_and_shows_detected_vs_scored():
    pc = pd.DataFrame(
        {
            "clip_id": ["aaaaaaaaaa", "bbbbbbbbbb"],
            "within_clip_auc": [0.9, 0.4],
            "time_in_top3": [0.8, 0.1],
            "detected": [1.0, 0.7],
            "scored": [0.9, 0.2],
        }
    )
    fig = F.per_clip_results(pc)
    a1, _, a3 = fig.axes
    assert [t.get_text() for t in a1.get_yticklabels()] == ["bbbbbbbb", "aaaaaaaa"]
    assert len(a3.patches) == 4  # detected + scored per clip
    assert F.per_clip_results(pc.iloc[:0]) is None


def test_cross_run_comparison_groups_seeds_by_run_folder():
    runs = pd.DataFrame(
        {
            "run_folder": ["A", "A", "B", "parent"],
            "headline.within_clip_auc": [0.6, 0.7, 0.5, math.nan],
            "headline.random.within_clip_auc": [0.5, 0.5, 0.5, math.nan],
        }
    )
    fig = F.cross_run_comparison(runs)
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert labels == ["B", "A"]  # sorted by mean, parent row (no metric) dropped
    assert F.cross_run_comparison(runs.drop(columns="headline.within_clip_auc")) is None


def test_pose_figures():
    run = pd.DataFrame(
        {
            "model": ["m1", "m1", "m2"],
            "mode": ["whole", "tiled", "tiled"],
            "recall": [0.05, 0.3, 0.2],
            "false_pos": [0, 30, 10],
        }
    )
    assert isinstance(F.pose_benchmark(run), Figure)
    sweep = pd.DataFrame(
        {
            "model": ["m1"] * 3,
            "threshold": [0.05, 0.1, 0.15],
            "recall": [0.6, 0.5, 0.4],
            "false_pos": [150, 80, 40],
        }
    )
    fig = F.pose_sweep(sweep)
    assert len(fig.axes[0].lines) == 1
    assert F.pose_sweep(sweep.iloc[:0]) is None


# ---------------------------------------------------------------- tracking (needs mlflow)


def test_tracking_parent_child_series_figure_and_lookup(tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    from hydro_knight.eval import tracking

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'm.db'}")
    import mlflow

    mlflow.set_tracking_uri(tracking.default_uri())
    mlflow.create_experiment("t", artifact_location=str(tmp_path / "art"))
    with (
        tracking.run("parent", experiment="t", tags={"command": "x"}, note="hi") as p,
        tracking.run("child", nested=True, tags={"checkpoint_sha256": "abc"}),
    ):
        tracking.log_series("train_loss", [3.0, 2.0, math.nan, 1.0])
        tracking.log_figure(F.loss_curves({"s": {0: [1.0, 0.5]}}), "figs/l.png")
        tracking.log_figure(None, "figs/none.png")  # nothing to draw: skipped
        tracking.log_dataset(
            run_meta.split_table(["a"], ["a"]), "train-s0", "training", "m.jsonl"
        )
    client = mlflow.MlflowClient()
    (child,) = tracking.find_runs("t", "tags.checkpoint_sha256 = 'abc'")
    assert child.data.tags["mlflow.parentRunId"] == p.info.run_id
    hist = client.get_metric_history(child.info.run_id, "train_loss")
    assert [(m.step, m.value) for m in hist] == [(1, 3.0), (2, 2.0), (4, 1.0)]
    arts = [a.path for a in client.list_artifacts(child.info.run_id, "figs")]
    assert arts == ["figs/l.png"]
    assert child.inputs.dataset_inputs[0].dataset.name == "train-s0"
    parent = client.get_run(p.info.run_id)
    assert parent.data.tags["mlflow.note.content"] == "hi"
    assert tracking.find_runs("missing-experiment") == []


# ---------------------------------------------------------------- demo video


def test_demo_video_helpers():
    from hydro_knight.eval import demo_video as D

    assert D.pick_best_worst({"a": 0.9, "b": math.nan, "c": 0.4}) == [
        ("best", "a", 0.9),
        ("worst", "c", 0.4),
    ]
    assert D.pick_best_worst({"a": 0.7}) == [("best", "a", 0.7)]
    assert D.pick_best_worst({"a": math.nan}) == []
    lo, hi = 0.2, 2.0
    assert D.score_color(0.1, lo, hi) == D.score_color(lo, lo, hi)  # clipped green
    b, g, r = D.score_color(hi, lo, hi)
    assert r > 200 and g < 50  # red at the alarm line
    assert D.score_color(math.nan, lo, hi) == D.GREY
    row = {"test_median": 0.3, "detection": [{"fa_target": 120, "threshold": 1.5}]}
    assert D.score_scale(row, np.array([1.0])) == (0.3, 1.5)
    lo2, hi2 = D.score_scale({}, np.arange(1, 101, dtype=float))
    assert lo2 == pytest.approx(50.5) and hi2 == pytest.approx(99.01)
    tr = P.EventTruth("c", 2.0, 4.0, 9.0, frozenset({1}))
    assert D.clip_window(tr) == (2.0, 12.0)  # never before the trim start


def test_render_demo_writes_a_playable_mp4(tmp_path):
    import shutil

    import cv2

    from hydro_knight.eval import demo_video as D

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    vids, kps = tmp_path / "v", tmp_path / "k"
    vids.mkdir()
    kps.mkdir()
    w = cv2.VideoWriter(
        str(vids / "c1.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 180)
    )
    for _ in range(150):  # 5 s
        w.write(np.full((180, 320, 3), 120, np.uint8))
    w.release()
    frames = np.arange(150)
    pd.DataFrame(
        {
            "frame": frames,
            "track_id": 1,
            "bx1": 10.0,
            "by1": 10.0,
            "bx2": 50.0,
            "by2": 60.0,
        }
    ).to_parquet(kps / "c1.parquet")
    alarms = pd.DataFrame(
        {"clip_id": "c1", "track_id": 1, "t": frames / 30, "score": 1.0}
    )
    tr = {"c1": P.EventTruth("c1", 0.0, 1.0, 3.0, frozenset({1}))}
    out = D.render_demo(
        tmp_path / "demo.mp4", alarms, tr, [("best", "c1", 0.9)], kps, vids, 0.2, 2.0
    )
    assert out is not None and out.stat().st_size > 0
    cap = cv2.VideoCapture(str(out))
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # 2.5 s title card + the 0..6 s window, cut at the end of the 5 s video
    assert n == pytest.approx(30 * 2.5 + 30 * 5, abs=3)
    # a missing video file means nothing to render, not an error
    assert (
        D.render_demo(
            tmp_path / "x.mp4", alarms, tr, [("best", "zz", 0.9)], kps, vids, 0.2, 2.0
        )
        is None
    )
