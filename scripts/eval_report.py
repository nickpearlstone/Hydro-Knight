"""
Generate a full evaluation report from keypoint parquets + the manifest.

Runs identically on Colab (full keypoint set restored from Drive) and locally
(dev against a scratch parquet). Each invocation writes one self-contained
run folder: figures, stats parquets, summary.md.

Usage:
  # Saved weights (e.g. the first Colab run's checkpoint):
  python scripts/eval_report.py \
      --keypoints data/keypoints --manifest data/manifests/pool_footage.jsonl \
      --videos raw_local --ckpt tcn_ae.pt --out runs/tcn_baseline

  # Fresh train (holds out 20% of normal windows; saves the new weights):
  python scripts/eval_report.py \
      --keypoints data/keypoints --manifest data/manifests/pool_footage.jsonl \
      --videos raw_local --train-fresh --save-ckpt tcn_fresh.pt --out runs/tcn_fresh

  # Local smoke test on a scratch parquet:
  python scripts/eval_report.py --keypoints scratch/long.parquet \
      --train-fresh --epochs 40 --window 16 --stride 4 --out runs/smoke

Timing correctness: event windows are annotated in *seconds*, so frame->time
conversion needs each clip's true fps. Pass --videos to read fps per clip from
<clip_id>.mp4; otherwise the single --fps value (default 30) is used for all.

Honesty guard: with --train-fresh, a fraction of normal windows (--holdout,
default 0.2) is excluded from training and window-level ROC/PR/percentiles are
computed ONLY on held-out + distress windows — never on windows the model
trained on. Event-level recall/latency use all windows (a catch is a catch).

Threshold: defaults to the 99th percentile of (held-out) normal-window scores
— "alarm on the weirdest 1%" — override with --threshold.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hydro_knight.eval.data_health import (
    dataset_census,
    frame_coverage_in_events,
    track_stats,
)
from hydro_knight.eval.metrics import ClipEval, detections_from_windows, split_scores
from hydro_knight.eval.report import generate_report
from hydro_knight.features.windows import make_windows
from hydro_knight.ingest.manifest import Manifest
from hydro_knight.models.tcn_autoencoder import (
    TCNAutoencoder,
    reconstruction_error,
    train_tcn,
)


def _load_model(ckpt_path: str):
    """Rebuild the TCN from a checkpoint {'model': state_dict, 'scaler': ...}."""
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = TCNAutoencoder(n_feat=34)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["scaler"]


def _save_model(model, scaler, path: str) -> None:
    import torch

    torch.save({"model": model.state_dict(), "scaler": scaler}, path)


def _clip_fps(clip_id: str, videos_dir: Path | None, default: float) -> float:
    """True per-clip fps from the video when available, else the default."""
    if videos_dir is None:
        return default
    video = videos_dir / f"{clip_id}.mp4"
    if not video.exists():
        return default
    import cv2

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps and fps > 0 else default


def _window_is_normal(f0: int, window: int, fps: float, events: list[dict]) -> bool:
    t0, t1 = f0 / fps, (f0 + window) / fps
    return not any(t0 < ev["end"] and t1 > ev["start"] for ev in events)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--keypoints",
        required=True,
        help="dir of <clip_id>.parquet files, or a single parquet",
    )
    ap.add_argument(
        "--manifest",
        default=None,
        help="manifest JSONL (for events + census); optional",
    )
    ap.add_argument("--out", required=True, help="run folder to write")
    ap.add_argument("--ckpt", default=None, help="trained checkpoint (tcn_ae.pt)")
    ap.add_argument(
        "--train-fresh",
        action="store_true",
        help="train a TCN on this data's normal windows (see --holdout)",
    )
    ap.add_argument(
        "--save-ckpt",
        default=None,
        help="with --train-fresh: save the trained weights here",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="training epochs for --train-fresh (300 = the Colab baseline)",
    )
    ap.add_argument(
        "--holdout",
        type=float,
        default=0.2,
        help="fraction of normal windows held out of --train-fresh training",
    )
    ap.add_argument(
        "--videos",
        default=None,
        help="dir of <clip_id>.mp4 — read true per-clip fps (recommended)",
    )
    ap.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="fallback fps when --videos is absent or a video is missing",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="alarm threshold; default = p99 of (held-out) normal scores",
    )
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--stride", type=int, default=8)
    args = ap.parse_args()

    kp = Path(args.keypoints)
    paths = sorted(kp.glob("*.parquet")) if kp.is_dir() else [kp]
    if not paths:
        raise SystemExit(f"no parquet files found under {kp}")
    videos_dir = Path(args.videos) if args.videos else None

    records = {}
    census = None
    if args.manifest:
        recs = Manifest(Path(args.manifest)).load()
        records = {r.clip_id: r for r in recs}
        census = dataset_census(recs)

    # Pass 1 — windows per clip (shared by training and scoring).
    per_clip = []  # (clip_id, df, windows, info, events, fps, duration_s)
    for p in paths:
        df = pd.read_parquet(p)
        if df.empty:
            continue
        w, info = make_windows(df, window=args.window, stride=args.stride)
        rec = records.get(p.stem)
        events = rec.events if rec else []
        fps = _clip_fps(p.stem, videos_dir, args.fps)
        duration = float(df["frame"].max() + 1) / fps
        per_clip.append((p.stem, df, w, info, events, fps, duration))
    if not per_clip:
        raise SystemExit("no usable clips (all parquets empty)")

    # Model + (for fresh training) which windows were trained on.
    loss_history = None
    trained_sel: set[tuple[int, int]] = set()  # (clip_index, window_index)
    if args.ckpt:
        model, scaler = _load_model(args.ckpt)
    elif args.train_fresh:
        normal_idx = [
            (ci, wi)
            for ci, (_cid, _df, w, info, events, fps, _dur) in enumerate(per_clip)
            for wi, (_tid, f0) in enumerate(info)
            if _window_is_normal(f0, args.window, fps, events)
        ]
        if len(normal_idx) < 10:
            raise SystemExit("not enough normal windows to train on")
        order = np.random.RandomState(0).permutation(len(normal_idx))
        n_train = int(round((1.0 - args.holdout) * len(normal_idx)))
        trained_sel = {normal_idx[i] for i in order[:n_train]}
        train_w = np.stack([per_clip[ci][2][wi] for ci, wi in trained_sel])
        print(
            f"training on {len(train_w)} normal windows "
            f"({len(normal_idx) - len(train_w)} held out, {args.epochs} epochs)"
        )
        model, scaler = train_tcn(train_w, epochs=args.epochs)
        loss_history = model.loss_history_
        if args.save_ckpt:
            _save_model(model, scaler, args.save_ckpt)
            print(f"weights saved: {args.save_ckpt}")
    else:
        raise SystemExit("provide --ckpt or --train-fresh")

    # Pass 2 — score every clip, build the neutral detections + health views.
    clips, coverage_rows, track_frames = [], [], []
    for ci, (cid, df, w, info, events, fps, duration) in enumerate(per_clip):
        errors = reconstruction_error(model, w, scaler) if len(w) else np.array([])
        det = detections_from_windows(info, errors, span=args.window)
        if trained_sel and len(det):
            det["trained_on"] = [(ci, wi) in trained_sel for wi in range(len(det))]
        clips.append(ClipEval(cid, det, events, fps, duration))
        coverage_rows += frame_coverage_in_events(df, events, fps)
        track_frames.append(track_stats(df))
    tracks = pd.concat(track_frames, ignore_index=True) if track_frames else None

    threshold = args.threshold
    if threshold is None:
        err_n, _ = split_scores(clips)  # trained-on windows already excluded
        threshold = float(np.percentile(err_n, 99)) if len(err_n) else 1.0

    summary = generate_report(
        Path(args.out),
        clips,
        threshold,
        loss_history=loss_history,
        coverage_rows=coverage_rows,
        tracks=tracks,
        census=census,
        run_name=Path(args.out).name,
    )
    print(f"report written: {summary}")


if __name__ == "__main__":
    main()
