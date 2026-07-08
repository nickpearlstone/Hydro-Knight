"""
Generate a full evaluation report from keypoint parquets + the manifest.

Runs identically on Colab (full keypoint set restored from Drive) and locally
(dev against a scratch parquet). Each invocation writes one self-contained
run folder: figures, stats parquets, summary.md.

Usage:
  # Colab / full dataset, using a trained checkpoint:
  uv run python scripts/eval_report.py \
      --keypoints data/keypoints --manifest data/manifests/pool_footage.jsonl \
      --ckpt tcn_ae.pt --out runs/tcn_baseline

  # Local smoke test on a scratch parquet (trains a throwaway model):
  uv run python scripts/eval_report.py \
      --keypoints scratch/test_keypoints.parquet --train-fresh --out runs/smoke

Threshold: defaults to the 99th percentile of scores on *normal* (outside-
event) windows — i.e. "alarm on the weirdest 1%" — override with --threshold.
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
    """Rebuild the TCN from a Colab checkpoint {'model': state_dict, 'scaler': ...}."""
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = TCNAutoencoder(n_feat=34)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["scaler"]


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
        help="train a throwaway TCN on this data (local smoke tests)",
    )
    ap.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="frames per second (event windows are in seconds)",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="alarm threshold; default = p99 of normal-window scores",
    )
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--stride", type=int, default=8)
    args = ap.parse_args()

    kp = Path(args.keypoints)
    paths = sorted(kp.glob("*.parquet")) if kp.is_dir() else [kp]
    if not paths:
        raise SystemExit(f"no parquet files found under {kp}")

    records = {}
    census = None
    if args.manifest:
        recs = Manifest(Path(args.manifest)).load()
        records = {r.clip_id: r for r in recs}
        census = dataset_census(recs)

    # Pass 1 — windows per clip (shared by training and scoring).
    per_clip = []  # (clip_id, df, windows, info, events, duration_s)
    for p in paths:
        df = pd.read_parquet(p)
        if df.empty:
            continue
        w, info = make_windows(df, window=args.window, stride=args.stride)
        rec = records.get(p.stem)
        events = rec.events if rec else []
        duration = float(df["frame"].max() + 1) / args.fps
        per_clip.append((p.stem, df, w, info, events, duration))
    if not per_clip:
        raise SystemExit("no usable clips (all parquets empty)")

    # Model: checkpoint, or train fresh on the normal (outside-event) windows.
    loss_history = None
    if args.ckpt:
        model, scaler = _load_model(args.ckpt)
    elif args.train_fresh:
        train_parts = []
        for _cid, _df, w, info, events, _dur in per_clip:
            if len(w) == 0:
                continue
            keep = np.ones(len(w), dtype=bool)
            for i, (_tid, f0) in enumerate(info):
                t0, t1 = f0 / args.fps, (f0 + args.window) / args.fps
                if any(t0 < ev["end"] and t1 > ev["start"] for ev in events):
                    keep[i] = False
            train_parts.append(w[keep])
        train_w = np.concatenate(train_parts) if train_parts else None
        if train_w is None or len(train_w) < 10:
            raise SystemExit("not enough normal windows to train on")
        model, scaler = train_tcn(train_w, epochs=120)
        loss_history = model.loss_history_
    else:
        raise SystemExit("provide --ckpt or --train-fresh")

    # Pass 2 — score every clip, build the neutral detections + health views.
    clips, coverage_rows, track_frames = [], [], []
    for cid, df, w, info, events, duration in per_clip:
        errors = reconstruction_error(model, w, scaler) if len(w) else np.array([])
        det = detections_from_windows(info, errors, span=args.window)
        clips.append(ClipEval(cid, det, events, args.fps, duration))
        coverage_rows += frame_coverage_in_events(df, events, args.fps)
        track_frames.append(track_stats(df))
    tracks = pd.concat(track_frames, ignore_index=True) if track_frames else None

    threshold = args.threshold
    if threshold is None:
        err_n, _ = split_scores(clips)
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
