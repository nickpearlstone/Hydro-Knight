"""Synthetic keypoint tables for the v2 feature/model tests (no video, no YOLO)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

KP_COLS = [f"{a}{i}" for i in range(17) for a in ("x", "y", "c")]


def track_rows(track_id, frames, box=(100, 100, 140, 180), conf=0.9, dx_per_frame=0.0):
    """Rows for one track: a fixed-size box sliding right by dx_per_frame px per frame,
    every keypoint at the box center with confidence `conf`."""
    rows = []
    for f in frames:
        x1, y1, x2, y2 = box
        x1, x2 = x1 + dx_per_frame * f, x2 + dx_per_frame * f
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        row = {"frame": int(f), "track_id": int(track_id), "box_conf": conf}
        for i in range(17):
            row[f"x{i}"], row[f"y{i}"], row[f"c{i}"] = cx, cy, conf
        row.update({"bx1": x1, "by1": y1, "bx2": x2, "by2": y2})
        rows.append(row)
    cols = ["frame", "track_id", "box_conf", *KP_COLS, "bx1", "by1", "bx2", "by2"]
    return pd.DataFrame(rows, columns=cols)


def write_keypoints(path, df, fps, width=1920, height=1080):
    """Write a keypoint Parquet carrying the provenance build_tracks records."""
    meta = {"video": {"fps": fps, "width": width, "height": height}}
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata({b"hydro_knight": json.dumps(meta).encode()})
    path = Path(path)
    pq.write_table(table, path)
    return path


def series_of(n, n_feat=58, seed=0, detected=None):
    """A random TrackSeries-like (feats, detected) pair of length n for windowing tests."""
    rng = np.random.default_rng(seed)
    feats = rng.normal(size=(n, n_feat)).astype(np.float32)
    det = np.ones(n, bool) if detected is None else np.asarray(detected, bool)
    feats[:, 52] = det
    return feats, det
