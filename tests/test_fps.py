"""
fps source of truth: backfill snapping and the order evaluation reads fps in.

Clicks and events are seconds, tracks are frames, so the tracker and the evaluation
must agree on each clip's fps. These pin that eval reads the fps the tracker used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from backfill_fps import _snap  # noqa: E402

from hydro_knight.eval.timing import clip_fps  # noqa: E402


@pytest.mark.parametrize(
    ("reported", "snapped"),
    [
        (60.0, 60.0),
        (59.94, 59.94),
        (24.0, 24.0),
        (23.976, 23.976),
        (29.97, 29.97),
        (30.0, 30.0),
        (60000 / 1001, 59.94),
        (17.3, 17.3),
    ],
)
def test_snap_picks_the_nearest_standard_rate(reported, snapped):
    # The old _snap took the first rate within 2%, so 60 became 59.94.
    assert _snap(reported) == snapped


class _Rec:
    fps = 59.94


def _parquet(path, video_fps=None):
    table = pa.Table.from_pandas(pd.DataFrame({"frame": [0, 1]}))
    if video_fps is not None:
        meta = {b"hydro_knight": json.dumps({"video": {"fps": video_fps}}).encode()}
        table = table.replace_schema_metadata(meta)
    pq.write_table(table, path)
    return path


def test_eval_uses_the_fps_the_tracker_used(tmp_path):
    p = _parquet(tmp_path / "clip.parquet", video_fps=60.0)
    assert clip_fps(p, None, 30.0, record=_Rec()) == (60.0, False)


def test_falls_back_to_manifest_then_default(tmp_path):
    p = _parquet(tmp_path / "old.parquet")  # written before provenance existed
    assert clip_fps(p, None, 30.0, record=_Rec()) == (59.94, False)
    assert clip_fps(p, None, 30.0, record=None) == (30.0, True)
