"""
Match the labeler's victim clicks to tracker tracks, by position and time.

Labels are clicks (source-video seconds, frame pixels), never track ids, because ids
change whenever tracking is rerun. Each click is converted to a file frame with the
clip's own clock, and matched to a confirmed track near that frame: a box containing
the click wins (closest center first), otherwise the nearest box center within
`radius` box heights. A click with nothing nearby means the victim was not detected
at that moment; it is reported, never forced onto another swimmer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..annotate.server import needs_victim_marks
from ..ingest.manifest import Label


def is_labeled(record) -> bool:
    """True for a distress clip with onset, contact, save and both victim clicks set."""
    return record.label == Label.DISTRESS and not needs_victim_marks(record)


@dataclass
class VictimMatch:
    """Victim track ids found for a clip, plus clicks that matched no track."""

    tracks: set[int] = field(default_factory=set)
    unmatched: list[dict] = field(default_factory=list)


def match_victim_tracks(
    df: pd.DataFrame, record, fps: float, frame_tol: int = 2, radius: float = 0.75
) -> VictimMatch:
    """Track ids under the record's victim clicks.

    Args:
        df: the clip's keypoint table (frame, track_id, bx1..by2).
        record: the clip's ClipRecord (events with "victim" clicks, source seconds).
        fps: the clip's source fps (the one tracking used).
        frame_tol: frames either side of the click's frame to search.
        radius: max distance from a box center, in box heights, when no box contains
            the click.
    Returns:
        VictimMatch.
    """
    det = df[df["track_id"] >= 0]
    out = VictimMatch()
    for ev in record.events:
        for click in ev.get("victim", []):
            frame = round(record.to_file_time(click["t"]) * fps)
            # reset_index: the table's index repeats across tracks, idxmin needs one row
            near = det[(det["frame"] - frame).abs() <= frame_tol].reset_index(drop=True)
            if near.empty:
                out.unmatched.append(click)
                continue
            cx = (near["bx1"] + near["bx2"]) / 2
            cy = (near["by1"] + near["by2"]) / 2
            h = (near["by2"] - near["by1"]).clip(lower=1e-6)
            dist = np.hypot(cx - click["x"], cy - click["y"]) / h
            inside = (
                (near["bx1"] <= click["x"])
                & (click["x"] <= near["bx2"])
                & (near["by1"] <= click["y"])
                & (click["y"] <= near["by2"])
            )
            best = int((dist - 1e6 * inside).idxmin())  # a containing box always wins
            if inside[best] or dist[best] <= radius:
                out.tracks.add(int(near.at[best, "track_id"]))
            else:
                out.unmatched.append(click)
    return out
