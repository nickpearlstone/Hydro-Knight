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


def near_path_tracks(
    df: pd.DataFrame,
    record,
    fps: float,
    exclude=frozenset(),
    radius: float = 1.0,
    min_s: float = 0.3,
) -> set[int]:
    """Tracks that follow the victim between clicks: likely victim fragments.

    The tracker splits one swimmer into ~1 s pieces, and clicks ~1 s apart miss the
    pieces between them. The victim's path is the straight line between consecutive
    clicks (first to last click of each event); a track whose box center stays within
    `radius` box heights of that path, at the same moment, for at least `min_s` seconds
    is treated as the victim too. A bystander right beside the victim can be caught,
    which costs a little normal data and never mislabels the victim as normal.

    Args:
        df: the clip's keypoint table (frame, track_id, bx1..by2).
        record: the clip's ClipRecord (events with "victim" clicks, source seconds).
        fps: the clip's source fps (the one tracking used).
        exclude: track ids already known to be the victim (left out of the result).
        radius: max distance from the path, in the track's box heights.
        min_s: min time near the path.
    Returns:
        Set of track ids.
    """
    det = df[(df["track_id"] >= 0) & ~df["track_id"].isin(exclude)]
    out: set[int] = set()
    for ev in record.events:
        clicks = sorted(ev.get("victim", []), key=lambda c: c["t"])
        if len(clicks) < 2:
            continue
        t = np.array([record.to_file_time(c["t"]) for c in clicks])
        x = np.array([c["x"] for c in clicks], float)
        y = np.array([c["y"] for c in clicks], float)
        sub = det[(det["frame"] >= t[0] * fps) & (det["frame"] <= t[-1] * fps)]
        if sub.empty:
            continue
        ts = sub["frame"].to_numpy() / fps
        px, py = np.interp(ts, t, x), np.interp(ts, t, y)
        cx = (sub["bx1"] + sub["bx2"]).to_numpy() / 2
        cy = (sub["by1"] + sub["by2"]).to_numpy() / 2
        h = np.maximum((sub["by2"] - sub["by1"]).to_numpy(), 1e-6)
        near = np.hypot(cx - px, cy - py) <= radius * h
        counts = sub[near].groupby("track_id").size()
        out |= {int(k) for k, n in counts.items() if n / fps >= min_s}
    return out
