"""Match the labeler's victim clicks to tracker tracks, by position and time."""

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
    oversize: set[int] = field(default_factory=set)


def local_height(det: pd.DataFrame, frame: float, y: float, fps: float, k: int = 9):
    """Typical box height of the swimmers near a click, or None if too few."""
    win = det[(det["frame"] - frame).abs() <= fps]
    if win.empty:
        return None
    per = (
        win.assign(h=win["by2"] - win["by1"], cy=(win["by1"] + win["by2"]) / 2)
        .groupby("track_id")[["h", "cy"]]
        .median()
    )
    if len(per) < 3:
        return None
    nearest = np.argsort(np.abs(per["cy"].to_numpy() - y))[:k]
    return float(per["h"].iloc[nearest].median())


def _track_heights(det: pd.DataFrame) -> pd.Series:
    return (det["by2"] - det["by1"]).groupby(det["track_id"]).median()


def match_victim_tracks(
    df: pd.DataFrame,
    record,
    fps: float,
    frame_tol: int = 2,
    radius: float = 0.75,
    max_ratio: float = 3.0,
) -> VictimMatch:
    """Track ids under the record's victim clicks."""
    det = df[df["track_id"] >= 0]
    track_h = _track_heights(det)
    out = VictimMatch()
    for ev in record.events:
        for click in ev.get("victim", []):
            frame = round(record.to_file_time(click["t"]) * fps)
            # reset_index first: idxmin needs a unique index
            near = det[(det["frame"] - frame).abs() <= frame_tol].reset_index(drop=True)
            ref = local_height(det, frame, click["y"], fps)
            if ref is not None:
                h_now = near["by2"] - near["by1"]
                big = (
                    np.maximum(h_now, near["track_id"].map(track_h)) > max_ratio * ref
                ).to_numpy()
                out.oversize |= {int(t) for t in near.loc[big, "track_id"]}
                near = near[~big].reset_index(drop=True)
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
    max_ratio: float = 3.0,
    frame_step: int = 1,
) -> set[int]:
    """Tracks that follow the victim between clicks: likely victim fragments."""
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
        refs = [
            local_height(det, ti * fps, yi, fps) for ti, yi in zip(t, y, strict=True)
        ]
        ok = [i for i, r in enumerate(refs) if r is not None]
        if ok:
            ref = np.interp(ts, t[ok], np.array([refs[i] for i in ok]))
            th = sub["track_id"].map(_track_heights(det)).to_numpy()
            near &= np.maximum(h, th) <= max_ratio * ref
        counts = sub[near].groupby("track_id").size()
        out |= {int(k) for k, n in counts.items() if n * frame_step / fps >= min_s}
    return out
