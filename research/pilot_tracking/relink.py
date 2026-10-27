"""
Offline relinking of track fragments across short gaps (OC-SORT style "ghost" tracks).

When a track ends and another starts soon after, near the first one's LAST OBSERVED box
(not a Kalman prediction, which drifts while the person is unseen) and at a similar size,
the two are given one id. Candidate joins are taken best first (smallest distance), one to
one, and chained, so A -> B -> C all become A's id.

Detector-agnostic: the input is any track table with frame, track_id, bx1, by1, bx2, by2.
Confidence is never read. track_id < 0 rows are unassigned and left alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BOX = ["bx1", "by1", "bx2", "by2"]


@dataclass(frozen=True)
class RelinkSettings:
    max_gap_s: float = (
        1.0  # G: latest start of the later track after the earlier one ends
    )
    max_dist_h: float = (
        1.0  # R: center distance, in heights of the earlier track's last box
    )
    max_height_ratio: float = 1.5  # S: larger / smaller box height at the join


def track_ends(table: pd.DataFrame) -> pd.DataFrame:
    """One row per track: first/last frame and the box at each end."""
    t = table[table["track_id"] >= 0].sort_values(["track_id", "frame"], kind="stable")
    g = t.groupby("track_id", sort=True)
    first, last = g.head(1).set_index("track_id"), g.tail(1).set_index("track_id")
    out = pd.DataFrame({"start": first["frame"], "end": last["frame"]})
    for c in BOX:
        out["s_" + c] = first[c]
        out["e_" + c] = last[c]
    return out


def candidate_pairs(
    ends: pd.DataFrame, fps: float, s: RelinkSettings, geometry: bool = True
) -> pd.DataFrame:
    """All (a, b) where b starts within the gap window after a ends; with distance and size gates if geometry."""
    max_gap = s.max_gap_s * fps
    starts = ends.sort_values("start")
    st = starts["start"].to_numpy()
    rows = []
    for a, r in ends.iterrows():
        lo = np.searchsorted(st, r["end"], side="right")
        hi = np.searchsorted(st, r["end"] + max_gap, side="right")
        if hi <= lo:
            continue
        cand = starts.iloc[lo:hi]
        ha = r["e_by2"] - r["e_by1"]
        hb = (cand["s_by2"] - cand["s_by1"]).to_numpy()
        dx = (cand["s_bx1"] + cand["s_bx2"]).to_numpy() / 2 - (
            r["e_bx1"] + r["e_bx2"]
        ) / 2
        dy = (cand["s_by1"] + cand["s_by2"]).to_numpy() / 2 - (
            r["e_by1"] + r["e_by2"]
        ) / 2
        dist = np.hypot(dx, dy) / max(ha, 1e-6)
        ratio = np.maximum(hb, ha) / np.maximum(np.minimum(hb, ha), 1e-6)
        ok = np.ones(len(cand), bool)
        if geometry:
            ok = (dist <= s.max_dist_h) & (ratio <= s.max_height_ratio)
        rows += [(a, b, d) for b, d in zip(cand.index[ok], dist[ok], strict=True)]
    return pd.DataFrame(rows, columns=["a", "b", "dist"])


def greedy_links(
    pairs: pd.DataFrame, order: np.ndarray, limit: int | None = None
) -> dict:
    """One-to-one joins taken in `order` (indices into pairs); returns successor map a -> b."""
    succ, pred = {}, set()
    a_col, b_col = pairs["a"].to_numpy(), pairs["b"].to_numpy()
    for i in order:
        if limit is not None and len(succ) >= limit:
            break
        a, b = int(a_col[i]), int(b_col[i])
        if a in succ or b in pred:
            continue
        succ[a] = b
        pred.add(b)
    return succ


def find_links(table: pd.DataFrame, fps: float, s: RelinkSettings) -> dict:
    """Successor map a -> b chosen by the relinking rule (closest first)."""
    pairs = candidate_pairs(track_ends(table), fps, s)
    if pairs.empty:
        return {}
    return greedy_links(pairs, np.argsort(pairs["dist"].to_numpy(), kind="stable"))


def random_links(
    table: pd.DataFrame,
    fps: float,
    s: RelinkSettings,
    n: int,
    rng: np.random.Generator,
    gated: bool = False,
) -> dict:
    """Baseline: n one-to-one joins, partners at random among time-eligible tracks.

    gated=False: any distance. gated=True: only pairs passing the same distance and size
    gates as the method, so it tests "closest first" against "anything nearby".
    """
    pairs = candidate_pairs(track_ends(table), fps, s, geometry=gated)
    if pairs.empty or n == 0:
        return {}
    return greedy_links(pairs, rng.permutation(len(pairs)), limit=n)


def root_map(succ: dict) -> dict:
    """track id -> id of the first track in its chain (only for ids in a chain)."""
    pred = {b: a for a, b in succ.items()}
    out = {}
    for t in set(succ) | set(pred):
        r = t
        while r in pred:
            r = pred[r]
        out[t] = r
    return out


def apply_links(table: pd.DataFrame, succ: dict, fill: bool = False) -> pd.DataFrame:
    """Relabel joined tracks with their chain's first id; if fill, add straight-line boxes across gaps.

    Filled rows carry virtual=True and NaN for every other column.
    """
    out = table.copy()
    out["virtual"] = False
    if fill and succ:
        ends = track_ends(table)
        rows = []
        for a, b in succ.items():
            f0, f1 = ends.at[a, "end"], ends.at[b, "start"]
            frames = np.arange(f0 + 1, f1)
            if not len(frames):
                continue
            w = ((frames - f0) / (f1 - f0))[:, None]
            b0 = ends.loc[a, ["e_" + c for c in BOX]].to_numpy(float)
            b1 = ends.loc[b, ["s_" + c for c in BOX]].to_numpy(float)
            boxes = b0 + w * (b1 - b0)
            rows.append(
                pd.DataFrame(
                    {
                        "frame": frames,
                        "track_id": a,
                        **dict(zip(BOX, boxes.T, strict=True)),
                    }
                ).assign(virtual=True)
            )
        if rows:
            out = pd.concat([out, *rows], ignore_index=True)
    roots = root_map(succ)
    if roots:
        out["track_id"] = out["track_id"].map(lambda t: roots.get(t, t)).astype("int64")
    return out.sort_values(["frame", "track_id"], kind="stable", ignore_index=True)
