"""Stitching head-track fragments across short gaps (offline relinking, OC-SORT style).

Ported from the 10-26 tracking pilot (branch claude/research-pilot-tracking-1b1ffb,
research/pilot_tracking/relink.py) to the head-track columns. When a track ends and another
starts within `max_gap_s`, near the first one's LAST OBSERVED box (not a Kalman prediction,
which drifts while the head is unseen) and at a similar size, the two get one id. Joins are
taken closest first, one to one, and chained (A -> B -> C all take A's id). Gaps are not filled.

Why: features need >= 2 s of one track, and in 5 of 46 clips the victim's head track breaks
into pieces shorter than that, so v3 never scores the victim there.

Result (2026-10-28, whole-frame heads, gap 2 s / 0.5 head heights): no gain. Victim scored 48% -> 49%,
v3 + appearance AUC 0.89 either way. The victim's breaks don't fit the rule: a third overlap in time
(two boxes on the victim at once) and the rest reappear a median 1.4 head heights away; only 5% fall
inside the gates, and gates wide enough to catch them join the wrong people ~1 time in 5. Kept as an
option (heads-rows --stitch), off by default.

Settings are chosen without victim labels (`gap_check`): cut long tracks, see whether
stitching rejoins the two halves (correct) or joins either half to someone else (wrong).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BOX = ["x0", "y0", "x1", "y1"]


@dataclass(frozen=True)
class StitchSettings:
    max_gap_s: float = 1.0  # latest start of the later track after the earlier one ends
    max_dist_h: float = (
        1.0  # centre distance, in heights of the earlier track's last box
    )
    max_height_ratio: float = 1.5  # larger / smaller box height at the join


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


def candidate_pairs(ends: pd.DataFrame, fps: float, s: StitchSettings) -> pd.DataFrame:
    """(a, b, dist): b starts within the gap after a ends and passes the distance and size gates."""
    starts = ends.sort_values("start")
    st = starts["start"].to_numpy()
    sh = (starts["s_y1"] - starts["s_y0"]).to_numpy()
    scx = (starts["s_x0"] + starts["s_x1"]).to_numpy() / 2
    scy = (starts["s_y0"] + starts["s_y1"]).to_numpy() / 2
    rows = []
    for a, r in ends.iterrows():
        lo = np.searchsorted(st, r["end"], side="right")
        hi = np.searchsorted(st, r["end"] + s.max_gap_s * fps, side="right")
        if hi <= lo:
            continue
        ha = max(r["e_y1"] - r["e_y0"], 1e-6)
        dist = (
            np.hypot(
                scx[lo:hi] - (r["e_x0"] + r["e_x1"]) / 2,
                scy[lo:hi] - (r["e_y0"] + r["e_y1"]) / 2,
            )
            / ha
        )
        hb = np.maximum(sh[lo:hi], 1e-6)
        ratio = np.maximum(hb, ha) / np.minimum(hb, ha)
        ok = (dist <= s.max_dist_h) & (ratio <= s.max_height_ratio)
        rows += [
            (a, b, d) for b, d in zip(starts.index[lo:hi][ok], dist[ok], strict=True)
        ]
    return pd.DataFrame(rows, columns=["a", "b", "dist"])


def find_links(table: pd.DataFrame, fps: float, s: StitchSettings) -> dict:
    """Successor map a -> b, closest pairs first, one to one."""
    pairs = candidate_pairs(track_ends(table), fps, s)
    succ, pred = {}, set()
    for i in np.argsort(pairs["dist"].to_numpy(), kind="stable"):
        a, b = int(pairs.at[i, "a"]), int(pairs.at[i, "b"])
        if a not in succ and b not in pred:
            succ[a] = b
            pred.add(b)
    return succ


def root_map(succ: dict) -> dict:
    """track id -> id of the first track in its chain (only ids in a chain)."""
    pred = {b: a for a, b in succ.items()}
    out = {}
    for t in set(succ) | set(pred):
        r = t
        while r in pred:
            r = pred[r]
        out[t] = r
    return out


def stitch(table: pd.DataFrame, fps: float, s: StitchSettings) -> pd.DataFrame:
    """The table with joined tracks relabelled to their chain's first id (track_id < 0 untouched)."""
    roots = root_map(find_links(table, fps, s))
    if not roots:
        return table
    tid = table["track_id"].to_numpy()
    new = np.array([roots.get(int(t), int(t)) for t in tid], dtype=tid.dtype)
    return table.assign(track_id=new)


def cut_tracks(
    table: pd.DataFrame, fps: float, rng: np.random.Generator, min_s: float = 3.0
) -> tuple[pd.DataFrame, dict]:
    """Every track >= min_s long loses a 0.3-3 s stretch from its middle; part two gets a new id.

    Returns (cut table, {first part id: second part id}).
    """
    t = table.copy()
    ends = track_ends(t)
    long_ = ends[(ends.end - ends.start) / fps >= min_s]
    next_id = int(max(t.track_id.max(), 0)) + 1
    truth = {}
    keep = np.ones(len(t), bool)
    tid = t.track_id.to_numpy().copy()
    fr = t.frame.to_numpy()
    for a, e in long_.iterrows():
        gap = rng.uniform(0.3, 3.0) * fps
        mid = (e.start + e.end) / 2
        g0, g1 = mid - gap / 2, mid + gap / 2
        on = tid == a
        if not ((on & (fr < g0)).any() and (on & (fr > g1)).any()):
            continue
        keep &= ~(on & (fr >= g0) & (fr <= g1))
        tid[on & (fr > g1)] = next_id
        truth[int(a)] = next_id
        next_id += 1
    t["track_id"] = tid
    return t[keep], truth


def gap_check(
    table: pd.DataFrame, fps: float, s: StitchSettings, seed: int = 0
) -> tuple[int, int, int]:
    """(cut tracks, correct rejoins, wrong joins) on artificially cut tracks; no victim labels."""
    cut, truth = cut_tracks(table, fps, np.random.default_rng(seed))
    succ = find_links(cut, fps, s)
    pred = {b: a for a, b in succ.items()}
    correct = sum(succ.get(a) == b for a, b in truth.items())
    wrong = sum(
        (a in succ and succ[a] != b) or (b in pred and pred[b] != a)
        for a, b in truth.items()
    )
    return len(truth), correct, wrong
