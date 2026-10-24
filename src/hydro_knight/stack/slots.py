"""Per-swimmer rows every 0.5 s, for visible swimmers and for ghosts (where a head vanished).

One pass over the frames keeps a memory of every confirmed track: where its head was last
seen, how big it was, and when. A track that lasted at least MIN_TRACK_S and vanishes
mid-pool (inside the zone, not near the frame edge) becomes a ghost: a spot that keeps
emitting rows, with the same track id, until any head is seen within KILL_L body lengths
of it, the same track comes back, or GHOST_CAP_S passes. Everything is causal: a row at
time t uses only frames up to t.

Columns: clip_id, track_id, t, kind (0 visible, 1 ghost), x, y (head, full-res pixels),
h (box height = the local body length L), w (box width), box_conf (NaN for ghosts), unseen_s,
dunks_30s (disappearances of at least MIN_GAP_S starting within DUNK_L body lengths of
this spot in the last 30 s, any track), and victim / victim_click flags.

A ghost measures "time the detector didn't see a head there". With the pose detector
finding 30-40% of countable swimmers that is mostly misses, not time under water
(stack review, 2026-10-24), so it is one input to the ranker, never an alarm on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..annotate import zones as zones_mod
from .data import BOX, Clip, head_points


@dataclass(frozen=True)
class SlotParams:
    """Hand-set constants, frozen before any K-fold run (taken from the review's measurements)."""

    step_s: float = 0.5  # row spacing (the protocol's rank timeline step)
    visible_s: float = 0.2  # seen within this long counts as visible (detector flicker)
    min_track_s: float = 0.5  # a track this long can leave a ghost (77% of victim gaps)
    kill_l: float = 0.5  # any head within this many body lengths ends a ghost
    edge_l: float = 1.0  # ghosts closer than this to the frame edge are not started
    ghost_cap_s: float = 15.0
    dunk_l: float = 1.3  # ~7 head widths
    dunk_window_s: float = 30.0
    min_gap_s: float = 0.5  # a disappearance this long counts as a dunk
    lifeguard_ratio: float = (
        3.0  # tracks taller than this x the clip median are dropped
    )


@dataclass
class _Track:
    first_t: float
    last_t: float
    x: float
    y: float
    h: float
    w: float
    ghost: bool = False  # currently a live ghost
    dead: bool = (
        False  # ghost ended by a nearby head or the cap; waits for its id to return
    )
    episode: int = -1  # index into the episode log while a ghost


def build_rows(clip: Clip, frame_size: tuple[int, int], p: SlotParams | None = None):
    """Rows for one clip from its footage start to guard contact (see module docstring)."""
    p = p or SlotParams()
    W, H = frame_size
    d = clip.df[clip.df["track_id"] >= 0]
    h_all = (d["by2"] - d["by1"]).to_numpy(np.float64)
    med_h = float(np.median(h_all)) if len(h_all) else 1.0
    tall = d.assign(h=h_all).groupby("track_id")["h"].median()
    lifeguards = set(tall.index[tall > p.lifeguard_ratio * med_h].tolist())
    d = d[~d["track_id"].isin(lifeguards)]
    hx, hy = head_points(d)
    d = d.assign(
        hx=hx,
        hy=hy,
        h=(d["by2"] - d["by1"]).to_numpy(np.float64),
        w=(d["bx2"] - d["bx1"]).to_numpy(np.float64),
    )
    d = d.assign(
        zone=zones_mod.in_zone(clip.zone, d["hx"].to_numpy(), d["hy"].to_numpy())
    )
    f0, f1 = int(round(clip.start * clip.fps)), int(round(clip.contact * clip.fps))
    d = d[(d["frame"] >= f0) & (d["frame"] <= f1)]
    by_frame = {
        int(f): g[["track_id", "hx", "hy", "h", "box_conf", "zone", "w"]].to_numpy(
            np.float64
        )
        for f, g in d.groupby("frame")
    }
    sample_every = max(1, int(round(p.step_s * clip.fps)))
    tracks: dict[int, _Track] = {}
    ep_start, ep_end, ep_x, ep_y = [], [], [], []  # episode log (end = nan while open)
    rows = []
    for f in range(f0, f1 + 1):
        t = f / clip.fps
        det = by_frame.get(f)
        if det is not None:
            for tid, x, y, h, _, _, w in det:
                tr = tracks.get(int(tid))
                if tr is None or tr.dead:
                    tracks[int(tid)] = _Track(t, t, x, y, h, w)
                    continue
                if tr.ghost:  # the same id came back: the ghost ends
                    ep_end[tr.episode] = t
                    tr.ghost, tr.episode = False, -1
                tr.last_t, tr.x, tr.y, tr.h, tr.w = t, x, y, h, w
            # any head near a live ghost ends it (a soft claim comes later; MVP = nearest rule)
            gx = [(tid, tr) for tid, tr in tracks.items() if tr.ghost]
            if gx:
                px = np.array([tr.x for _, tr in gx])
                py = np.array([tr.y for _, tr in gx])
                pl = np.array([tr.h for _, tr in gx])
                dist = np.hypot(
                    det[:, 1][None] - px[:, None], det[:, 2][None] - py[:, None]
                )
                own = det[:, 0][None] == np.array([tid for tid, _ in gx])[:, None]
                hit = ((dist <= p.kill_l * pl[:, None]) & ~own).any(1)
                for (_, tr), k in zip(gx, hit, strict=True):
                    if k:
                        ep_end[tr.episode] = t
                        tr.ghost, tr.dead, tr.episode = False, True, -1
        # start ghosts for tracks that just went unseen; cap old ones
        for tr in tracks.values():
            if tr.dead:
                continue
            gone = t - tr.last_t
            if tr.ghost and gone > p.ghost_cap_s:
                ep_end[tr.episode] = t
                tr.ghost, tr.dead, tr.episode = False, True, -1
            elif (
                not tr.ghost
                and gone > p.visible_s
                and tr.last_t - tr.first_t >= p.min_track_s
                and _mid_pool(tr, clip.zone, W, H, p.edge_l)
            ):
                tr.ghost, tr.episode = True, len(ep_start)
                ep_start.append(tr.last_t)
                ep_end.append(np.nan)
                ep_x.append(tr.x)
                ep_y.append(tr.y)
        if (f - f0) % (
            sample_every * 20
        ) == 0:  # forget tracks gone longer than any ghost lives
            for tid in [
                k for k, tr in tracks.items() if t - tr.last_t > p.ghost_cap_s + 1
            ]:
                if not tracks[tid].ghost:
                    del tracks[tid]
        if (f - f0) % sample_every:
            continue
        snap = []
        if det is not None:
            for tid, x, y, h, conf, zone, w in det:
                if zone:
                    snap.append((int(tid), 0, x, y, h, conf, 0.0, w))
        seen_now = {s[0] for s in snap}
        for tid, tr in tracks.items():
            if tid in seen_now or tr.dead:
                continue
            gone = t - tr.last_t
            if tr.ghost:
                snap.append((tid, 1, tr.x, tr.y, tr.h, np.nan, gone, tr.w))
            elif (
                gone <= p.visible_s and zones_mod.in_zone(clip.zone, [tr.x], [tr.y])[0]
            ):
                snap.append(
                    (tid, 0, tr.x, tr.y, tr.h, np.nan, 0.0, tr.w)
                )  # flicker: still visible
        if not snap:
            continue
        arr = np.array([s[1:] for s in snap], np.float64)
        dunks = _dunks(
            arr[:, 1], arr[:, 2], arr[:, 3], t, ep_start, ep_end, ep_x, ep_y, p
        )
        for (tid, *_), a, n in zip(snap, arr, dunks, strict=True):
            rows.append((tid, t, int(a[0]), a[1], a[2], a[3], a[6], a[4], a[5], n))
    out = pd.DataFrame(
        rows,
        columns=[
            "track_id",
            "t",
            "kind",
            "x",
            "y",
            "h",
            "w",
            "box_conf",
            "unseen_s",
            "dunks_30s",
        ],
    )
    out.insert(0, "clip_id", clip.clip_id)
    out["victim"] = out["track_id"].isin(clip.victim_tracks)
    out["victim_click"] = out["track_id"].isin(clip.victim_click_tracks)
    return out


def _mid_pool(tr: _Track, zone, W: int, H: int, edge_l: float) -> bool:
    m = edge_l * tr.h
    inside = m <= tr.x <= W - m and m <= tr.y <= H - m
    return bool(inside and zones_mod.in_zone(zone, [tr.x], [tr.y])[0])


def _dunks(x, y, h, t, ep_start, ep_end, ep_x, ep_y, p: SlotParams) -> np.ndarray:
    """Disappearances of at least min_gap_s that started near each (x, y) in the last window."""
    if not ep_start:
        return np.zeros(len(x), int)
    s, e = np.asarray(ep_start), np.asarray(ep_end)
    length = np.where(np.isnan(e), t, e) - s
    keep = (s >= t - p.dunk_window_s) & (s <= t) & (length >= p.min_gap_s)
    if not keep.any():
        return np.zeros(len(x), int)
    ex, ey = np.asarray(ep_x)[keep], np.asarray(ep_y)[keep]
    dist = np.hypot(ex[None] - x[:, None], ey[None] - y[:, None])
    return (dist <= p.dunk_l * h[:, None]).sum(1)


__all__ = ["SlotParams", "build_rows", "BOX"]
