"""Head tracks (RT-DETR head boxes) and the two features that rank the victim: churn and size flicker.

Input: data/detections_heads/raw/<clip>.npz + .json from the Kaggle head-detection kernel
(hk-heads-detect: RT-DETR v7, 640 px tiles, ~10 fps, conf floor 0.05; frame = video frame).

- Tracks: the same cross-tile merge and ByteTrack settings as build_tracks, with the tracker
  stepping once per sampled frame (fps / step), so the 1 s lost buffer stays 1 s.
- Weak kept (Nick, 2026-10-27): every box on a confirmed track stays, flagged `weak` under
  WEAK_BELOW; boxes on no track are kept only at WEAK_BELOW or above. 37% of the victim's boxes
  are weak, so a hard cut would hide a third of the victim's sightings.
- Victim tracks: tracks within MATCH head heights of a click (Nick clicks on heads: 87% within
  one head height), plus fragments that stay that close to the click path for >= 0.3 s.
- Features over the last WINDOW_S of a track (causal; >= 4 samples spanning the window):
  churn = smoothed path length - net displacement, in head heights ("busy but going nowhere");
  size_flick = std of log box height (the head dipping and reappearing).
  Same-moment victim AUC on 51 clips: churn 0.78, size flicker 0.79, both 0.82.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import supervision as sv

from ..preprocess.build_tracks import TrackSettings, make_tracker
from ..preprocess.tiled_pose import merge_detections

HEADS = Path("data/detections_heads")
FLOOR = 0.1  # ByteTrack's second round never looks below this
WEAK_BELOW = 0.4
MATCH = 1.5  # head heights
PATH_R = (
    2.0  # a victim-track row counts as the victim only this close to the click path
)
VICTIM_ID = 999_999  # one track id for the victim's rows (the tracker's ids never get this high)
WINDOW_S = 2.0
STEP_S = 0.5  # row spacing (the protocol's rank timeline step)
STITCH = None  # stitch.StitchSettings to join track fragments before features (heads-rows --stitch)


def read_meta(clip_id: str) -> dict:
    return json.loads((HEADS / "raw" / f"{clip_id}.json").read_text())


def track_clip(clip_id: str) -> pd.DataFrame:
    """Merge + track one clip's raw head boxes; weak-kept rows (frame, track_id, conf, box, weak)."""
    meta = read_meta(clip_id)
    d = np.load(HEADS / "raw" / f"{clip_id}.npz")
    k = d["conf"] >= FLOOR
    fr, box, conf, crop = (
        d["frame"][k],
        d["box"][k].astype(np.float32),
        d["conf"][k],
        d["crop"][k],
    )
    o = np.argsort(fr, kind="stable")
    fr, box, conf, crop = fr[o], box[o], conf[o], crop[o]
    tracker = make_tracker(meta["fps"] / meta["step"], TrackSettings())
    frames = np.arange(0, meta["frames_decoded"], meta["step"])
    lo_all = np.searchsorted(fr, frames, "left")
    hi_all = np.searchsorted(fr, frames, "right")
    rows = []
    for f, lo, hi in zip(frames, lo_all, hi_all, strict=True):
        if hi > lo:
            idx = lo + merge_detections(box[lo:hi], conf[lo:hi], crop[lo:hi])
            det = sv.Detections(
                xyxy=box[idx].astype(float),
                confidence=conf[idx].astype(float),
                class_id=np.zeros(len(idx), dtype=int),
            )
            det.data["row"] = idx  # the tracker reorders its output
        else:
            det = sv.Detections.empty()
        t = tracker.update(det)
        if len(t):
            r = t.data["row"].astype(int)
            rows.append(
                np.column_stack([np.full(len(r), f), t.tracker_id, conf[r], box[r]])
            )
    df = pd.DataFrame(
        np.concatenate(rows),
        columns=["frame", "track_id", "conf", "x0", "y0", "x1", "y1"],
    )
    df = df.astype(
        {"frame": "int64", "track_id": "int64"} | {c: "float32" for c in df.columns[2:]}
    )
    df = df[(df.track_id >= 0) | (df.conf >= WEAK_BELOW)].reset_index(drop=True)
    return df.assign(weak=(df.track_id >= 0) & (df.conf < WEAK_BELOW))


def write_eval_tracks(df: pd.DataFrame, clip_id: str, out: Path) -> None:
    """Confirmed (frame, track_id) with fps provenance, the form `eval protocol --tracks-dir` reads."""
    meta = read_meta(clip_id)
    out.mkdir(parents=True, exist_ok=True)
    tab = pa.Table.from_pandas(
        df.loc[df.track_id >= 0, ["frame", "track_id"]].reset_index(drop=True),
        preserve_index=False,
    )
    prov = {"video": {"fps": meta["fps"]}, "frame_step": meta["step"]}
    md = dict(tab.schema.metadata or {}) | {b"hydro_knight": json.dumps(prov).encode()}
    pq.write_table(tab.replace_schema_metadata(md), out / f"{clip_id}.parquet")


def with_geometry(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(cx=(df.x0 + df.x1) / 2, cy=(df.y0 + df.y1) / 2, hh=df.y1 - df.y0)


def victim_tracks(
    d: pd.DataFrame, record, fps: float, step: int
) -> tuple[list[int], list[int]]:
    """(click-matched track ids, path fragments) for one clip; d = confirmed rows with geometry."""
    clicks = sorted(record.events[0]["victim"], key=lambda c: c["t"])
    cf = np.array([record.to_file_time(c["t"]) * fps for c in clicks])
    frames = np.unique(d.frame)
    click = set()
    for c, f in zip(clicks, cf, strict=True):
        nb = d[d.frame == frames[np.argmin(abs(frames - f))]]
        if nb.empty:
            continue
        dist = np.hypot(nb.cx - c["x"], nb.cy - c["y"]).to_numpy() / np.maximum(
            nb.hh.to_numpy(), 1
        )
        if dist.min() <= MATCH:
            click.add(int(nb.track_id.to_numpy()[dist.argmin()]))
    seg = d[(d.frame >= cf[0]) & (d.frame <= cf[-1])]
    px = np.interp(seg.frame, cf, [c["x"] for c in clicks])
    py = np.interp(seg.frame, cf, [c["y"] for c in clicks])
    near = np.hypot(seg.cx - px, seg.cy - py) <= MATCH * seg.hh
    n = seg[near].groupby("track_id").size()
    frags = {int(k) for k, v in n.items() if v * step / fps >= 0.3} - click
    return sorted(click), sorted(frags)


def _window(fr, sx, sy, lh, conf, h, fps, w):
    """churn, size flicker, confidence flicker over the last w s of one track (NaN until full)."""
    cp = np.cumsum(np.r_[0, np.hypot(np.diff(sx), np.diff(sy))])
    j = np.arange(len(fr))
    i0 = np.searchsorted(fr, fr - w * fps - 0.5)
    ok = (fr[i0] <= fr - w * fps + 0.15 * fps) & (j - i0 >= 3)
    n = j - i0 + 1

    def sd(v):
        c1, c2 = np.r_[0, np.cumsum(v)], np.r_[0, np.cumsum(v**2)]
        s1, s2 = c1[j + 1] - c1[i0], c2[j + 1] - c2[i0]
        return np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0))

    churn = (cp[j] - cp[i0]) / h - np.hypot(sx - sx[i0], sy - sy[i0]) / h
    return (np.where(ok, x, np.nan) for x in (churn, sd(lh), sd(conf)))


def add_features(d: pd.DataFrame, fps: float) -> pd.DataFrame:
    """d (confirmed rows with geometry) + per-sample features, NaN until their window fills.

    churn, size_flick, conf_flick over WINDOW_S; churn_4 over 4 s (slower drift-free struggle);
    aspect = median log(width / height) over the last 1 s: the box widens as the face tips back
    toward the sky or the head sits low in the water (Nick's head-tilt cue; AUC 0.74 alone).
    """
    d = d.sort_values(["track_id", "frame"]).reset_index(drop=True)
    cols = {
        k: np.full(len(d), np.nan)
        for k in ("churn", "size_flick", "conf_flick", "churn_4", "aspect")
    }
    for _, idx in d.groupby("track_id", sort=False).indices.items():
        t = d.iloc[idx]
        fr = t.frame.to_numpy()
        sx = (
            t.cx.rolling(3, min_periods=1).mean().to_numpy()
        )  # last 3 samples: cuts jitter
        sy = t.cy.rolling(3, min_periods=1).mean().to_numpy()
        lh = np.log(np.maximum(t.hh.to_numpy(), 1))
        conf = t.conf.to_numpy(np.float64)
        h = t.hh.median()
        ch, sf, cf = _window(fr, sx, sy, lh, conf, h, fps, WINDOW_S)
        cols["churn"][idx], cols["size_flick"][idx], cols["conf_flick"][idx] = (
            ch,
            sf,
            cf,
        )
        cols["churn_4"][idx] = next(_window(fr, sx, sy, lh, conf, h, fps, 4.0))
        ar = pd.Series(
            np.log(np.maximum((t.x1 - t.x0).to_numpy(), 1)) - lh,
            index=pd.to_datetime(fr / fps, unit="s"),
        )
        cols["aspect"][idx] = ar.rolling("1s").median().to_numpy()
    return d.assign(**cols)


def victim_label(d: pd.DataFrame, record, fps: float, tracks) -> np.ndarray:
    """Per row: 1 victim, 0 someone else, -1 unknown (dropped). d needs frame, track_id, cx, cy, hh.

    Whole-track labels go wrong when a track switches people or a neighbour drifts along the
    victim's path (644cb555b4d9: a matched track started 250 px away, 7 s before onset). So a row
    of a victim track is the victim only within PATH_R head heights of where the clicks put the
    victim at that time: the interpolated click path between the first and last click, else the
    nearest click if it is at most 1 s away. One row per frame at most (the nearest); the other
    rows of victim tracks are unknown. Rows of other tracks stay 0.
    """
    clicks = sorted(record.events[0]["victim"], key=lambda c: c["t"])
    ct = np.array([record.to_file_time(c["t"]) for c in clicks])
    t = d.frame.to_numpy() / fps
    px = np.interp(t, ct, [c["x"] for c in clicks])
    py = np.interp(t, ct, [c["y"] for c in clicks])
    dist = np.hypot(d.cx.to_numpy() - px, d.cy.to_numpy() - py) / np.maximum(
        d.hh.to_numpy(), 1
    )
    gap = np.where(
        (t >= ct[0]) & (t <= ct[-1]), 0, np.minimum(abs(t - ct[0]), abs(t - ct[-1]))
    )
    on = d.track_id.isin(tracks).to_numpy()
    ok = on & (dist <= PATH_R) & (gap <= 1.0)
    label = np.where(on, -1, 0)
    if ok.any():
        cand = pd.DataFrame(
            {"frame": d.frame.to_numpy()[ok], "dist": dist[ok]},
            index=np.flatnonzero(ok),
        )
        label[cand.groupby("frame").dist.idxmin().to_numpy()] = 1
    return label


def clip_rows(record, zone, in_zone) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scored rows every ~STEP_S in the zone, and the clip's (frame, track_id) for the protocol.

    Rows: clip_id, track_id (VICTIM_ID for the victim), src_track (the tracker's id), t, features,
    head box, victim. Rows victim_label() can't attribute are dropped from both tables.
    """
    cid = record.clip_id
    meta = read_meta(cid)
    fps, step = meta["fps"], meta["step"]
    d = pd.read_parquet(HEADS / "tracks_kept" / f"{cid}.parquet")
    d = d[d.track_id >= 0]
    if STITCH is not None:
        from .stitch import stitch

        d = stitch(d, fps, STITCH)
    d = with_geometry(d)
    click, frags = victim_tracks(d, record, fps, step)
    d = add_features(d, fps)  # per tracker track, before any relabelling
    d["label"] = victim_label(d, record, fps, click + frags)
    d = d[d.label >= 0]
    d["src_track"] = d.track_id
    d.loc[d.label == 1, "track_id"] = VICTIM_ID
    tracks = d[["frame", "track_id"]].reset_index(drop=True)
    start = 0.0 if record.trim_start is None else record.to_file_time(record.trim_start)
    end = (
        d.frame.max() / fps
        if record.trim_end is None
        else record.to_file_time(record.trim_end)
    )
    frames = np.unique(d.frame)
    grid = np.arange(start, end, STEP_S)
    picks = (
        frames[np.abs(frames[None, :] / fps - grid[:, None]).argmin(1)]
        if len(grid)
        else []
    )
    w = d[d.frame.isin(np.unique(picks))].dropna(subset=["churn", "size_flick"])
    w = w[np.asarray(in_zone(zone, w.cx.to_numpy(), w.cy.to_numpy()))]
    rows = pd.DataFrame(
        {
            "clip_id": cid,
            "track_id": w.track_id.to_numpy(),
            "src_track": w.src_track.to_numpy(),
            "t": w.frame.to_numpy() / fps,
            "churn": w.churn.to_numpy(),
            "size_flick": w.size_flick.to_numpy(),
            "conf_flick": w.conf_flick.to_numpy(),
            "churn_4": w.churn_4.to_numpy(),
            "aspect": w.aspect.to_numpy(),
            "x": w.cx.to_numpy(),
            "y": w.cy.to_numpy(),
            "hh": w.hh.to_numpy(),
            "hw": (w.x1 - w.x0).to_numpy(),
            "victim": (w.label == 1).to_numpy(),
        }
    )
    return rows, tracks


def rank_score(rows: pd.DataFrame) -> np.ndarray:
    """Within-moment percentile of churn + of size flicker (unfitted)."""
    m = rows.clip_id + "_" + rows.t.round(3).astype(str)
    return (
        rows.groupby(m).churn.rank(pct=True) + rows.groupby(m).size_flick.rank(pct=True)
    ).to_numpy()


def raw_score(rows: pd.DataFrame, folds: dict[str, int]) -> np.ndarray:
    """log1p(feature) scaled by the median / IQR of the other folds' rows, summed (fixed scale)."""
    fold = rows.clip_id.map(folds).to_numpy()
    out = np.full(len(rows), np.nan)
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        s = np.zeros(te.sum())
        for c in ("churn", "size_flick"):
            x = np.log1p(np.maximum(rows[c].to_numpy(), 0))
            q1, med, q3 = np.quantile(x[tr], [0.25, 0.5, 0.75])
            s += (x[te] - med) / (q3 - q1)
        out[te] = s
    return out


# v3: the within-moment ranker (stack.rank.fit_ranker) on head-track features.
V3_FEATURES = [
    "f_churn",
    "f_churn_4",
    "f_size",
    "f_conf",
    "f_splash",
    "f_aspect",
    "f_height",
]
V3_SIGNED = [
    "f_churn",
    "f_churn_4",
    "f_size",
    "f_conf",
    "f_splash",
    "f_aspect",
]  # weight >= 0


def v3_features(rows: pd.DataFrame) -> pd.DataFrame:
    """log1p of each measurement (all >= 0, right-skewed); log head height as a free control."""
    f = {
        "f_churn": "churn",
        "f_churn_4": "churn_4",
        "f_size": "size_flick",
        "f_conf": "conf_flick",
        "f_splash": "splash",
    }
    out = {
        k: np.log1p(np.maximum(rows[c].to_numpy(np.float64), 0)) for k, c in f.items()
    }
    out["f_aspect"] = rows["aspect"].to_numpy(
        np.float64
    )  # already a log ratio, can be < 0
    out["f_height"] = np.log(np.maximum(rows["hh"].to_numpy(np.float64), 1.0))
    return pd.DataFrame(out, index=rows.index)
