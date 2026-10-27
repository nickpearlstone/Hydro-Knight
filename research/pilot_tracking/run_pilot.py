"""
Pilot experiment runner (see PREREG.md).

  uv run python research/pilot_tracking/run_pilot.py gaps OUT_DIR      # artificial-gap check
  uv run python research/pilot_tracking/run_pilot.py victims OUT_DIR   # victim scoring, full grid
  uv run python research/pilot_tracking/run_pilot.py summary OUT_DIR G R

Reads the main checkout's data read-only; writes CSVs to OUT_DIR.
"""

from __future__ import annotations

import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from clickscore import ClickScorer, clicks_for, shifted_paths, translated_paths
from hydro_knight.eval.protocol import ProtocolConfig, truths_from_records, victim_detected_share
from hydro_knight.eval.victims import is_labeled, match_victim_tracks, near_path_tracks
from hydro_knight.ingest.manifest import Manifest
from hydro_knight.preprocess.build_tracks import provenance_frame_step, read_provenance
from relink import RelinkSettings, apply_links, find_links, random_links, track_ends

MAIN = Path("/Users/nickpearlstone/dev/repos/Hydro-Knight")
TRACKS = MAIN / "data/keypoints_tiled"
COLS = ["frame", "track_id", "bx1", "by1", "bx2", "by2"]
GRID = [RelinkSettings(g, r) for g, r in itertools.product([0.5, 1, 2, 4], [0.5, 1, 2])]
N_PATHS, N_RANDOM_RELINK = 20, 5


def clips():
    reviews = json.loads((MAIN / "data/annotations/label_reviews.json").read_text())
    for r in Manifest(MAIN / "data/manifests/pool_footage.jsonl").load():
        if "[HOLD" in (r.notes or "") or not is_labeled(r):
            continue
        if reviews.get(r.clip_id, {}).get("verdict") == "not_good":
            continue
        p = TRACKS / f"{r.clip_id}.parquet"
        if not p.exists():
            continue
        prov = read_provenance(p)
        fps = prov["video"]["fps"] / provenance_frame_step(prov)
        yield r, pd.read_parquet(p, columns=COLS), fps


# ---------------------------------------------------------------- artificial gaps


def cut_gaps(table: pd.DataFrame, fps: float, rng: np.random.Generator):
    """Delete a 0.3-3 s stretch from the middle of every track lasting >= 3 s; new id after the gap."""
    ends = track_ends(table)
    nxt = int(table["track_id"].max()) + 1
    out, pairs = table.copy(), []
    for tid, e in ends.iterrows():
        dur = (e["end"] - e["start"]) / fps
        if dur < 3.0:
            continue
        g = rng.uniform(0.3, 3.0)
        if dur < g + 1.0:
            continue
        c = rng.uniform(e["start"] / fps + 0.5, e["end"] / fps - 0.5 - g) * fps
        rows = out["track_id"] == tid
        out = out[~(rows & (out["frame"] >= c) & (out["frame"] < c + g * fps))]
        tail = (out["track_id"] == tid) & (out["frame"] >= c + g * fps)
        out.loc[tail, "track_id"] = nxt
        pairs.append((int(tid), nxt, g))
        nxt += 1
    return out, pairs


def run_gaps(out_dir: Path):
    rows = []
    data = list(clips())
    for seed in (0, 1):
        for r, table, fps in data:
            rng = np.random.default_rng([seed, int(r.clip_id, 16) % 2**32])
            cut, pairs = cut_gaps(table[table["track_id"] >= 0], fps, rng)
            for s in GRID:
                succ = find_links(cut, fps, s)
                pred = {b: a for a, b in succ.items()}
                for head, tail, g in pairs:
                    if succ.get(head) == tail:
                        res = "correct"
                    elif head in succ or tail in pred:
                        res = "wrong"
                    else:
                        res = "missed"
                    rows.append(dict(seed=seed, clip=r.clip_id, gap_s=g, result=res, **asdict(s)))
        print("seed", seed, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "gaps.csv", index=False)
    summ = (
        df.groupby(["max_gap_s", "max_dist_h"])["result"]
        .value_counts(normalize=True)
        .unstack()
        .round(3)
    )
    summ["n"] = df.groupby(["max_gap_s", "max_dist_h"]).size() // 1
    print(summ)
    summ.to_csv(out_dir / "gaps_summary.csv")


# ---------------------------------------------------------------- victims


def victim_detected(table, record, fps):
    """The existing protocol number: victim tracks re-matched on this table's ids."""
    m = match_victim_tracks(table, record, fps)
    frag = near_path_tracks(table, record, fps, exclude=m.tracks)
    truth = truths_from_records([record], [record.clip_id], {record.clip_id: m.tracks | frag})
    if record.clip_id not in truth:
        return np.nan
    t = table.assign(t=table["frame"] / fps)
    return victim_detected_share(t, truth[record.clip_id], ProtocolConfig())


def path_scores(table, paths):
    sc = ClickScorer(table)
    vals = [sc.score(p) for p in paths]
    if not vals:
        return np.nan, np.nan
    return np.nanmean([v["coverage"] for v in vals]), np.nanmean([v["continuity"] for v in vals])


def run_victims(out_dir: Path):
    zones = json.loads((MAIN / "data/annotations/zones.json").read_text())
    rows = []
    for r, table, fps in clips():
        c = clicks_for(r, fps)
        if not len(c.frame):
            continue
        rng = np.random.default_rng(int(r.clip_id, 16) % 2**32)
        lo = 0 if r.trim_start is None else round(r.to_file_time(r.trim_start) * fps)
        hi = round(r.to_file_time(r.events[0]["contact"]) * fps)
        shifted = shifted_paths(c, lo, hi, N_PATHS, rng)
        zone = zones.get(r.clip_id)
        moved = translated_paths(c, np.array(zone["points"]), N_PATHS, rng) if zone else []

        def row(name, tbl, n_links, s=None, **extra):
            v = ClickScorer(tbl).score(c)
            mcov, mcon = path_scores(tbl, moved)
            scov, scon = path_scores(tbl, shifted)
            return dict(
                clip=r.clip_id, method=name, n_clicks=len(c.frame), n_links=n_links,
                max_gap_s=s.max_gap_s if s else np.nan, max_dist_h=s.max_dist_h if s else np.nan,
                coverage=v["coverage"], continuity=v["continuity"],
                moved_coverage=mcov, moved_continuity=mcon, n_moved=len(moved),
                shifted_coverage=scov, shifted_continuity=scon, **extra,
            )

        rows.append(row("today", table, 0, detected=victim_detected(table, r, fps)))
        for s in GRID:
            succ = find_links(table, fps, s)
            linked = apply_links(table, succ)
            rows.append(row("relink", linked, len(succ), s, detected=victim_detected(linked, r, fps)))
            rows.append(row("relink_fill", apply_links(table, succ, fill=True), len(succ), s))
            for k in range(N_RANDOM_RELINK):
                rs = random_links(table, fps, s, len(succ), rng)
                rows.append(row("random_relink", apply_links(table, rs), len(rs), s, draw=k))
        print(r.clip_id, len(c.frame), "clicks", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "victims.csv", index=False)


# ---------------------------------------------------------------- summary


def boot(x: np.ndarray, rng, n=2000):
    x = x[~np.isnan(x)]
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def summary(out_dir: Path, g: float, r: float):
    df = pd.read_csv(out_dir / "victims.csv")
    rng = np.random.default_rng(0)
    today = df[df.method == "today"].set_index("clip")
    sel = df[(df.max_gap_s == g) & (df.max_dist_h == r)]
    num = [c for c in df.columns if c not in ("clip", "method") and df[c].dtype != object]
    per = {m: sel[sel.method == m].groupby("clip")[num].mean() for m in sel.method.unique()}
    per["today"] = today[num]
    lines = [f"Config G={g} s, R={r} box heights; {len(today)} clips\n"]
    for m in ["today", "relink", "relink_fill", "random_relink"]:
        p = per[m]
        for col in ["continuity", "coverage", "moved_continuity", "shifted_continuity", "detected"]:
            if col in p and p[col].notna().any():
                mu, lo_, hi_ = boot(p[col].to_numpy(float), rng)
                lines.append(f"{m:14s} {col:20s} {mu:.3f} [{lo_:.3f}, {hi_:.3f}]")
    for m in ["relink", "relink_fill", "random_relink"]:
        for col in ["continuity", "coverage", "moved_continuity", "shifted_continuity"]:
            d = (per[m][col] - per["today"][col]).dropna().to_numpy(float)
            mu, lo_, hi_ = boot(d, rng)
            lines.append(
                f"diff {m:14s} {col:20s} {mu:+.3f} [{lo_:+.3f}, {hi_:+.3f}]  "
                f"better {(d > 1e-9).sum()} same {(np.abs(d) <= 1e-9).sum()} worse {(d < -1e-9).sum()}"
            )
    d = (per["relink"]["continuity"] - per["random_relink"]["continuity"]).to_numpy(float)
    mu, lo_, hi_ = boot(d, rng)
    lines.append(f"relink minus random_relink continuity {mu:+.3f} [{lo_:+.3f}, {hi_:+.3f}]")
    txt = "\n".join(lines)
    print(txt)
    (out_dir / f"summary_G{g}_R{r}.txt").write_text(txt + "\n")


if __name__ == "__main__":
    cmd, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    if cmd == "gaps":
        run_gaps(out)
    elif cmd == "victims":
        run_victims(out)
    else:
        summary(out, float(sys.argv[3]), float(sys.argv[4]))
