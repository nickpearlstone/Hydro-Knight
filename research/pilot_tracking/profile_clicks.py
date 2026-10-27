"""Stage-1 profile: at each victim click before guard contact, what did the detector and tracker hold?"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hydro_knight.eval.victims import is_labeled
from hydro_knight.ingest.manifest import Manifest
from hydro_knight.preprocess.extract_pose import load_detections

MAIN = Path("/Users/nickpearlstone/dev/repos/Hydro-Knight")
REVIEWS = json.loads((MAIN / "data/annotations/label_reviews.json").read_text())


def active_clips():
    for r in Manifest(MAIN / "data/manifests/pool_footage.jsonl").load():
        if "[HOLD" in (r.notes or "") or not is_labeled(r):
            continue
        if REVIEWS.get(r.clip_id, {}).get("verdict") == "not_good":
            continue
        if not (MAIN / "data/keypoints_tiled" / f"{r.clip_id}.parquet").exists():
            continue
        yield r


def near(df, frame, x, y, tol=2, radius=0.75):
    s = df[(df.frame - frame).abs() <= tol]
    if s.empty:
        return s
    h = (s.by2 - s.by1).clip(lower=1)
    inside = (s.bx1 <= x) & (x <= s.bx2) & (s.by1 <= y) & (y <= s.by2)
    d = np.hypot((s.bx1 + s.bx2) / 2 - x, (s.by1 + s.by2) / 2 - y) / h
    return s[inside | (d <= radius)]


rows = []
for r in active_clips():
    meta, dets, _ = load_detections(MAIN / "data/detections" / r.clip_id)
    fps = meta["video"]["fps"] / meta["settings"].get("frame_step", 1)
    kp = pd.read_parquet(MAIN / "data/keypoints_tiled" / f"{r.clip_id}.parquet")
    ev = r.events[0]
    contact = r.to_file_time(ev["contact"])
    for c in ev.get("victim", []):
        t = r.to_file_time(c["t"])
        if t > contact:
            continue
        f = round(t * fps)
        raw = near(dets, f, c["x"], c["y"])
        trk = near(kp[kp.track_id >= 0], f, c["x"], c["y"])
        rows.append(
            dict(
                clip=r.clip_id,
                kind=c.get("kind"),
                raw=len(raw) > 0,
                raw_conf=raw.conf.max() if len(raw) else np.nan,
                raw_h=float((raw.by2 - raw.by1).median()) if len(raw) else np.nan,
                tracked=len(trk) > 0,
            )
        )
df = pd.DataFrame(rows)
df.to_csv(sys.argv[1], index=False)
print(len(df), "clicks in", df["clip"].nunique(), "clips")
print("raw det near click (conf>=.25):", df.raw.mean().round(3))
print("raw det conf>=0.5 near click:", (df.raw_conf >= 0.5).mean().round(3))
print("confirmed track near click:", df.tracked.mean().round(3))
print("raw but not tracked:", (df.raw & ~df.tracked).mean().round(3))
print(
    "raw conf quantiles:",
    df.raw_conf.quantile([0.1, 0.25, 0.5, 0.75]).round(2).tolist(),
)
print(
    "victim box h quantiles:",
    df.raw_h.quantile([0.1, 0.25, 0.5, 0.75]).round(0).tolist(),
)
pc = df.groupby("clip")[["raw", "tracked"]].mean()
print(
    "per clip raw/tracked median:",
    pc.median().round(2).tolist(),
    "clips with 0 tracked:",
    (pc.tracked == 0).sum(),
)
