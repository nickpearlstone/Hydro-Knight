"""
Write one Claude label to the manifest from a small JSON spec, then log it.

    uv run hydroknight write-label SPEC.json

Rescue spec (times are FILE seconds; converted to source seconds and frame-snapped):
    {"clip": ID, "onset": t, "contact": t, "saved": t | null,
     "trim_start": t | null | "keep", "trim_end": t | null | "keep",
     "clicks": [[t, x, y, "onset" | "extra" | "last_seen"], ...], "notes": "..."}
Normal-footage spec:
    {"clip": ID, "kind": "normal", "trim_start": ..., "trim_end": ..., "notes": "..."}

Refuses to overwrite a clip that is already labeled or already in the Claude log. Every
label is recorded in data/annotations/labeled_by_claude.json, so it stays separate from
Nick's and is held out of training until a review marks it good. fps comes from the
keypoint file if the clip is extracted, otherwise from the video (the same value).
If keypoints exist, the victim clicks are matched to tracks as a check.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
from pathlib import Path

import cv2
import pandas as pd

from hydro_knight.annotate.review import CLAUDE_PATH
from hydro_knight.eval.timing import clip_fps
from hydro_knight.eval.victims import is_labeled, match_victim_tracks
from hydro_knight.ingest.download import local_path
from hydro_knight.ingest.manifest import Label, Manifest

MANIFEST = Path("data/manifests/pool_footage.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("spec", type=Path, help="label spec JSON (format above)")
    spec = json.loads(ap.parse_args().spec.read_text())
    clip = spec["clip"]
    kind = spec.get("kind", "rescue")
    m = Manifest(MANIFEST)
    rec = {r.clip_id: r for r in m.load()}[clip]
    log = json.loads(CLAUDE_PATH.read_text()) if CLAUDE_PATH.exists() else {}
    if clip in log or is_labeled(rec):
        raise SystemExit(f"{clip} is already labeled; refusing to overwrite")

    pq = Path("data/keypoints_tiled") / f"{clip}.parquet"
    if pq.exists():
        fps, _ = clip_fps(pq, None, default=rec.fps or 30.0, record=rec)
    else:
        cap = cv2.VideoCapture(str(local_path(rec)))
        fps = cap.get(cv2.CAP_PROP_FPS) or rec.fps or 30.0
        cap.release()
    off = rec.file_offset

    def src(t):
        return None if t is None else round((int(t * fps) + 0.5) / fps + off, 6)

    def trim(key):
        v = spec.get(key)
        return getattr(rec, key) if v == "keep" else src(v)

    ts, te = trim("trim_start"), trim("trim_end")
    if kind == "normal":
        m.modify(
            clip,
            lambda r: dataclasses.replace(
                r, label=Label.NORMAL, events=[], trim_start=ts, trim_end=te
            ),
        )
    else:
        ev = {
            "start": src(spec["onset"]),
            "contact": src(spec["contact"]),
            "end": src(spec["saved"]),
            "label": "distress",
            "victim": [
                {"t": src(t), "x": x, "y": y, "kind": k}
                for t, x, y, k in spec["clicks"]
            ],
        }
        if spec["saved"] is None:
            ev["saved_off_camera"] = True
        m.modify(
            clip,
            lambda r: dataclasses.replace(
                r, label=Label.DISTRESS, events=[ev], trim_start=ts, trim_end=te
            ),
        )

    rec = {r.clip_id: r for r in m.load()}[clip]
    print(f"{clip}: {kind}, fps {fps:.3f}, trim {rec.trim_start}..{rec.trim_end}")
    if kind != "normal":
        print(f"  labeled: {is_labeled(rec)}; event {rec.events[0]}")
        if pq.exists():
            print(f"  {match_victim_tracks(pd.read_parquet(pq), rec, fps)}")
        else:
            print("  not extracted yet: victim matching runs after extraction")

    log[clip] = {
        "date": datetime.date.today().isoformat(),
        "kind": kind,
        "notes": spec.get("notes", ""),
    }
    CLAUDE_PATH.write_text(json.dumps(log, indent=2) + "\n")


if __name__ == "__main__":
    main()
