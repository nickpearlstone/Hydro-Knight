"""Pick head validation frames: one frame per rescue clip at drowning onset, from clips the
benchmark frames don't use.

    uv run hydroknight heads frames              # 40 frames -> data/annotations/head_val_gt.json
    uv run hydroknight heads frames --n 20 --seed 1 --force

These frames are for choosing a head detector's checkpoint, seed and threshold without
touching the benchmark (test) frames, so no clip is in both sets. Label them in the labeler's
Swimmer count tab (Frames: Head validation, Head boxes mode). Images are cached under
raw_local/pose_bench/ like the benchmark's; the label file holds no pixels.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2

from hydro_knight.annotate import count
from hydro_knight.ingest import download
from hydro_knight.ingest.manifest import Label, Manifest

MANIFEST = Path("data/manifests/pool_footage.jsonl")


def pick_val_frames(
    records, bench_ids: set[str], n: int, seed: int, fps_of
) -> list[dict]:
    """One onset frame per eligible clip, n clips sampled with a fixed seed.

    Eligible: a distress clip, not on [HOLD, with a local video, an onset time, and not one of
    the benchmark clips.

    Args:
        records: manifest ClipRecords.
        bench_ids: clip ids used by the benchmark (test) frames.
        n: frames wanted (fewer if fewer clips are eligible).
        seed: sampling seed.
        fps_of: clip_id -> frame rate of the local video file.

    Returns:
        Frame records in the count-file format, ordered by clip id.
    """
    eligible = [
        r
        for r in records
        if r.label == Label.DISTRESS
        and "[HOLD" not in (r.notes or "")
        and r.clip_id not in bench_ids
        and r.events
        and r.events[0].get("start") is not None
        and (download.RAW_LOCAL / f"{r.clip_id}.mp4").exists()
    ]
    eligible.sort(key=lambda r: r.clip_id)
    chosen = random.Random(seed).sample(eligible, min(n, len(eligible)))
    frames = []
    for r in sorted(chosen, key=lambda r: r.clip_id):
        t = r.to_file_time(r.events[0]["start"])  # source time -> seconds into the file
        idx = max(0, int(round(t * fps_of(r.clip_id))))
        frames.append(
            {
                "clip_id": r.clip_id,
                "frame": idx,
                "label": r.label.value,
                "water": [],
                "deck": [],
            }
        )
    return frames


def file_fps(clip_id: str) -> float:
    """Frame rate of raw_local/<clip_id>.mp4, as the decoder that reads the frame sees it."""
    cap = cv2.VideoCapture(str(download.RAW_LOCAL / f"{clip_id}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("frames", help="pick the head validation frames")
    f.add_argument("--n", type=int, default=40)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--force", action="store_true", help="overwrite existing labels")
    args = ap.parse_args()
    if count.VAL_PATH.exists() and not args.force:
        raise SystemExit(
            f"{count.VAL_PATH} exists (may hold your boxes); pass --force to re-pick"
        )
    bench = {f["clip_id"] for f in count.load(count.COUNT_PATH)["frames"]}
    frames = pick_val_frames(
        Manifest(MANIFEST).load(), bench, args.n, args.seed, file_fps
    )
    for item in frames:
        count.read_frame(item)  # decode once and cache, so the labeler opens instantly
    count.save({"frames": frames}, count.VAL_PATH)
    print(
        f"picked {len(frames)} frames from {len(frames)} clips (none of the {len(bench)} "
        f"benchmark clips) -> {count.VAL_PATH}, images in {count.frame_dir()}/"
    )


if __name__ == "__main__":
    main()
