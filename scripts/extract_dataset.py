"""
Extract raw tiled pose detections for every clip in the manifest (the GPU step).

Walks the manifest, skips held clips and clips whose video is missing or
unreadable, and runs `extract_raw` on each. Labeled clips go first and
unlabeled/review clips last, so the clips that training and eval need are
finished early in a long run. Finished clips are skipped and interrupted ones
resume, so rerunning the same command after a Colab disconnect just continues.
A clip whose video cannot be decoded is reported at the end and left incomplete,
so it is retried on the next run.

Usage:
    uv run python scripts/extract_dataset.py --device 0                # full run (GPU)
    uv run python scripts/extract_dataset.py --clips b4b2db5ffb48 --max-frames 50   # smoke test
    uv run python scripts/extract_dataset.py --out /content/drive/MyDrive/hydroknight/detections
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import cv2

from hydro_knight.ingest.manifest import Label, Manifest
from hydro_knight.preprocess.extract_pose import VideoReadError, extract_raw

_LATER = (Label.UNLABELED, Label.REVIEW)


def _readable(path: Path) -> bool:
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
    cap.release()
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", default="data/manifests/pool_footage.jsonl")
    ap.add_argument("--videos", default="raw_local", help="dir of <clip_id>.mp4")
    ap.add_argument("--out", default="data/detections", help="one subfolder per clip")
    ap.add_argument("--clips", nargs="*", help="only these clip ids")
    ap.add_argument("--device", default=None, help="e.g. 0 for the first GPU")
    ap.add_argument("--model", default="yolo11n-pose.pt")
    ap.add_argument(
        "--conf", type=float, default=0.25, help="detection confidence floor"
    )
    ap.add_argument("--max-frames", type=int, default=None, help="debugging only")
    args = ap.parse_args()
    warnings.simplefilter("ignore")

    videos, out = Path(args.videos), Path(args.out)
    records = Manifest(Path(args.manifest)).load()
    if args.clips:
        records = [r for r in records if r.clip_id in set(args.clips)]
    records = [r for r in records if "[HOLD" not in r.notes]
    records.sort(
        key=lambda r: r.label in _LATER
    )  # stable: labeled first, manifest order kept

    todo = []
    for r in records:
        video = videos / f"{r.clip_id}.mp4"
        if not video.exists() or not _readable(video):
            print(f"skip {r.clip_id}: video missing or unreadable")
            continue
        todo.append((r, video))

    device = int(args.device) if args.device and args.device.isdigit() else args.device
    failed = []
    for i, (r, video) in enumerate(todo, 1):
        try:
            meta = extract_raw(
                video,
                out / r.clip_id,
                model_name=args.model,
                conf=args.conf,
                device=device,
                max_frames=args.max_frames,
            )
        except (
            VideoReadError
        ) as e:  # keep going; the folder stays incomplete for a retry
            print(f"[{i}/{len(todo)}] FAILED {e}")
            failed.append(r.clip_id)
            continue
        print(
            f"[{i}/{len(todo)}] {r.clip_id} ({r.label}): "
            f"{meta['frames_processed']} frames, complete={meta['complete']}"
        )

    if failed:
        raise SystemExit(
            f"{len(failed)} clip(s) could not be read: {' '.join(failed)}. "
            "AV1 videos: run scripts/transcode_av1.py, then rerun this command."
        )


if __name__ == "__main__":
    main()
