"""
Backfill the manifest's `fps` field from source metadata.

Events are stored in SECONDS; keypoint Parquets index by FRAME. Every conversion
between them needs the clip's frame rate, and until now nothing recorded it — so
eval fell back to a flat fps=30. That is wrong for most of this dataset (many
clips are 60fps), which shifts every event label by 2x and quietly destroys the
ground truth. Recording fps once, in the manifest, removes the guess.

YouTube clips are read with a single metadata-only yt-dlp call (no video is
downloaded, and one process avoids the rate limiting that repeated queries hit).
Local clips fall back to reading the file with OpenCV, when it is present.

Usage:
    uv run python scripts/backfill_fps.py --dry-run   # show what would change
    uv run python scripts/backfill_fps.py             # write it
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import re
import subprocess
import sys
from pathlib import Path

from hydro_knight.ingest.manifest import Manifest

_YTDLP = [sys.executable, "-m", "yt_dlp"]
_ENV = {
    **os.environ,
    "PATH": f"/opt/homebrew/bin:/usr/local/bin:{os.environ.get('PATH', '')}",
}

# Real-world frame rates. Reported values are matched to the nearest of these so
# a 59.94 NTSC reading and a clean 60 don't end up as two different numbers.
STANDARD_FPS = [23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0, 120.0]


def _youtube_id(url: str) -> str | None:
    """Extract the 11-character video id from a YouTube URL, or None."""
    m = re.search(r"(?:v=|youtu\.be/|/shorts/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None


def _snap(fps: float, tol: float = 0.02) -> float:
    """Snap a reported rate to the nearest standard frame rate within `tol`."""
    for std in STANDARD_FPS:
        if abs(fps - std) / std <= tol:
            return std
    return round(fps, 3)


def fetch_youtube_fps(urls: list[str]) -> dict[str, tuple[float, float]]:
    """Fetch (fps, duration) per YouTube video id in ONE metadata-only call.

    Args:
        urls: YouTube watch URLs.
    Returns:
        {video_id: (fps, duration_sec)} for every video that answered.
    """
    if not urls:
        return {}
    cmd = [
        *_YTDLP,
        *urls,
        "--print",
        "%(id)s|%(fps)s|%(duration)s",
        "--skip-download",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "--ignore-errors",  # one dead video must not abort the whole batch
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=_ENV)

    out: dict[str, tuple[float, float]] = {}
    for line in result.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 3:
            continue
        vid, fps_s, dur_s = parts
        try:
            out[vid] = (_snap(float(fps_s)), float(dur_s))
        except ValueError:
            continue  # "NA" for fields yt-dlp could not determine
    return out


def probe_local_fps(path: Path) -> float:
    """Read a local video's frame rate with OpenCV; 0.0 if unreadable."""
    import cv2

    if not path.exists():
        return 0.0
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return _snap(float(fps)) if fps and fps > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/manifests/pool_footage.jsonl")
    ap.add_argument("--videos-dir", default="raw_local")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change without writing the manifest",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="refetch even for records that already carry an fps",
    )
    args = ap.parse_args()

    manifest = Manifest(Path(args.manifest))
    records = manifest.load()
    todo = [r for r in records if args.force or not r.fps]
    print(f"{len(records)} records, {len(todo)} needing fps\n")
    if not todo:
        return

    yt = [r for r in todo if _youtube_id(r.source_url)]
    print(f"fetching metadata for {len(yt)} YouTube clips (one batched call)...")
    meta = fetch_youtube_fps([r.source_url for r in yt])
    print(f"  got {len(meta)} responses\n")

    videos_dir = Path(args.videos_dir)
    resolved, failed = [], []
    for r in todo:
        vid = _youtube_id(r.source_url)
        if vid and vid in meta:
            fps, dur = meta[vid]
            resolved.append((r, fps, dur, "youtube"))
            continue
        fps = probe_local_fps(videos_dir / f"{r.clip_id}.mp4")
        if fps:
            resolved.append((r, fps, 0.0, "local"))
        else:
            failed.append(r)

    span = "span_s"
    print(f"{'clip':14}{'fps':>8}{'source':>10}{span:>9}{'src_dur':>9}  trim?")
    for r, fps, dur, how in sorted(resolved, key=lambda x: -x[1]):
        clip_span = (r.end_sec - r.start_sec) if r.end_sec > 0 else 0.0
        # A source noticeably longer than the trimmed span means extraction that
        # ran on the whole video produced frames outside this record's window.
        trim = "yes" if dur and clip_span and dur > clip_span * 1.1 else ""
        print(
            f"{r.clip_id[:12]:14}{fps:>8}{how:>10}{clip_span:>9.1f}{dur:>9.1f}  {trim}"
        )

    print(f"\nresolved {len(resolved)}, unresolved {len(failed)}")
    for r in failed:
        print(f"  ! {r.clip_id[:12]}  {r.source_url[:56]}")

    if args.dry_run:
        print("\n--dry-run: manifest not modified")
        return

    for r, fps, _dur, _how in resolved:
        # modify() changes only fps on the latest saved record, so labels written by an
        # open labeler while this ran are kept.
        manifest.modify(
            r.clip_id, lambda rec, fps=fps: dataclasses.replace(rec, fps=fps)
        )
    print(f"\nwrote fps for {len(resolved)} records -> {args.manifest}")


if __name__ == "__main__":
    main()
