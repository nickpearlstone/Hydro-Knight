"""
Convert AV1 source videos to H.264 so every extraction environment can decode them.

The 10 clips that produced empty keypoint files are exactly the 10 AV1-encoded
videos: Colab's OpenCV has no AV1 decoder, so it read 0 frames and nothing was
extracted. H.264 decodes everywhere.

Frame-exact on purpose. Event windows are in seconds and keypoints index frames,
so a conversion that drops or duplicates a single frame would silently shift
labels. Every source frame is passed through unchanged in count and timing
(-fps_mode passthrough), then both files' frames are counted by full decode, and
the original is only replaced if the counts match. Quality: CRF 18, visually
lossless. Audio is dropped (unused). Originals are kept in <videos>/av1_originals/.

Needs ffmpeg/ffprobe with an AV1 decoder (Homebrew's build has libdav1d).

Usage:
    uv run python scripts/transcode_av1.py --dry-run   # list AV1 clips
    uv run python scripts/transcode_av1.py             # convert, verify, replace
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def _probe(path: Path, entries: str, count: bool = False) -> str:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    if count:
        cmd.append("-count_frames")
    cmd += ["-show_entries", f"stream={entries}", "-of", "csv=p=0", str(path)]
    return subprocess.run(
        cmd, capture_output=True, text=True, check=True
    ).stdout.strip()


def codec(path: Path) -> str:
    """Video stream codec name as ffprobe reports it (e.g. "av1", "h264")."""
    return _probe(path, "codec_name")


def decoded_frames(path: Path) -> int:
    """Frame count by decoding the whole stream (exact, unlike container metadata)."""
    return int(_probe(path, "nb_read_frames", count=True))


def transcode(src: Path, dst: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-i", str(src),
            "-map", "0:v:0", "-an",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-fps_mode", "passthrough",
            "-movflags", "+faststart",
            str(dst),
        ],
        check=True,
    )  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--videos", default="raw_local")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    videos = Path(args.videos)
    backup = videos / "av1_originals"
    av1 = [p for p in sorted(videos.glob("*.mp4")) if codec(p) == "av1"]
    print(f"{len(av1)} AV1 clip(s): {' '.join(p.stem for p in av1)}")
    if args.dry_run or not av1:
        return

    backup.mkdir(exist_ok=True)
    failed = []
    for src in av1:
        tmp = src.with_suffix(".h264.tmp.mp4")
        transcode(src, tmp)
        n_src, n_new = decoded_frames(src), decoded_frames(tmp)
        if n_src != n_new or codec(tmp) != "h264":
            print(
                f"FAIL {src.stem}: {n_src} source frames vs {n_new} converted; original kept"
            )
            tmp.unlink()
            failed.append(src.stem)
            continue
        shutil.move(src, backup / src.name)
        tmp.replace(src)
        print(f"ok   {src.stem}: {n_new} frames, now h264")

    if failed:
        raise SystemExit(f"{len(failed)} clip(s) not converted: {' '.join(failed)}")


if __name__ == "__main__":
    main()
