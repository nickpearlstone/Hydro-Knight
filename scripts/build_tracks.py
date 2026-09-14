"""
Build keypoint Parquets from raw detections: merge duplicates, then track (CPU, no video needed).

Reads every complete clip folder under --detections and writes
<out>/<clip_id>.parquet, the table `features/` and `eval/` consume. Existing
keypoint files are left alone unless --overwrite is passed, so older extractions
are never replaced by accident.

Usage:
    uv run python scripts/build_tracks.py                         # fixed defaults
    uv run python scripts/build_tracks.py --legacy --out data/keypoints_legacy   # original behavior
    uv run python scripts/build_tracks.py --lost-track-seconds 2 --overwrite
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from tqdm import tqdm

from hydro_knight.preprocess.build_tracks import LEGACY, TrackSettings, build_tracks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--detections", default="data/detections")
    ap.add_argument("--out", default="data/keypoints")
    ap.add_argument("--clips", nargs="*", help="only these clip ids")
    ap.add_argument(
        "--legacy", action="store_true", help="reproduce the original merge + tracker"
    )
    ap.add_argument("--lost-track-seconds", type=float, default=None)
    ap.add_argument(
        "--overwrite", action="store_true", help="replace existing keypoint files"
    )
    args = ap.parse_args()

    settings = LEGACY if args.legacy else TrackSettings()
    if args.lost_track_seconds is not None:
        settings = replace(settings, lost_track_seconds=args.lost_track_seconds)

    folders = sorted(
        p for p in Path(args.detections).iterdir() if (p / "meta.json").exists()
    )
    if args.clips:
        folders = [p for p in folders if p.name in set(args.clips)]

    out = Path(args.out)
    for folder in tqdm(folders, desc="build tracks"):
        dest = out / f"{folder.name}.parquet"
        if dest.exists() and not args.overwrite:
            print(f"skip {folder.name}: {dest} exists (use --overwrite)")
            continue
        try:
            df = build_tracks(folder, dest, settings)
        except RuntimeError as e:  # extraction not finished yet
            print(f"skip {folder.name}: {e}")
            continue
        n_tracks = df.loc[df["track_id"] >= 0, "track_id"].nunique()
        print(f"{folder.name}: {len(df)} rows, {n_tracks} tracks -> {dest}")


if __name__ == "__main__":
    main()
