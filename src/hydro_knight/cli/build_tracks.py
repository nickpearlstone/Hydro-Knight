"""
Build keypoint Parquets from raw detections: merge duplicates, then track (CPU, no video needed).

Reads every complete clip folder under --detections and writes
<out>/<clip_id>.parquet, the table `features/` and `eval/` consume. Existing
keypoint files are left alone unless --overwrite is passed, so older extractions
are never replaced by accident.

Usage:
    uv run hydroknight tracks                         # fixed defaults
    uv run hydroknight tracks --legacy --out data/keypoints_legacy   # original behavior
    uv run hydroknight tracks --lost-track-seconds 2 --overwrite
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
        "--activation", type=float, default=None, help="confidence to start a track"
    )
    ap.add_argument(
        "--high-conf", type=float, default=None, help="first-round matching confidence"
    )
    ap.add_argument(
        "--same-crop-conf",
        type=float,
        default=None,
        help="merge same-tile overlaps below this confidence (for low-floor extractions)",
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="replace existing keypoint files"
    )
    args = ap.parse_args()

    settings = LEGACY if args.legacy else TrackSettings()
    if args.lost_track_seconds is not None:
        settings = replace(settings, lost_track_seconds=args.lost_track_seconds)
    if args.activation is not None:
        settings = replace(settings, activation_threshold=args.activation)
    if args.high_conf is not None:
        settings = replace(settings, high_conf_threshold=args.high_conf)
    if args.same_crop_conf is not None:
        settings = replace(settings, same_crop_conf=args.same_crop_conf)

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
