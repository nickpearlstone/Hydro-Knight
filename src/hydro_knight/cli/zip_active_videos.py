"""
Zip the videos of every active clip (no "[HOLD" in its notes) for upload to Google Drive.

    uv run hydroknight zip-videos      # -> scratch/colab_upload/videos_active.zip

Videos are stored uncompressed in the zip (they are already compressed), and the zip
lands in scratch/, which is gitignored: footage never enters the repo.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from hydro_knight.ingest.manifest import Manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", default="data/manifests/pool_footage.jsonl")
    ap.add_argument("--videos", default="raw_local")
    ap.add_argument("--out", default="scratch/colab_upload/videos_active.zip")
    args = ap.parse_args()

    records = [
        r for r in Manifest(Path(args.manifest)).load() if "[HOLD" not in r.notes
    ]
    paths = [Path(args.videos) / f"{r.clip_id}.mp4" for r in records]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} active clip video(s) missing: {' '.join(missing)}"
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:
        for p in paths:
            zf.write(p, arcname=p.name)
    size = out.stat().st_size / 1e9
    print(f"{len(paths)} videos, {size:.1f} GB -> {out}")


if __name__ == "__main__":
    main()
