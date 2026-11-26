"""Download video files for clips registered in the manifest."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .manifest import ClipRecord, Manifest

_ENV = {
    **os.environ,
    "PATH": f"/opt/homebrew/bin:/usr/local/bin:{os.environ.get('PATH', '')}",
}

# Run yt-dlp via sys.executable, never the bare name.
_YTDLP = [sys.executable, "-m", "yt_dlp"]


RAW_LOCAL = Path("raw_local")


def local_path(record: ClipRecord) -> Path:
    """Expected local video path for a clip (raw_local/<clip_id>.mp4)."""
    return RAW_LOCAL / f"{record.clip_id}.mp4"


def done_marker(record: ClipRecord) -> Path:
    """Path to the sidecar marking a clip as fully downloaded (raw_local/<clip_id>.done)."""
    return RAW_LOCAL / f"{record.clip_id}.done"


def is_downloaded(record: ClipRecord) -> bool:
    """True if the clip's .done marker exists (i.e. it downloaded successfully)."""
    return done_marker(record).exists()


def download_clip(
    record: ClipRecord,
    cookies_file: Path | None = None,
    cookies_from_browser: str | None = None,
) -> bool:
    """Download one clip's video to raw_local/, optionally just its [start,end] section."""
    RAW_LOCAL.mkdir(parents=True, exist_ok=True)
    out_path = local_path(record)

    cmd = [
        *_YTDLP,
        record.source_url,
        "--output",
        str(out_path),
        "--format",
        # keep H.264 (avc1) first
        "bestvideo[ext=mp4][vcodec^=avc1][height<=1080]/bestvideo[ext=mp4][height<=1080]"
        "/bestvideo[height<=1080]/136/135/134/best[ext=mp4][height<=1080]",
        "--js-runtimes",
        "node",
        # Requires yt-dlp's remote EJS solver via Node.
        "--remote-components",
        "ejs:github",
        "--concurrent-fragments",
        "5",
        "--quiet",
        "--no-playlist",
    ]
    if cookies_file:
        cmd += ["--cookies", str(cookies_file)]
    elif cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]

    # File time t = source time start_sec + t; convert with ClipRecord.to_file_time/to_source_time.
    if (
        record.start_sec
        and record.start_sec > 0
        and record.end_sec
        and record.end_sec > 0
    ):
        # keep --force-keyframes-at-cuts; file frame 0 must be exactly start_sec
        cmd += [
            "--download-sections",
            f"*{record.start_sec}-{record.end_sec}",
            "--force-keyframes-at-cuts",
        ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=_ENV)

    if result.returncode != 0:
        print(
            f"  Failed ({record.clip_id}): {result.stderr.strip().splitlines()[-1] if result.stderr.strip() else 'unknown error'}"
        )
        return False

    done_marker(record).touch()
    return True


def download(
    manifest_path: Path,
    limit: int | None = None,
    cookies_file: Path | None = None,
    cookies_from_browser: str | None = None,
) -> None:
    """Download every not-yet-downloaded clip in the manifest."""
    manifest = Manifest(manifest_path)
    records = manifest.load()

    pending = [r for r in records if not is_downloaded(r)]

    if not pending:
        print("All clips already downloaded.")
        return

    if limit is not None:
        pending = pending[:limit]

    print(f"{len(pending)} clips to download.")

    succeeded = failed = 0
    for i, record in enumerate(pending, start=1):
        title = record.notes[:60] if record.notes else record.clip_id
        print(f"[{i}/{len(pending)}] {title}")
        if download_clip(
            record, cookies_file=cookies_file, cookies_from_browser=cookies_from_browser
        ):
            print(f"  OK → {local_path(record)}")
            succeeded += 1
        else:
            failed += 1

    print(f"\nDone. {succeeded} downloaded, {failed} failed.")


def cleanup_orphans(manifest_path: Path) -> None:
    """Delete raw_local/ files whose clip_id no longer appears in the manifest."""
    if not RAW_LOCAL.exists():
        print("raw_local/ does not exist — nothing to clean.")
        return

    manifest = Manifest(manifest_path)
    known_ids = {r.clip_id for r in manifest.load()}

    removed = 0
    for f in RAW_LOCAL.iterdir():
        clip_id = f.stem
        if clip_id not in known_ids:
            f.unlink()
            removed += 1

    print(f"Cleanup done. {removed} orphaned files removed from raw_local/.")


if __name__ == "__main__":
    cleanup_orphans(Path("data/manifests/pool_footage.jsonl"))
    download(
        manifest_path=Path("data/manifests/pool_footage.jsonl"),
        limit=5,
        cookies_file=Path.home() / "Downloads/www.youtube.com_cookies.txt",
    )
