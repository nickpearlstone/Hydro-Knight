"""
Download video files for clips registered in the manifest.

Videos are saved to raw_local/ which is gitignored — they never enter the repo.
A sidecar file (<clip_id>.done) is written next to each video on success so
re-running this script skips already-downloaded clips.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .manifest import ClipRecord, Manifest

# Ensure Homebrew binaries (ffmpeg, node) are visible to subprocesses even
# when the shell PATH isn't inherited by the venv.
_ENV = {
    **os.environ,
    "PATH": f"/opt/homebrew/bin:/usr/local/bin:{os.environ.get('PATH', '')}",
}

# Invoke yt-dlp through THIS interpreter (the venv's python) rather than the
# bare "yt-dlp" name. Several yt-dlp copies of different ages exist on this
# machine; the bare name resolved to an old 2023 build without --js-runtimes.
# "python -m yt_dlp" guarantees the venv's up-to-date version is used.
_YTDLP = [sys.executable, "-m", "yt_dlp"]


# Where downloaded videos land. Gitignored.
RAW_LOCAL = Path("raw_local")


def local_path(record: ClipRecord) -> Path:
    """
    Return the expected local file path for a clip.

    We name the file by clip_id so the filename is stable and unique
    regardless of what the original video was called on YouTube.
    The extension is .mp4 — yt-dlp will convert to this format on download.
    """
    return RAW_LOCAL / f"{record.clip_id}.mp4"


def done_marker(record: ClipRecord) -> Path:
    """
    Return the path to the sidecar file that marks a clip as downloaded.

    We use a separate .done file rather than checking whether the .mp4 exists
    because a partial download leaves a real (but broken) .mp4 behind.
    The .done file is only written after yt-dlp exits successfully.
    """
    return RAW_LOCAL / f"{record.clip_id}.done"


def is_downloaded(record: ClipRecord) -> bool:
    return done_marker(record).exists()


def download_clip(
    record: ClipRecord,
    cookies_file: Path | None = None,
    cookies_from_browser: str | None = None,
) -> bool:
    """
    Download one clip to raw_local/.

    If start_sec and end_sec are set, yt-dlp downloads only that segment
    rather than the full video — saves disk space for long source videos.

    Returns True on success, False on failure.
    """
    RAW_LOCAL.mkdir(parents=True, exist_ok=True)
    out_path = local_path(record)

    cmd = [
        *_YTDLP,
        record.source_url,
        "--output",
        str(out_path),
        "--format",
        "bestvideo[ext=mp4][height<=1080]/bestvideo[height<=1080]/136/135/134",  # mp4 video-only up to 1080p
        "--js-runtimes",
        "node",  # use installed Node to run YouTube's JS
        # YouTube's upgraded "n challenge" now needs yt-dlp's remote EJS solver
        # script (fetched from its official GitHub) run via Node. Without this,
        # extraction yields "only images" -> "requested format is not available".
        "--remote-components",
        "ejs:github",
        # Download DASH fragments in parallel to beat YouTube's per-connection
        # throttling (single connection was ~0.5 MB/s; this multiplies it).
        "--concurrent-fragments",
        "5",
        "--quiet",
        "--no-playlist",
    ]
    if cookies_file:
        cmd += ["--cookies", str(cookies_file)]
    elif cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]

    # Section download for clips with a real start offset (e.g. a 9.5h slice of
    # a 12h livestream). Downloading the full source would be wasteful/huge.
    # Convention: the local file always corresponds to the manifest's
    # [start_sec, end_sec] window. For a sectioned download the file is
    # re-based to 0, so the pose-extraction step maps file-time t to
    # source-time (start_sec + t); events (source timeline) map in the same way.
    # Clips with start_sec == 0 download whole (the common short-clip case).
    if (
        record.start_sec
        and record.start_sec > 0
        and record.end_sec
        and record.end_sec > 0
    ):
        # Cut at nearest keyframes (no --force-keyframes-at-cuts): a stream copy
        # that's fast and avoids a multi-hour re-encode on long sections. Start/
        # end may be off by a few seconds, which is fine for our purposes.
        cmd += ["--download-sections", f"*{record.start_sec}-{record.end_sec}"]

    result = subprocess.run(cmd, capture_output=True, text=True, env=_ENV)

    if result.returncode != 0:
        print(
            f"  Failed ({record.clip_id}): {result.stderr.strip().splitlines()[-1] if result.stderr.strip() else 'unknown error'}"
        )
        return False

    # Write the sidecar marker so future runs skip this clip.
    done_marker(record).touch()
    return True


def download(
    manifest_path: Path,
    limit: int | None = None,
    cookies_file: Path | None = None,
    cookies_from_browser: str | None = None,
) -> None:
    """
    Download all undownloaded clips in the manifest.

    Parameters
    ----------
    manifest_path :
        Path to the JSONL manifest file to read from.
    limit :
        If set, download at most this many clips. Useful for test runs.
    """
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
    """
    Delete any files in raw_local/ whose clip_id has no matching record
    in the manifest.

    This fixes the case where the manifest was regenerated (new clip IDs)
    while old downloaded files were still sitting on disk. Without cleanup,
    those orphaned files waste disk space and can never be opened by the
    annotator since nothing in the manifest points to them.
    """
    if not RAW_LOCAL.exists():
        print("raw_local/ does not exist — nothing to clean.")
        return

    manifest = Manifest(manifest_path)
    known_ids = {r.clip_id for r in manifest.load()}

    removed = 0
    for f in RAW_LOCAL.iterdir():
        # Each file is named <clip_id>.mp4 or <clip_id>.done
        # Strip the suffix to get the clip_id
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
