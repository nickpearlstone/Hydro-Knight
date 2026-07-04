"""
Search YouTube for pool footage and register clips in the manifest.

This script never downloads video. It uses yt-dlp in metadata-only mode to
find videos — either by keyword search or by pulling whole channels — then
writes a ClipRecord for each result into the manifest JSONL file.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .blocklist import Blocklist
from .manifest import (
    CameraView,
    ClipRecord,
    Label,
    Manifest,
    Setting,
    TimeOfDay,
    Weather,
    make_clip_id,
)

# Make Homebrew binaries (node, ffmpeg) visible to yt-dlp subprocesses.
_ENV = {
    **os.environ,
    "PATH": f"/opt/homebrew/bin:/usr/local/bin:{os.environ.get('PATH', '')}",
}

# Invoke yt-dlp via this interpreter's module so the venv's current version is
# always used. The bare "yt-dlp" name can resolve to a stale system build that
# lacks --js-runtimes and gets throttled by YouTube. (Same fix as download.py.)
_YTDLP = [sys.executable, "-m", "yt_dlp"]

# Seconds to wait between search queries to avoid YouTube rate limiting.
# Many queries in rapid succession reliably triggers 400 errors.
QUERY_SLEEP_SEC = 3


# --- Search configuration ----------------------------------------------------

# Duration filter for keyword searches.
# Floor lowered to 20s so short surveillance/rescue clips (e.g. the Lifeguard
# Rescue "Spot the Drowning" series, ~30s–3min) are not filtered out.
# Ceiling rejects construction timelapses, livestream archives, music videos.
MIN_DURATION_SEC = 20  # 20 seconds
MAX_DURATION_SEC = 45 * 60  # 45 minutes

# Keyword queries grouped by intent. Written to avoid YouTube's popularity bias
# — specific descriptive phrases surface niche uploads rather than viral hits.
DEFAULT_QUERIES = [
    # Overhead / wide angle — most useful camera angle for pose detection
    "swimming pool overhead view people swimming",
    "pool deck camera angle swimmers laps",
    "aquatic center wide angle swim practice",
    # Outdoor recreational — the primary training distribution
    "outdoor public pool swimmers summer",
    "community pool open swim session",
    "backyard pool party swimming people",
    "hotel pool guests swimming vacation",
    # Indoor recreational
    "indoor lap pool swimmers training session",
    "ymca pool open swim",
    "leisure centre pool swimming",
    # Competitive — different body positions, useful for stroke variety
    "swim meet 50m pool side view",
    "masters swimming competition pool",
    "age group swim meet outdoor pool",
    # Lifeguard / surveillance perspective — closest to a real safety camera.
    # "lifeguards view" is the term that surfaced the Lifeguard Rescue channel.
    "lifeguards view pool",
    "lifeguard stand view pool swimmers",
    "lifeguard pov pool surveillance",
    "pool safety swim lesson children",
    # Rescue / distress footage — the rare POSITIVE class. These queries target
    # real surveillance-angle rescue clips, our hardest-to-find training data.
    "spot the drowning lifeguard",
    "lifeguard rescue wavepool",
    "waterpark lifeguard rescue",
    "pool rescue caught on camera",
    "drowning rescue pool surveillance",
    "lifeguard rescue compilation",
    # Varied conditions
    "outdoor pool cloudy day swimmers",
    "evening outdoor pool swimmers dusk",
    "crowded public pool summer swimmers",
]

# Whole channels worth pulling in their entirety — concentrated sources of the
# rare positive (distress) class at realistic surveillance camera angles.
DEFAULT_CHANNELS = [
    "https://www.youtube.com/@LifeguardRescue/videos",
]


# --- yt-dlp metadata fetch ---------------------------------------------------


def fetch_metadata(
    query: str,
    max_results: int = 10,
    min_duration: int = MIN_DURATION_SEC,
    max_duration: int = MAX_DURATION_SEC,
) -> list[dict]:
    """
    Search YouTube for a keyword query and return filtered video metadata.

    We over-fetch (3x) to absorb videos dropped by the duration filter, then
    trim to max_results after filtering.
    """
    fetch_n = max_results * 3

    cmd = [
        *_YTDLP,
        f"ytsearch{fetch_n}:{query}",
        "--dump-json",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "--ignore-errors",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=_ENV)

    # Always parse stdout regardless of return code: yt-dlp exits non-zero if
    # ANY video in the batch is unavailable, even when others returned fine.
    all_results = _parse_json_lines(result.stdout)

    if not all_results and result.returncode != 0:
        first_err = (
            result.stderr.strip().splitlines()[-1]
            if result.stderr.strip()
            else "unknown error"
        )
        print(f"  (query yielded nothing: {first_err})")

    filtered = []
    for meta in all_results:
        duration = float(meta.get("duration") or 0.0)
        # duration may be missing (0.0) in search results; keep those rather
        # than dropping them, since the floor is only meant to reject shorts.
        if duration == 0.0 or min_duration <= duration <= max_duration:
            filtered.append(meta)
        if len(filtered) >= max_results:
            break

    return filtered


def fetch_channel_videos(channel_url: str, limit: int | None = None) -> list[dict]:
    """
    List every video on a channel (or playlist) without downloading.

    Uses --flat-playlist, which returns lightweight entries (id, title, url,
    sometimes duration) for the whole channel in a single fast call rather
    than fully extracting each video. No duration filter is applied — channel
    ingest is opt-in and assumed relevant.
    """
    cmd = [
        *_YTDLP,
        channel_url,
        "--flat-playlist",
        "--dump-json",
        "--no-warnings",
        "--ignore-errors",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=_ENV)
    videos = _parse_json_lines(result.stdout)

    if not videos and result.returncode != 0:
        first_err = (
            result.stderr.strip().splitlines()[-1]
            if result.stderr.strip()
            else "unknown error"
        )
        print(f"  (channel yielded nothing: {first_err})")

    if limit is not None:
        videos = videos[:limit]
    return videos


def _parse_json_lines(stdout: str) -> list[dict]:
    """Parse yt-dlp --dump-json output: one JSON object per line."""
    out = []
    for line in stdout.strip().splitlines():
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# --- Metadata → ClipRecord ---------------------------------------------------


def metadata_to_record(
    meta: dict,
    guess_setting: Setting,
    guess_time_of_day: TimeOfDay,
    guess_weather: Weather,
) -> ClipRecord:
    """
    Convert a raw yt-dlp metadata dict (from search OR flat-playlist) into a
    ClipRecord. Condition fields are guesses based on source intent and are
    marked UNLABELED so the annotation pass verifies them.
    """
    # Flat-playlist entries use "url"; full extractions use "webpage_url".
    url = meta.get("webpage_url") or meta.get("url") or ""
    # Flat-playlist sometimes gives only the video id — normalise to a watch URL.
    if url and not url.startswith("http"):
        url = f"https://www.youtube.com/watch?v={url}"
    if not url and meta.get("id"):
        url = f"https://www.youtube.com/watch?v={meta['id']}"

    duration = float(meta.get("duration") or 0.0)
    platform = meta.get("extractor") or ("youtube" if "youtube" in url else "unknown")

    start_sec = 0.0
    end_sec = (
        duration if duration > 0 else -1.0
    )  # -1 = "to end"; annotator reads real length

    clip_id = make_clip_id(url, start_sec, end_sec)

    return ClipRecord(
        clip_id=clip_id,
        source_url=url,
        platform=platform,
        start_sec=start_sec,
        end_sec=end_sec,
        camera_view=CameraView.UNKNOWN,
        setting=guess_setting,
        time_of_day=guess_time_of_day,
        weather=guess_weather,
        label=Label.UNLABELED,
        notes=meta.get("title", ""),
    )


def _register(
    manifest: Manifest, blocklist: Blocklist, records: list[ClipRecord]
) -> None:
    """Append records to the manifest, skipping blocklisted URLs and dupes."""
    keep = [r for r in records if not blocklist.contains(r.source_url)]
    blocked = len(records) - len(keep)
    written, skipped = manifest.append_many(keep)
    print(
        f"\nDone. {written} new clips registered, {skipped} duplicates skipped, {blocked} blocklisted."
    )


# --- Collection entry points -------------------------------------------------


def collect(
    manifest_path: Path,
    queries: list[str] | None = None,
    max_results_per_query: int = 10,
    guess_setting: Setting = Setting.OUTDOOR,
    guess_time_of_day: TimeOfDay = TimeOfDay.DAY,
    guess_weather: Weather = Weather.UNKNOWN,
) -> None:
    """Search for pool footage by keyword and register results in the manifest."""
    if queries is None:
        queries = DEFAULT_QUERIES

    manifest = Manifest(manifest_path)
    blocklist = Blocklist()
    all_records: list[ClipRecord] = []

    for i, query in enumerate(queries):
        if i > 0:
            time.sleep(QUERY_SLEEP_SEC)
        print(f"Searching: '{query}'")
        results = fetch_metadata(query, max_results=max_results_per_query)
        print(f"  Found {len(results)} results")
        for meta in results:
            all_records.append(
                metadata_to_record(
                    meta, guess_setting, guess_time_of_day, guess_weather
                )
            )

    _register(manifest, blocklist, all_records)
    print(f"Manifest: {manifest_path}")


def collect_channels(
    manifest_path: Path,
    channels: list[str] | None = None,
    limit_per_channel: int | None = None,
    guess_setting: Setting = Setting.UNKNOWN,
    guess_time_of_day: TimeOfDay = TimeOfDay.UNKNOWN,
    guess_weather: Weather = Weather.UNKNOWN,
) -> None:
    """
    Pull whole channels and register every video in the manifest.

    Condition guesses default to UNKNOWN here because a single channel often
    mixes indoor/outdoor and varied conditions — the annotation pass sets them.
    """
    if channels is None:
        channels = DEFAULT_CHANNELS

    manifest = Manifest(manifest_path)
    blocklist = Blocklist()
    all_records: list[ClipRecord] = []

    for i, channel in enumerate(channels):
        if i > 0:
            time.sleep(QUERY_SLEEP_SEC)
        print(f"Pulling channel: {channel}")
        videos = fetch_channel_videos(channel, limit=limit_per_channel)
        print(f"  Found {len(videos)} videos")
        for meta in videos:
            all_records.append(
                metadata_to_record(
                    meta, guess_setting, guess_time_of_day, guess_weather
                )
            )

    _register(manifest, blocklist, all_records)
    print(f"Manifest: {manifest_path}")


# --- CLI entry point ---------------------------------------------------------

if __name__ == "__main__":
    manifest = Path("data/manifests/pool_footage.jsonl")
    collect(manifest)  # keyword searches
    collect_channels(manifest)  # whole-channel pulls (Lifeguard Rescue, etc.)
