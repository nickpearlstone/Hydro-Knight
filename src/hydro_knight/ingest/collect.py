"""Search YouTube for pool footage and register clips in the manifest."""

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

_ENV = {
    **os.environ,
    "PATH": f"/opt/homebrew/bin:/usr/local/bin:{os.environ.get('PATH', '')}",
}

# Run yt-dlp via sys.executable, never the bare name.
_YTDLP = [sys.executable, "-m", "yt_dlp"]

# Seconds between queries; keep nonzero.
QUERY_SLEEP_SEC = 3


MIN_DURATION_SEC = 20
MAX_DURATION_SEC = 45 * 60

DEFAULT_QUERIES = [
    "swimming pool overhead view people swimming",
    "pool deck camera angle swimmers laps",
    "aquatic center wide angle swim practice",
    "outdoor public pool swimmers summer",
    "community pool open swim session",
    "backyard pool party swimming people",
    "hotel pool guests swimming vacation",
    "indoor lap pool swimmers training session",
    "ymca pool open swim",
    "leisure centre pool swimming",
    "swim meet 50m pool side view",
    "masters swimming competition pool",
    "age group swim meet outdoor pool",
    "lifeguards view pool",
    "lifeguard stand view pool swimmers",
    "lifeguard pov pool surveillance",
    "pool safety swim lesson children",
    "spot the drowning lifeguard",
    "lifeguard rescue wavepool",
    "waterpark lifeguard rescue",
    "pool rescue caught on camera",
    "drowning rescue pool surveillance",
    "lifeguard rescue compilation",
    "outdoor pool cloudy day swimmers",
    "evening outdoor pool swimmers dusk",
    "crowded public pool summer swimmers",
]

DEFAULT_CHANNELS = [
    "https://www.youtube.com/@LifeguardRescue/videos",
]


def fetch_metadata(
    query: str,
    max_results: int = 10,
    min_duration: int = MIN_DURATION_SEC,
    max_duration: int = MAX_DURATION_SEC,
) -> list[dict]:
    """Search YouTube for a keyword query and return duration-filtered video metadata."""
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

    # Parse stdout regardless of return code.
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
        # Keep entries with missing duration (0.0).
        if duration == 0.0 or min_duration <= duration <= max_duration:
            filtered.append(meta)
        if len(filtered) >= max_results:
            break

    return filtered


def fetch_channel_videos(channel_url: str, limit: int | None = None) -> list[dict]:
    """List every video on a channel/playlist (metadata only, no download)."""
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
    out = []
    for line in stdout.strip().splitlines():
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def metadata_to_record(
    meta: dict,
    guess_setting: Setting,
    guess_time_of_day: TimeOfDay,
    guess_weather: Weather,
) -> ClipRecord:
    """Convert a raw yt-dlp metadata dict into a ClipRecord (label UNLABELED, conditions guessed)."""
    url = meta.get("webpage_url") or meta.get("url") or ""
    if url and not url.startswith("http"):
        url = f"https://www.youtube.com/watch?v={url}"
    if not url and meta.get("id"):
        url = f"https://www.youtube.com/watch?v={meta['id']}"

    duration = float(meta.get("duration") or 0.0)
    platform = meta.get("extractor") or ("youtube" if "youtube" in url else "unknown")

    start_sec = 0.0
    end_sec = duration if duration > 0 else -1.0  # -1 = "to end"

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
    keep = [r for r in records if not blocklist.contains(r.source_url)]
    blocked = len(records) - len(keep)
    written, skipped = manifest.append_many(keep)
    print(
        f"\nDone. {written} new clips registered, {skipped} duplicates skipped, {blocked} blocklisted."
    )


def collect(
    manifest_path: Path,
    queries: list[str] | None = None,
    max_results_per_query: int = 10,
    guess_setting: Setting = Setting.OUTDOOR,
    guess_time_of_day: TimeOfDay = TimeOfDay.DAY,
    guess_weather: Weather = Weather.UNKNOWN,
) -> None:
    """Search for pool footage by keyword and register the results in the manifest."""
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
    """Pull whole channels and register every video in the manifest."""
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


if __name__ == "__main__":
    manifest = Path("data/manifests/pool_footage.jsonl")
    collect(manifest)
    collect_channels(manifest)
