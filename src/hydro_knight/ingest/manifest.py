"""
Manifest schema and read/write logic for clip records.

A manifest is a JSONL file (one JSON object per line) where each line
describes one video clip: where it came from, what conditions it was filmed
in, and what label it has been given.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

# --- Controlled vocabularies -------------------------------------------------
# These Enums define the only legal values for categorical fields.
# Using an Enum instead of a plain string means a typo ("outdooor") becomes
# an immediate crash rather than silent bad data in the manifest.


class CameraView(StrEnum):
    OVERHEAD = "overhead"  # camera mounted directly above the pool
    ELEVATED = "elevated"  # camera on a stand or high wall, angled down
    DECK_LEVEL = "deck_level"  # roughly eye-level with the water surface
    UNDERWATER = "underwater"  # below the surface
    UNKNOWN = "unknown"


class Setting(StrEnum):
    OUTDOOR = "outdoor"
    INDOOR = "indoor"
    UNKNOWN = "unknown"


class TimeOfDay(StrEnum):
    DAY = "day"
    DUSK = "dusk"
    NIGHT = "night"
    UNKNOWN = "unknown"


class Weather(StrEnum):
    CLEAR = "clear"
    OVERCAST = "overcast"
    RAIN = "rain"
    UNKNOWN = "unknown"


class Label(StrEnum):
    NORMAL = "normal"  # confirmed normal swimming activity
    DISTRESS = "distress"  # confirmed drowning / distress event
    SUBMERGED = "submerged"  # person submerged but outcome unknown
    FACE_DOWN = "face_down"  # prone face-down beyond normal duration
    REVIEW = "review"  # needs a human to watch before labeling
    UNLABELED = "unlabeled"  # not yet looked at


# --- Clip record -------------------------------------------------------------


@dataclass
class ClipRecord:
    """One row in the manifest — represents a single video clip."""

    clip_id: str  # deterministic hash, computed from source + timestamps
    source_url: str  # original URL the clip came from
    platform: str  # e.g. "youtube", "vimeo", "local"
    # The DOWNLOAD window: which part of the source video the local file holds.
    # start_sec > 0 means only that section was downloaded (file frame 0 is source
    # time start_sec); start_sec == 0 means the whole video. Only ingest writes
    # these; the "part that counts" is trim_start/trim_end below.
    start_sec: float  # where in the source video this clip starts (seconds)
    end_sec: float  # where it ends; use -1.0 to mean "to the end"
    camera_view: CameraView
    setting: Setting
    time_of_day: TimeOfDay
    weather: Weather
    label: Label
    notes: str = ""  # free-text, optional

    # Source frame rate, recorded once at collection time so nothing downstream
    # has to guess it. Events are stored in SECONDS but keypoint Parquets index
    # by FRAME, so every frame<->time conversion needs this number; a wrong fps
    # silently shifts every event label (a flat fps=30 assumption on 60fps
    # footage puts the drowning at twice its real timestamp).
    # 0.0 = unknown / not yet fetched — callers must fall back, never assume.
    fps: float = 0.0

    # Event windows: typed time spans (in the same source-video timeline as
    # start_sec/end_sec) marking WHERE a specific anomaly is visible.
    # Each element is a dict: {"start": float, "end": float, "label": str}.
    # Labeler events may also carry "contact", "victim" and "saved_off_camera";
    # "start"/"end" can be None while half-labeled or when the save is off camera
    # (eval uses eval.metrics.resolve_events to turn those into usable spans).
    # The per-event "label" is one of the anomaly Label values
    # (distress / submerged / face_down), so a single clip can contain
    # multiple events of DIFFERENT types — e.g. a submersion that becomes
    # a distress rescue. This also yields a frame-level multi-class mask
    # for evaluation.
    #
    # Empty list = no marked events:
    #   - a `normal` clip has no events (the whole trim span is normal)
    #   - an anomaly clip's frames OUTSIDE these windows are reusable as
    #     normal training data; frames INSIDE are the typed positive for eval
    #
    # default_factory=list gives each ClipRecord its own empty list rather
    # than sharing one mutable list across all instances (a classic bug).
    events: list[dict] = field(default_factory=list)

    # The TRIM: the part of the clip that counts (cuts intros, post-save footage,
    # end cards). Source-video seconds, like events; None = no trim on that side.
    # Kept separate from start_sec/end_sec so trimming never changes what a
    # download fetches, and never invalidates frame numbers already extracted.
    trim_start: float | None = None
    trim_end: float | None = None

    @property
    def file_offset(self) -> float:
        """Source-video time of the local file's frame 0 (0.0 for whole downloads).

        Section downloads are cut exactly at start_sec (see ingest/download.py), so
        this is exact, not a guess.
        """
        return self.start_sec if self.start_sec > 0 else 0.0

    def to_file_time(self, t: float) -> float:
        """Convert a source-video time (events, trims) to seconds into the local file."""
        return t - self.file_offset

    def to_source_time(self, t: float) -> float:
        """Convert seconds into the local file to source-video time."""
        return t + self.file_offset


def make_clip_id(source_url: str, start_sec: float, end_sec: float) -> str:
    """Deterministic short id for a clip, hashed from (source_url, start_sec, end_sec).

    Same inputs always hash to the same id, so re-running collection de-duplicates correctly;
    truncated to 12 hex chars — collision-safe for thousands of clips, readable in filenames.
    Args:
        source_url: the clip's source URL.
        start_sec: clip start offset in the source (seconds).
        end_sec: clip end offset (-1.0 = "to end").
    Returns:
        A 12-character hex id.
    """
    key = f"{source_url}|{start_sec}|{end_sec}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


# --- Manifest reader / writer ------------------------------------------------


def _to_row(record: ClipRecord) -> dict:
    """A record as a JSON-ready dict: Enums become their string values (JSON has no Enums)."""
    row = asdict(record)
    for key in ("camera_view", "setting", "time_of_day", "weather", "label"):
        row[key] = getattr(record, key).value
    return row


class Manifest:
    """
    Reads and writes a JSONL manifest file.

    JSONL ("JSON Lines") means each line of the file is its own complete JSON
    object. This format is append-safe: you can add new clips by writing a new
    line without rewriting the whole file. It also diffs cleanly in git
    because each clip is on its own line.

    Every write holds an exclusive lock on a sidecar "<manifest>.lock" file for its
    whole read-change-write, so the labeler and scripts like backfill_fps can run at
    the same time without one silently overwriting the other's change. Use modify()
    to change one clip based on its latest saved state.
    """

    def __init__(self, path: Path) -> None:
        # Store the path but don't open the file yet.
        # The file is created lazily on the first write.
        self.path = Path(path)

    def load(self) -> list[ClipRecord]:
        """Read all records from the manifest file (Enum fields restored).

        Returns:
            List of ClipRecord; empty list if the file doesn't exist yet (callers need
            not check for the file first).
        """
        if not self.path.exists():
            return []

        records = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                # Convert the raw string values back into their Enum types
                # so callers always get ClipRecord objects with Enum fields,
                # never bare strings.
                records.append(
                    ClipRecord(
                        clip_id=data["clip_id"],
                        source_url=data["source_url"],
                        platform=data["platform"],
                        start_sec=data["start_sec"],
                        end_sec=data["end_sec"],
                        camera_view=CameraView(data["camera_view"]),
                        setting=Setting(data["setting"]),
                        time_of_day=TimeOfDay(data["time_of_day"]),
                        weather=Weather(data["weather"]),
                        label=Label(data["label"]),
                        notes=data.get("notes", ""),
                        # .get with default keeps older manifests (written before
                        # the events / fps fields existed) loading without error.
                        fps=data.get("fps", 0.0),
                        events=data.get("events", []),
                        trim_start=data.get("trim_start"),
                        trim_end=data.get("trim_end"),
                    )
                )
        return records

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Hold an exclusive advisory lock across one read-change-write (blocks until free)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _write_all(self, records: list[ClipRecord]) -> None:
        """Rewrite the whole file atomically: a uniquely named temp file, then rename.

        An interrupted write can't corrupt the manifest, and two writers never share a temp file.
        """
        fd, tmp = tempfile.mkstemp(
            dir=self.path.parent, prefix=self.path.name + ".", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(_to_row(record)) + "\n")
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def modify(
        self, clip_id: str, change: Callable[[ClipRecord], ClipRecord]
    ) -> ClipRecord | None:
        """Apply change() to one clip's latest saved record and write it back, all under the lock.

        Args:
            clip_id: the clip to change.
            change: takes the current record, returns the replacement.
        Returns:
            The written record, or None if no clip has that id.
        """
        with self._locked():
            records = self.load()
            for i, record in enumerate(records):
                if record.clip_id == clip_id:
                    records[i] = change(record)
                    self._write_all(records)
                    return records[i]
        return None

    def append(self, record: ClipRecord) -> bool:
        """Append one record, skipping it if its clip_id already exists.

        De-dup is by clip_id (a deterministic hash), so re-running the same search won't double-add.
        Args:
            record: the ClipRecord to write.
        Returns:
            True if written, False if it was a duplicate.
        """
        with self._locked():
            if record.clip_id in {r.clip_id for r in self.load()}:
                return False
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(_to_row(record)) + "\n")
        return True

    def update(self, updated: ClipRecord) -> bool:
        """Replace an existing record (matched by clip_id) with an atomic full-file rewrite.

        This overwrites every field with `updated`. To change some fields of the latest saved
        record without clobbering a concurrent writer's change, use modify().
        Args:
            updated: the replacement ClipRecord (matched on its clip_id).
        Returns:
            True if a matching record was found and replaced, False otherwise.
        """
        return self.modify(updated.clip_id, lambda _old: updated) is not None

    def delete(self, clip_id: str) -> bool:
        """Remove a record by clip_id via the same locked, atomic rewrite as modify().

        Recoverable from git history if the record was ever committed.
        Args:
            clip_id: id of the record to remove.
        Returns:
            True if a matching record was removed, False otherwise.
        """
        with self._locked():
            records = self.load()
            kept = [r for r in records if r.clip_id != clip_id]
            if len(kept) == len(records):
                return False
            self._write_all(kept)
        return True

    def append_many(self, records: list[ClipRecord]) -> tuple[int, int]:
        """Append multiple records, skipping duplicates.

        Args:
            records: the ClipRecords to write.
        Returns:
            (written_count, skipped_count).
        """
        written = skipped = 0
        for record in records:
            if self.append(record):
                written += 1
            else:
                skipped += 1
        return written, skipped
