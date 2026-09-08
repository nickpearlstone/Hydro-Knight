"""
Manifest schema and read/write logic for clip records.

A manifest is a JSONL file (one JSON object per line) where each line
describes one video clip: where it came from, what conditions it was filmed
in, and what label it has been given.
"""

from __future__ import annotations

import hashlib
import json
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
    start_sec: float  # where in the source video this clip starts (seconds)
    end_sec: float  # where it ends; use -1.0 to mean "to the end"
    camera_view: CameraView
    setting: Setting
    time_of_day: TimeOfDay
    weather: Weather
    label: Label
    notes: str = ""  # free-text, optional

    # Event windows: typed time spans (in the same source-video timeline as
    # start_sec/end_sec) marking WHERE a specific anomaly is visible.
    # Each element is a dict: {"start": float, "end": float, "label": str}.
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


class Manifest:
    """
    Reads and writes a JSONL manifest file.

    JSONL ("JSON Lines") means each line of the file is its own complete JSON
    object. This format is append-safe: you can add new clips by writing a new
    line without rewriting the whole file. It also diffs cleanly in git
    because each clip is on its own line.
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
                        # the events field existed) loading without error.
                        events=data.get("events", []),
                    )
                )
        return records

    def append(self, record: ClipRecord) -> bool:
        """Append one record, skipping it if its clip_id already exists.

        De-dup is by clip_id (a deterministic hash), so re-running the same search won't double-add.
        Args:
            record: the ClipRecord to write.
        Returns:
            True if written, False if it was a duplicate.
        """
        existing_ids = {r.clip_id for r in self.load()}
        if record.clip_id in existing_ids:
            return False

        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self.path.open("a", encoding="utf-8") as f:
            # asdict() converts the dataclass to a plain dict.
            # We then convert each Enum to its .value string so JSON can
            # serialize it (JSON doesn't know what an Enum is).
            row = asdict(record)
            row["camera_view"] = record.camera_view.value
            row["setting"] = record.setting.value
            row["time_of_day"] = record.time_of_day.value
            row["weather"] = record.weather.value
            row["label"] = record.label.value
            f.write(json.dumps(row) + "\n")

        return True

    def update(self, updated: ClipRecord) -> bool:
        """Replace an existing record (matched by clip_id) via an atomic full-file rewrite.

        Writes to a temp file then renames, so an interrupted write can't corrupt the manifest
        (JSONL has no "edit line N"; whole-file rewrite is the only safe edit, and stays fast).
        Args:
            updated: the replacement ClipRecord (matched on its clip_id).
        Returns:
            True if a matching record was found and replaced, False otherwise.
        """
        records = self.load()
        found = False

        for i, record in enumerate(records):
            if record.clip_id == updated.clip_id:
                records[i] = updated
                found = True
                break

        if not found:
            return False

        # Write all records back to the file from scratch.
        # We write to a temporary file first, then replace the original.
        # This protects against data loss if the process is interrupted
        # mid-write — without this, a crash halfway through would leave
        # a half-written, corrupted manifest.
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for record in records:
                row = asdict(record)
                row["camera_view"] = record.camera_view.value
                row["setting"] = record.setting.value
                row["time_of_day"] = record.time_of_day.value
                row["weather"] = record.weather.value
                row["label"] = record.label.value
                f.write(json.dumps(row) + "\n")

        tmp.replace(self.path)
        return True

    def delete(self, clip_id: str) -> bool:
        """Remove a record by clip_id via the same atomic temp-file rewrite as update().

        Recoverable from git history if the record was ever committed.
        Args:
            clip_id: id of the record to remove.
        Returns:
            True if a matching record was removed, False otherwise.
        """
        records = self.load()
        filtered = [r for r in records if r.clip_id != clip_id]

        if len(filtered) == len(records):
            return False

        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for record in filtered:
                row = asdict(record)
                row["camera_view"] = record.camera_view.value
                row["setting"] = record.setting.value
                row["time_of_day"] = record.time_of_day.value
                row["weather"] = record.weather.value
                row["label"] = record.label.value
                f.write(json.dumps(row) + "\n")

        tmp.replace(self.path)
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
