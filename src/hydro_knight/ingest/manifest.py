"""Manifest schema and read/write logic for clip records."""

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


class CameraView(StrEnum):
    OVERHEAD = "overhead"
    ELEVATED = "elevated"
    DECK_LEVEL = "deck_level"
    UNDERWATER = "underwater"
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
    NORMAL = "normal"
    DISTRESS = "distress"
    SUBMERGED = "submerged"
    FACE_DOWN = "face_down"
    REVIEW = "review"
    UNLABELED = "unlabeled"


@dataclass
class ClipRecord:
    """One row in the manifest — represents a single video clip."""

    # CAUTION[manifest-clip-id]: clip_id must equal make_clip_id(source_url, start_sec, end_sec).
    clip_id: str
    source_url: str
    platform: str
    # CAUTION[manifest-download-window]: start_sec/end_sec are source seconds; start_sec > 0 means file frame 0 is source time start_sec.
    start_sec: float
    # CAUTION[manifest-end-sec-sentinel]: end_sec -1.0 means "to the end"; never treat as a time.
    end_sec: float
    camera_view: CameraView
    setting: Setting
    time_of_day: TimeOfDay
    weather: Weather
    label: Label
    notes: str = ""

    # CAUTION[manifest-fps-unknown]: fps 0.0 = unknown; callers must fall back, never assume.
    fps: float = 0.0

    # CAUTION[manifest-events-source-time]: events are source-video seconds, not file time; start/end may be None.
    events: list[dict] = field(default_factory=list)

    # CAUTION[manifest-trim-source-time]: trim is source seconds; None = no trim; independent of start_sec/end_sec.
    trim_start: float | None = None
    trim_end: float | None = None

    @property
    def file_offset(self) -> float:
        """Source-video time of the local file's frame 0 (0.0 for whole downloads)."""
        return self.start_sec if self.start_sec > 0 else 0.0

    def to_file_time(self, t: float) -> float:
        """Convert a source-video time (events, trims) to seconds into the local file."""
        return t - self.file_offset

    def to_source_time(self, t: float) -> float:
        """Convert seconds into the local file to source-video time."""
        return t + self.file_offset


def make_clip_id(source_url: str, start_sec: float, end_sec: float) -> str:
    """Deterministic short id for a clip, hashed from (source_url, start_sec, end_sec)."""
    key = f"{source_url}|{start_sec}|{end_sec}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _to_row(record: ClipRecord) -> dict:
    row = asdict(record)
    for key in ("camera_view", "setting", "time_of_day", "weather", "label"):
        row[key] = getattr(record, key).value
    return row


class Manifest:
    """Reads and writes a JSONL manifest file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> list[ClipRecord]:
        """Read all records from the manifest file (Enum fields restored)."""
        if not self.path.exists():
            return []

        records = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
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
                        fps=data.get("fps", 0.0),
                        events=data.get("events", []),
                        trim_start=data.get("trim_start"),
                        trim_end=data.get("trim_end"),
                    )
                )
        return records

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _write_all(self, records: list[ClipRecord]) -> None:
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
        """Apply change() to one clip's latest saved record and write it back, all under the lock."""
        with self._locked():
            records = self.load()
            for i, record in enumerate(records):
                if record.clip_id == clip_id:
                    records[i] = change(record)
                    self._write_all(records)
                    return records[i]
        return None

    def append(self, record: ClipRecord) -> bool:
        """Append one record, skipping it if its clip_id already exists."""
        with self._locked():
            if record.clip_id in {r.clip_id for r in self.load()}:
                return False
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(_to_row(record)) + "\n")
        return True

    def update(self, updated: ClipRecord) -> bool:
        """Replace an existing record (matched by clip_id) with an atomic full-file rewrite."""
        return self.modify(updated.clip_id, lambda _old: updated) is not None

    def delete(self, clip_id: str) -> bool:
        """Remove a record by clip_id via the same locked, atomic rewrite as modify()."""
        with self._locked():
            records = self.load()
            kept = [r for r in records if r.clip_id != clip_id]
            if len(kept) == len(records):
                return False
            self._write_all(kept)
        return True

    def append_many(self, records: list[ClipRecord]) -> tuple[int, int]:
        """Append multiple records, skipping duplicates."""
        written = skipped = 0
        for record in records:
            if self.append(record):
                written += 1
            else:
                skipped += 1
        return written, skipped
