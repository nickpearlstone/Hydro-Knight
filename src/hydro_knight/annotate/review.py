"""
Review of Claude's labels: one verdict per clip, so Nick can check them quickly.

Claude labels rescue clips and lists each one in data/annotations/labeled_by_claude.json
({clip_id: {"date", "notes"}}). The labeler's Review tab plays each of those clips with
the victim marked, and records Nick's verdict here, in data/annotations/label_reviews.json:
{clip_id: {"verdict": "good" | "not_good", "note": str, "date": "YYYY-MM-DD"}}.

A "not_good" clip keeps Claude's label until someone fixes it in the Rescue timeline tab;
the note says what is wrong. Neither file holds pixels, so both are safe to commit.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

CLAUDE_PATH = Path("data/annotations/labeled_by_claude.json")
REVIEW_PATH = Path("data/annotations/label_reviews.json")
VERDICTS = ("good", "not_good")
MAX_NOTE = 2000


def load(path: Path) -> dict:
    """A {clip_id: {...}} JSON file, or {} if it doesn't exist yet."""
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def save(data: dict, path: Path = REVIEW_PATH) -> None:
    """Write the review file atomically (temp file + rename), so a crash never truncates it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def clean_verdict(body: dict) -> dict | None:
    """Validate {"verdict", "note"} posted by the UI.

    Returns:
        The stored form with today's date, or None when verdict is null (clear the review).
    Raises:
        ValueError: an unknown verdict, or a note that isn't a string or is too long.
    """
    verdict = body.get("verdict")
    if verdict is None:
        return None
    if verdict not in VERDICTS:
        raise ValueError(f"unknown verdict {verdict!r}")
    note = body.get("note") or ""
    if not isinstance(note, str) or len(note) > MAX_NOTE:
        raise ValueError("note must be text, at most 2000 characters")
    return {
        "verdict": verdict,
        "note": note.strip(),
        "date": datetime.date.today().isoformat(),
    }
