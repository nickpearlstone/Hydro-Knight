"""Review of Claude's labels: one good / not_good verdict per clip (label_reviews.json)."""

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
    """Write the review file atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def clean_verdict(body: dict) -> dict | None:
    """Validate a posted {"verdict", "note"}; None when verdict is null (clear the review)."""
    if not isinstance(body, dict):
        raise ValueError("expected a JSON object")
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
