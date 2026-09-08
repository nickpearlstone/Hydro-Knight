"""
Blocklist of source URLs that have been reviewed and rejected.

When a clip is deleted from the manifest, its source URL is added here.
The collector checks this list before registering new clips so rejected
videos are never re-added by a future search run.

The blocklist is a plain text file — one URL per line. It is committed
to the repo so the exclusion is permanent and shared.
"""

from __future__ import annotations

from pathlib import Path

BLOCKLIST_PATH = Path("data/blocklist.txt")


class Blocklist:
    """Reader/writer for the committed reject-list of source URLs (one per line)."""

    def __init__(self, path: Path = BLOCKLIST_PATH) -> None:
        self.path = Path(path)

    def load(self) -> set[str]:
        """Return the set of all blocked URLs. Empty set if file doesn't exist."""
        if not self.path.exists():
            return set()
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return {line.strip() for line in lines if line.strip()}

    def add(self, url: str) -> None:
        """Add a URL to the blocklist if not already present."""
        existing = self.load()
        if url in existing:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(url + "\n")

    def contains(self, url: str) -> bool:
        """True if `url` is on the blocklist."""
        return url in self.load()
