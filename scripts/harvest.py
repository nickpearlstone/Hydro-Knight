"""
Harvest new clips: find candidates, fetch, accept or reject, extract.

The mechanical half of the unattended sourcing pipeline (`make harvest`). Judging and
labeling are done by Claude following docs/HARVEST.md; this script
only does what needs no judgment. Candidates live in data/harvest/candidates.jsonl
(metadata only, safe to commit), separate from the manifest: a clip enters the
manifest only once accepted.

    uv run python scripts/harvest.py search             # find new candidates (metadata only)
    uv run python scripts/harvest.py next               # print the next pending candidate
    uv run python scripts/harvest.py fetch URL_ID       # download it to raw_local/ for judging
    uv run python scripts/harvest.py accept URL_ID --kind rescue|normal [--section A B]
    uv run python scripts/harvest.py reject URL_ID --reason "..."
    uv run python scripts/harvest.py extract            # pose + tracks for accepted clips (Mac GPU)
    uv run python scripts/harvest.py status

URL_ID is the candidate's short id from `next`. Rejected URLs go on the blocklist, so no
search ever brings them back. Every accepted clip is tagged "[AUTO <date>]" in its notes.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from hydro_knight.ingest import collect
from hydro_knight.ingest.blocklist import Blocklist
from hydro_knight.ingest.download import done_marker, download_clip, local_path
from hydro_knight.ingest.manifest import Label, Manifest, make_clip_id

MANIFEST = Path("data/manifests/pool_footage.jsonl")
CANDIDATES = Path("data/harvest/candidates.jsonl")
MAX_FETCH_S = 10 * 60  # longer videos are skipped: too slow to judge and extract
MAX_NORMAL_S = 3 * 60  # normal footage is kept as a section at most this long
QUERIES = [
    "spot the drowning lifeguard",
    "lifeguard rescue wavepool",
    "wave pool lifeguard rescue",
    "waterpark lifeguard rescue wave pool",
    "wave pool surveillance camera",
    "wave pool crowd overhead view",
]


def _load() -> list[dict]:
    if not CANDIDATES.exists():
        return []
    return [json.loads(line) for line in CANDIDATES.read_text().splitlines() if line]


def _save(rows: list[dict]) -> None:
    CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
    tmp = CANDIDATES.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(CANDIDATES)


def _cid(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:10]


def _find(rows, cid) -> dict:
    for r in rows:
        if r["id"] == cid:
            return r
    raise SystemExit(f"no candidate {cid}")


def _record(row: dict, start: float = 0.0, end: float | None = None):
    """A manifest ClipRecord for a candidate (whole video, or a section)."""
    end = row["duration"] if end is None else end
    meta = {
        "webpage_url": row["url"],
        "duration": row["duration"],
        "title": row["title"],
    }
    rec = collect.metadata_to_record(
        meta, collect.Setting.OUTDOOR, collect.TimeOfDay.DAY, collect.Weather.UNKNOWN
    )
    return dataclasses.replace(
        rec, start_sec=start, end_sec=end, clip_id=make_clip_id(row["url"], start, end)
    )


def cmd_search(args) -> None:
    rows = _load()
    known = {r["url"] for r in rows}
    known |= {r.source_url for r in Manifest(MANIFEST).load()}
    block = Blocklist()
    found = []
    for ch in collect.DEFAULT_CHANNELS:
        found += [("channel", m) for m in collect.fetch_channel_videos(ch)]
    for q in QUERIES:
        time.sleep(collect.QUERY_SLEEP_SEC)
        found += [
            (f"search: {q}", m)
            for m in collect.fetch_metadata(q, max_results=args.per_query)
        ]
    new = 0
    for source, meta in found:
        rec = collect.metadata_to_record(
            meta,
            collect.Setting.OUTDOOR,
            collect.TimeOfDay.DAY,
            collect.Weather.UNKNOWN,
        )
        url = rec.source_url
        if not url or url in known or block.contains(url):
            continue
        known.add(url)
        rows.append(
            {
                "id": _cid(url),
                "url": url,
                "title": rec.notes,
                "duration": float(meta.get("duration") or 0.0),
                "source": source,
                "found": datetime.date.today().isoformat(),
                "status": "pending",
            }
        )
        new += 1
    _save(rows)
    print(
        f"{new} new candidates; {sum(r['status'] == 'pending' for r in rows)} pending"
    )


def cmd_next(args) -> None:
    rows = _load()
    for r in rows:
        if r["status"] != "pending":
            continue
        if not 0 < r["duration"] <= MAX_FETCH_S:
            r.update(status="skipped", reason="duration unknown or over 10 min")
            continue
        _save(rows)
        print(json.dumps(r))
        return
    _save(rows)
    print("NONE")


def _download(rec) -> bool:
    """download_clip with a clean start: a leftover .part from a failed try makes yt-dlp
    resume with an expired link (HTTP 403). One retry after a pause for transient 403s."""
    for attempt in range(2):
        for part in local_path(rec).parent.glob(f"{rec.clip_id}.mp4*.part"):
            part.unlink()
        if download_clip(rec):
            return True
        if attempt == 0:
            time.sleep(20)
    return False


def cmd_fetch(args) -> None:
    rows = _load()
    row = _find(rows, args.id)
    rec = _record(row)
    ok = local_path(rec).exists() or _download(rec)
    if not ok:
        row.update(status="failed", reason="download failed")
        _save(rows)
        raise SystemExit("download failed")
    row["fetched_clip_id"] = rec.clip_id
    _save(rows)
    print(
        f"fetched {local_path(rec)}  (sheets: scripts/label_sheets.py {rec.clip_id} overview)"
    )


def _drop_video(clip_id_rec) -> None:
    for p in (local_path(clip_id_rec), done_marker(clip_id_rec)):
        if p.exists():
            p.unlink()


def cmd_accept(args) -> None:
    rows = _load()
    row = _find(rows, args.id)
    whole = _record(row)
    if args.section:
        a, b = args.section
        if args.kind == "normal" and b - a > MAX_NORMAL_S:
            raise SystemExit(f"normal sections are at most {MAX_NORMAL_S} s")
        rec = _record(row, a, b)
        _drop_video(whole)
        if not _download(rec):
            raise SystemExit("section download failed")
    else:
        if args.kind == "normal" and row["duration"] > MAX_NORMAL_S:
            raise SystemExit(f"normal footage over {MAX_NORMAL_S} s needs --section")
        rec = whole
    tag = f"[AUTO {datetime.date.today().isoformat()}]"
    rec = dataclasses.replace(
        rec,
        label=Label.DISTRESS if args.kind == "rescue" else Label.NORMAL,
        notes=f"{row['title']} {tag}",
    )
    written, _ = Manifest(MANIFEST).append_many([rec])
    if not written:
        raise SystemExit(f"{rec.clip_id} is already in the manifest")
    row.update(status="accepted", kind=args.kind, clip_id=rec.clip_id)
    _save(rows)
    print(f"accepted {rec.clip_id} ({args.kind})")


def cmd_reject(args) -> None:
    rows = _load()
    row = _find(rows, args.id)
    _drop_video(_record(row))
    Blocklist().add(row["url"])
    row.update(status="rejected", reason=args.reason)
    _save(rows)
    print(f"rejected {row['id']}: {args.reason}")


def cmd_extract(args) -> None:
    clips = [r["clip_id"] for r in _load() if r["status"] == "accepted"]
    todo = [
        c for c in clips if not (Path("data/keypoints_tiled") / f"{c}.parquet").exists()
    ]
    if not todo:
        print("nothing to extract")
        return
    py = [sys.executable]
    run = lambda cmd: subprocess.run(py + cmd, check=True)  # noqa: E731
    run(["scripts/extract_dataset.py", "--clips", *todo, "--device", args.device])
    run(["scripts/build_tracks.py", "--clips", *todo, "--out", "data/keypoints_tiled"])
    print(f"extracted {len(todo)} clips")


def cmd_status(args) -> None:
    from collections import Counter

    rows = _load()
    print(Counter(r["status"] for r in rows))
    acc = [r for r in rows if r["status"] == "accepted"]
    print(f"accepted: {[(r['clip_id'], r['kind']) for r in acc]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--per-query", type=int, default=15)
    sub.add_parser("next")
    f = sub.add_parser("fetch")
    f.add_argument("id")
    a = sub.add_parser("accept")
    a.add_argument("id")
    a.add_argument("--kind", choices=["rescue", "normal"], required=True)
    a.add_argument("--section", nargs=2, type=float, metavar=("START", "END"))
    r = sub.add_parser("reject")
    r.add_argument("id")
    r.add_argument("--reason", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--device", default="mps")
    sub.add_parser("status")
    args = p.parse_args()
    globals()[f"cmd_{args.cmd}"](args)


if __name__ == "__main__":
    main()
