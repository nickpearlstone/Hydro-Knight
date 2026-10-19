"""
Harvest new clips: find candidates, fetch, accept or reject, extract.

The mechanical half of the unattended sourcing pipeline (`make harvest`). Judging and
labeling are done by Claude following docs/HARVEST.md; this script
only does what needs no judgment. Candidates live in data/harvest/candidates.jsonl
(metadata only, safe to commit), separate from the manifest: a clip enters the
manifest only once accepted.

    uv run hydroknight harvest search             # find new candidates (metadata only)
    uv run hydroknight harvest next               # print the next pending candidate
    uv run hydroknight harvest fetch URL_ID       # download it to raw_local/ for judging
    uv run hydroknight harvest accept URL_ID --kind rescue|normal [--section A B]
    uv run hydroknight harvest reject URL_ID --reason "..."
    uv run hydroknight harvest extract            # pose + tracks for accepted clips (Mac GPU)
    uv run hydroknight harvest status

URL_ID is the candidate's short id from `next`. Rejected URLs go on the blocklist, so no
search ever brings them back. Every accepted clip is tagged "[AUTO <date>]" in its notes.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import ssl
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import certifi
import cv2
import numpy as np

from hydro_knight.ingest import collect
from hydro_knight.ingest.blocklist import Blocklist
from hydro_knight.ingest.download import done_marker, download_clip, local_path
from hydro_knight.ingest.manifest import Label, Manifest, make_clip_id

MANIFEST = Path("data/manifests/pool_footage.jsonl")
CANDIDATES = Path("data/harvest/candidates.jsonl")
MAX_FETCH_S = 10 * 60  # longer videos are skipped: too slow to judge and extract
MAX_NORMAL_S = 3 * 60  # normal footage is kept as a section at most this long
# Only these sites are ever searched or downloaded from (all through yt-dlp, never a
# direct file link). Anything else is refused, so nothing unvetted reaches the disk.
TRUSTED_SITES = {
    "youtube.com": "youtube",
    "youtu.be": "youtube",
    "dailymotion.com": "dailymotion",
    "dai.ly": "dailymotion",
    "vimeo.com": "vimeo",
}
SOURCES = ("youtube", "dailymotion")  # searchable. Vimeo has no public search and its
# videos now need a logged-in account to download, so it only works via `add` with login.
QUERIES = [
    "spot the drowning lifeguard",
    "lifeguard rescue wavepool",
    "wave pool lifeguard rescue",
    "waterpark lifeguard rescue wave pool",
    "wave pool surveillance camera",
    "wave pool lifeguard view",
    "pool rescue caught on camera",
    "lifeguard rescue public pool",
    "swimming pool security camera swimmers",
    "lifeguard stand view pool",
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


def site_of(url: str) -> str | None:
    """The trusted site a URL belongs to, or None (https only)."""
    u = urllib.parse.urlparse(url)
    if u.scheme != "https":
        return None
    host = (u.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    for domain, site in TRUSTED_SITES.items():
        if host == domain or host.endswith("." + domain):
            return site
    return None


def dailymotion_search(query: str, limit: int) -> list[dict]:
    """Dailymotion's public search API (no key): metadata only, shaped like yt-dlp's."""
    q = urllib.parse.urlencode(
        {"search": query, "fields": "id,title,duration,url", "limit": limit}
    )
    ctx = ssl.create_default_context(cafile=certifi.where())  # verified HTTPS
    with urllib.request.urlopen(
        f"https://api.dailymotion.com/videos?{q}", timeout=30, context=ctx
    ) as r:
        data = json.load(r)
    return [
        {"webpage_url": v["url"], "title": v["title"], "duration": v["duration"]}
        for v in data.get("list", [])
    ]


def cmd_search(args) -> None:
    rows = _load()
    known = {r["url"] for r in rows}
    known |= {r.source_url for r in Manifest(MANIFEST).load()}
    block = Blocklist()
    found = []
    sources = SOURCES if args.source == "all" else (args.source,)
    if "youtube" in sources and not args.queries:
        for ch in collect.DEFAULT_CHANNELS:
            found += [("channel", m) for m in collect.fetch_channel_videos(ch)]
    for q in args.queries or QUERIES:
        if "youtube" in sources:
            time.sleep(collect.QUERY_SLEEP_SEC)
            found += [
                (f"youtube: {q}", m)
                for m in collect.fetch_metadata(q, max_results=args.per_query)
            ]
        if "dailymotion" in sources:
            found += [
                (f"dailymotion: {q}", m) for m in dailymotion_search(q, args.per_query)
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
        if not url or not site_of(url) or url in known or block.contains(url):
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


def cmd_add(args) -> None:
    """Queue a hand-picked URL (e.g. a Vimeo link found by web search)."""
    site = site_of(args.url)
    if not site:
        raise SystemExit(f"refused: only https links from {sorted(TRUSTED_SITES)}")
    rows = _load()
    if any(r["url"] == args.url for r in rows) or Blocklist().contains(args.url):
        raise SystemExit("already a candidate or blocklisted")
    meta = collect._parse_json_lines(
        subprocess.run(
            [*collect._YTDLP, args.url, "--dump-json", "--no-playlist", "--quiet"],
            capture_output=True,
            text=True,
            env=collect._ENV,
        ).stdout
    )
    if not meta:
        hint = (
            " (Vimeo now requires a logged-in account; not set up)"
            if site == "vimeo"
            else ""
        )
        raise SystemExit(f"could not read the video's metadata{hint}")
    m = meta[0]
    rows.append(
        {
            "id": _cid(args.url),
            "url": args.url,
            "title": m.get("title", ""),
            "duration": float(m.get("duration") or 0.0),
            "source": f"added: {site}",
            "found": datetime.date.today().isoformat(),
            "status": "pending",
        }
    )
    _save(rows)
    print(f"queued {_cid(args.url)}: {m.get('title')}")


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


def _is_video(path: Path) -> bool:
    """True if ffprobe sees a real video stream in an .mp4 container. Anything else is
    deleted: only video files from trusted sites are ever kept, and nothing is executed."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type:format=format_name",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        env=collect._ENV,
    )
    try:
        info = json.loads(out.stdout)
    except json.JSONDecodeError:
        return False
    kinds = {s.get("codec_type") for s in info.get("streams", [])}
    fmt = info.get("format", {}).get("format_name", "")
    return "video" in kinds and "mp4" in fmt


def _download(rec) -> bool:
    """download_clip with a clean start: a leftover .part from a failed try makes yt-dlp
    resume with an expired link (HTTP 403). One retry after a pause for transient 403s."""
    for attempt in range(2):
        for part in local_path(rec).parent.glob(f"{rec.clip_id}.mp4*.part"):
            part.unlink()
        if not site_of(rec.source_url):
            print(f"refused: {rec.source_url} is not on a trusted site")
            return False
        if download_clip(rec):
            if _is_video(local_path(rec)):
                return True
            print("downloaded file is not a valid mp4 video; deleted")
            _drop_video(rec)
            return False
        if attempt == 0:
            time.sleep(20)
    return False


HASH_DIR = Path("raw_local/frame_hashes")  # gitignored cache, one .npy per clip


def frame_hashes(path: Path, step_s: float = 1.0) -> np.ndarray:
    """64-bit difference hashes of one frame per step_s (robust to re-encoding/resizing)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for f in np.arange(0, n, fps * step_s).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if not ok:
            break
        g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 8))
        out.append(np.packbits((g[:, 1:] > g[:, :-1]).ravel()))
    cap.release()
    return np.array(out, np.uint8).reshape(-1, 8)


def _cached_hashes(clip_id: str, path: Path) -> np.ndarray:
    cache = HASH_DIR / f"{clip_id}.npy"
    if cache.exists():
        return np.load(cache)
    h = frame_hashes(path)
    HASH_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, h)
    return h


def duplicates_of(path: Path, min_matches: int = 5, max_bits: int = 8) -> list[tuple]:
    """Manifest clips sharing >= min_matches near-identical frames with this video."""
    cand = frame_hashes(path)
    if not len(cand):
        return []
    cbits = np.unpackbits(cand, axis=1).astype(bool)
    hits = []
    for r in Manifest(MANIFEST).load():
        p = local_path(r)
        long = r.end_sec and r.end_sec - (r.start_sec or 0) > MAX_FETCH_S
        if (
            not p.exists() or long
        ):  # long recordings can't hold a short re-upload's event
            continue
        ref = _cached_hashes(r.clip_id, p)
        if not len(ref):
            continue
        rbits = np.unpackbits(ref, axis=1).astype(bool)
        dist = (cbits[:, None, :] != rbits[None, :, :]).sum(-1)
        matches = int((dist.min(axis=1) <= max_bits).sum())
        if matches >= min_matches:
            hits.append((r.clip_id, matches, len(cand), r.notes[:50]))
    return sorted(hits, key=lambda h: -h[1])


def cmd_fetch(args) -> None:
    rows = _load()
    row = _find(rows, args.id)
    if row["status"] != "pending":
        raise SystemExit(
            f"{args.id} is {row['status']}; only pending candidates are fetched"
        )
    rec = _record(row)
    ok = local_path(rec).exists() or _download(rec)
    if not ok:
        row.update(status="failed", reason="download failed")
        _save(rows)
        raise SystemExit("download failed")
    row["fetched_clip_id"] = rec.clip_id
    dups = duplicates_of(local_path(rec))
    row["possible_duplicates"] = [h[0] for h in dups]
    _save(rows)
    for cid, m, n, title in dups:
        print(f"POSSIBLE DUPLICATE of {cid}: {m}/{n} frames match ({title})")
    print(
        f"fetched {local_path(rec)}  (sheets: hydroknight sheets {rec.clip_id} overview)"
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
    py = [sys.executable, "-m", "hydro_knight.cli"]
    run = lambda cmd: subprocess.run(py + cmd, check=True)  # noqa: E731
    run(["extract", "--clips", *todo, "--device", args.device])
    run(["tracks", "--clips", *todo, "--out", "data/keypoints_tiled"])
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
    s.add_argument("--queries", nargs="+", help="search these instead of the defaults")
    s.add_argument("--source", default="all", choices=[*SOURCES, "all"])
    ad = sub.add_parser("add")
    ad.add_argument("url")
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
