"""
Browser-based clip labeler: a small local Flask server plus a single-page UI.

Run `uv run python -m hydro_knight.annotate` and a browser tab opens. Everything
stays on this machine: the server only listens on 127.0.0.1 and serves videos
straight from raw_local/.

What gets labeled, per clip:
- label: distress or normal
- trim: start_sec / end_sec, the usable part of the clip (excludes end cards)
- per event (distress clips):
    start   drowning onset
    contact guard makes contact           (new)
    end     victim saved
    victim  clicked points on the victim  (new), each {"t", "x", "y", "kind"}
            kind is "onset", "last_seen" or "extra"; x/y are pixels in the
            original video frame, so they stay valid for any display size

Victim marks are positions, never track ids. Tracking is rerun with different
settings over time and a victim's id changes every second or so; a clicked
position can be matched to whatever the tracker produced afterwards, and a
click with no detection nearby records that the victim was not detected.

A second tab, Swimmer count (/count), labels every person on a few still frames for
the pose benchmark; see annotate/count.py for that file's format.

Times: the manifest stores event and trim times in the source-video timeline.
For clips downloaded as a section, the local file starts at start_sec, so the
UI works in file time and converts with `offset` (0 for whole downloads).
"""

from __future__ import annotations

import dataclasses
import threading
import webbrowser
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, abort, jsonify, request, send_file

from ..ingest.blocklist import Blocklist
from ..ingest.download import done_marker, local_path
from ..ingest.manifest import Label, Manifest
from ..preprocess.extract_pose import load_detections
from ..preprocess.tiled_pose import merge_detections
from . import count

STATIC = Path(__file__).parent / "static"
VICTIM_KINDS = ("onset", "last_seen", "extra")
UI_LABELS = (Label.DISTRESS, Label.NORMAL)


def _video_facts(path: Path) -> dict:
    """fps, frame count, duration and size of a local video file (zeros if unreadable)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return {
        "fps": fps,
        "frames": n,
        "duration": n / fps if fps else 0.0,
        "width": w,
        "height": h,
    }


def file_offset(record, duration: float) -> float:
    """Seconds to subtract from source-timeline times to get local-file times.

    A sectioned download starts at start_sec, so its file is about (end - start) long; a whole
    download is much longer than that, and its file time already equals source time.
    Args:
        record: the ClipRecord.
        duration: local file duration in seconds.
    Returns:
        start_sec for sectioned downloads, else 0.0.
    """
    s, e = record.start_sec, record.end_sec
    if s > 0 and e > 0 and abs(duration - (e - s)) < 2.0:
        return float(s)
    return 0.0


def needs_victim_marks(record) -> bool:
    """True for a distress clip with an event missing contact or an onset/last_seen victim click."""
    if record.label != Label.DISTRESS:
        return False
    if not record.events:
        return True
    for ev in record.events:
        kinds = {m.get("kind") for m in ev.get("victim", [])}
        if ev.get("contact") is None or not {"onset", "last_seen"} <= kinds:
            return True
    return False


def _clean_events(events: list) -> list[dict]:
    """Validate events posted by the UI and return them in manifest form.

    Raises:
        ValueError: malformed event or victim mark.
    """
    out = []
    for ev in events:
        start, end = float(ev["start"]), float(ev["end"])
        if end < start:
            raise ValueError("event end is before its start")
        contact = ev.get("contact")
        contact = None if contact is None else float(contact)
        victim = []
        for m in ev.get("victim", []):
            if m.get("kind") not in VICTIM_KINDS:
                raise ValueError(f"unknown victim mark kind {m.get('kind')!r}")
            victim.append(
                {
                    "t": float(m["t"]),
                    "x": float(m["x"]),
                    "y": float(m["y"]),
                    "kind": m["kind"],
                }
            )
        victim.sort(key=lambda m: m["t"])
        clean = {"start": start, "end": end, "label": ev.get("label", "distress")}
        if contact is not None:
            clean["contact"] = contact
        if victim:
            clean["victim"] = victim
        out.append(clean)
    return sorted(out, key=lambda e: e["start"])


def create_app(
    manifest_path: Path,
    detections_dir: Path = Path("data/detections"),
    blocklist: Blocklist | None = None,
    count_path: Path = count.COUNT_PATH,
) -> Flask:
    """Build the labeler app around one manifest file.

    Args:
        manifest_path: JSONL manifest to read and write.
        detections_dir: raw tiled detections (for the optional box overlay).
        blocklist: URL blocklist for clip deletion (default: the project blocklist).
        count_path: swimmer-count label file for the Swimmer count tab.
    Returns:
        The Flask app.
    """
    app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")
    manifest = Manifest(Path(manifest_path))
    blocklist = blocklist or Blocklist()
    lock = threading.Lock()  # manifest rewrites are whole-file; serialize them

    def get_record(clip_id: str):
        for r in manifest.load():
            if r.clip_id == clip_id:
                return r
        abort(404, f"no clip {clip_id}")

    def detections_by_frame(clip_id: str):
        """Merged raw detections per frame, or None if the clip was never extracted.

        Cached per meta.json modification time, so a clip extracted (or extended) while the
        labeler is running shows its boxes without a restart.
        """
        meta = Path(detections_dir) / clip_id / "meta.json"
        if not meta.exists():
            return None
        return _load_boxes(clip_id, meta.stat().st_mtime_ns)

    @lru_cache(maxsize=4)
    def _load_boxes(clip_id: str, _mtime: int):
        folder = Path(detections_dir) / clip_id
        _, dets, _ = load_detections(folder, require_complete=False)
        if dets.empty:
            return {}
        boxes = dets[["bx1", "by1", "bx2", "by2"]].to_numpy(np.float32)
        conf = dets["conf"].to_numpy(np.float32)
        crop = dets["crop"].to_numpy()
        frames = dets["frame"].to_numpy()
        out = {}
        for f in np.unique(frames):
            idx = np.flatnonzero(frames == f)
            keep = idx[merge_detections(boxes[idx], conf[idx], crop[idx])]
            out[int(f)] = [
                {"box": boxes[i].round(1).tolist(), "conf": round(float(conf[i]), 3)}
                for i in keep
            ]
        return out

    @app.get("/")
    def index():
        return send_file(STATIC / "index.html")

    @app.get("/api/clips")
    def list_clips():
        which = request.args.get("filter", "needs_victim")
        rows = []
        for r in manifest.load():
            if "[HOLD" in r.notes:
                continue
            if not local_path(r).exists():
                continue
            if which == "needs_victim" and not needs_victim_marks(r):
                continue
            if which == "unlabeled" and r.label not in (Label.UNLABELED, Label.REVIEW):
                continue
            if which == "distress" and r.label != Label.DISTRESS:
                continue
            rows.append(
                {
                    "clip_id": r.clip_id,
                    "label": r.label.value,
                    "notes": r.notes,
                    "done": not needs_victim_marks(r) and r.label in UI_LABELS,
                }
            )
        return jsonify(rows)

    @app.get("/api/clip/<clip_id>")
    def get_clip(clip_id: str):
        r = get_record(clip_id)
        facts = _video_facts(local_path(r))
        return jsonify(
            {
                "clip_id": r.clip_id,
                "label": r.label.value,
                "notes": r.notes,
                "source_url": r.source_url,
                "start_sec": r.start_sec,
                "end_sec": r.end_sec,
                "events": r.events,
                "offset": file_offset(r, facts["duration"]),
                "video": facts,
                "has_detections": (
                    Path(detections_dir) / clip_id / "meta.json"
                ).exists(),
            }
        )

    @app.post("/api/clip/<clip_id>")
    def save_clip(clip_id: str):
        body = request.get_json(force=True)
        with lock:
            r = get_record(clip_id)
            try:
                label = Label(body.get("label", r.label.value))
                events = _clean_events(body.get("events", r.events))
                start = float(body.get("start_sec", r.start_sec))
                end = float(body.get("end_sec", r.end_sec))
            except (ValueError, KeyError, TypeError) as e:
                return jsonify({"error": str(e)}), 400
            updated = dataclasses.replace(
                r, label=label, events=events, start_sec=start, end_sec=end
            )
            manifest.update(updated)
        return jsonify({"ok": True, "events": events})

    @app.post("/api/clip/<clip_id>/delete")
    def delete_clip(clip_id: str):
        with lock:
            r = get_record(clip_id)
            blocklist.add(r.source_url)
            manifest.delete(r.clip_id)
            for p in (local_path(r), done_marker(r)):
                if p.exists():
                    p.unlink()
        return jsonify({"ok": True})

    @app.get("/video/<clip_id>")
    def video(clip_id: str):
        path = local_path(get_record(clip_id))
        if not path.exists():
            abort(404)
        # conditional=True answers HTTP Range requests, which the browser needs to seek.
        return send_file(path.resolve(), mimetype="video/mp4", conditional=True)

    @app.get("/api/boxes/<clip_id>")
    def boxes(clip_id: str):
        frame = int(request.args.get("frame", 0))
        by_frame = detections_by_frame(clip_id)
        if by_frame is None:
            return jsonify({"available": False, "boxes": []})
        return jsonify({"available": True, "boxes": by_frame.get(frame, [])})

    # ------------------------------------------------------------ swimmer count tab

    def count_frames() -> list[dict]:
        if not Path(count_path).exists():
            abort(404, f"no {count_path}: run scripts/benchmark_pose.py frames first")
        return count.load(count_path)["frames"]

    @app.get("/count")
    def count_page():
        return send_file(STATIC / "count.html")

    @app.get("/api/count")
    def count_list():
        return jsonify(count_frames())

    @app.get("/api/count/<int:i>/image")
    def count_image(i: int):
        frames = count_frames()
        if not 0 <= i < len(frames):
            abort(404)
        try:
            img = count.read_frame(frames[i])
        except FileNotFoundError as e:
            abort(404, str(e))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return app.response_class(buf.tobytes(), mimetype="image/jpeg")

    @app.post("/api/count/<int:i>")
    def count_save(i: int):
        with lock:
            data = count.load(count_path) if Path(count_path).exists() else None
            if data is None or not 0 <= i < len(data["frames"]):
                abort(404)
            try:
                points = count.clean_points(request.get_json(force=True))
            except (ValueError, TypeError) as e:
                return jsonify({"error": str(e)}), 400
            data["frames"][i].update(points)
            count.save(data, count_path)
        return jsonify({"ok": True})

    return app


def main(port: int = 8765, open_browser: bool = True) -> None:
    """Start the labeler on http://127.0.0.1:<port> and open it in the default browser."""
    app = create_app(Path("data/manifests/pool_footage.jsonl"))
    url = f"http://127.0.0.1:{port}"
    print(f"Labeler running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
