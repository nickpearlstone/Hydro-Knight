"""
Tests for the browser labeler's server (annotate/server.py), via Flask's test client.

The labeler writes the manifest every other file depends on, so these pin that a
save keeps every field it doesn't own, that bad marks are refused instead of
written, and that the "needs victim marks" queue matches what the UI treats as done.
Videos and manifests are throwaway files under tmp_path.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd
import pytest

from hydro_knight.annotate import server
from hydro_knight.ingest import download
from hydro_knight.ingest.blocklist import Blocklist
from hydro_knight.ingest.manifest import (
    CameraView,
    ClipRecord,
    Label,
    Manifest,
    Setting,
    TimeOfDay,
    Weather,
)
from hydro_knight.preprocess.extract_pose import DET_COLUMNS, write_chunk


def _record(clip_id, label=Label.DISTRESS, events=None, notes="", start=0.0, end=-1.0):
    return ClipRecord(
        clip_id=clip_id,
        source_url=f"https://example.com/{clip_id}",
        platform="youtube",
        start_sec=start,
        end_sec=end,
        camera_view=CameraView.ELEVATED,
        setting=Setting.OUTDOOR,
        time_of_day=TimeOfDay.DAY,
        weather=Weather.CLEAR,
        label=label,
        notes=notes,
        fps=30.0,
        events=events or [],
    )


def _write_video(path, n=30, fps=30.0, size=(64, 48)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n):
        w.write(np.full((size[1], size[0], 3), i * 8 % 255, np.uint8))
    w.release()


DONE_EVENT = {
    "start": 1.0,
    "end": 5.0,
    "label": "distress",
    "contact": 3.0,
    "victim": [
        {"t": 1.0, "x": 10, "y": 20, "kind": "onset"},
        {"t": 2.5, "x": 12, "y": 22, "kind": "last_seen"},
    ],
}


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    raw = tmp_path / "raw_local"
    raw.mkdir()
    monkeypatch.setattr(download, "RAW_LOCAL", raw)
    manifest_path = tmp_path / "manifest.jsonl"
    m = Manifest(manifest_path)
    m.append(_record("todo", events=[{"start": 1.0, "end": 5.0, "label": "distress"}]))
    m.append(_record("done", events=[DONE_EVENT]))
    m.append(_record("calm", label=Label.NORMAL))
    m.append(_record("held", notes="[HOLD indoor]"))
    m.append(_record("novideo"))
    for cid in ("todo", "done", "calm", "held"):
        _write_video(raw / f"{cid}.mp4")
    blocklist = Blocklist(tmp_path / "blocklist.txt")
    app = server.create_app(manifest_path, tmp_path / "detections", blocklist)
    return app.test_client(), manifest_path, tmp_path, blocklist


def _ids(client, which):
    return [c["clip_id"] for c in client.get(f"/api/clips?filter={which}").get_json()]


def test_needs_victim_queue_skips_done_normal_held_and_missing_video(app_env):
    client, *_ = app_env
    assert _ids(client, "needs_victim") == ["todo"]
    assert _ids(client, "all") == ["todo", "done", "calm"]
    rows = {
        c["clip_id"]: c["done"] for c in client.get("/api/clips?filter=all").get_json()
    }
    assert rows == {"todo": False, "done": True, "calm": True}


def test_save_writes_marks_and_keeps_fields_it_does_not_own(app_env):
    client, manifest_path, *_ = app_env
    res = client.post(
        "/api/clip/todo",
        json={
            "label": "distress",
            "trim_start": 0.5,
            "trim_end": 6.0,
            "start_sec": 99.0,  # the download window is not the labeler's to change
            "events": [DONE_EVENT],
        },
    )
    assert res.status_code == 200

    rec = {r.clip_id: r for r in Manifest(manifest_path).load()}["todo"]
    assert rec.events[0]["contact"] == 3.0
    assert [m["kind"] for m in rec.events[0]["victim"]] == ["onset", "last_seen"]
    assert (rec.trim_start, rec.trim_end) == (0.5, 6.0)
    assert (rec.start_sec, rec.end_sec) == (0.0, -1.0)  # download window untouched
    assert rec.fps == 30.0 and rec.camera_view == CameraView.ELEVATED  # untouched
    client_ids = _ids(client, "needs_victim")
    assert "todo" not in client_ids


def _event(manifest_path, cid="todo"):
    return {r.clip_id: r for r in Manifest(manifest_path).load()}[cid].events


def test_clearing_saved_keeps_the_rest_of_the_event(app_env):
    # The old UI dropped an event with no "saved" time, deleting contact and clicks.
    client, manifest_path, *_ = app_env
    half = {**DONE_EVENT, "end": None}
    assert client.post("/api/clip/todo", json={"events": [half]}).status_code == 200
    (ev,) = _event(manifest_path)
    assert ev["end"] is None and ev["contact"] == 3.0 and len(ev["victim"]) == 2
    assert "todo" in _ids(client, "needs_victim")  # unfinished until saved is set


def test_saved_off_camera_counts_as_done(app_env):
    client, manifest_path, *_ = app_env
    off = {**DONE_EVENT, "end": None, "saved_off_camera": True}
    client.post("/api/clip/todo", json={"events": [off]})
    (ev,) = _event(manifest_path)
    assert ev["end"] is None and ev["saved_off_camera"] is True
    assert "todo" not in _ids(client, "needs_victim")


def test_off_camera_flag_dropped_when_a_save_time_exists(app_env):
    client, manifest_path, *_ = app_env
    client.post(
        "/api/clip/todo", json={"events": [{**DONE_EVENT, "saved_off_camera": True}]}
    )
    (ev,) = _event(manifest_path)
    assert ev["end"] == 5.0 and "saved_off_camera" not in ev


def test_event_without_onset_is_kept(app_env):
    client, manifest_path, *_ = app_env
    no_onset = {**DONE_EVENT, "start": None}
    assert client.post("/api/clip/todo", json={"events": [no_onset]}).status_code == 200
    (ev,) = _event(manifest_path)
    assert ev["start"] is None and ev["contact"] == 3.0
    assert "todo" in _ids(client, "needs_victim")


@pytest.mark.parametrize(
    "bad",
    [
        {"start": 5.0, "end": 1.0},  # end before start
        {
            "start": 1.0,
            "end": 5.0,
            "victim": [{"t": 1, "x": 1, "y": 1, "kind": "guess"}],
        },
    ],
)
def test_malformed_events_are_refused_and_nothing_is_written(app_env, bad):
    client, manifest_path, *_ = app_env
    before = manifest_path.read_text()
    res = client.post("/api/clip/todo", json={"events": [bad]})
    assert res.status_code == 400
    assert manifest_path.read_text() == before


@pytest.mark.parametrize(
    "trims",
    [
        {"trim_start": 6.0, "trim_end": 2.0},  # end before start
        {"trim_start": -1.0},  # negative
    ],
)
def test_bad_trims_are_refused_and_nothing_is_written(app_env, trims):
    client, manifest_path, *_ = app_env
    before = manifest_path.read_text()
    assert client.post("/api/clip/todo", json=trims).status_code == 400
    assert manifest_path.read_text() == before


def test_clearing_a_trim_saves_none(app_env):
    client, manifest_path, *_ = app_env
    client.post("/api/clip/todo", json={"trim_start": 1.0, "trim_end": 4.0})
    client.post("/api/clip/todo", json={"trim_start": None, "trim_end": 4.0})
    rec = {r.clip_id: r for r in Manifest(manifest_path).load()}["todo"]
    assert (rec.trim_start, rec.trim_end) == (None, 4.0)


def test_video_supports_range_requests_for_seeking(app_env):
    client, *_ = app_env
    res = client.get("/video/todo", headers={"Range": "bytes=0-99"})
    assert res.status_code == 206 and len(res.data) == 100


def test_clip_payload_reports_video_facts_and_offset(app_env):
    client, *_ = app_env
    c = client.get("/api/clip/todo").get_json()
    assert c["video"]["width"] == 64 and c["video"]["frames"] == 30
    assert c["offset"] == 0.0 and c["has_detections"] is False


def test_boxes_are_merged_per_frame_and_absent_when_not_extracted(app_env):
    client, _, tmp_path, _ = app_env
    assert client.get("/api/boxes/todo?frame=0").get_json() == {
        "available": False,
        "boxes": [],
    }

    folder = tmp_path / "detections" / "todo"
    folder.mkdir(parents=True)

    def det(frame, box, conf, crop):
        row = dict.fromkeys(DET_COLUMNS, 0.5)
        row.update(
            frame=frame,
            crop=crop,
            conf=conf,
            bx1=box[0],
            by1=box[1],
            bx2=box[2],
            by2=box[3],
        )
        return row

    dets = pd.DataFrame(
        [
            det(0, (10, 10, 20, 40), 0.9, 0),
            det(0, (10, 10, 20, 22), 0.6, 1),
            det(1, (30, 5, 40, 30), 0.8, -1),
        ],
        columns=DET_COLUMNS,
    )
    frames = pd.DataFrame({"frame": [0, 1], "n_dets": [2, 1], "scene_diff": [0.0, 0.0]})
    write_chunk(folder, 0, dets, frames)
    (folder / "meta.json").write_text(json.dumps({"chunks_done": 1, "complete": True}))

    f0 = client.get("/api/boxes/todo?frame=0").get_json()
    assert f0["available"] and len(f0["boxes"]) == 1  # half-body duplicate merged away
    assert client.get("/api/boxes/todo?frame=1").get_json()["boxes"][0]["conf"] == 0.8


def test_delete_blocklists_url_and_removes_video(app_env):
    client, manifest_path, tmp_path, blocklist = app_env
    assert client.post("/api/clip/calm/delete").status_code == 200
    assert "calm" not in {r.clip_id for r in Manifest(manifest_path).load()}
    assert not (tmp_path / "raw_local" / "calm.mp4").exists()
    assert blocklist.contains("https://example.com/calm")


def test_offset_comes_from_the_download_window_not_from_trims(tmp_path, monkeypatch):
    # The old labeler guessed "sectioned" from file length vs. start/end, so a small
    # trim on a whole download could flip the offset. Now trims never move it.
    raw = tmp_path / "raw_local"
    raw.mkdir()
    monkeypatch.setattr(download, "RAW_LOCAL", raw)
    m = Manifest(tmp_path / "m.jsonl")
    trimmed = _record("trimmed")
    trimmed.trim_start, trimmed.trim_end = 0.5, 0.9  # trims that fooled the old guess
    m.append(trimmed)
    m.append(_record("section", start=14.0, end=15.0))
    for cid in ("trimmed", "section"):
        _write_video(raw / f"{cid}.mp4")
    client = server.create_app(
        m.path, tmp_path / "d", Blocklist(tmp_path / "b")
    ).test_client()
    assert client.get("/api/clip/trimmed").get_json()["offset"] == 0.0
    assert client.get("/api/clip/section").get_json()["offset"] == 14.0
