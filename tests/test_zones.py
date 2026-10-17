"""
Tests for coverage zones (annotate/zones.py + the labeler's /api/zones routes).

These pin the point-in-polygon rule evaluation will rely on, that bad polygons are
refused instead of written, and that a saved zone round-trips through the labeler.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from hydro_knight.annotate import server, zones
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

SQUARE = {"points": [[10, 10], [110, 10], [110, 110], [10, 110]], "camera_moves": False}


def test_in_zone_square_and_concave_shape():
    assert zones.in_zone(SQUARE, [50, 5, 120, 109.9], [50, 50, 50, 109.9]).tolist() == [
        True,
        False,
        False,
        True,
    ]
    # an L shape: the notch at top right is outside
    ell = {"points": [[0, 0], [50, 0], [50, 50], [100, 50], [100, 100], [0, 100]]}
    assert zones.in_zone(ell, [25, 75, 75], [25, 25, 75]).tolist() == [
        True,
        False,
        True,
    ]


def test_no_zone_or_moving_camera_contains_everything():
    assert zones.in_zone(None, [1, 999], [1, 999]).all()
    assert zones.in_zone({"points": [], "camera_moves": True}, [5], [5]).all()


@pytest.mark.parametrize(
    "body",
    [
        {"points": [[0, 0], [1, 1]], "frame": 0},  # 2 points on a fixed camera
        {"points": [[0, 0], [1, 1], [-1, 2]], "frame": 0},  # negative
        {"points": [[0, 0], [1, 1], [2, float("nan")]], "frame": 0},
        {"points": [[0, 0], [1, 1], [2, 2]], "frame": -3},
        {"points": [[0, 0], [1, 1], [2, 2]], "frame": True},
        [1, 2, 3],
    ],
)
def test_bad_zones_are_refused(body):
    with pytest.raises(ValueError):
        zones.clean_zone(body)


def test_moving_camera_needs_no_points():
    z = zones.clean_zone({"points": [], "frame": 4, "camera_moves": True})
    assert z["camera_moves"] and z["points"] == [] and z["frame"] == 4


@pytest.fixture
def client(tmp_path, monkeypatch):
    raw = tmp_path / "raw_local"
    raw.mkdir()
    monkeypatch.setattr(download, "RAW_LOCAL", raw)
    rec = ClipRecord(
        clip_id="aaa",
        source_url="u",
        platform="youtube",
        start_sec=0.0,
        end_sec=-1.0,
        camera_view=CameraView.ELEVATED,
        setting=Setting.OUTDOOR,
        time_of_day=TimeOfDay.DAY,
        weather=Weather.CLEAR,
        label=Label.DISTRESS,
        fps=10.0,
        events=[{"start": 1.0, "end": 3.0, "label": "distress", "contact": 2.0}],
    )
    Manifest(tmp_path / "manifest.jsonl").append(rec)
    vw = cv2.VideoWriter(
        str(raw / "aaa.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48)
    )
    for _ in range(30):
        vw.write(np.zeros((48, 64, 3), np.uint8))
    vw.release()
    zones_path = tmp_path / "zones.json"
    app = server.create_app(
        tmp_path / "manifest.jsonl",
        tmp_path / "detections",
        Blocklist(tmp_path / "blocklist.txt"),
        zones_path=zones_path,
    )
    return app.test_client(), zones_path


def test_zone_round_trips_and_clears(client):
    c, path = client
    row = c.get("/api/zones").get_json()[0]
    assert row["clip_id"] == "aaa" and row["zone"] is None
    assert row["default_frame"] == 10  # event onset, 1.0 s at 10 fps
    assert c.post("/api/zones/aaa", json=SQUARE | {"frame": 10}).status_code == 200
    assert zones.load(path)["aaa"]["points"][1] == [110.0, 10.0]
    assert c.get("/api/zones").get_json()[0]["zone"]["frame"] == 10
    assert c.post("/api/zones/aaa", json={"clear": True}).status_code == 200
    assert "aaa" not in zones.load(path)


def test_bad_zone_is_refused_and_nothing_written(client):
    c, path = client
    assert (
        c.post("/api/zones/aaa", json={"points": [[1, 1]], "frame": 0}).status_code
        == 400
    )
    assert c.post("/api/zones/nope", json=SQUARE | {"frame": 0}).status_code == 404
    assert zones.load(path) == {}


def test_frame_image_served(client):
    c, _ = client
    res = c.get("/api/zones/aaa/image?frame=5")
    assert res.status_code == 200 and res.mimetype == "image/jpeg"
    assert c.get("/api/zones/aaa/image?frame=999").status_code == 404
