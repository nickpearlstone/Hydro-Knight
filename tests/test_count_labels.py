"""
Tests for the labeler's Swimmer count tab (annotate/count.py + its /api/count routes).

These pin that clicks round-trip to the label file, that malformed points are refused
instead of written, and that frame images come from the raw_local cache.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from hydro_knight.annotate import count, server
from hydro_knight.ingest import download
from hydro_knight.ingest.blocklist import Blocklist
from hydro_knight.ingest.manifest import Manifest

FRAMES = [
    {"clip_id": "aaa", "frame": 5, "label": "distress", "water": [], "deck": []},
    {"clip_id": "bbb", "frame": 9, "label": "normal", "water": [[1, 2]], "deck": []},
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    raw = tmp_path / "raw_local"
    monkeypatch.setattr(download, "RAW_LOCAL", raw)
    for f in FRAMES:  # cached frame images, so no video is needed
        path = count.frame_path(f)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.zeros((48, 64, 3), np.uint8))
    count_path = tmp_path / "gt.json"
    count.save({"frames": FRAMES}, count_path)
    Manifest(tmp_path / "manifest.jsonl")
    app = server.create_app(
        tmp_path / "manifest.jsonl",
        tmp_path / "detections",
        Blocklist(tmp_path / "blocklist.txt"),
        count_path=count_path,
    )
    return app.test_client(), count_path


def test_save_round_trips_and_leaves_other_frames_alone(client):
    c, path = client
    res = c.post(
        "/api/count/0", json={"water": [[10.04, 20], [30, 40]], "deck": [[5, 6]]}
    )
    assert res.status_code == 200
    frames = count.load(path)["frames"]
    assert frames[0]["water"] == [[10.0, 20.0], [30.0, 40.0]]
    assert frames[0]["deck"] == [[5.0, 6.0]]
    assert frames[0]["clip_id"] == "aaa" and frames[0]["frame"] == 5
    assert frames[1] == FRAMES[1]
    assert c.get("/api/count").get_json()[0]["water"] == [[10.0, 20.0], [30.0, 40.0]]


@pytest.mark.parametrize(
    "body",
    [
        {"water": [[1, 2]]},  # missing deck list
        {"water": [[1]], "deck": []},  # not a pair
        {"water": [[-1, 2]], "deck": []},  # negative
        {"water": [["nan", 2]], "deck": []},  # not finite
    ],
)
def test_bad_points_are_refused_and_nothing_is_written(client, body):
    c, path = client
    before = path.read_text()
    assert c.post("/api/count/0", json=body).status_code == 400
    assert path.read_text() == before


def test_image_served_from_cache_and_unknown_frame_404s(client):
    c, _ = client
    res = c.get("/api/count/1/image")
    assert res.status_code == 200 and res.mimetype == "image/jpeg"
    assert c.get("/api/count/7/image").status_code == 404
    assert c.post("/api/count/7", json={"water": [], "deck": []}).status_code == 404
