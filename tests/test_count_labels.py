"""
Tests for the labeler's Swimmer count tab (annotate/count.py + its /api/count routes).

These pin that clicks round-trip to the label file, that malformed points are refused
instead of written, and that frame images come from the raw_local cache. Head boxes and
the second frame set (?set=val) follow the same rules.
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
VAL_FRAMES = [
    {
        "clip_id": "ccc",
        "frame": 3,
        "label": "distress",
        "water": [],
        "deck": [],
        "heads": {"water": [[1, 1, 5, 5]], "deck": [], "no_head": []},
        "heads_reviewed": False,
        "heads_source": "model",
    }
]
HEADS = {"water": [[10.04, 20, 18, 29]], "deck": [[1, 1, 4, 4]], "no_head": [[7, 8]]}


@pytest.fixture
def client(tmp_path, monkeypatch):
    raw = tmp_path / "raw_local"
    monkeypatch.setattr(download, "RAW_LOCAL", raw)
    for f in FRAMES + VAL_FRAMES:  # cached frame images, so no video is needed
        path = count.frame_path(f)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.zeros((48, 64, 3), np.uint8))
    count_path = tmp_path / "gt.json"
    count.save({"frames": FRAMES}, count_path)
    count.save({"frames": VAL_FRAMES}, tmp_path / "val.json")
    Manifest(tmp_path / "manifest.jsonl")
    app = server.create_app(
        tmp_path / "manifest.jsonl",
        tmp_path / "detections",
        Blocklist(tmp_path / "blocklist.txt"),
        count_path=count_path,
        head_val_path=tmp_path / "val.json",
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


def test_head_boxes_and_reviewed_flag_round_trip(client):
    c, path = client
    body = {"water": [[1, 2]], "deck": [], "heads": HEADS, "heads_reviewed": True}
    assert c.post("/api/count/0", json=body).status_code == 200
    f = count.load(path)["frames"][0]
    assert f["heads"] == {
        "water": [[10.0, 20.0, 18.0, 29.0]],
        "deck": [[1.0, 1.0, 4.0, 4.0]],
        "no_head": [[7.0, 8.0]],
    }
    assert f["heads_reviewed"] is True
    assert f["water"] == [[1.0, 2.0]]  # the dots are saved alongside, unchanged


def test_dots_only_save_keeps_existing_head_boxes(client):
    c, path = client
    c.post("/api/count/0", json={"water": [], "deck": [], "heads": HEADS})
    c.post("/api/count/0", json={"water": [[3, 4]], "deck": []})
    assert count.load(path)["frames"][0]["heads"]["no_head"] == [[7.0, 8.0]]


@pytest.mark.parametrize(
    "heads",
    [
        {"water": [[5, 5, 1, 9]], "deck": [], "no_head": []},  # x1 > x2
        {"water": [[1, 1, 5]], "deck": [], "no_head": []},  # not four numbers
        {"water": [[-1, 1, 5, 5]], "deck": [], "no_head": []},  # negative
        {"water": [], "deck": []},  # missing no_head list
        [[1, 1, 5, 5]],  # not an object
    ],
)
def test_bad_head_boxes_are_refused_and_nothing_is_written(client, heads):
    c, path = client
    before = path.read_text()
    res = c.post("/api/count/0", json={"water": [], "deck": [], "heads": heads})
    assert res.status_code == 400
    assert path.read_text() == before


def test_reviewed_flag_must_be_a_boolean(client):
    c, path = client
    body = {"water": [], "deck": [], "heads_reviewed": "yes"}
    assert c.post("/api/count/0", json=body).status_code == 400


def test_validation_set_reads_and_writes_its_own_file(client, tmp_path):
    c, path = client
    bench_before = path.read_text()
    assert c.get("/api/count?set=val").get_json()[0]["clip_id"] == "ccc"
    assert c.get("/api/count/0/image?set=val").status_code == 200
    body = {"water": [], "deck": [], "heads": HEADS, "heads_reviewed": True}
    assert c.post("/api/count/0?set=val", json=body).status_code == 200
    val = count.load(tmp_path / "val.json")["frames"][0]
    assert val["heads_reviewed"] is True and val["heads_source"] == "model"
    assert path.read_text() == bench_before
    assert c.get("/api/count?set=nope").status_code == 400


def test_missing_validation_file_404s_with_the_command_to_run(client, tmp_path):
    c, _ = client
    (tmp_path / "val.json").unlink()
    res = c.get("/api/count?set=val")
    assert res.status_code == 404
    assert b"hydroknight heads frames" in res.data


def test_hidden_dots_round_trip_and_are_optional(client):
    c, path = client
    body = {"water": [[1, 2]], "deck": [], "hidden": [[5.04, 6]]}
    assert c.post("/api/count/0", json=body).status_code == 200
    assert count.load(path)["frames"][0]["hidden"] == [[5.0, 6.0]]
    # an older client that doesn't send "hidden" leaves the saved list alone
    assert c.post("/api/count/0", json={"water": [], "deck": []}).status_code == 200
    assert count.load(path)["frames"][0]["hidden"] == [[5.0, 6.0]]


def test_bad_hidden_dot_is_refused(client):
    c, path = client
    before = path.read_text()
    body = {"water": [], "deck": [], "hidden": [[-1, 2]]}
    assert c.post("/api/count/0", json=body).status_code == 400
    assert path.read_text() == before


def test_ignored_is_deck_plus_hidden():
    assert count.ignored({"deck": [[1, 1]], "hidden": [[2, 2]]}) == [[1, 1], [2, 2]]
    assert count.ignored({"deck": [[1, 1]]}) == [[1, 1]]


def test_head_truth_uses_reviewed_boxes_else_dots():
    dots = {"water": [[1, 1]], "deck": [[2, 2]], "hidden": [[3, 3]]}
    assert count.head_truth(dots) == ([[1, 1]], [[2, 2], [3, 3]])
    boxed = dots | {
        "heads": {
            "water": [[0, 0, 4, 2]],
            "deck": [[10, 10, 12, 12]],
            "no_head": [[7, 7]],
        },
        "heads_reviewed": True,
    }
    assert count.head_truth(boxed) == ([[2.0, 1.0]], [[11.0, 11.0], [7, 7]])
    unreviewed = boxed | {"heads_reviewed": False}
    assert count.head_truth(unreviewed) == count.head_truth(dots)


def test_pages_are_never_cached(client):
    c, _ = client
    assert c.get("/count").headers["Cache-Control"] == "no-store"
