"""
Pose extraction (Rung 1): video -> per-swimmer keypoint time-series (Parquet).

Runs YOLO-pose with tracking over a clip. Every detected swimmer in every
frame becomes one row: frame index, track id, box confidence, and the 17
COCO keypoints as (x, y, confidence). That table is the training data — group
by track_id, sort by frame, and you have each swimmer's motion over time.

Non-tiled version (SAHI tiling is a later upgrade). Model is a parameter so
the YOLO version/size is swappable without code changes.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from trackers import ByteTrackTracker
from ultralytics import YOLO

from .tiled_pose import detect_tiled

# 17 COCO keypoints, in the order YOLO returns them.
KEYPOINT_NAMES = [
    "nose",
    "l_eye",
    "r_eye",
    "l_ear",
    "r_ear",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "l_hip",
    "r_hip",
    "l_knee",
    "r_knee",
    "l_ankle",
    "r_ankle",
]

# Column layout of the output table: metadata first, then x/y/conf per keypoint.
COLUMNS = ["frame", "track_id", "box_conf"] + [
    f"{axis}{i}" for i in range(17) for axis in ("x", "y", "c")
]


def extract(
    video_path: Path,
    out_path: Path,
    model_name: str = "yolo11n-pose.pt",
    imgsz: int = 1280,
    conf: float = 0.20,
    tracker: str = "bytetrack.yaml",
    device: str | None = None,
    max_frames: int | None = None,
) -> int:
    """Extract pose keypoints from one clip into a Parquet file (whole-frame YOLO + tracking).

    Args:
        video_path: input clip.
        out_path: destination .parquet (parent dirs created).
        model_name: YOLO-pose weights (version/size is swappable).
        imgsz: inference resolution (>=1280 was the biggest recall lever).
        conf: detection confidence threshold.
        tracker: Ultralytics tracker config (e.g. bytetrack.yaml).
        device: torch device, or None to let Ultralytics choose.
        max_frames: stop after this many frames (None = whole clip; never cap for real runs).
    Returns:
        Number of rows (swimmer-detections) written.
    """
    video_path = Path(video_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)

    # model.track(... stream=True) yields one Results object per frame, while
    # maintaining a persistent track_id per swimmer across frames (the tracker).
    # stream=True processes frames lazily so we don't hold the whole video in RAM.
    results = model.track(
        source=str(video_path),
        stream=True,
        imgsz=imgsz,
        conf=conf,
        tracker=tracker,
        device=device,
        verbose=False,
    )

    rows: list[list[float]] = []
    for frame_idx, result in enumerate(results):
        if max_frames is not None and frame_idx >= max_frames:
            break

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue  # no swimmers this frame

        # Track IDs: present once the tracker has locked on. If a detection has
        # no id yet (e.g. first sighting), fall back to -1.
        ids = boxes.id.int().tolist() if boxes.id is not None else [-1] * len(boxes)
        confs = boxes.conf.tolist()

        # keypoints.data is a tensor (n_people, 17, 3) of (x, y, confidence).
        kpts = result.keypoints.data.cpu().numpy()

        for p in range(len(boxes)):
            row = [frame_idx, ids[p], float(confs[p])]
            row.extend(kpts[p].flatten().tolist())  # 17*(x,y,c) = 51 values
            rows.append(row)

    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_parquet(out_path, index=False)

    n_tracks = df["track_id"].nunique() if len(df) else 0
    print(
        f"{video_path.name}: {len(df)} detections across "
        f"{df['frame'].nunique() if len(df) else 0} frames, "
        f"{n_tracks} unique tracks -> {out_path}"
    )
    return len(df)


def extract_tiled(
    video_path: Path,
    out_path: Path,
    model_name: str = "yolo11n-pose.pt",
    tile: int = 480,
    imgsz: int = 1280,
    overlap: float = 0.25,
    conf: float = 0.25,
    max_frames: int | None = None,
) -> int:
    """SAHI extraction: tiled detection + ByteTrack -> Parquet (same schema as extract()).

    Recovers far more distant swimmers via tiling, but much slower (many inferences per frame) —
    GPU territory for full clips. ByteTrack runs on the merged tiled detections (YOLO's built-in
    tracker can't); a keypoint index is carried through so tracked boxes map back to keypoints
    (track_id -1 = not yet confirmed).
    Args:
        video_path: input clip.
        out_path: destination .parquet.
        model_name: YOLO-pose weights.
        tile: tile size in pixels.
        imgsz: per-tile inference resolution (must exceed `tile` for SAHI to help).
        overlap: fractional tile overlap.
        conf: detection confidence threshold.
        max_frames: stop after this many frames (None = whole clip).
    Returns:
        Number of rows (swimmer-detections) written.
    """
    video_path = Path(video_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)
    tracker = ByteTrackTracker(
        track_activation_threshold=0.3, minimum_consecutive_frames=2
    )
    cap = cv2.VideoCapture(str(video_path))

    rows: list[list[float]] = []
    frame_idx = 0
    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break

        dets = detect_tiled(
            model, frame, tile=tile, overlap=overlap, imgsz=imgsz, conf=conf
        )
        if dets:
            detections = sv.Detections(
                xyxy=np.array([d[0] for d in dets], dtype=float),
                confidence=np.array([d[1] for d in dets], dtype=float),
                class_id=np.zeros(len(dets), dtype=int),
            )
            detections.data["kp_idx"] = np.arange(len(dets))  # map tracked -> keypoints
        else:
            detections = sv.Detections.empty()

        tracked = tracker.update(detections, frame)
        for j in range(len(tracked)):
            kpts = dets[int(tracked.data["kp_idx"][j])][2]
            row = [frame_idx, int(tracked.tracker_id[j]), float(tracked.confidence[j])]
            row.extend(kpts.flatten().tolist())
            rows.append(row)
        frame_idx += 1

    cap.release()
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_parquet(out_path, index=False)
    print(
        f"{video_path.name} (tiled): {len(df)} detections across {frame_idx} "
        f"frames, {df['track_id'].nunique() if len(df) else 0} tracks -> {out_path}"
    )
    return len(df)


if __name__ == "__main__":
    import sys

    video = Path(sys.argv[1])
    tiled = "--tiled" in sys.argv
    out = Path("data/keypoints") / f"{video.stem}.parquet"
    (extract_tiled if tiled else extract)(video, out)
