"""
Generate the privacy-safe pose-comparison figures used in the README.

Backgrounds are heavily pixelated (faces unrecognizable) and crisp pose
skeletons are drawn on top — so the figures show detection quality without
committing identifiable footage, consistent with the project's data ethics.

Reads from a gitignored local clip; writes committable PNGs to docs/images/.
Run: PYTHONPATH=src .venv/bin/python scripts/make_pose_figure.py
"""

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

CLIP = "raw_local/b4b2db5ffb48.mp4"  # a wavepool rescue clip (crowded normal lead-in)
FRAME = 1197
OUT = Path("docs/images")
MP_MODEL = "raw_local/pose_landmarker.task"


def pixelate(img, blocks=48):
    h, w = img.shape[:2]
    small = cv2.resize(
        img, (blocks, max(1, int(blocks * h / w))), interpolation=cv2.INTER_LINEAR
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(img, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def draw_mp(img, poses, conns):
    h, w = img.shape[:2]
    for lms in poses:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
        for c in conns:
            cv2.line(img, pts[c.start], pts[c.end], (0, 255, 0), 2)
        for p in pts:
            cv2.circle(img, p, 3, (0, 0, 255), -1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(CLIP)
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {FRAME} from {CLIP}")

    bg = pixelate(frame)

    yolo = YOLO("yolo11n-pose.pt")
    r640 = yolo(frame, imgsz=640, verbose=False)[0]
    r1280 = yolo(frame, imgsz=1280, verbose=False)[0]

    base = mp_python.BaseOptions(model_asset_path=MP_MODEL)
    opts = vision.PoseLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=10,
        min_pose_detection_confidence=0.3,
    )
    lm = vision.PoseLandmarker.create_from_options(opts)
    conns = vision.PoseLandmarksConnections.POSE_LANDMARKS
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mpres = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    lm.close()

    # Figure 1 — resolution: YOLO @640 vs @1280 (same frame, same pixelated bg)
    a = r640.plot(img=bg.copy(), boxes=False, labels=False)
    label(a, f"YOLO @ 640px: {len(r640.boxes)} swimmers")
    b = r1280.plot(img=bg.copy(), boxes=False, labels=False)
    label(b, f"YOLO @ 1280px: {len(r1280.boxes)} swimmers")
    cv2.imwrite(str(OUT / "pose_resolution.png"), np.hstack([a, b]))

    # Figure 2 — model: YOLO vs MediaPipe (same frame, both @ best settings)
    c = r1280.plot(img=bg.copy(), boxes=False, labels=False)
    label(c, f"YOLO-pose: {len(r1280.boxes)} swimmers")
    d = bg.copy()
    draw_mp(d, mpres.pose_landmarks, conns)
    label(d, f"MediaPipe: {len(mpres.pose_landmarks)} swimmers")
    cv2.imwrite(str(OUT / "pose_model.png"), np.hstack([c, d]))

    print(f"wrote {OUT / 'pose_resolution.png'} and {OUT / 'pose_model.png'}")


if __name__ == "__main__":
    main()
