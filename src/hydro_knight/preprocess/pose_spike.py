"""
Rung 1 spike: compare pose backends on the same clip.

Generalized comparison tool. A "backend" is a config — a YOLO model of any
version/size/settings, or MediaPipe. For each sampled frame it runs every
backend, draws the detected skeletons, and saves an N-up side-by-side PNG so
you can eyeball which finds the swimmers. This is what captures (reproducibly)
the yolo11-vs-yolo26-vs-MediaPipe experiments.

  backend spec = {"kind": "yolo", "model": "yolo11n-pose.pt", "imgsz": 1280, "conf": 0.25}
               | {"kind": "mediapipe", "conf": 0.3, "num_poses": 10}
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

MP_MODEL = Path("raw_local/pose_landmarker.task")

# Default comparison: the three backends we evaluated. Editing this list (or
# passing your own) is how you compare any models/settings reproducibly.
DEFAULT_BACKENDS = [
    {"kind": "yolo", "model": "yolo11n-pose.pt", "imgsz": 1280, "conf": 0.25},
    {"kind": "yolo", "model": "yolo26n-pose.pt", "imgsz": 1280, "conf": 0.25},
    {"kind": "mediapipe", "conf": 0.3, "num_poses": 10},
]


def _label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def _resize_h(img, target_h):
    scale = target_h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * scale), target_h))


def _pixelate(img, blocks=48):
    h, w = img.shape[:2]
    small = cv2.resize(
        img, (blocks, max(1, int(blocks * h / w))), interpolation=cv2.INTER_LINEAR
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _draw_mp_pose(img, poses, connections):
    h, w = img.shape[:2]
    for landmarks in poses:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for c in connections:
            cv2.line(img, pts[c.start], pts[c.end], (0, 255, 0), 2)
        for p in pts:
            cv2.circle(img, p, 3, (0, 0, 255), -1)


def _make_runner(spec):
    """
    Turn one backend spec into (label, run_fn). run_fn(frame, bg) detects on the
    full-quality `frame` and draws results onto `bg`, returning (image, count).
    Models/detectors are created once here, not per frame.
    """
    if spec["kind"] == "yolo":
        model = YOLO(spec["model"])
        imgsz, conf = spec.get("imgsz", 1280), spec.get("conf", 0.25)
        label = spec.get("label", f"{spec['model'].split('-')[0]} @{imgsz} c{conf}")

        def run(frame, bg, _m=model, _i=imgsz, _c=conf):
            r = _m(frame, imgsz=_i, conf=_c, verbose=False)[0]
            return r.plot(img=bg.copy(), boxes=False, labels=False), len(r.boxes)

        return label, run

    if spec["kind"] == "mediapipe":
        base = mp_python.BaseOptions(model_asset_path=str(MP_MODEL))
        opts = vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=spec.get("num_poses", 10),
            min_pose_detection_confidence=spec.get("conf", 0.3),
        )
        landmarker = vision.PoseLandmarker.create_from_options(opts)
        conns = vision.PoseLandmarksConnections.POSE_LANDMARKS
        label = spec.get("label", "MediaPipe")

        def run(frame, bg, _lm=landmarker, _conns=conns):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = _lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            img = bg.copy()
            _draw_mp_pose(img, res.pose_landmarks, _conns)
            return img, len(res.pose_landmarks)

        return label, run

    raise ValueError(f"unknown backend kind: {spec['kind']}")


def compare(
    video_path: Path,
    out_dir: Path,
    backends=None,
    n_frames: int = 8,
    pixelate: bool = False,
) -> None:
    """
    Run every backend on n_frames sampled across the clip; save N-up PNGs.

    pixelate=True blurs the background (faces unrecognizable) for shareable
    output; default False since output goes to gitignored raw_local/.
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    backends = backends if backends is not None else DEFAULT_BACKENDS
    runners = [_make_runner(s) for s in backends]

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total - 1, n_frames, dtype=int)

    for i, idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue

        bg = _pixelate(frame) if pixelate else frame.copy()

        panels, counts = [], []
        for label, run in runners:
            img, count = run(frame, bg)
            _label(img, f"{label}: {count}")
            panels.append(img)
            counts.append(f"{label}={count}")

        h = min(p.shape[0] for p in panels)
        combined = np.hstack([_resize_h(p, h) for p in panels])

        out_path = out_dir / f"compare_{i:02d}_frame{int(idx)}.png"
        cv2.imwrite(str(out_path), combined)
        print(f"[{i + 1}/{len(frame_indices)}] frame {idx}: " + "  ".join(counts))

    cap.release()
    print(f"\nDone. Open the PNGs in {out_dir} to compare.")


if __name__ == "__main__":
    import sys

    video = Path(sys.argv[1])
    out = Path("raw_local/pose_spike_out") / video.stem
    compare(video, out, n_frames=8)
