"""Head detection on the whole frame (no tiles), on this machine: the Jetson-shaped input.

Same RT-DETR head model and output format as the Kaggle tiled kernel (hk-heads-detect), so
`stack heads-track --heads-dir <out>` and `heads-rows --heads-dir <out>` run on it unchanged:
<out>/raw/<clip>.npz (frame, crop, box, conf; crop is always 0) + <clip>.json (fps, step, ...).
Every round(fps / 10)-th frame, boxes at FLOOR or above, in full-frame pixels.

The frame is resized to `width` (aspect kept, height rounded to a multiple of 32). On one clip
(0168422d9be4, 2026-10-28) a 640-wide pass found 76% of the tiled in-zone heads and boxed the
victim 86% of the time (tiled 93%), at 88 ms per frame on the M1 GPU vs ~1.4 s for 16 tiles; the
misses are mostly heads under 15 px at the far edge. Needs the `heads` extra (transformers 5.21.0:
5.16.1 leaves RT-DETR's last decoder layer untrained).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

FLOOR = 0.05
TARGET_FPS = 10
BATCH = 8


def input_size(w: int, h: int, width: int) -> tuple[int, int]:
    """(width, height) for the model: aspect kept, height a multiple of 32."""
    return width, max(32, int(round(h * width / w / 32)) * 32)


def detect_clip(video: Path, m, proc, out: Path, width: int, dev: str) -> dict:
    import torch

    cid = video.stem
    t0 = time.time()
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, round(fps / TARGET_FPS))
    frames, boxes, confs = [], [], []
    pending: list[tuple[int, np.ndarray]] = []
    W = H = None

    def flush():
        imgs = [p[1] for p in pending]
        x = proc(images=imgs, size={"height": sh, "width": sw}, return_tensors="pt")
        with torch.no_grad():
            o = m(pixel_values=x["pixel_values"].to(dev))
        res = proc.post_process_object_detection(
            o, threshold=FLOOR, target_sizes=[(H, W)] * len(imgs)
        )
        for (f, _), r in zip(pending, res, strict=True):
            n = len(r["scores"])
            frames.append(np.full(n, f, np.int32))
            boxes.append(r["boxes"].cpu().numpy().astype(np.float32))
            confs.append(r["scores"].cpu().numpy().astype(np.float32))
        pending.clear()

    f = read = 0
    while cap.grab():
        if f % step == 0:
            ok, img = cap.retrieve()
            if not ok:
                raise RuntimeError(f"decode failed {cid} frame {f}")
            if W is None:
                H, W = img.shape[:2]
                sw, sh = input_size(W, H, width)
            pending.append((f, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            if len(pending) >= BATCH:
                flush()
            read += 1
        f += 1
    if pending:
        flush()
    cap.release()
    if read == 0:
        raise RuntimeError(f"no frames decoded from {cid}")

    def cat(xs, shape, dt):
        return np.concatenate(xs) if xs else np.zeros(shape, dt)

    fr = cat(frames, (0,), np.int32)
    np.savez_compressed(
        out / f"{cid}.npz",
        frame=fr,
        crop=np.zeros(len(fr), np.int8),
        box=cat(boxes, (0, 4), np.float32),
        conf=cat(confs, (0,), np.float32),
    )
    meta = {
        "clip_id": cid,
        "fps": fps,
        "step": step,
        "frames_decoded": f,
        "frames_detected": read,
        "width": W,
        "height": H,
        "tile": None,
        "input": [sw, sh],
        "floor": FLOOR,
        "grid": [[0, 0]],
        "model": "rtdetr_v7_mix_seed1/last",
        "device": dev,
        "seconds": round(time.time() - t0),
    }
    (out / f"{cid}.json").write_text(json.dumps(meta))
    return meta


def detect_clips(
    clips: list[str] | None,
    model: Path,
    out: Path,
    width: int,
    base: str,
    videos: Path = Path("raw_local"),
) -> None:
    """Whole-frame head boxes for `clips` (default: the base run's clips); resumes per clip."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    if not clips:
        reports = json.loads((Path(base) / "summary.json").read_text())["reports"]
        clips = reports[next(iter(reports))]["used_test"]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(model)
    m = AutoModelForObjectDetection.from_pretrained(model).to(dev).eval()
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for cid in clips:
        if (raw / f"{cid}.npz").exists():
            continue
        meta = detect_clip(videos / f"{cid}.mp4", m, proc, raw, width, dev)
        print(
            f"{cid}: {meta['frames_detected']} frames, {meta['seconds']} s", flush=True
        )
