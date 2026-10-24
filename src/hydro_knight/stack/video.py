"""Check video for the ghost piece: what the stack's memory sees, drawn on the real clip.

Each 0.5 s row is held on screen until the next one:
- visible swimmers: a small grey dot on the head (the victim's rows: a white ring);
- ghosts: an orange circle the size of the claim radius (half a body length) where the head
  was last seen, labelled with seconds unseen and the dunk count near that spot;
- corner: clip time, and onset / guard-contact markers.

For Nick to time by hand against the video. Written under runs/ (gitignored) and never
committed: the footage shows identifiable people, often minors.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .data import VIDEOS

ORANGE, GREY, WHITE = (0, 140, 255), (170, 170, 170), (255, 255, 255)


def render(
    rows: pd.DataFrame,
    meta: dict,
    clip_id: str,
    out: Path,
    kill_l: float = 0.5,
    pre_s: float = 10.0,
    post_s: float = 2.0,
    width: int = 1280,
) -> Path | None:
    """Write the check video for one clip, from onset - pre_s to contact + post_s."""
    if shutil.which("ffmpeg") is None:
        return None
    r = rows[rows["clip_id"] == clip_id]
    fps = meta["fps"]
    t0 = max(meta["start"], meta["onset"] - pre_s)
    t1 = meta["contact"] + post_s
    cap = cv2.VideoCapture(str(VIDEOS / f"{clip_id}.mp4"))
    W0, H0 = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    sc = width / W0
    size = (width, int(round(H0 * sc / 2)) * 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{size[0]}x{size[1]}", "-r", f"{fps:.3f}", "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26",
            "-movflags", "+faststart", str(out),
        ],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    times = np.sort(r["t"].unique())
    by_t = {t: g for t, g in r.groupby("t")}
    f0, f1 = int(round(t0 * fps)), int(round(t1 * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    for f in range(f0, f1 + 1):
        ok, img = cap.read()
        if not ok:
            break
        t = f / fps
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        k = np.searchsorted(times, t, side="right") - 1
        if k >= 0:
            _draw(img, by_t[times[k]], sc, kill_l)
        phase = (
            "before onset"
            if t < meta["onset"]
            else ("ONSET -> contact" if t <= meta["contact"] else "after contact")
        )
        cv2.putText(
            img,
            f"t={t:6.2f}s  {phase}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            WHITE,
            2,
            cv2.LINE_AA,
        )
        proc.stdin.write(np.ascontiguousarray(img).tobytes())
    cap.release()
    proc.stdin.close()
    proc.wait()
    return out if proc.returncode == 0 else None


def _draw(img, g: pd.DataFrame, sc: float, kill_l: float) -> None:
    for row in g.itertuples():
        x, y = int(row.x * sc), int(row.y * sc)
        if row.kind == 1:
            rad = max(4, int(kill_l * row.h * sc))
            cv2.circle(img, (x, y), rad, ORANGE, 2, cv2.LINE_AA)
            label = f"{row.unseen_s:.1f}s d{int(row.dunks_30s)}"
            cv2.putText(
                img,
                label,
                (x + rad + 2, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                ORANGE,
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.circle(img, (x, y), 3, GREY, -1, cv2.LINE_AA)
        if row.victim:
            cv2.circle(
                img, (x, y), max(8, int(0.6 * row.h * sc)), WHITE, 1, cv2.LINE_AA
            )
