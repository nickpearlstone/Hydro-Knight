"""
Short demo video of a trained model on held-out footage: every swimmer's box coloured by
its anomaly score (green = normal, red = at or past the alarm line, grey = not scored),
for the best and the worst held-out clip by within-clip AUC, from a few seconds before
drowning onset to just after guard contact.

Causal view: each box shows the latest score whose window ended at or before the current
moment (what a live system would show): a colour describes the past 3 s window.
The victim's boxes carry a "victim" tag; banners mark onset and guard contact.

The output shows identifiable people (often minors). It is for local viewing only: it
is written under runs/ and attached to local MLflow runs, both gitignored, and must never
be committed or published (anonymised figures go through scripts/make_pose_figure.py).
Needs ffmpeg on PATH (H.264 output that browsers and the MLflow UI can play).
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import protocol as P

STALE_S = 1.0  # a score older than this is not shown (box goes grey)
OUT_FPS = 30
WHITE, GREY, BLACK = (255, 255, 255), (150, 150, 150), (0, 0, 0)


def score_scale(
    row: dict, scores: np.ndarray, fa_target: float = 120
) -> tuple[float, float]:
    """(low, high) of the colour scale: median normal score, and the alarm threshold at `fa_target` FA/h.

    Falls back to the 50th and 99th percentile of `scores` when the run has no such value.
    """
    s = np.asarray(scores, float)
    s = s[np.isfinite(s) & (s > 0)]
    lo = row.get("test_median") or (float(np.median(s)) if len(s) else 1.0)
    hi = next(
        (
            d["threshold"]
            for d in row.get("detection", [])
            if d.get("fa_target") == fa_target and d.get("threshold") is not None
        ),
        None,
    )
    if hi is None or not np.isfinite(hi):
        hi = float(np.percentile(s, 99)) if len(s) else lo * 2
    return float(lo), float(max(hi, lo * 1.01))


def score_color(score: float, lo: float, hi: float) -> tuple[int, int, int]:
    """BGR colour on a log scale from green (score <= lo) to red (score >= hi)."""
    if not (score > 0):
        return GREY
    u = float(
        np.clip((math.log(score) - math.log(lo)) / (math.log(hi) - math.log(lo)), 0, 1)
    )
    hue = int(round(60 * (1 - u)))  # OpenCV hue 60 = green, 0 = red
    b, g, r = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def pick_best_worst(auc_by_clip: dict[str, float]) -> list[tuple[str, str, float]]:
    """[("best", clip, auc), ("worst", clip, auc)] over clips with a finite AUC; one entry if only one clip."""
    ok = {c: a for c, a in auc_by_clip.items() if a is not None and np.isfinite(a)}
    if not ok:
        return []
    best = max(ok, key=lambda c: (ok[c], c))
    worst = min(ok, key=lambda c: (ok[c], c))
    out = [("best", best, ok[best])]
    if worst != best:
        out.append(("worst", worst, ok[worst]))
    return out


def clip_window(
    truth: P.EventTruth, pre_s: float = 5.0, post_s: float = 3.0
) -> tuple[float, float]:
    """(start, end) in file seconds: pre_s before onset (not before the trim) to post_s after contact."""
    return max(truth.start, truth.onset_lo - pre_s), truth.contact + post_s


def _text(img, txt, org, scale=0.55, color=WHITE, thick=1):
    cv2.putText(
        img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA
    )


def _legend(img, lo, hi):
    h_img, w_img = img.shape[:2]
    w, h = 220, 10
    x0, y0 = w_img - w - 20, 14
    cv2.rectangle(img, (x0 - 8, y0 - 6), (x0 + w + 8, y0 + h + 40), BLACK, -1)
    for i in range(w):
        s = math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * i / (w - 1))
        cv2.line(img, (x0 + i, y0), (x0 + i, y0 + h), score_color(s, lo, hi), 1)
    _text(img, "normal", (x0, y0 + h + 15), 0.42)
    _text(img, "alarm line", (x0 + w - 70, y0 + h + 15), 0.42)
    _text(img, "grey = not scored", (x0, y0 + h + 32), 0.4, (180, 180, 180))


def _banner(img, now, truth):
    """Phase banner at the bottom: time from onset, and which phase of the rescue this is."""
    dt = now - truth.onset
    if now < truth.onset_lo:
        txt, col = f"before onset   {dt:+.1f} s", (200, 200, 200)
    elif now < truth.contact:
        txt, col = (
            f"DROWNING ONSET {dt:+.1f} s   (guard contact in {truth.contact - now:.1f} s)",
            (60, 60, 255),
        )
    else:
        txt, col = f"GUARD CONTACT   {dt:+.1f} s from onset", WHITE
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - 34), (w, h), BLACK, -1)
    _text(img, txt, (12, h - 11), 0.6, col, 1)


def _title_card(write, size, lines, seconds=2.5):
    img = np.zeros((size[1], size[0], 3), np.uint8)
    for i, ln in enumerate(lines):
        _text(img, ln, (40, size[1] // 2 - 40 + 40 * i), 0.8, WHITE, 2 if i == 0 else 1)
    for _ in range(int(OUT_FPS * seconds)):
        write(img)


def _render_clip(
    write, size, cid, truth, alarms, keypoints_dir, videos_dir, lo, hi, pre_s, post_s
):
    cap = cv2.VideoCapture(str(Path(videos_dir) / f"{cid}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sx, sy = (
        size[0] / cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        size[1] / cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
    )
    t0, t1 = clip_window(truth, pre_s, post_s)
    f = int(t0 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    kp = pd.read_parquet(
        Path(keypoints_dir) / f"{cid}.parquet",
        columns=["frame", "track_id", "bx1", "by1", "bx2", "by2"],
    )
    kp = kp[(kp["frame"] >= f) & (kp["frame"] <= t1 * fps + 1) & (kp["track_id"] >= 0)]
    by_frame = dict(tuple(kp.groupby("frame")))
    a = alarms[alarms["clip_id"] == cid].sort_values("t")
    per_track = {
        k: (g["t"].to_numpy(), g["score"].to_numpy()) for k, g in a.groupby("track_id")
    }
    step, f_next = fps / OUT_FPS, float(f)
    while True:
        ok, frame = cap.read()
        if not ok or f / fps > t1:
            break
        if f + 1e-6 < f_next:  # downsample 60 fps sources to OUT_FPS
            f += 1
            continue
        f_next += step
        now = f / fps
        img = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        rows = by_frame.get(f)
        if rows is not None:
            for r in rows.itertuples():
                p1, p2 = (
                    (int(r.bx1 * sx), int(r.by1 * sy)),
                    (int(r.bx2 * sx), int(r.by2 * sy)),
                )
                c, th = GREY, 1
                if r.track_id in per_track:
                    ts, ss = per_track[r.track_id]
                    i = np.searchsorted(ts, now, side="right") - 1
                    if i >= 0 and now - ts[i] <= STALE_S:
                        c, th = score_color(ss[i], lo, hi), (3 if ss[i] >= hi else 2)
                cv2.rectangle(img, p1, p2, c, th)
                if r.track_id in truth.victim_tracks:
                    _text(img, "victim", (p1[0], max(12, p1[1] - 5)), 0.45, WHITE, 1)
        cv2.rectangle(img, (0, 0), (260, 28), BLACK, -1)
        _text(img, f"{cid}  t={now:5.1f}s", (8, 19), 0.5)
        _legend(img, lo, hi)
        _banner(img, now, truth)
        write(img)
        f += 1
    cap.release()


def render_demo(
    out_path: Path,
    alarms: pd.DataFrame,
    truths: dict[str, P.EventTruth],
    picks: list[tuple[str, str, float]],
    keypoints_dir: Path,
    videos_dir: Path,
    lo: float,
    hi: float,
    subtitle: str = "",
    width: int = 960,
    pre_s: float = 5.0,
    post_s: float = 3.0,
) -> Path | None:
    """Write the best/worst demo as an H.264 mp4.

    Args:
        alarms: alarm table (clip_id, track_id, t in file seconds, score).
        truths: clip_id -> EventTruth (file seconds, victim track ids).
        picks: pick_best_worst output.
        lo, hi: colour scale (score_scale).
        subtitle: second title-card line, e.g. "train-v2 run, small seed 0".
    Returns:
        out_path, or None (nothing written) when ffmpeg, the picks or their video files are missing.
    """
    picks = [
        p
        for p in picks
        if p[1] in truths
        and (Path(videos_dir) / f"{p[1]}.mp4").exists()
        and (Path(keypoints_dir) / f"{p[1]}.parquet").exists()
    ]
    if not picks or shutil.which("ffmpeg") is None:
        return None
    size = (width, int(round(width * 9 / 16 / 2)) * 2)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{size[0]}x{size[1]}", "-r", str(OUT_FPS), "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28",
            "-movflags", "+faststart", str(out_path),
        ],
        stdin=subprocess.PIPE,
    )  # fmt: skip

    def write(img):
        proc.stdin.write(np.ascontiguousarray(img).tobytes())

    try:
        for which, cid, auc in picks:
            _title_card(
                write,
                size,
                [
                    f"{which.upper()} held-out clip: {cid}",
                    f"within-clip AUC {auc:.2f} (0.5 = chance)",
                    subtitle,
                    "box colour = how unusual that swimmer's last 3 s looked (live view)",
                ],
            )
            _render_clip(
                write, size, cid, truths[cid], alarms, keypoints_dir, videos_dir,
                lo, hi, pre_s, post_s,
            )  # fmt: skip
    finally:
        proc.stdin.close()
        proc.wait()
    return out_path if proc.returncode == 0 else None
