"""
Draw each swimmer's v2 anomaly score on the video: box color = score.

    uv run hydroknight score-video runs/v2_ckpt_after scratch/scores.mp4 7096dfde4791@0 cc9bd9e965e3@0
    uv run hydroknight score-video runs/v2_ckpt_after scratch/scores.mp4 829a5075e439@1 --window

Reads a `train_v2.py --detection` run (alarms_small_s<seed>.parquet + summary.json),
the tracked boxes in data/keypoints_tiled/ and the video in raw_local/. Each clip is
given as clip_id@seed, the seed whose test split held it. Colors run from green (the
run's median held-out score) to red (its 120 false-alarms/hour threshold) on a log
scale; red boxes are drawn thicker, grey = no score. Causal view: a box shows the
latest score already available at that moment, so color lags motion by about 4.5 s
(window plus right context), as a live system would.

Output is unblurred footage of identifiable people, often minors: write it under
scratch/ and never commit or publish it. `--pixelate N` pixelates the footage (N blocks
across) before the boxes are drawn; only pixelated output may go in docs/images/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from hydro_knight.cli.make_pose_figure import pixelate

OUT_W, OUT_H, OUT_FPS = 1280, 720, 30
STALE_S = 1.0  # drop a track's score once no newer window has arrived for this long
WHITE, GREY = (255, 255, 255), (150, 150, 150)


class Scale:
    """Log color scale from a run's median held-out score to its alarm threshold."""

    def __init__(self, lo: float, hi: float):
        self.lo, self.hi = lo, hi

    def color(self, score: float) -> tuple[int, int, int]:
        u = (np.log(score) - np.log(self.lo)) / (np.log(self.hi) - np.log(self.lo))
        hue = int(60 * (1 - np.clip(u, 0, 1)))  # OpenCV hue 60 = green, 0 = red
        b, g, r = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
        return int(b), int(g), int(r)

    def legend(self, img):
        x0, y0, w, h = OUT_W - 330, 16, 300, 14
        cv2.rectangle(img, (x0 - 10, y0 - 8), (x0 + w + 10, y0 + h + 42), (0, 0, 0), -1)
        for i in range(w):
            s = np.exp(np.log(self.lo) + np.log(self.hi / self.lo) * i / (w - 1))
            cv2.line(img, (x0 + i, y0), (x0 + i, y0 + h), self.color(s), 1)
        _text(img, "normal", (x0, y0 + h + 18), 0.5)
        _text(img, "alarm line", (x0 + w - 85, y0 + h + 18), 0.5)
        _text(img, "grey = not scored", (x0, y0 + h + 36), 0.45, (180, 180, 180))


def _text(img, txt, xy, size, color=WHITE, thick=1):
    cv2.putText(img, txt, xy, 0, size, color, thick, cv2.LINE_AA)


def run_scale(results: list[dict], seed: int) -> Scale:
    row = next(r for r in results if r["seed"] == seed)
    hi = next(d["threshold"] for d in row["detection"] if d["fa_target"] == 120)
    return Scale(row["test_median"], hi)


def clip_span(rec: dict, window: bool) -> tuple[float, float]:
    """(start, end) in file seconds: the trims, or onset - 10 s to contact + 5 s."""
    off = rec["start_sec"]
    start = rec["trim_start"] - off if rec["trim_start"] is not None else 0.0
    end = rec["trim_end"] - off if rec["trim_end"] is not None else float("inf")
    if window:
        ev = rec["events"][0]
        start = max(start, ev["start"] - off - 10)
        end = min(end, ev["contact"] - off + 5)
    return start, end


def title_card(writer, lines):
    img = np.zeros((OUT_H, OUT_W, 3), np.uint8)
    for i, ln in enumerate(lines):
        _text(img, ln, (60, 260 + 50 * i), 1.0, thick=2)
    for _ in range(3 * OUT_FPS):
        writer.stdin.write(img.tobytes())


def render_clip(
    writer, clip: str, alarms: pd.DataFrame, scale: Scale, span, blocks: int = 0
):
    start, end = span
    cap = cv2.VideoCapture(f"raw_local/{clip}.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    sx = OUT_W / cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    sy = OUT_H / cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    kp = pd.read_parquet(
        f"data/keypoints_tiled/{clip}.parquet",
        columns=["frame", "track_id", "bx1", "by1", "bx2", "by2"],
    )
    by_frame = dict(tuple(kp.groupby("frame")))
    a = alarms[alarms.clip_id == clip].sort_values("t")
    per_track = {
        k: (g.t.to_numpy(), g.score.to_numpy()) for k, g in a.groupby("track_id")
    }
    first = int(np.ceil(start * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    f, f_next = first, float(first)
    while f / fps <= end:
        ok, frame = cap.read()
        if not ok:
            break
        if f + 1e-6 >= f_next:  # resample any source fps to OUT_FPS
            f_next += fps / OUT_FPS
            now = f / fps
            img = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
            if blocks:
                img = pixelate(img, blocks)
            rows = by_frame.get(f)
            for r in rows.itertuples() if rows is not None else ():
                color, thick = GREY, 1
                if r.track_id in per_track:
                    ts, ss = per_track[r.track_id]
                    i = np.searchsorted(ts, now, side="right") - 1
                    if i >= 0 and now - ts[i] <= STALE_S:
                        color = scale.color(ss[i])
                        thick = 4 if ss[i] >= scale.hi else 2
                p1 = (int(r.bx1 * sx), int(r.by1 * sy))
                p2 = (int(r.bx2 * sx), int(r.by2 * sy))
                cv2.rectangle(img, p1, p2, color, thick)
            cv2.rectangle(img, (0, 0), (300, 34), (0, 0, 0), -1)
            _text(img, f"{clip}  t={now:5.1f}s", (10, 24), 0.65)
            scale.legend(img)
            writer.stdin.write(img.tobytes())
        f += 1


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run", type=Path, help="a train_v2.py --detection output folder")
    p.add_argument("out", help="output .mp4 (keep it under scratch/)")
    p.add_argument("clips", nargs="+", help="clip_id@seed")
    p.add_argument(
        "--window", action="store_true", help="onset - 10 s to contact + 5 s"
    )
    p.add_argument("--manifest", default="data/manifests/pool_footage.jsonl")
    p.add_argument(
        "--pixelate",
        type=int,
        default=0,
        metavar="N",
        help="pixelate the footage to N blocks across before drawing boxes",
    )
    args = p.parse_args()

    results = json.loads((args.run / "summary.json").read_text())["results"]
    with open(args.manifest) as fh:
        manifest = {r["clip_id"]: r for r in map(json.loads, fh)}
    # fmt: off
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{OUT_W}x{OUT_H}", "-r", str(OUT_FPS), "-i", "-", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-crf", "23", args.out]
    # fmt: on
    writer = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for spec in args.clips:
        clip, _, seed = spec.partition("@")
        if not seed:
            raise SystemExit(
                f"{spec}: give the seed whose test split held it (clip@seed)"
            )
        seed = int(seed)
        alarms = pd.read_parquet(args.run / f"alarms_small_s{seed}.parquet")
        if clip not in set(alarms.clip_id):
            raise SystemExit(
                f"{clip} was not a test clip for seed {seed} in {args.run}"
            )
        title_card(
            writer,
            [
                f"Clip {clip}",
                f"v2 small, {args.run.name} seed {seed}, held-out (never trained on)",
                "Box color = anomaly score (live view, about 4.5 s lag)",
            ],
        )
        span = clip_span(manifest[clip], args.window)
        render_clip(writer, clip, alarms, run_scale(results, seed), span, args.pixelate)
    writer.stdin.close()
    writer.wait()


if __name__ == "__main__":
    main()
