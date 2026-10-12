"""Contact sheets for labeling one clip. Images go to scratch/lbl/ (gitignored): never
commit them, footage shows minors.

  uv run python scripts/label_sheets.py CLIP_ID overview             # 1 fps, 36 s per sheet -> scratch/lbl/<clip>/sheet_*.jpg
  uv run python scripts/label_sheets.py CLIP_ID zoom T0 T1 X0 Y0 X1 Y1 [STEP]
      # STEP s (default 0.25) from T0..T1, crop full-res box (X0,Y0)-(X1,Y1) at half size,
      # white grid every 100 full-res px -> scratch/lbl/<clip>/zoom_*.jpg
Reading positions in a zoom tile: full_x = X0 + k*tile_x, full_y = Y0 + k*tile_y
(k = 2 for crops >= 600 px wide, else 1)
(grid lines sit at full-res multiples of 100). Times are file seconds (= source seconds
when the clip's start_sec is 0; otherwise source = file + start_sec).
"""

import sys
from pathlib import Path

import cv2
import numpy as np

clip, mode = sys.argv[1], sys.argv[2]
out = Path("scratch/lbl") / clip
out.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(f"raw_local/{clip}.mp4")
fps, n = cap.get(5), int(cap.get(7))
print(f"{clip}: fps {fps:.3f}, {n / fps:.1f} s, {int(cap.get(3))}x{int(cap.get(4))}")


def grab(t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, f = cap.read()
    return f if ok else None


def sheets(tiles, prefix, cols=3, per=12):
    for k in range(0, len(tiles), per):
        ch = tiles[k : k + per]
        ch += [np.zeros_like(tiles[0])] * (per - len(ch))
        rows = [np.hstack(ch[i : i + cols]) for i in range(0, per, cols)]
        cv2.imwrite(
            str(out / f"{prefix}_{k:03d}.jpg"),
            np.vstack(rows),
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )
    print(f"{len(tiles)} tiles -> {out}/{prefix}_*.jpg")


if mode == "overview":
    tiles, t = [], 0.0
    while t < n / fps:
        f = grab(t)
        if f is None:
            break
        s = cv2.resize(f, (320, 180))
        cv2.putText(s, f"{t:.0f}s", (4, 18), 0, 0.6, (0, 255, 255), 2)
        tiles.append(s)
        t += 1.0
    sheets(
        tiles, "sheet", cols=6, per=36
    )  # 36 s per sheet: read EVERY sheet (overlays)
else:
    t0, t1, x0, y0, x1, y1 = map(float, sys.argv[3:9])
    step = float(sys.argv[9]) if len(sys.argv) > 9 else 0.25
    x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
    tiles, t = [], t0
    while t <= t1 + 1e-6:
        f = grab(t)
        if f is None:
            break
        k = (
            2 if (x1 - x0) >= 600 else 1
        )  # small crops stay full-res: full = X0 + k*tile
        c = cv2.resize(f[y0:y1, x0:x1], ((x1 - x0) // k, (y1 - y0) // k))
        for gx in range((x0 // 100 + 1) * 100, x1, 100):
            cv2.line(
                c, ((gx - x0) // k, 0), ((gx - x0) // k, c.shape[0]), (255, 255, 255), 1
            )
        for gy in range((y0 // 100 + 1) * 100, y1, 100):
            cv2.line(
                c, (0, (gy - y0) // k), (c.shape[1], (gy - y0) // k), (255, 255, 255), 1
            )
        cv2.putText(c, f"{t:.2f}s", (4, 20), 0, 0.6, (0, 255, 255), 2)
        tiles.append(c)
        t += step
    sheets(tiles, "zoom")
