"""Image-statistics diagnostics, real vs render (tells which knob to turn; not a pass/fail gate).

Measured on the water (cyan pixels) and on the whole frame:
  - colour: mean and std of L, a, b over water pixels
  - noise: std of the high-pass residual in the flattest 10% of 32 px water tiles
  - sharpness: median Laplacian variance over water tiles (crowd + ripples)
  - blockiness: |gradient| on the 8 px grid / off the grid (> ~1.05 = visible H.264 blocking)
  - spectrum slope: log-log slope of the radially averaged power spectrum (renders run smoother)

    uv run python synthetic/tools/stats.py raw_local/pose_bench/6f5c67826a76_720.png synthetic/renders/b01/rgbv_0000.png
"""

import sys

import cv2
import numpy as np


def water_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h > 80) & (h < 105) & (s > 60) & (v > 90)


def tiles(mask, t=32):
    H, W = mask.shape
    for y in range(0, H - t, t):
        for x in range(0, W - t, t):
            if mask[y : y + t, x : x + t].mean() > 0.9:
                yield y, x


def stats(path):
    img = cv2.imread(path)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    m = water_mask(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 1.5)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    gx = np.abs(np.diff(g, axis=1))
    tl = list(tiles(m))
    grad = np.array(
        [np.abs(np.gradient(g[y : y + 32, x : x + 32])).mean() for y, x in tl]
    )
    flat = [tl[i] for i in np.argsort(grad)[: max(1, len(tl) // 10)]]
    noise = float(np.median([hp[y : y + 32, x : x + 32].std() for y, x in flat]))
    sharp = float(np.median([lap[y : y + 32, x : x + 32].var() for y, x in tl]))
    on = gx[:, 7::8].mean()
    off = np.delete(gx, np.s_[7::8], axis=1).mean()
    f = np.fft.fftshift(np.abs(np.fft.fft2(g - g.mean())) ** 2)
    H, W = g.shape
    yy, xx = np.indices(f.shape)
    r = np.hypot(yy - H / 2, xx - W / 2).astype(int)
    prof = np.bincount(r.ravel(), f.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    k = np.arange(10, min(H, W) // 2)
    slope = float(np.polyfit(np.log(k), np.log(prof[k] + 1e-9), 1)[0])
    wl = lab[m]
    return {
        "water_frac": round(float(m.mean()), 3),
        "water_Lab_mean": [round(float(v), 1) for v in wl.mean(0)],
        "water_Lab_std": [round(float(v), 1) for v in wl.std(0)],
        "noise_flat": round(noise, 2),
        "lapvar_water": round(sharp, 1),
        "blockiness": round(float(on / off), 3),
        "spectrum_slope": round(slope, 2),
        "frame_L_mean_std": [
            round(float(lab[..., 0].mean()), 1),
            round(float(lab[..., 0].std()), 1),
        ],
    }


if __name__ == "__main__":
    rows = {p: stats(p) for p in sys.argv[1:]}
    keys = list(next(iter(rows.values())))
    for k in keys:
        print(f"{k:18s}", "  ".join(f"{str(rows[p][k]):>22s}" for p in rows))
