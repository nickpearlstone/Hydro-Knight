"""Camera + video degradation for clean renders, so they look like a frame from a YouTube upload.

Geometry never moves on the kept frame (labels and segmentation stay exact).

    uv run python synthetic/tools/post.py synthetic/renders/b01 [--params '{"sat": 1.2}']
writes rgbv_<n>.png beside each rgb_<n>.png.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

DEFAULTS = {
    "gain": None,
    # CAUTION[post-gain-backend-default]: gain None = per-backend default
    "gain_by_backend": {"METAL": 0.9, "CPU": 0.96, "OPTIX": 0.96, "CUDA": 0.96},
    # CAUTION[post-wb-bgr-order]: wb gains are B, G, R
    "wb": [0.95, 1.05, 1.0],
    "clahe": 1.2,
    "sat": 1.4,
    "black": 0.08,
    "blur": 0.9,
    "sharpen": 0.6,
    "sharpen_sigma": 1.4,
    "shot": 6e-3,
    "read": 1e-4,
    "fps": 60,
    "cam_mbps": 40,
    "yt_mbps": 2.5,
    "n_frames": 16,
    "keep": 11,
    "shake_px": 2.0,
    "wobble_px": 2.5,
    "grain": 0.45,
    "grain_blur": 0.6,
}


def tone(img, p):
    out = img.astype(np.float32) / 255.0
    out *= p.get("gain", 1.0) * np.array(p["wb"], np.float32)[None, None, :]
    out = np.clip(out, 0, 1)
    if p["clahe"] > 0:
        lab = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
        cl = cv2.createCLAHE(clipLimit=p["clahe"], tileGridSize=(8, 8))
        lab[..., 0] = cl.apply(lab[..., 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * p["sat"], 0, 1)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if p.get("black", 0) > 0:
        out = np.clip((out - p["black"]) / (1 - p["black"]), 0, 1)
    if p["blur"] > 0:
        out = cv2.GaussianBlur(out, (0, 0), p["blur"])
    if p["sharpen"] > 0:
        out = out + p["sharpen"] * (
            out - cv2.GaussianBlur(out, (0, 0), p["sharpen_sigma"])
        )
    return np.clip(out, 0, 1)


def noisy(img, p, rng):
    lin = img**2.2
    var = p["shot"] * lin + p["read"]
    lin = np.clip(
        lin + rng.normal(0, 1, lin.shape).astype(np.float32) * np.sqrt(var), 0, 1
    )
    return lin ** (1 / 2.2)


def shaken(img, dx, dy, wob, rng):
    h, w = img.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    if wob > 0:
        fx = cv2.resize(
            rng.normal(0, 1, (h // 64 + 2, w // 64 + 2)).astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_CUBIC,
        )
        fy = cv2.resize(
            rng.normal(0, 1, (h // 64 + 2, w // 64 + 2)).astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_CUBIC,
        )
        gx, gy = gx + wob * fx, gy + wob * fy
    mx, my = (gx + dx).astype(np.float32), (gy + dy).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def encode(frames, p):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, f in enumerate(frames):
            cv2.imwrite(str(td / f"f_{i:03d}.png"), (f * 255 + 0.5).astype(np.uint8))
        q = ["-hide_banner", "-loglevel", "error", "-y"]
        subprocess.run(
            [
                "ffmpeg",
                *q,
                "-framerate",
                str(p["fps"]),
                "-i",
                str(td / "f_%03d.png"),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-b:v",
                f"{p['cam_mbps']}M",
                "-g",
                str(p["fps"]),
                "-pix_fmt",
                "yuv420p",
                str(td / "cam.mp4"),
            ],
            check=True,
        )
        yt = p["yt_mbps"]
        subprocess.run(
            [
                "ffmpeg",
                *q,
                "-i",
                str(td / "cam.mp4"),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-b:v",
                f"{yt}M",
                "-maxrate",
                f"{yt * 1.5}M",
                "-bufsize",
                f"{yt * 2}M",
                "-bf",
                "3",
                "-g",
                str(2 * p["fps"]),
                "-pix_fmt",
                "yuv420p",
                str(td / "yt.mp4"),
            ],
            check=True,
        )
        cap = cv2.VideoCapture(str(td / "yt.mp4"))
        out = None
        for _ in range(p["keep"] + 1):
            ok, out = cap.read()
            if not ok:
                raise RuntimeError("decode failed")
        return out


def grain(img, p, rng):
    """Visible grain on the decoded frame; strength in 8-bit levels."""
    if p.get("grain", 0) <= 0:
        return img
    g = rng.normal(0, 1, img.shape[:2]).astype(np.float32)
    if p.get("grain_blur", 0) > 0:
        g = cv2.GaussianBlur(g, (0, 0), p["grain_blur"])
        g /= g.std() + 1e-6
    chroma = rng.normal(0, 1, img.shape).astype(np.float32) * p.get("grain_chroma", 0.3)
    out = img.astype(np.float32) + p["grain"] * (g[..., None] + chroma)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def degrade(img, p, rng, clip=None, clip_keep=0):
    """clip = rendered neighbouring frames (moving water) in order, img's own slot at clip_keep."""
    clip = clip or [img]
    bases = [tone(c, p) for c in clip]
    n, keep = p["n_frames"], p["keep"]
    # CAUTION[post-shake-zero-offset]: shake random walk keeps zero offset on the kept frame
    steps = rng.normal(0, p["shake_px"] / 2, (n, 2))
    walk = np.cumsum(steps, axis=0)
    walk -= walk[keep]
    frames = []
    for i in range(n):
        r = i - keep + clip_keep
        rendered = 0 <= r < len(bases)
        f = noisy(bases[min(max(r, 0), len(bases) - 1)], p, rng)
        if i != keep:
            f = shaken(
                f, walk[i, 0], walk[i, 1], 0.0 if rendered else p["wobble_px"], rng
            )
        frames.append(f.astype(np.float32))
    return encode(frames, p)


def fisheye_maps(W, H, k):
    """Barrel ("fish-eye") distortion that keeps the frame full: r_in = r * (1 + k r^2) / (1 + k), r = 1 at the corners."""
    cx, cy, R = W / 2, H / 2, np.hypot(W / 2, H / 2)
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    dx, dy = (gx - cx) / R, (gy - cy) / R
    f = (1 + k * (dx * dx + dy * dy)) / (1 + k)
    return (cx + dx * f * R).astype(np.float32), (cy + dy * f * R).astype(np.float32)


def fisheye_point(u, v, W, H, k):
    """Where a clean-render pixel (u, v) lands after fisheye_maps: invert r_u = r_d (1 + k r_d^2) / (1 + k)."""
    cx, cy, R = W / 2, H / 2, np.hypot(W / 2, H / 2)
    xu, yu = (u - cx) / R, (v - cy) / R
    ru = np.hypot(xu, yu)
    rd = ru
    for _ in range(30):
        g = rd * (1 + k * rd * rd) / (1 + k) - ru
        rd = rd - g / ((1 + 3 * k * rd * rd) / (1 + k))
    sc = rd / ru if ru > 0 else 1.0 / (1 / (1 + k))
    return cx + xu * sc * R, cy + yu * sc * R


def apply_fisheye(d, n, k, W, H):
    """Distort this frame's labels and masks once (recorded in the labels as lens_k_applied)."""
    lf = d / f"labels_{n}.json"
    lab = json.loads(lf.read_text())
    if k <= 0 or lab.get("lens_k_applied"):
        return
    mx, my = fisheye_maps(W, H, k)
    for p in lab["people"]:
        u, v = fisheye_point(p["u"], p["v"], W, H, k)
        r_out = np.hypot(
            *(
                np.array(fisheye_point(p["u"] + p["head_r_px"], p["v"], W, H, k))
                - (u, v)
            )
        )
        p["u"], p["v"], p["head_r_px"] = float(u), float(v), float(r_out)
    for name in ("seg", "headseg"):
        f = d / f"{name}_{n}.npy"
        if f.exists():
            np.save(f, cv2.remap(np.load(f), mx, my, cv2.INTER_NEAREST))
    lab["lens_k_applied"] = k
    lf.write_text(json.dumps(lab))


def supersample(d, n, img, target_w):
    """Renders made larger than target_w are area-downscaled; labels and seg are rescaled once to match."""
    H, W = img.shape[:2]
    if target_w >= W:
        return img
    k = target_w / W
    lf = d / f"labels_{n}.json"
    lab = json.loads(lf.read_text())
    if not lab.get("downscaled"):
        for p in lab["people"]:
            p["u"], p["v"], p["head_r_px"] = p["u"] * k, p["v"] * k, p["head_r_px"] * k
            p["visible_px"] = int(p["visible_px"] * k * k)
        lab["downscaled"] = k
        lf.write_text(json.dumps(lab))
        sf = d / f"seg_{n}.npy"
        seg = np.load(sf)
        np.save(
            sf,
            cv2.resize(seg, (target_w, round(H * k)), interpolation=cv2.INTER_NEAREST),
        )
    return cv2.resize(img, (target_w, round(H * k)), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--params", default="{}")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    p0 = {**DEFAULTS, **json.loads(a.params)}
    p = p0
    rng = np.random.default_rng(a.seed)
    d = Path(a.dir)
    for f in sorted(d.glob("rgb_*.png")):
        img = cv2.imread(str(f))
        n = f.stem.split("_")[1]
        lab = json.loads((d / f"labels_{n}.json").read_text())
        p = {
            **p0,
            "gain": p0["gain"]
            or p0["gain_by_backend"].get(lab.get("backend", "METAL"), 0.9),
        }
        img = supersample(d, n, img, p.get("target_w", 1920))
        clip, ck = None, 0
        meta = d / f"clip_{n}.json"
        if meta.exists():
            m = json.loads(meta.read_text())
            ck = m["keep"]
            clip = [
                img
                if k == ck
                else cv2.resize(
                    cv2.imread(str(d / f"clip_{n}_{k:02d}.png")),
                    img.shape[1::-1],
                    interpolation=cv2.INTER_AREA,
                )
                for k in range(m["n"])
            ]
        k = lab.get("config", {}).get("camera", {}).get("lens_k", 0.0)
        # CAUTION[post-lens-same-curve]: same curve on the image, every clip frame, labels and masks
        if k > 0:
            mx, my = fisheye_maps(img.shape[1], img.shape[0], k)
            img = cv2.remap(
                img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
            )
            clip = (
                [
                    cv2.remap(
                        c, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
                    )
                    for c in clip
                ]
                if clip
                else None
            )
            if clip:
                clip[ck] = img
            apply_fisheye(d, n, k, img.shape[1], img.shape[0])
        out = degrade(img, p, rng, clip, ck)
        out = grain(out, p, rng)
        cv2.imwrite(str(d / f.name.replace("rgb_", "rgbv_")), out)
        print("wrote", f.name.replace("rgb_", "rgbv_"))
    (d / "post_params.json").write_text(json.dumps(p, indent=1))


if __name__ == "__main__":
    main()
