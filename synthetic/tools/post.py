"""Camera + video degradation for clean renders, so they look like a frame from a YouTube upload.

Geometry never moves on the kept frame, so labels and segmentation stay exact. Steps:
  1. camera tone: white-balance gains, local tone mapping (CLAHE on L), saturation
  2. optics: soft lens blur, then in-camera sharpening (unsharp mask, gives halos)
  3. sensor noise, signal dependent (shot + read, in linear light; Brooks et al. 2019)
  4. codec: a short clip of the frame with independent noise, a sub-pixel camera-shake random
     walk and a ~1 px wobble, encoded camera-like (x264 40 Mbps) and then YouTube-like
     (x264 at the reference clip's measured ~5.8 Mbps, B-frames); the kept frame sits mid-GOP
     with zero shake. A single encoded still is unrealistically clean (it becomes an I-frame).

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

DEFAULTS = {  # tuned against 6f5c67826a76 #720 (batch b02, see docs/research synthetic log)
    "gain": 0.9,
    "wb": [0.95, 1.05, 1.0],  # B, G, R gains
    "clahe": 1.2,  # 0 = off
    "sat": 1.4,
    "black": 0.08,
    "blur": 0.9,
    "sharpen": 0.6,
    "sharpen_sigma": 1.4,
    "shot": 6e-3,
    "read": 1e-4,
    "fps": 60,
    "cam_mbps": 40,
    "yt_mbps": 2.5,  # b13 sweep: best detector-level match (1.8 too hard, 3.5+ too easy)
    "n_frames": 16,
    "keep": 11,
    "shake_px": 2.0,
    "wobble_px": 2.5,
    "grain": 0.45,  # post-codec grain, 8-bit levels (real flat-water high-pass ~0.42)
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
    if p.get("black", 0) > 0:  # consumer cameras crush shadows to true black
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
    """Visible grain on the decoded frame (survives the codec, like grain in real uploads).

    Slightly blurred, luma-weighted gaussian noise; strength in 8-bit levels.
    """
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
    """clip = rendered neighbouring frames (moving water) in order, img's own slot at clip_keep.

    They sit around the kept slot of the encoded sequence; slots beyond them reuse the nearest
    rendered frame with extra wobble. Without a clip every slot is the still plus wobble.
    """
    clip = clip or [img]
    bases = [tone(c, p) for c in clip]
    n, keep = p["n_frames"], p["keep"]
    # shake: random walk anchored so the kept frame has zero offset (labels stay exact)
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
    p = {**DEFAULTS, **json.loads(a.params)}
    rng = np.random.default_rng(a.seed)
    d = Path(a.dir)
    for f in sorted(d.glob("rgb_*.png")):
        img = cv2.imread(str(f))
        n = f.stem.split("_")[1]
        img = supersample(d, n, img, p.get("target_w", 1920))
        clip, ck = None, 0
        meta = d / f"clip_{n}.json"
        if meta.exists():  # moving-water neighbours rendered by scene.py --clip
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
        out = degrade(img, p, rng, clip, ck)
        out = grain(out, p, rng)
        cv2.imwrite(str(d / f.name.replace("rgb_", "rgbv_")), out)
        print("wrote", f.name.replace("rgb_", "rgbv_"))
    (d / "post_params.json").write_text(json.dumps(p, indent=1))


if __name__ == "__main__":
    main()
