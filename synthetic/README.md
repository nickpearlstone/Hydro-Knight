# synthetic/: Blender wave-pool renders with exact labels

Rendered crowded wave-pool frames, matched to a real reference frame and checked against it
by the same detectors. Progress log and numbers: `docs/research/2026-10-19-synthetic-realism-loop.md`.

```
synthetic/
├── blender/
│   ├── scene.py            config → frame: fan-shaped pool, water, tubes, crowd, labels + seg
│   ├── humans.py           procedural fallback people, tubes, deck clutter, splash
│   ├── make_human_lib.py   builds the posed MPFB human library (synthetic/mpfb/humans.blend)
│   ├── mpfb_env.sh         runs Blender with the repo-local MPFB user dirs
│   └── pack_lib.py         shrinks + packs library textures into one portable .blend (Kaggle)
├── configs/ref_6f5c.json   camera, pool, water, light and crowd matched to real frame 6f5c…#720
├── tools/
│   ├── post.py             camera + two-stage x264 degradation (rgb_ → rgbv_)
│   ├── score.py            YOLO11n-pose + RT-DETRv2 recall / false boxes, real vs render
│   ├── realism.py          DINOv2 crop KID + real-vs-render classifier AUC
│   ├── stats.py            image-statistics diagnostics
│   ├── side_by_side.py     real frame above a render
│   └── evaluate.sh         post → stats → score → realism for one render folder
├── kaggle/                 private Kaggle GPU job (render timing) + its metadata
├── renders/                outputs (gitignored)
├── blender-bin/            local Blender 4.2.1 (gitignored, per machine)
└── mpfb/                   MPFB extension + CC0 asset pack + humans.blend (gitignored)
```

## Setup (once per machine)

- Blender 4.2.1 (macOS arm64 or Linux x64) unpacked or linked at `synthetic/blender-bin/`.
- MPFB 2.0.18 + the CC0 `makehuman_system_assets` pack, installed into `synthetic/mpfb/bl`
  (zips kept in `synthetic/mpfb/dl`):

```bash
synthetic/blender/mpfb_env.sh synthetic/blender-bin/Blender.app/Contents/MacOS/Blender --factory-startup --command extension install-file -r user_default -e synthetic/mpfb/dl/mpfb-2.0.18.zip
```

then extract the asset pack with MPFB's `AssetService.fix_and_extract_asset_pack_zip`.

## Run

```bash
synthetic/blender/mpfb_env.sh synthetic/blender-bin/Blender.app/Contents/MacOS/Blender -b -P synthetic/blender/make_human_lib.py -- --out synthetic/mpfb/humans.blend --per-pose 14
```

```bash
synthetic/blender-bin/Blender.app/Contents/MacOS/Blender -b -P synthetic/blender/scene.py -- --config synthetic/configs/ref_6f5c.json --out synthetic/renders/b10 --seed 3
```

```bash
synthetic/tools/evaluate.sh synthetic/renders/b10
```

The real reference is scored once with `uv run --with transformers python synthetic/tools/score.py real 6f5c67826a76_720`.

## Licences

MPFB code is GPL-3.0 (a tool we run, not shipped). Base mesh, rigs and the system asset pack
are CC0. No SMPL, no NC data. Swimwear is painted onto the body, so no clothing assets are needed.
