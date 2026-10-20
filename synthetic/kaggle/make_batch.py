"""Write the two private Kaggle notebook folders for the batch (synthetic/kaggle/batch_a, batch_b).

uv run python synthetic/kaggle/make_batch.py
uvx --from "kaggle>=1.8" kaggle kernels push -p synthetic/kaggle/batch_a   # and batch_b
"""

import json
from pathlib import Path

here = Path(__file__).parent
src = (here / "batch_kernel.py").read_text()
for plan in ("a", "b"):
    d = here / f"batch_{plan}"
    d.mkdir(exist_ok=True)
    (d / "batch_kernel.py").write_text(src.replace('PLAN = "a"', f'PLAN = "{plan}"', 1))
    meta = {
        "id": f"npstone/hk-synth-batch-{plan}",
        "title": f"hk-synth-batch-{plan}",
        "code_file": "batch_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,  # every Kaggle item for this project stays private
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["npstone/hk-synth"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", d)
