"""
The `hydroknight` command: one entry point for every pipeline step and tool.

    hydroknight --help                  # every command, one line each
    hydroknight train --help            # the models `train` knows
    hydroknight train tcn-v2 --help     # that command's own options

Each command is a module in this package with a `main()` that parses `sys.argv`.
The dispatcher below only picks the module and hands it the remaining arguments
unchanged, so options live in one place: the module's own argparse. Modules are
imported only when their command runs, so `--help` never loads torch, MLflow or
MediaPipe.
"""

from __future__ import annotations

import difflib
import importlib
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    name: str  # what you type after `hydroknight`, e.g. "train tcn-v2"
    module: str  # module in hydro_knight.cli that runs it
    help: str
    section: str


PIPELINE = "Pipeline"
DATA = "Data and labeling"
LEGACY = "Legacy (kept to reproduce earlier results)"

COMMANDS = [
    Command(
        "extract",
        "extract_dataset",
        "GPU step: raw tiled YOLO-pose detections for every manifest clip",
        PIPELINE,
    ),
    Command(
        "tracks",
        "build_tracks",
        "CPU step: merge duplicate detections and track them into keypoint Parquets",
        PIPELINE,
    ),
    Command(
        "train tcn-v2",
        "train_v2",
        "train the v2 TCN autoencoder sizes and score them on held-out clips",
        PIPELINE,
    ),
    Command(
        "stack",
        "stack",
        "two-level stack MVP (spike): rows, fit, video",
        PIPELINE,
    ),
    Command(
        "eval protocol",
        "eval_protocol",
        "score a saved training run with the two-tier evaluation protocol",
        PIPELINE,
    ),
    Command(
        "benchmark pose",
        "benchmark_pose",
        "score pose backends against hand-clicked swimmers (frames/run/sweep/log)",
        PIPELINE,
    ),
    Command(
        "score-video",
        "render_scores",
        "draw each swimmer's anomaly score on the video (box color = score)",
        PIPELINE,
    ),
    Command(
        "labeler", "labeler", "start the browser labeler (http://127.0.0.1:8765)", DATA
    ),
    Command(
        "harvest",
        "harvest",
        "find, fetch, accept or reject, and extract new clips (see docs/HARVEST.md)",
        DATA,
    ),
    Command("sheets", "label_sheets", "contact sheets for labeling one clip", DATA),
    Command(
        "heads",
        "head_frames",
        "head-box labels: `heads frames` picks validation frames from non-benchmark clips",
        DATA,
    ),
    Command(
        "write-label",
        "write_label",
        "write one Claude label to the manifest from a JSON spec",
        DATA,
    ),
    Command(
        "backfill-fps",
        "backfill_fps",
        "fill in the manifest's fps field from source metadata",
        DATA,
    ),
    Command(
        "transcode-av1",
        "transcode_av1",
        "convert AV1 videos to H.264, frame-exact, so every environment decodes them",
        DATA,
    ),
    Command(
        "zip-videos",
        "zip_active_videos",
        "zip the videos of every active clip for upload to Google Drive",
        DATA,
    ),
    Command(
        "legacy eval-report",
        "eval_report",
        "first-model evaluation report (replaced by `eval protocol`)",
        LEGACY,
    ),
    Command(
        "legacy rung3-demo",
        "rung3_demo",
        "v1 autoencoder sanity demo: does it learn temporal order?",
        LEGACY,
    ),
    Command(
        "legacy pose-figure",
        "make_pose_figure",
        "regenerate the README pose figures (needs `uv sync --extra spike`)",
        LEGACY,
    ),
]

HELP_FLAGS = {"-h", "--help", "help"}


def _listing(commands: list[Command]) -> str:
    width = max(len(c.name) for c in commands) + 2
    lines, section = [], None
    for c in commands:
        if c.section != section:
            section = c.section
            lines.append(f"\n{section}:")
        lines.append(f"  {c.name.ljust(width)}{c.help}")
    return "\n".join(lines)


def usage(prefix: str = "") -> str:
    """Help text for `hydroknight [prefix] --help`: every command under `prefix`."""
    shown = [c for c in COMMANDS if c.name.startswith(prefix)]
    where = f"hydroknight {prefix}".strip()
    return (
        f"usage: {where} <command> [options]\n"
        f"Run `{where} <command> --help` for a command's own options.\n"
        + _listing(shown)
    )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in HELP_FLAGS:
        print(usage())
        return 0 if argv else 2

    by_name = {c.name: c for c in COMMANDS}
    for n_words in (2, 1):  # "train tcn-v2" before "train"
        name = " ".join(argv[:n_words])
        if len(argv) >= n_words and name in by_name:
            cmd = by_name[name]
            module = importlib.import_module(f"{__name__}.{cmd.module}")
            # The module's argparse reads sys.argv; argv[0] becomes its `prog`.
            sys.argv = [f"hydroknight {cmd.name}", *argv[n_words:]]
            module.main()
            return 0

    group = argv[0] + " "
    if any(c.name.startswith(group) for c in COMMANDS):
        # A group (`train`, `eval`, ...) with no or an unknown choice after it.
        wanted = argv[1:2]
        if wanted and wanted[0] not in HELP_FLAGS:
            print(
                f"hydroknight: unknown {argv[0]} choice '{wanted[0]}'\n",
                file=sys.stderr,
            )
        print(usage(group))
        return 0 if wanted and wanted[0] in HELP_FLAGS else 2

    first_words = sorted({c.name.split()[0] for c in COMMANDS})
    close = difflib.get_close_matches(argv[0], first_words, n=1)
    hint = f" Did you mean '{close[0]}'?" if close else ""
    print(f"hydroknight: unknown command '{argv[0]}'.{hint}\n", file=sys.stderr)
    print(usage(), file=sys.stderr)
    return 2
