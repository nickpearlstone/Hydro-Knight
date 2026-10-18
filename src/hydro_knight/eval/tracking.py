"""Thin wrapper around MLflow: the single place all experiment-tracking calls live, so
scripts stay clean and the backend can change in one spot. Importing this module imports
mlflow (the optional `tracking` extra); scripts import it only behind their --mlflow flag.

Run layout (MLflow parent/child runs): one parent run per script command, one nested
child run per (size, seed) or per model. Summary numbers and cross-seed figures sit on
the parent; per-step loss, datasets, checkpoints and per-seed figures on the children.
Tags follow the vocabulary in eval/run_meta.py; the run description shown in the UI is
the standard `mlflow.note.content` tag.

The backend is sqlite:///mlflow.db unless MLFLOW_TRACKING_URI is set (MLflow's own
variable), which is how a throwaway database is used for test runs.
"""

from __future__ import annotations

import contextlib
import os
import time
import warnings
from pathlib import Path

import mlflow
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"  # was "./mlruns"
DEFAULT_EXPERIMENT = "hydro-knight"  # v1 eval_report runs; v2 scripts name their own
BATCH = 1000  # MLflow's per-call metric limit (mlflow.utils.validation.MAX_METRICS_PER_BATCH)


def default_uri() -> str:
    """MLFLOW_TRACKING_URI if set, else the repo's SQLite file."""
    return os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI


def log_config(params: dict) -> None:
    """Log a settings dict as run params (values stringified by MLflow)."""
    mlflow.log_params(params)


def log_metrics(metrics: dict, step: int | None = None) -> None:
    """Log a flat {name: number} dict, optionally at one step."""
    mlflow.log_metrics(metrics, step=step)


def log_series(key: str, values, start_step: int = 1) -> None:
    """Log a whole per-step series (e.g. training loss) as one metric with step = 1, 2, ...

    Sent in batches of 1000 through MlflowClient.log_batch instead of one call per step.
    """
    run_id = mlflow.active_run().info.run_id
    now = int(time.time() * 1000)
    ms = [
        Metric(key, float(v), now, start_step + i)
        for i, v in enumerate(values)
        if v == v  # skip NaN
    ]
    client = MlflowClient()
    for i in range(0, len(ms), BATCH):
        client.log_batch(run_id, metrics=ms[i : i + BATCH])


def log_artifacts(run_dir, artifact_path: str | None = None) -> None:
    """Log every file in a directory as artifacts of the active run."""
    mlflow.log_artifacts(str(run_dir), artifact_path)


def log_artifact(path, artifact_path: str | None = None) -> None:
    """Log one file as an artifact of the active run."""
    mlflow.log_artifact(str(path), artifact_path)


def log_figure(fig, name: str) -> None:
    """Attach a matplotlib Figure as an image artifact (e.g. "figures/loss.png"); None is skipped.

    Figures from eval/figures.py are never registered with pyplot, so there is nothing
    to close; any pyplot figure passed in is closed after logging.
    """
    if fig is None:
        return
    mlflow.log_figure(fig, name)
    import matplotlib.pyplot as plt

    if plt.fignum_exists(getattr(fig, "number", -1)):
        plt.close(fig)


def log_dataset(df, name: str, context: str, source: str | Path) -> None:
    """Record a table (e.g. a train/test clip list) as a run input; MLflow hashes it into a digest."""
    # MLflow hints that int columns cannot hold NaN; ours never do, so silence it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ds = mlflow.data.from_pandas(df, source=str(source), name=name)
        mlflow.log_input(ds, context=context)


def set_tags(tags: dict) -> None:
    mlflow.set_tags({k: str(v) for k, v in tags.items() if v is not None})


def set_note(text: str) -> None:
    """The run's description in the UI (markdown), via the standard mlflow.note.content tag."""
    mlflow.set_tag("mlflow.note.content", text)


@contextlib.contextmanager
def run(
    name,
    tracking_uri: str | None = None,
    experiment: str = DEFAULT_EXPERIMENT,
    nested: bool = False,
    tags: dict | None = None,
    note: str | None = None,
):
    """Open an MLflow run (parent, or child of the active run with nested=True), closing it on exit.

    A run whose body raises is closed with status FAILED, so crashed runs are visible as such.
    """
    if not nested:
        mlflow.set_tracking_uri(tracking_uri or default_uri())
        mlflow.set_experiment(experiment)
    tags = {k: str(v) for k, v in (tags or {}).items() if v is not None}
    with mlflow.start_run(run_name=name, nested=nested, tags=tags) as r:
        if note:
            set_note(note)
        yield r


def find_runs(experiment: str, filter_string: str = "", tracking_uri=None) -> list:
    """Runs of an experiment matching an MLflow search filter, newest first; [] if the experiment is missing."""
    client = MlflowClient(tracking_uri or default_uri())
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        return []
    return client.search_runs(
        [exp.experiment_id], filter_string, order_by=["attributes.start_time DESC"]
    )


def runs_frame(experiment: str, filter_string: str = "", tracking_uri=None):
    """Same search as a pandas table (columns metrics.*, params.*, tags.*), via mlflow.search_runs."""
    mlflow.set_tracking_uri(tracking_uri or default_uri())
    return mlflow.search_runs(
        experiment_names=[experiment], filter_string=filter_string
    )
