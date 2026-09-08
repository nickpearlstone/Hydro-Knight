"""Thin wrapper around MLflow: the single place all experiment-tracking calls live,
so eval_report.py stays clean and the tracking backend can change in one spot."""

from __future__ import annotations

import contextlib
import os

import mlflow

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"   # was "./mlruns"
DEFAULT_EXPERIMENT = "hydro-knight"    # the named group all runs belong to

def log_config(params: dict) -> None:
    """this function takes a config dict and feeds it to mlflow(hyperparameter and setting logging)"""
    mlflow.log_params(params)

def log_metrics(metrics: dict) -> None:
    """this function takes a metrics dict and feeds it to mlflow(metric logging)"""
    mlflow.log_metrics(metrics)

def log_artifacts(run_dir) -> None:
    """this function takes a directory path(obj) and feeds it to mlflow(artifact logging)"""
    mlflow.log_artifacts(str(run_dir)) 

@contextlib.contextmanager
def run(name, tracking_uri=DEFAULT_TRACKING_URI, experiment=DEFAULT_EXPERIMENT):
    """open an MLflow run (sets URI + experiment), auto-closing on exit"""

    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=name) as run:
        yield run
  
  