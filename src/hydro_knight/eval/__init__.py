"""
Detector-agnostic evaluation: metrics, data-health views, and report generation.

Every detector (TCN autoencoder, future Plan A state machine) reduces to the
same neutral detections table (see metrics.py), so one harness evaluates and
compares them all.
"""

from .metrics import ClipEval, detections_from_windows
from .report import generate_report

__all__ = ["ClipEval", "detections_from_windows", "generate_report"]
