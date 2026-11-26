"""Detector-agnostic evaluation: metrics, data-health views, and report generation."""

from .metrics import ClipEval, detections_from_windows
from .report import generate_report

__all__ = ["ClipEval", "detections_from_windows", "generate_report"]
