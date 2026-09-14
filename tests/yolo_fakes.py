"""Minimal stand-ins for Ultralytics Results objects, shared by the extraction tests."""

from __future__ import annotations

import numpy as np


class T:
    """Stand-in for a torch tensor: .cpu().numpy()."""

    def __init__(self, a):
        self.a = np.asarray(a, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class Boxes:
    def __init__(self, xyxy, conf):
        self.xyxy, self.conf = T(xyxy), T(conf)

    def __len__(self):
        return len(self.xyxy.a)


class Result:
    def __init__(self, xyxy, conf):
        self.boxes = Boxes(xyxy, conf)
        kp = np.zeros((len(conf), 17, 3), np.float32)
        kp[:, :, :2] = 5.0  # every keypoint at (5, 5) in crop pixels
        kp[:, :, 2] = 0.7
        self.keypoints = type("K", (), {"data": T(kp)})()
