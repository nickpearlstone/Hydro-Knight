"""Stack the real reference frame above a render (both 960 px wide) for a visual check.

uv run python synthetic/tools/side_by_side.py synthetic/renders/p02/rgb_0000.png out.jpg [real_name]
"""

import sys

import cv2
import numpy as np

real = sys.argv[3] if len(sys.argv) > 3 else "6f5c67826a76_720"
a = cv2.resize(cv2.imread(f"raw_local/pose_bench/{real}.png"), (960, 540))
b = cv2.resize(cv2.imread(sys.argv[1]), (960, 540))
cv2.imwrite(sys.argv[2], np.vstack([a, b]), [cv2.IMWRITE_JPEG_QUALITY, 80])
