"""Blur detection used by best_snapshot.py to prefer sharp crops (and
by alert_policy's min_snapshot_sharpness to reject unusably blurry
alert images) — a fast, well-known technique: the variance of the
Laplacian drops sharply for out-of-focus / motion-blurred images.
"""

from __future__ import annotations

import cv2
import numpy as np


def sharpness_score(image: np.ndarray) -> float:
    """Higher = sharper. Typically: <20 very blurry, 20-60 soft, >60 sharp
    — but the right threshold depends on the camera/lens, so
    alert_policy.min_snapshot_sharpness is tunable in the field."""
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_blurry(image: np.ndarray, threshold: float = 40.0) -> bool:
    return sharpness_score(image) < threshold
