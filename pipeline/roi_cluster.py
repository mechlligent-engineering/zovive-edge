"""One enclosing box over every target, so SAHI slices once instead of per animal.

Blueprint Section 4.2: for N>=2 animals, Xmin=min(x1), Ymin=min(y1),
Xmax=max(x2), Ymax=max(y2). Slicing per animal would re-run the NPU N times over
largely the same pixels; one cluster box drops a 20-tile job to 2-4 tiles.
"""
from __future__ import annotations

import numpy as np


def cluster_box(boxes, frame_shape, pad_ratio: float = 0.10, min_size: int = 640):
    """Return (x1, y1, x2, y2) enclosing all boxes, padded and clamped to frame.

    min_size stops a single small distant animal producing a 40x30 pixel region
    that SAHI cannot tile meaningfully.
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    if len(boxes) == 0:
        return None
    h, w = frame_shape[:2]

    x1, y1 = boxes[:, 0].min(), boxes[:, 1].min()
    x2, y2 = boxes[:, 2].max(), boxes[:, 3].max()

    pw, ph = (x2 - x1) * pad_ratio, (y2 - y1) * pad_ratio
    x1, y1, x2, y2 = x1 - pw, y1 - ph, x2 + pw, y2 + ph

    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if (x2 - x1) < min_size:
        x1, x2 = cx - min_size / 2, cx + min_size / 2
    if (y2 - y1) < min_size:
        y1, y2 = cy - min_size / 2, cy + min_size / 2

    # Shift the window back inside the frame rather than truncating it. Truncating
    # would silently give SAHI a region smaller than one tile near the frame edge,
    # which is exactly where distant animals appear.
    x1, x2 = _fit(x1, x2, w)
    y1, y2 = _fit(y1, y2, h)
    return (int(x1), int(y1), int(x2), int(y2))


def _fit(lo, hi, limit):
    span = min(hi - lo, limit)
    if lo < 0:
        lo, hi = 0, span
    elif hi > limit:
        lo, hi = limit - span, limit
    return max(0, lo), min(limit, hi)


def cluster_area_ratio(cluster, frame_shape) -> float:
    """Fraction of the frame the cluster covers.

    If this approaches 1.0 the animals are spread across the whole scene and the
    cluster saves nothing - fall back to full-frame SAHI or split into groups.
    """
    if cluster is None:
        return 0.0
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = cluster
    return ((x2 - x1) * (y2 - y1)) / float(w * h)
