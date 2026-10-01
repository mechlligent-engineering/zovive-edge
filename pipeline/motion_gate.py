"""Cheap frame-differencing motion score.

This is ADVISORY only — it does NOT skip running the detector. A
sleeping or stationary tiger produces zero motion and must still be
detected, so gating detection on motion would create exactly the kind
of false negative this project can't afford. What it's actually used
for: (1) a metric logged alongside each detection cycle so false
positives can later be cross-checked against "was there even motion"
during dataset triage — never as a correctness gate.
"""

from __future__ import annotations

import cv2
import numpy as np


class MotionGate:
    def __init__(self, diff_threshold: int = 25, min_changed_ratio: float = 0.01):
        self.diff_threshold = diff_threshold
        self.min_changed_ratio = min_changed_ratio
        self._prev_gray: np.ndarray | None = None

    def score(self, frame: np.ndarray) -> float:
        """Returns the fraction (0..1) of pixels that changed significantly
        since the last call. Resets internal state to `frame` each call."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return 0.0

        diff = cv2.absdiff(gray, self._prev_gray)
        self._prev_gray = gray
        changed = np.count_nonzero(diff > self.diff_threshold)
        return changed / diff.size

    def has_motion(self, frame: np.ndarray) -> bool:
        return self.score(frame) >= self.min_changed_ratio

    def reset(self) -> None:
        self._prev_gray = None
