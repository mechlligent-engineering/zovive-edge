"""Keeps the best few crops seen for each track while it's active, so
stage2_verifier.py has good material to classify and alert_dispatcher
has a presentable snapshot to send — instead of just using whichever
frame happened to trigger the event.

"Best" = sharp and reasonably large, scored as sharpness * area, which
in practice favors close, in-focus crops over small or blurry ones.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from pipeline.sharpness_filter import sharpness_score


@dataclass
class ScoredCrop:
    crop: np.ndarray
    full_frame: np.ndarray
    score: float
    sharpness: float
    timestamp: float = field(default_factory=time.time)


class BestSnapshotTracker:
    """One instance per track_id (created/discarded by the pipeline as
    tracks appear/disappear), or reused across tracks via a dict keyed
    by track_id — see `BestSnapshotStore` below for that."""

    def __init__(self, keep_top_k: int = 5):
        self.keep_top_k = keep_top_k
        self._crops: list[ScoredCrop] = []

    def offer(self, crop: np.ndarray, full_frame: np.ndarray) -> None:
        if crop is None or crop.size == 0:
            return
        sharpness = sharpness_score(crop)
        area = crop.shape[0] * crop.shape[1]
        score = sharpness * (area ** 0.5)  # sqrt(area) so huge blurry crops don't dominate
        self._crops.append(ScoredCrop(crop=crop, full_frame=full_frame, score=score, sharpness=sharpness))
        self._crops.sort(key=lambda c: c.score, reverse=True)
        self._crops = self._crops[: self.keep_top_k]

    def best(self) -> ScoredCrop | None:
        return self._crops[0] if self._crops else None

    def top(self, n: int) -> list[ScoredCrop]:
        return self._crops[:n]

    def __len__(self) -> int:
        return len(self._crops)


class BestSnapshotStore:
    """Owns one BestSnapshotTracker per track_id and evicts them when
    the pipeline tells it a track died, so memory doesn't grow with
    every track that ever existed since boot."""

    def __init__(self, keep_top_k: int = 5):
        self.keep_top_k = keep_top_k
        self._by_track: dict[str, BestSnapshotTracker] = {}

    def offer(self, track_id: str, crop: np.ndarray, full_frame: np.ndarray) -> None:
        tracker = self._by_track.setdefault(track_id, BestSnapshotTracker(self.keep_top_k))
        tracker.offer(crop, full_frame)

    def top(self, track_id: str, n: int) -> list[ScoredCrop]:
        tracker = self._by_track.get(track_id)
        return tracker.top(n) if tracker else []

    def best(self, track_id: str) -> ScoredCrop | None:
        tracker = self._by_track.get(track_id)
        return tracker.best() if tracker else None

    def discard(self, track_id: str) -> None:
        self._by_track.pop(track_id, None)

    def active_track_ids(self) -> list[str]:
        return list(self._by_track.keys())
