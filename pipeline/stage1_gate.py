"""Stage 1: "is this track a real animal worth acting on" gate.

Every check has to pass before a track is promoted to an event:
  1. Presence vote — the track has been seen as an animal in at least
     `min_hits_in_window` of the last `window_size` frames (its
     `presence_history`, maintained by inference/bytetrack_wrapper.py).
     This is what turns a single-frame false positive (e.g. "empty
     forest -> leopard") into noise instead of an alert, while a few
     missed frames (leaves, motion blur) don't reset the count.
  2. Duration — the track has existed for at least
     `min_track_duration_sec` (its first to last sighting), so a burst of
     frames inside a fraction of a second can't alert.
  3. Box area — the track's box covers at least `min_box_area_px` pixels
     of the original frame (0 = off), so specks of foliage can't alert.
  4. Mean confidence — the track's leading species, averaged over its
     recent frames the way stage2_verifier's "average" vote does (a frame
     that saw another species counts as 0), reaches that species'
     threshold (inference/class_thresholds.py). A track that keeps
     changing species doesn't confirm. Off when `class_thresholds` is None.
  5. Per-track cooldown — the same track hasn't already triggered an
     alert within `track_alert_cooldown_sec` (a sleeping tiger
     shouldn't re-alert every few seconds).

Stage1Config's defaults are the original presence-and-cooldown gate;
configs/inference_config.yaml turns the other checks on.

Pure logic, no I/O, so it's cheap to unit test with fake Track objects.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

from inference.bytetrack_wrapper import Track
from inference.class_thresholds import ClassThresholds


@dataclass
class Stage1Config:
    min_hits_in_window: int = 3
    window_size: int = 5
    track_alert_cooldown_sec: float = 300.0
    min_track_duration_sec: float = 0.0
    min_box_area_px: float = 0.0
    class_thresholds: ClassThresholds | None = None

    @classmethod
    def from_config(cls, inference_cfg: dict) -> Stage1Config:
        det = inference_cfg.get("detector", {})
        return cls(
            min_hits_in_window=int(det.get("min_hits_in_window", 3)),
            window_size=int(det.get("window_size", 5)),
            track_alert_cooldown_sec=float(det.get("track_alert_cooldown_sec", 300.0)),
            min_track_duration_sec=float(det.get("min_track_duration_sec", 0.0)),
            min_box_area_px=float(det.get("min_box_area_px", 0.0)),
            class_thresholds=ClassThresholds.from_config(det),
        )


class Stage1Gate:
    def __init__(self, config: Stage1Config | None = None):
        self.config = config or Stage1Config()
        self._last_alert_ts: dict[str, float] = {}

    def presence_vote_passed(self, track: Track) -> bool:
        window = track.presence_history[-self.config.window_size :]
        return sum(1 for v in window if v) >= self.config.min_hits_in_window

    def duration_passed(self, track: Track) -> bool:
        return (track.last_seen_ts - track.first_seen_ts) >= self.config.min_track_duration_sec

    def box_area_passed(self, track: Track) -> bool:
        if self.config.min_box_area_px <= 0:
            return True
        x1, y1, x2, y2 = track.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1) >= self.config.min_box_area_px

    def mean_confidence_passed(self, track: Track) -> bool:
        thresholds = self.config.class_thresholds
        if thresholds is None:
            return True
        if not track.class_votes:
            return False
        totals: dict[str, float] = defaultdict(float)
        for name, score in track.class_votes:
            totals[name] += score
        leader = max(totals, key=totals.__getitem__)  # a tie goes to the species seen first
        return totals[leader] / len(track.class_votes) >= thresholds.for_class(leader)

    def in_cooldown(self, track: Track, now: float | None = None) -> bool:
        last = self._last_alert_ts.get(track.track_id)
        if last is None:
            return False
        now = now if now is not None else time.time()
        return (now - last) < self.config.track_alert_cooldown_sec

    def evaluate(self, track: Track, now: float | None = None) -> bool:
        """True if this track should be promoted to an event right now."""
        if not self.presence_vote_passed(track):
            return False
        if not self.duration_passed(track):
            return False
        if not self.box_area_passed(track):
            return False
        if not self.mean_confidence_passed(track):
            return False
        if self.in_cooldown(track, now):
            return False
        return True

    def mark_alerted(self, track_id: str, now: float | None = None) -> None:
        self._last_alert_ts[track_id] = now if now is not None else time.time()

    def evaluate_all(self, tracks: list[Track], now: float | None = None) -> list[Track]:
        return [t for t in tracks if self.evaluate(t, now)]
