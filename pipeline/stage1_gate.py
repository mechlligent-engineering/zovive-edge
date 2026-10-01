"""Stage 1: "is this track a real animal worth acting on" gate.

Two independent checks have to pass before a track is promoted to an
event:
  1. Presence vote — the track has been seen as an animal in at least
     `min_hits_in_window` of the last `window_size` frames (its
     `presence_history`, maintained by inference/bytetrack_wrapper.py).
     This is what turns a single-frame false positive (e.g. "empty
     forest -> leopard") into noise instead of an alert.
  2. Per-track cooldown — the same track hasn't already triggered an
     alert within `track_alert_cooldown_sec` (a sleeping tiger
     shouldn't re-alert every few seconds).

Pure logic, no I/O, so it's cheap to unit test with fake Track objects.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from inference.bytetrack_wrapper import Track


@dataclass
class Stage1Config:
    min_hits_in_window: int = 3
    window_size: int = 5
    track_alert_cooldown_sec: float = 300.0

    @classmethod
    def from_config(cls, inference_cfg: dict) -> Stage1Config:
        det = inference_cfg.get("detector", {})
        return cls(
            min_hits_in_window=int(det.get("min_hits_in_window", 3)),
            window_size=int(det.get("window_size", 5)),
            track_alert_cooldown_sec=float(det.get("track_alert_cooldown_sec", 300.0)),
        )


class Stage1Gate:
    def __init__(self, config: Stage1Config | None = None):
        self.config = config or Stage1Config()
        self._last_alert_ts: dict[str, float] = {}

    def presence_vote_passed(self, track: Track) -> bool:
        window = track.presence_history[-self.config.window_size:]
        return sum(1 for v in window if v) >= self.config.min_hits_in_window

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
        if self.in_cooldown(track, now):
            return False
        return True

    def mark_alerted(self, track_id: str, now: float | None = None) -> None:
        self._last_alert_ts[track_id] = now if now is not None else time.time()

    def evaluate_all(self, tracks: list[Track], now: float | None = None) -> list[Track]:
        return [t for t in tracks if self.evaluate(t, now)]
