"""Exponential backoff policy used by rtsp_reader.py when the stream
drops. Forest network links flap, so reconnect attempts back off
rather than hammering the camera, but reset instantly once a frame
comes through again.
"""

from __future__ import annotations

import random
import time


class ReconnectPolicy:
    def __init__(
        self,
        initial_backoff_sec: float = 1.0,
        max_backoff_sec: float = 30.0,
        multiplier: float = 2.0,
        jitter: float = 0.2,
    ):
        self.initial = initial_backoff_sec
        self.max = max_backoff_sec
        self.multiplier = multiplier
        self.jitter = jitter
        self._current = initial_backoff_sec
        self.attempt_count = 0

    def reset(self) -> None:
        self._current = self.initial
        self.attempt_count = 0

    def next_delay(self) -> float:
        """Return the delay (seconds) to wait before the next attempt,
        and advance the backoff for the attempt after that."""
        delay = self._current
        # Non-cryptographic retry jitter to prevent thundering herd reconnection spikes
        jittered = delay * (1.0 + random.uniform(-self.jitter, self.jitter))  # nosec B311
        self.attempt_count += 1
        self._current = min(self._current * self.multiplier, self.max)
        return max(0.0, jittered)

    def sleep_next(self) -> float:
        delay = self.next_delay()
        time.sleep(delay)
        return delay

    @classmethod
    def from_config(cls, cfg: dict) -> ReconnectPolicy:
        r = cfg.get("reconnect", {})
        return cls(
            initial_backoff_sec=float(r.get("initial_backoff_sec", 1.0)),
            max_backoff_sec=float(r.get("max_backoff_sec", 30.0)),
            multiplier=float(r.get("backoff_multiplier", 2.0)),
        )
