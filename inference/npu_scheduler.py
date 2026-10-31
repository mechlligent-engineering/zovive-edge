"""Serializes access to the Hailo-8 device.

The Hailo M.2 HAT is a single shared NPU. The node runs one model on it
(the species detector, every frame), but InferVStreams calls must still
never overlap from different threads (e.g. the inference thread and a
benchmark or SAHI tile pass). Every hailo_inference.py call goes through
this lock.

Also tracks basic latency stats for the health heartbeat.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


@dataclass
class _Stats:
    calls: int = 0
    total_latency_sec: float = 0.0
    recent_latencies: deque = None  # type: ignore[assignment]

    def __post_init__(self):
        self.recent_latencies = deque(maxlen=50)


class NpuScheduler:
    """One instance per physical NPU device, shared by all backends
    that touch it."""

    def __init__(self, name: str = "hailo8"):
        self.name = name
        self._lock = threading.Lock()
        self._stats = _Stats()
        self._stats_lock = threading.Lock()

    def run(self, fn: Callable[[], Any]) -> Any:
        """Run `fn` (a zero-arg callable doing the actual device call)
        with exclusive access to the device, recording latency."""
        with self._lock:
            start = time.monotonic()
            try:
                return fn()
            finally:
                elapsed = time.monotonic() - start
                with self._stats_lock:
                    self._stats.calls += 1
                    self._stats.total_latency_sec += elapsed
                    self._stats.recent_latencies.append(elapsed)

    @contextmanager
    def exclusive(self, timeout: float = -1) -> Iterator[bool]:
        """The device lock without a latency sample, for setup and teardown.
        Yields whether the lock was taken within `timeout` seconds (-1: wait
        as long as it takes)."""
        acquired = self._lock.acquire(timeout=timeout)
        try:
            yield acquired
        finally:
            if acquired:
                self._lock.release()

    def snapshot(self) -> dict:
        with self._stats_lock:
            recent = list(self._stats.recent_latencies)
            avg_recent = sum(recent) / len(recent) if recent else 0.0
            return {
                "device": self.name,
                "calls": self._stats.calls,
                "avg_latency_ms_recent": round(avg_recent * 1000, 2),
                "avg_latency_ms_overall": round(
                    (self._stats.total_latency_sec / self._stats.calls * 1000) if self._stats.calls else 0.0,
                    2,
                ),
            }


# Process-wide singleton: every HailoDetector imports and shares this so
# all calls queue behind the same lock rather than each opening their own
# device handle.
default_scheduler = NpuScheduler()
