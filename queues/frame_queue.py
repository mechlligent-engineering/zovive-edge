"""Single-slot "latest frame" hand-off between the capture thread and
the inference thread.

A normal queue is wrong here: if inference falls behind, we want the
*newest* frame next time it asks, not a backlog of stale ones (RTSP
lag only grows if you queue every frame). So this holds exactly one
frame, always overwritten by the latest `put`, with a monotonically
increasing sequence number so a consumer can tell whether it already
processed the current slot.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from queues.queue_monitor import registry


@dataclass
class FrameEnvelope:
    frame: Any  # numpy ndarray (BGR), kept generic to avoid a hard cv2 dep here
    seq: int
    timestamp: float
    camera_id: str = ""


class LatestFrameSlot:
    def __init__(self, name: str = "frame_slot"):
        self.name = name
        self._lock = threading.Condition()
        self._envelope: FrameEnvelope | None = None
        self._seq = 0
        self._stats = registry.register(name, maxsize=1)

    def put(self, frame: Any, camera_id: str = "") -> int:
        """Overwrite the slot with a new frame. Returns the new sequence number."""
        with self._lock:
            self._seq += 1
            self._envelope = FrameEnvelope(
                frame=frame, seq=self._seq, timestamp=time.time(), camera_id=camera_id
            )
            self._stats.put_count += 1
            self._stats.current_size = 1
            self._stats.last_put_ts = self._envelope.timestamp
            self._lock.notify_all()
            return self._seq

    def get_latest(self, timeout: float | None = None) -> FrameEnvelope | None:
        """Return whatever frame is currently in the slot (non-blocking read).
        Blocks only until the *first* frame ever arrives, up to `timeout`.
        """
        with self._lock:
            if self._envelope is None:
                self._lock.wait(timeout=timeout)
            if self._envelope is not None:
                self._stats.get_count += 1
            return self._envelope

    def wait_for_next(self, after_seq: int, timeout: float | None = None) -> FrameEnvelope | None:
        """Block until a frame newer than `after_seq` is available."""
        with self._lock:
            deadline = None if timeout is None else time.monotonic() + timeout
            while self._envelope is None or self._envelope.seq <= after_seq:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                if not self._lock.wait(timeout=remaining):
                    return None
            self._stats.get_count += 1
            return self._envelope

    @property
    def last_seq(self) -> int:
        with self._lock:
            return self._seq
