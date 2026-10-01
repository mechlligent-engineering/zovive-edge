"""Tracks liveness/quality metrics for one camera stream.

rtsp_reader.py updates this on every frame and every reconnect;
health_main.py reads `snapshot()` for the heartbeat sent to the base
station so a stalled or flapping camera is visible remotely instead of
only in local logs.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class _Counters:
    frames_received: int = 0
    reconnect_count: int = 0
    stalled_count: int = 0
    last_frame_ts: float | None = None
    last_reconnect_ts: float | None = None
    stream_started_ts: float = field(default_factory=time.time)


class StreamHealth:
    def __init__(self, camera_id: str, stall_timeout_sec: float = 5.0):
        self.camera_id = camera_id
        self.stall_timeout_sec = stall_timeout_sec
        self._lock = threading.Lock()
        self._c = _Counters()
        self._fps_window: list[float] = []

    def record_frame(self) -> None:
        now = time.time()
        with self._lock:
            if self._c.last_frame_ts is not None:
                self._fps_window.append(now - self._c.last_frame_ts)
                if len(self._fps_window) > 30:
                    self._fps_window.pop(0)
            self._c.frames_received += 1
            self._c.last_frame_ts = now

    def record_reconnect(self) -> None:
        with self._lock:
            self._c.reconnect_count += 1
            self._c.last_reconnect_ts = time.time()

    def record_stall(self) -> None:
        with self._lock:
            self._c.stalled_count += 1

    def is_stalled(self) -> bool:
        with self._lock:
            if self._c.last_frame_ts is None:
                # not stalled until the stream had a chance to start
                return (time.time() - self._c.stream_started_ts) > self.stall_timeout_sec
            return (time.time() - self._c.last_frame_ts) > self.stall_timeout_sec

    def fps(self) -> float:
        with self._lock:
            if len(self._fps_window) < 2:
                return 0.0
            avg_interval = sum(self._fps_window) / len(self._fps_window)
            return 1.0 / avg_interval if avg_interval > 0 else 0.0

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "camera_id": self.camera_id,
                "frames_received": self._c.frames_received,
                "reconnect_count": self._c.reconnect_count,
                "stalled_count": self._c.stalled_count,
                "last_frame_ts": self._c.last_frame_ts,
                "last_reconnect_ts": self._c.last_reconnect_ts,
                "fps": round(self.fps(), 2),
                "is_stalled": self.is_stalled(),
                "uptime_sec": round(time.time() - self._c.stream_started_ts, 1),
            }
