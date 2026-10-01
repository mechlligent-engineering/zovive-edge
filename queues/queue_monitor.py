"""Shared bounded, drop-oldest queue plus a process-wide metrics registry.

Every inter-thread queue in this codebase (detection_queue, alert_queue,
offline_queue) is built on `BoundedDropOldestQueue` rather than
`queue.Queue` directly, because the failure mode we want under load is
"lose the stalest item and keep moving", not "block the producer
forever" or "grow without bound". health_main.py reads `snapshot()`
for the heartbeat, so an operator can see queues filling up and drops
happening before anything crashes.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class QueueStats:
    name: str
    maxsize: int
    put_count: int = 0
    drop_count: int = 0
    get_count: int = 0
    current_size: int = 0
    last_put_ts: float | None = None
    last_drop_ts: float | None = None


class _Registry:
    """Process-wide, thread-safe map of queue name -> QueueStats."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats: dict[str, QueueStats] = {}

    def register(self, name: str, maxsize: int) -> QueueStats:
        with self._lock:
            stats = self._stats.setdefault(name, QueueStats(name=name, maxsize=maxsize))
            return stats

    def snapshot(self) -> dict[str, dict]:
        with self._lock:
            return {
                name: {
                    "maxsize": s.maxsize,
                    "current_size": s.current_size,
                    "put_count": s.put_count,
                    "drop_count": s.drop_count,
                    "get_count": s.get_count,
                    "last_put_ts": s.last_put_ts,
                    "last_drop_ts": s.last_drop_ts,
                }
                for name, s in self._stats.items()
            }


registry = _Registry()


class BoundedDropOldestQueue(Generic[T]):
    """Thread-safe FIFO with a fixed capacity that drops the OLDEST item
    (not the newest) when full, and records every drop in `registry`.

    Why drop-oldest: the newest frame/detection/alert is the one most
    likely to still be relevant. Blocking the producer (like a normal
    bounded queue.Queue.put would) instead risks stalling capture or
    inference, which is worse than losing a stale item.
    """

    def __init__(self, name: str, maxsize: int):
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.name = name
        self.maxsize = maxsize
        self._dq: deque[T] = deque()
        self._cv = threading.Condition()
        self._closed = False
        self._stats = registry.register(name, maxsize)

    def put(self, item: T) -> bool:
        """Push an item. Returns True if something was dropped to make room."""
        dropped = False
        with self._cv:
            if len(self._dq) >= self.maxsize:
                self._dq.popleft()
                dropped = True
                self._stats.drop_count += 1
                self._stats.last_drop_ts = time.time()
            self._dq.append(item)
            self._stats.put_count += 1
            self._stats.current_size = len(self._dq)
            self._stats.last_put_ts = time.time()
            self._cv.notify()
        return dropped

    def get(self, timeout: float | None = None) -> T | None:
        """Pop the oldest item, blocking up to `timeout` seconds. None on timeout/close."""
        with self._cv:
            deadline = None if timeout is None else time.monotonic() + timeout
            while not self._dq and not self._closed:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._cv.wait(timeout=remaining)
            if not self._dq:
                return None
            item = self._dq.popleft()
            self._stats.get_count += 1
            self._stats.current_size = len(self._dq)
            return item

    def qsize(self) -> int:
        with self._cv:
            return len(self._dq)

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()


def snapshot() -> dict[str, dict]:
    """All registered queues' stats, for health_main.py's heartbeat payload."""
    return registry.snapshot()
