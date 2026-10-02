"""Rolling in-memory buffer of the most recent N seconds of frames.

Listed as deferred in the batch-1 README (`pipeline/ram_buffer`) — this
is that piece, and it exists for exactly one reason: by the time a
track is CONFIRMED and an alert fires, the *interesting* moment (the
animal stepping into frame) already happened a few seconds ago. Without
a pre-event buffer, an event video could only start recording from the
moment of the alert, missing that lead-in entirely. This buffer is fed
every pipeline cycle (cheap: an append + maybe a pop) so
`pipeline/clip_extractor.py` always has the last `window_seconds`
worth of frames on hand to prepend to a new clip the instant a track
gets confirmed.

Sized in frames rather than a fixed time.sleep-style window because
that's what a deque needs, computed from `window_seconds * fps` once
at construction (see `RamBuffer.for_duration`).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class BufferedFrame:
    frame: Any
    timestamp: float


class RamBuffer:
    """Bounded, drop-oldest ring buffer of (frame, timestamp) pairs.
    Not a queue — there's no "consumer" that removes items; every
    reader (`frames()` / `frames_since()`) just gets a snapshot copy
    of what's currently held, and old entries are evicted automatically
    as new ones arrive.
    """

    def __init__(self, max_frames: int):
        if max_frames < 1:
            raise ValueError("max_frames must be >= 1")
        self.max_frames = max_frames
        self._buf: deque[BufferedFrame] = deque(maxlen=max_frames)

    @classmethod
    def for_duration(cls, window_seconds: float, fps: float) -> RamBuffer:
        frames = max(1, int(round(window_seconds * max(fps, 0.1))))
        return cls(frames)

    def offer(self, frame: Any, timestamp: float | None = None) -> None:
        self._buf.append(
            BufferedFrame(frame=frame, timestamp=timestamp if timestamp is not None else time.time())
        )

    def frames(self) -> list[BufferedFrame]:
        """A chronological snapshot of everything currently buffered.
        Returns a new list each call — safe for the caller to hold onto
        even as more frames are offered afterwards."""
        return list(self._buf)

    def frames_since(self, since_ts: float) -> list[BufferedFrame]:
        return [f for f in self._buf if f.timestamp >= since_ts]

    def __len__(self) -> int:
        return len(self._buf)
