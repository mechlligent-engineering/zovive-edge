"""Alerts raised recently, by place and class, so the same animal (or the
same false positive) isn't alerted again under a new track id.

The tracker starts a new track whenever it misses a detection for more than
max_age_frames cycles, and the stage-1 cooldown is per track id, so without
this a detection that stays in view alerts once per track fragment
(docs/REALERT_SUPPRESSION_SPEC.md: 59 alerts in one night from a static
object). Every dispatched alert is remembered here, wide view or zoomed; a
new track whose centre falls inside a remembered box of the same class is
the same thing again. An entry lives for `window_sec` after the last time a
live track was seen at its place, so the same place alerts again only after
it has been out of view that long.

Boxes are wide-view pixel coordinates: zoomed-view boxes are never compared.
Not thread-safe; the pipeline thread is its only user.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

Box = tuple[float, float, float, float]

DEFAULT_WINDOW_SEC = 300.0
DEFAULT_GROW = 0.5  # the remembered box, grown by half its size on each side


@dataclass
class RecentAlert:
    box: Box
    class_name: str
    species: str | None
    confidence: float
    event_id: str | None
    expires_ts: float


def center_inside(box: Box, region: Box, grow: float) -> bool:
    x1, y1, x2, y2 = region
    gx, gy = (x2 - x1) * grow, (y2 - y1) * grow
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    return (x1 - gx) <= cx <= (x2 + gx) and (y1 - gy) <= cy <= (y2 + gy)


class RecentAlerts:
    def __init__(
        self,
        window_sec: float = DEFAULT_WINDOW_SEC,
        grow: float = DEFAULT_GROW,
        clock: Callable[[], float] = time.monotonic,
    ):
        if window_sec <= 0:
            raise ValueError("window_sec must be > 0")
        self.window_sec = window_sec
        self.grow = grow
        self.clock = clock
        self._entries: list[RecentAlert] = []

    def __len__(self) -> int:
        return len(self._entries)

    def _expire(self, now: float) -> None:
        self._entries = [e for e in self._entries if e.expires_ts > now]

    def remember(
        self,
        box: Box,
        class_name: str,
        species: str | None,
        confidence: float,
        event_id: str | None,
        now: float | None = None,
    ) -> None:
        now = now if now is not None else self.clock()
        self._entries.append(
            RecentAlert(box, class_name, species, confidence, event_id, now + self.window_sec)
        )

    def match(self, class_name: str, box: Box, now: float | None = None) -> RecentAlert | None:
        """The live entry this box is probably the same thing as."""
        now = now if now is not None else self.clock()
        self._expire(now)
        for e in self._entries:
            if e.class_name == class_name and center_inside(box, e.box, self.grow):
                return e
        return None

    def seen(self, class_name: str, box: Box, now: float | None = None) -> bool:
        """A live track is at this place: keep every entry it matches alive
        for another window. Returns whether any entry matched."""
        now = now if now is not None else self.clock()
        self._expire(now)
        matched = False
        for e in self._entries:
            if e.class_name == class_name and center_inside(box, e.box, self.grow):
                e.expires_ts = now + self.window_sec
                matched = True
        return matched
