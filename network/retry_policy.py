"""Upload retry rules for transfer_main.py. The retry *state* lives on the
outbox rows (db/outbox.py claim_due / record_send_failure), so it survives
restarts and can be shared by several senders; this module only decides when
the next try happens.

A failed send is one of three kinds (network/base_station_client.py):

- rejected (HTTP 4xx other than 429): the Base Station refused this item, so
  it costs one retry. After the n-th rejection the item waits
  ``schedule_sec[n-1]`` (+/- ``jitter``); after more than ``max_retries``
  rejections it is dead-lettered. With the defaults that is the first send
  plus 5 retries after 30 s, 1 m, 2 m, 5 m and 15 m.
- unreachable (DNS failure, connection refused, connect timeout, TLS): costs
  nothing; every due item of both queues waits and the Base Station is probed
  about every ``pause_sec``.
- busy (HTTP 5xx, 429, 3xx, read timeout): costs nothing; that queue waits
  ``pause_sec``, or a 429's Retry-After capped at ``max_pause_sec``. The item
  that failed goes ``BEHIND_SEC`` behind the rest, so one item the Base
  Station always fails on cannot block the queue.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# The failing item's next try comes this long after the queue's pause ends, so
# the next probe tries a different item first.
BEHIND_SEC = 1.0

# Added to the longest legitimate wait when deciding that a next_retry_ts is so
# far ahead that the clock must have jumped back (db/outbox.py _due); covers
# any send lease.
_SKEW_MARGIN_SEC = 3600.0


@dataclass(frozen=True)
class RetryPolicy:
    schedule_sec: tuple[float, ...] = (30.0, 60.0, 120.0, 300.0, 900.0)
    max_retries: int = 5
    jitter: float = 0.2
    pause_sec: float = 60.0
    max_pause_sec: float = 900.0

    def __post_init__(self) -> None:
        s = self.schedule_sec
        if not s or any(step <= 0 for step in s) or list(s) != sorted(s):
            raise ValueError(
                f"transfer.retry.schedule_sec must be positive and non-decreasing, got {list(s)}"
            )
        if self.max_retries < 0:
            raise ValueError(f"transfer.retry.max_retries must be >= 0, got {self.max_retries}")
        if not 0 <= self.jitter < 1:
            raise ValueError(f"transfer.retry.jitter must be in [0, 1), got {self.jitter}")
        if not 0 < self.pause_sec <= self.max_pause_sec:
            raise ValueError(
                "transfer.retry needs 0 < pause_sec <= max_pause_sec, "
                f"got {self.pause_sec} and {self.max_pause_sec}"
            )

    @classmethod
    def from_config(cls, transfer_cfg: Mapping[str, Any]) -> RetryPolicy:
        """From transfer_config.yaml's ``transfer.retry`` section (defaults for
        anything missing). Raises ValueError on invalid values."""
        r = transfer_cfg.get("retry") or {}
        d = cls()
        return cls(
            schedule_sec=tuple(float(x) for x in r.get("schedule_sec", d.schedule_sec)),
            max_retries=int(r.get("max_retries", d.max_retries)),
            jitter=float(r.get("jitter", d.jitter)),
            pause_sec=float(r.get("pause_sec", d.pause_sec)),
            max_pause_sec=float(r.get("max_pause_sec", d.max_pause_sec)),
        )

    def retry_at(self, now: float, rejections: int, rng: random.Random) -> float | None:
        """When to try again after an item's ``rejections``-th rejection, or
        None when it is out of retries (dead-letter it)."""
        if rejections > self.max_retries:
            return None
        step = self.schedule_sec[min(rejections, len(self.schedule_sec)) - 1]
        return now + step * rng.uniform(1 - self.jitter, 1 + self.jitter)

    def pause_until(self, now: float, rng: random.Random, retry_after: float | None = None) -> float:
        """End of a queue pause (Base Station unreachable or busy). A
        Retry-After is honoured, never shortened, and capped at max_pause_sec."""
        if retry_after is not None:
            return now + min(max(retry_after, 0.0), self.max_pause_sec) * rng.uniform(1, 1 + self.jitter)
        return now + self.pause_sec * rng.uniform(1 - self.jitter, 1 + self.jitter)

    @property
    def max_wait_sec(self) -> float:
        """No legitimate next_retry_ts is further than this ahead of now."""
        longest = max(self.schedule_sec[-1], self.max_pause_sec)
        return longest * (1 + self.jitter) + BEHIND_SEC + _SKEW_MARGIN_SEC
