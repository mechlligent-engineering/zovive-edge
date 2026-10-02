"""Per-event retry backoff bookkeeping for transfer_main.py, separate
from db.outbox's status column: the outbox says pending/sent/failed
for anyone inspecting the DB, while this in-memory helper decides
*when* a failed event is worth retrying again, using the same
exponential-backoff shape as capture/reconnect.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class _RetryEntry:
    next_attempt_ts: float
    backoff_sec: float


class TransferState:
    def __init__(
        self, initial_backoff_sec: float = 2.0, max_backoff_sec: float = 300.0, multiplier: float = 2.0
    ):
        self.initial = initial_backoff_sec
        self.max = max_backoff_sec
        self.multiplier = multiplier
        self._entries: dict[str, _RetryEntry] = {}

    def is_ready(self, event_id: str, now: float | None = None) -> bool:
        entry = self._entries.get(event_id)
        if entry is None:
            return True
        return (now if now is not None else time.time()) >= entry.next_attempt_ts

    def record_failure(self, event_id: str, now: float | None = None) -> float:
        now = now if now is not None else time.time()
        entry = self._entries.get(event_id)
        backoff = self.initial if entry is None else min(entry.backoff_sec * self.multiplier, self.max)
        self._entries[event_id] = _RetryEntry(next_attempt_ts=now + backoff, backoff_sec=backoff)
        return backoff

    def clear(self, event_id: str) -> None:
        self._entries.pop(event_id, None)
