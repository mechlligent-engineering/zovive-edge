"""Bounded queue carrying finished alert payloads from the pipeline
thread to transfer_main.py's sender.

Durability does NOT depend on this queue: alert_dispatcher.py writes
each alert to the SQLite outbox (db/) *before* pushing it here, so a
drop from this in-memory queue under extreme load just means the
sender picks it up a little later by scanning the outbox instead of
being pushed it directly — nothing is lost. We still log drops at
WARNING (louder than the generic drop-oldest queue) because a drop
here usually means the sender is stuck.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import paths
from config_loader import load
from queues.queue_monitor import BoundedDropOldestQueue

log = logging.getLogger(__name__)


@dataclass
class AlertPayload:
    event_id: str
    track_id: str
    camera_id: str
    species: str
    confidence: float
    snapshot_path: str
    timestamp: float
    priority: str = "normal"  # normal | high
    extra: dict[str, Any] | None = None


def _default_size() -> int:
    cfg = load(paths.QUEUE_CONFIG, required=False)
    return int(cfg.get("queues", {}).get("alert_queue_size", 64))


class AlertQueue(BoundedDropOldestQueue[AlertPayload]):
    def __init__(self, maxsize: int | None = None):
        super().__init__("alert_queue", maxsize or _default_size())

    def put(self, item: AlertPayload) -> bool:
        dropped = super().put(item)
        if dropped:
            log.warning(
                "alert_queue full, dropped oldest pending alert",
                extra={"event_id": item.event_id, "camera_id": item.camera_id},
            )
        return dropped
