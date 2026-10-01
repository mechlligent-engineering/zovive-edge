"""Bounded, drop-oldest queue carrying inference results from the
Detector thread to the tracking/pipeline thread.

Sized from configs/queue_config.yaml (`detection_queue_size`, default
small — 4) on purpose: a growing backlog here means the pipeline
thread is falling behind the detector, and we would rather drop a
stale detection batch than let latency creep up silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import paths
from config_loader import load
from inference.base import Detection
from queues.queue_monitor import BoundedDropOldestQueue


@dataclass
class DetectionBatch:
    """One inference cycle's output: the frame it came from plus the
    Detector's already-decoded, already-NMS'd Detection objects."""

    seq: int
    timestamp: float
    frame: Any
    detections: list[Detection] = field(default_factory=list)
    camera_id: str = ""


def _default_size() -> int:
    cfg = load(paths.QUEUE_CONFIG, required=False)
    return int(cfg.get("queues", {}).get("detection_queue_size", 4))


class DetectionQueue(BoundedDropOldestQueue[DetectionBatch]):
    def __init__(self, maxsize: int | None = None):
        super().__init__("detection_queue", maxsize or _default_size())
