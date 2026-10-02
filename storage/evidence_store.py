"""Saves alert snapshots to disk and keeps total snapshot usage under a
cap by deleting the oldest already-sent files first.

Full retention policy (age-based purge, orphan scanning, disk-guard
emergency eviction) lives in storage/retention_manager.py,
storage/purge_service.py, storage/disk_guard.py and
storage/orphan_scanner.py — deferred to the next batch. This module
covers what alert_dispatcher.py needs right now: write a snapshot,
get a stable path back, keep the directory from growing unbounded.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

import paths

log = logging.getLogger(__name__)


def _event_filename(camera_id: str, track_id: str, event_id: str, ts: float) -> str:
    t = time.strftime("%Y%m%dT%H%M%S", time.localtime(ts))
    return f"{t}_{camera_id}_{track_id}_{event_id[:8]}.jpg"


class EvidenceStore:
    def __init__(
        self, snapshot_dir: Path | None = None, max_bytes: int | None = None, jpeg_quality: int = 90
    ):
        self.snapshot_dir = snapshot_dir or paths.SNAPSHOT_DIR
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        # None = no cap enforced here (leave it to storage/disk_guard.py in batch 2).
        self.max_bytes = max_bytes
        self.jpeg_quality = jpeg_quality

    def save_snapshot(
        self, image: np.ndarray, camera_id: str, track_id: str, event_id: str, ts: float | None = None
    ) -> Path:
        ts = ts if ts is not None else time.time()
        filename = _event_filename(camera_id, track_id, event_id, ts)
        out_path = self.snapshot_dir / filename
        ok = cv2.imwrite(str(out_path), image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise OSError(f"failed to write snapshot to {out_path}")
        if self.max_bytes:
            self._enforce_cap()
        return out_path

    def _enforce_cap(self) -> None:
        files = sorted(
            (p for p in self.snapshot_dir.glob("*.jpg") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
        )
        total = sum(p.stat().st_size for p in files)
        i = 0
        while total > self.max_bytes and i < len(files):
            p = files[i]
            try:
                size = p.stat().st_size
                p.unlink()
                total -= size
                log.info("evicted snapshot to stay under disk cap", extra={"path": str(p)})
            except OSError:
                log.exception("failed to evict snapshot", extra={"path": str(p)})
            i += 1

    def usage_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.snapshot_dir.glob("*.jpg") if p.is_file())
