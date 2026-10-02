"""Drop-in stand-in for RtspReader that reads a local video file
(or a folder of images) on a loop instead of a live camera.

Not in the original skeleton, added because both the test suite and
`edge_main.py --source file:<path>` need a way to run the full
pipeline without a camera or a Pi. It writes into the same
LatestFrameSlot as RtspReader and exposes the same start()/stop()
shape, so edge_main.py can use either one interchangeably.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from capture.stream_health import StreamHealth
from queues.frame_queue import LatestFrameSlot

log = logging.getLogger(__name__)


class FileFrameSource:
    def __init__(
        self,
        frame_slot: LatestFrameSlot,
        path: str | Path,
        camera_id: str = "cam01",
        target_fps: float = 8.0,
        loop: bool = True,
    ):
        self.frame_slot = frame_slot
        self.path = Path(path)
        self.camera_id = camera_id
        self.loop = loop
        self._target_interval = 1.0 / target_fps if target_fps > 0 else 0.0
        self.health = StreamHealth(camera_id, stall_timeout_sec=9999)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.loop_ts: float | None = None  # see RtspReader.loop_ts

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self.run, name=f"file-source-{self.camera_id}", daemon=True
        )
        self._thread.start()

    def stop(self, join_timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=join_timeout)

    def _iter_frames(self):
        if self.path.is_dir():
            image_paths = sorted(
                p for p in self.path.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
            while not self._stop_event.is_set():
                if not image_paths:
                    # Nothing to read; yield a blank frame so callers don't spin forever.
                    yield np.zeros((360, 640, 3), dtype=np.uint8)
                    if not self.loop:
                        return
                    continue
                for p in image_paths:
                    if self._stop_event.is_set():
                        return
                    frame = cv2.imread(str(p))
                    if frame is not None:
                        yield frame
                if not self.loop:
                    return
        else:
            # Outer loop MUST check _stop_event before reopening — without
            # this check, once the video reaches EOF the inner while's own
            # `not self._stop_event.is_set()` condition (checked only while
            # the inner loop is running) never gets evaluated again: the
            # outer `while True` would reopen and release a fresh
            # VideoCapture in a tight, unthrottled loop forever, even after
            # stop() was called. That pegged a CPU core, meant stop()'s
            # join() never actually returned, and hammering cv2/FFmpeg from
            # a thread that outlives the rest of the program is exactly
            # what caused the native "terminate called without an active
            # exception" abort observed at interpreter shutdown.
            while not self._stop_event.is_set():
                cap = cv2.VideoCapture(str(self.path))
                try:
                    if not cap.isOpened():
                        log.warning("could not open video file", extra={"path": str(self.path)})
                        return
                    while not self._stop_event.is_set():
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            break
                        yield frame
                finally:
                    cap.release()
                if not self.loop:
                    return

    def run(self) -> None:
        last_time = 0.0
        self.loop_ts = time.time()
        for frame in self._iter_frames():
            self.loop_ts = time.time()
            if self._stop_event.is_set():
                break
            now = time.monotonic()
            if now - last_time < self._target_interval:
                time.sleep(max(0.0, self._target_interval - (now - last_time)))
            last_time = time.monotonic()
            self.health.record_frame()
            self.frame_slot.put(frame, camera_id=self.camera_id)

    def grab_main_stream_frame(self, timeout_sec: float = 3.0):
        """Mirrors RtspReader's API: just returns the current frame."""
        env = self.frame_slot.get_latest(timeout=timeout_sec)
        return env.frame if env else None
