"""Event video recorder: on alert, writes a short MP4 spanning
`pre_event_seconds` before the trigger (pulled from `RamBuffer`,
pipeline/ram_buffer.py) through `post_event_seconds` after it, next to
the JPEG snapshot already saved by storage/evidence_store.py.

Listed as deferred in the batch-1 README (`pipeline/clip_extractor`) —
this is that piece.

Design notes:

- `offer_frame()` is called once per pipeline cycle (same place
  `motion_gate.score()` already is in edge_main._pipeline_loop) with
  every frame, regardless of whether anything is recording — cheap
  (an append to a bounded deque, iterate a small "active recordings"
  dict) so it never becomes the pipeline's bottleneck.
- `start()` is called once, at the moment alert_dispatcher.dispatch()
  returns an event_id — it does NOT modify alert_dispatcher.py's own
  responsibility (write the durable DB row + snapshot); recording is
  bolted on alongside it in edge_main.py, keeping AlertDispatcher's
  contract exactly what it was in batch 1.
- The actual MP4 write (cv2.VideoWriter, several seconds of frames) is
  disk I/O and takes real wall-clock time, so it always runs on its own
  short-lived daemon thread (`_write_clip`) — never on the pipeline
  thread. `on_clip_ready(event_id, path)` is a callback the caller
  supplies to learn when a clip is ready (edge_main.py wires this to
  db.outbox.set_video_path so it can be uploaded — see
  network/base_station_client.py's send_video and transfer_main.py's
  video-send loop).
- A clip failing to write (disk full, bad codec) only loses the video;
  the image alert this clip is attached to was already durably written
  to db.outbox before start() is ever called, so this never risks
  losing an event the way a lost snapshot would.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

import paths
from config_loader import load
from pipeline.ram_buffer import BufferedFrame, RamBuffer

log = logging.getLogger(__name__)

OnClipReady = Callable[[str, str], None]


@dataclass
class _ActiveRecording:
    track_id: str
    camera_id: str
    pre_roll: list[BufferedFrame]
    post_roll: list[BufferedFrame] = field(default_factory=list)
    deadline_ts: float = 0.0
    started_ts: float = field(default_factory=time.time)


def _clip_filename(camera_id: str, track_id: str, event_id: str, ts: float) -> str:
    t = time.strftime("%Y%m%dT%H%M%S", time.localtime(ts))
    return f"{t}_{camera_id}_{track_id}_{event_id[:8]}.mp4"


class ClipExtractor:
    def __init__(
        self,
        config: dict | None = None,
        clip_dir: Path | None = None,
        on_clip_ready: OnClipReady | None = None,
    ):
        cfg = (config if config is not None else load(paths.CLIP_CONFIG, required=False)).get("clip", {})
        self.enabled = bool(cfg.get("enabled", True))
        self.pre_event_seconds = float(cfg.get("pre_event_seconds", 8.0))
        self.post_event_seconds = float(cfg.get("post_event_seconds", 8.0))
        self.fps = float(cfg.get("fps", 8.0))
        self.max_width = cfg.get("max_width") or None
        self.codec = cfg.get("codec", "mp4v")
        self.max_concurrent_recordings = int(cfg.get("max_concurrent_recordings", 4))

        self.clip_dir = clip_dir or paths.CLIP_DIR
        self.clip_dir.mkdir(parents=True, exist_ok=True)
        self.on_clip_ready = on_clip_ready

        self._ram_buffer = RamBuffer.for_duration(self.pre_event_seconds, self.fps)
        self._lock = threading.Lock()
        self._active: dict[str, _ActiveRecording] = {}

    # -- called every pipeline cycle -------------------------------------

    def offer_frame(self, frame: Any, timestamp: float | None = None) -> None:
        ts = timestamp if timestamp is not None else time.time()
        self._ram_buffer.offer(frame, ts)

        to_finalize: list[tuple[str, _ActiveRecording]] = []
        with self._lock:
            for event_id, rec in list(self._active.items()):
                rec.post_roll.append(BufferedFrame(frame=frame, timestamp=ts))
                if ts >= rec.deadline_ts:
                    to_finalize.append((event_id, rec))
                    del self._active[event_id]

        for event_id, rec in to_finalize:
            self._finalize_async(event_id, rec)

    # -- called once, at alert-dispatch time ------------------------------

    def start(self, event_id: str, track_id: str, camera_id: str = "") -> bool:
        """Begin recording pre-roll (already buffered) + post-roll
        (collected from here on via offer_frame) for `event_id`.
        Returns False (no-op) if disabled, already recording this
        event, or at `max_concurrent_recordings`."""
        if not self.enabled:
            return False
        with self._lock:
            if event_id in self._active:
                return False
            if len(self._active) >= self.max_concurrent_recordings:
                log.warning(
                    "clip recording skipped: max_concurrent_recordings reached",
                    extra={"event_id": event_id, "track_id": track_id},
                )
                return False
            now = time.time()
            self._active[event_id] = _ActiveRecording(
                track_id=track_id,
                camera_id=camera_id,
                pre_roll=self._ram_buffer.frames(),
                deadline_ts=now + self.post_event_seconds,
                started_ts=now,
            )
        log.info("clip recording started", extra={"event_id": event_id, "track_id": track_id})
        return True

    # -- finalization ------------------------------------------------------

    def _finalize_async(self, event_id: str, rec: _ActiveRecording) -> None:
        thread = threading.Thread(
            target=self._write_clip, args=(event_id, rec), name=f"clip-writer-{event_id[:8]}", daemon=True
        )
        thread.start()

    def _write_clip(self, event_id: str, rec: _ActiveRecording) -> None:
        try:
            out_path = self._write_mp4(event_id, rec)
        except Exception:
            log.exception("failed to write event clip", extra={"event_id": event_id})
            return
        log.info(
            "clip written",
            extra={
                "event_id": event_id,
                "path": str(out_path),
                "frames": len(rec.pre_roll) + len(rec.post_roll),
            },
        )
        if self.on_clip_ready:
            try:
                self.on_clip_ready(event_id, str(out_path))
            except Exception:
                log.exception("on_clip_ready callback failed", extra={"event_id": event_id})

    def _write_mp4(self, event_id: str, rec: _ActiveRecording) -> Path:
        all_frames = rec.pre_roll + rec.post_roll
        if not all_frames:
            raise ValueError("no frames buffered for clip")

        h, w = all_frames[0].frame.shape[:2]
        if self.max_width and w > self.max_width:
            scale = self.max_width / w
            w, h = int(self.max_width), int(round(h * scale))

        out_path = self.clip_dir / _clip_filename(rec.camera_id, rec.track_id, event_id, rec.started_ts)
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(str(out_path), fourcc, self.fps, (w, h))
        try:
            if not writer.isOpened():
                raise OSError(f"failed to open VideoWriter for {out_path} (codec={self.codec})")
            for buffered in all_frames:
                img = buffered.frame
                if (img.shape[1], img.shape[0]) != (w, h):
                    img = cv2.resize(img, (w, h))
                writer.write(img)
        finally:
            writer.release()
        return out_path

    # -- introspection / shutdown -------------------------------------------

    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    def flush_all(self) -> None:
        """Finalize every in-progress recording immediately with whatever
        post-roll it has so far, instead of waiting for its full
        post_event_seconds. Used at process shutdown so an event near
        the end of a run still gets a (shorter) clip instead of none."""
        with self._lock:
            pending = list(self._active.items())
            self._active.clear()
        for event_id, rec in pending:
            self._finalize_async(event_id, rec)
