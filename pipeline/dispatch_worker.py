"""Alert disk writes off the pipeline thread (docs/PIPELINE_HARDENING_SPEC.md,
change 3).

AlertDispatcher (pipeline/alert_dispatcher.py) saves an event's snapshot,
inserts its outbox row and announces it to transfer_main.py, in that
order. DispatchWorker runs those three disk steps on a thread of its own,
so a slow or stalled SD card never holds up the pipeline thread:

    pipeline thread   dispatch(): policy, event_id and priority
                      (AlertDispatcher.prepare(), no I/O), then a
                      non-blocking put on a bounded FIFO queue. Returns
                      the event_id at once, so the track is marked alerted
                      and its clip starts at the trigger, as before.
    dispatch thread   run(): for each job, save the snapshot, insert the
                      row, push the alert_queue payload.
    clip writers      video_path_ready(): a finished clip's path goes
                      through the same queue, so it is recorded after its
                      event row exists.

Kept from AlertDispatcher: snapshot before row (a row never points at a
missing snapshot), row before the alert_queue push, the same events
schema and one snapshot per event.

Nothing is dropped silently:
- Queue full (the disk has been stalled through `maxsize` alerts):
  dispatch() logs an ERROR naming the event, counts it as rejected and
  returns None, so the pipeline treats the track as not alerted and no
  clip is recorded without a row.
- A step fails: the job is retried from the step that failed (a saved
  snapshot is never saved twice; the evidence store refuses to overwrite
  one) up to `max_attempts` times with backoff, then logged as an ERROR
  naming the event and counted as failed.
- Shutdown: run() keeps writing after the stop event until the queue is
  empty or `drain_timeout_sec` has passed; anything still queued then is
  logged as an ERROR and counted as failed.
The counts are in the edge status file's `dispatch` section; the queue's
depth and rejections are in its `queues` section, like every other queue.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from db.outbox import EventRecord, set_video_path
from pipeline.alert_dispatcher import AlertDispatcher
from queues.queue_monitor import registry

log = logging.getLogger(__name__)

QUEUE_NAME = "dispatch_queue"
_MAX_ERROR_CHARS = 300


@dataclass
class _AlertJob:
    record: EventRecord
    # Held by reference, as BestSnapshotStore holds it: frames are never
    # reused after capture.
    snapshot_image: np.ndarray
    # Progress, so a retry resumes at the step that failed.
    snapshot_saved: bool = False
    inserted: bool = False


@dataclass(frozen=True)
class _VideoPathJob:
    event_id: str
    video_path: str


def _describe(record: EventRecord) -> dict[str, Any]:
    return {
        "event_id": record.event_id,
        "track_id": record.track_id,
        "species": record.animal_name,
        "confidence": round(record.confidence, 3),
        "priority": record.priority,
    }


class DispatchWorker:
    def __init__(
        self,
        dispatcher: AlertDispatcher,
        maxsize: int = 256,
        max_attempts: int = 3,
        retry_backoff_sec: float = 0.5,
        drain_timeout_sec: float = 10.0,
        video_put_timeout_sec: float = 30.0,
        on_video_path: Callable[[str, str], None] = set_video_path,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ):
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.dispatcher = dispatcher
        self.maxsize = maxsize
        self.max_attempts = max_attempts
        self.retry_backoff_sec = retry_backoff_sec
        self.drain_timeout_sec = drain_timeout_sec
        self.video_put_timeout_sec = video_put_timeout_sec
        self.on_video_path = on_video_path
        self.sleep = sleep
        self.clock = clock
        self.wall_clock = wall_clock

        self._queue: queue.Queue[_AlertJob | _VideoPathJob] = queue.Queue(maxsize=maxsize)
        self._stats = registry.register(QUEUE_NAME, maxsize)
        self._lock = threading.Lock()
        self._running = False
        # Liveness for watchdog/supervisor.py: refreshed every loop, idle or not.
        self.loop_ts: float | None = None
        self._dispatched = 0
        self._rejected = 0
        self._failed = 0
        self._retries = 0
        self._video_paths = 0
        self._video_path_failures = 0
        self._last_error: str | None = None
        self._last_error_ts: float | None = None

    # -- pipeline thread ---------------------------------------------------------

    def dispatch(
        self,
        track_id: str,
        species: str,
        confidence: float,
        snapshot_image: np.ndarray,
        preset_name: str = "",
        model_version: str = "",
    ) -> str | None:
        """AlertDispatcher.dispatch()'s contract, but returns as soon as the
        event is queued. None if policy suppresses it or the queue is full."""
        record = self.dispatcher.prepare(track_id, species, confidence, preset_name, model_version)
        if record is None:
            return None
        if not self._put(_AlertJob(record, snapshot_image)):
            log.error(
                "alert dispatch queue full; this alert is NOT recorded",
                extra={**_describe(record), "queue_size": self.maxsize},
            )
            with self._lock:
                self._rejected += 1
                self._note_error("dispatch queue full")
            return None
        return record.event_id

    # -- clip writer threads -----------------------------------------------------

    def video_path_ready(self, event_id: str, video_path: str) -> None:
        """ClipExtractor's on_clip_ready. Queued behind the event's own job,
        so the path is recorded after the row exists. Recorded directly when
        the worker isn't running (after shutdown its jobs are done) or the
        queue stays full for video_put_timeout_sec."""
        job = _VideoPathJob(event_id, video_path)
        if self._running and self._put(job, timeout=self.video_put_timeout_sec):
            return
        if self._running:
            log.warning(
                "dispatch queue full; recording the clip's path directly", extra={"event_id": event_id}
            )
        self._record_video_path(job)

    # -- dispatch thread ---------------------------------------------------------

    def run(self, stop_event: threading.Event) -> None:
        """The dispatch thread's loop (a SupervisedThread target). After
        `stop_event` it drains the queue, for at most drain_timeout_sec."""
        self._running = True
        try:
            while not stop_event.is_set():
                self.loop_ts = self.wall_clock()
                job = self._get(wait=1.0)
                if job is not None:
                    self._handle(job)
            self._drain()
        finally:
            self._running = False
            if stop_event.is_set():
                self._abandon_remaining()
        log.info("dispatch worker stopped")

    def _drain(self) -> None:
        deadline = self.clock() + self.drain_timeout_sec
        while self.clock() < deadline:
            self.loop_ts = self.wall_clock()
            job = self._get()
            if job is None:
                return
            self._handle(job)

    def _abandon_remaining(self) -> None:
        """After the drain: a clip path can still be recorded, an alert that
        never got its turn is reported, never silently left behind."""
        while (job := self._get()) is not None:
            if isinstance(job, _VideoPathJob):
                self._record_video_path(job)
            else:
                self._fail(
                    job.record, f"not written before shutdown (drain timeout {self.drain_timeout_sec} s)"
                )

    def _handle(self, job: _AlertJob | _VideoPathJob) -> None:
        if isinstance(job, _VideoPathJob):
            self._record_video_path(job)
            return
        for attempt in range(1, self.max_attempts + 1):
            try:
                self._write(job)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"[:_MAX_ERROR_CHARS]
                if attempt == self.max_attempts:
                    self._fail(job.record, error)
                    return
                log.warning(
                    "alert write failed; retrying",
                    extra={**_describe(job.record), "attempt": attempt, "error": error},
                )
                with self._lock:
                    self._retries += 1
                self.loop_ts = self.wall_clock()
                self.sleep(self.retry_backoff_sec * 2 ** (attempt - 1))
                continue
            with self._lock:
                self._dispatched += 1
            return

    def _write(self, job: _AlertJob) -> None:
        if not job.snapshot_saved:
            self.dispatcher.save_snapshot(job.record, job.snapshot_image)
            job.snapshot_saved = True
        if not job.inserted:
            self.dispatcher.insert(job.record)
            job.inserted = True
        self.dispatcher.announce(job.record)

    def _fail(self, record: EventRecord, error: str) -> None:
        log.error(
            "alert could not be written; this alert is NOT recorded",
            extra={**_describe(record), "error": error},
        )
        with self._lock:
            self._failed += 1
            self._note_error(error)

    def _record_video_path(self, job: _VideoPathJob) -> None:
        try:
            self.on_video_path(job.event_id, job.video_path)
        except Exception:
            log.exception("could not record the clip's video path", extra={"event_id": job.event_id})
            with self._lock:
                self._video_path_failures += 1
            return
        with self._lock:
            self._video_paths += 1

    # -- queue and counters ----------------------------------------------------------

    def _put(self, job: _AlertJob | _VideoPathJob, timeout: float | None = None) -> bool:
        try:
            if timeout is None:
                self._queue.put_nowait(job)
            else:
                self._queue.put(job, timeout=timeout)
        except queue.Full:
            with self._lock:
                self._stats.drop_count += 1
                self._stats.last_drop_ts = self.wall_clock()
            return False
        with self._lock:
            self._stats.put_count += 1
            self._stats.current_size = self._queue.qsize()
            self._stats.last_put_ts = self.wall_clock()
        return True

    def _get(self, wait: float = 0.0) -> _AlertJob | _VideoPathJob | None:
        """The next job, waiting up to `wait` seconds; None if there is none."""
        try:
            job = self._queue.get(timeout=wait) if wait > 0 else self._queue.get_nowait()
        except queue.Empty:
            return None
        with self._lock:
            self._stats.get_count += 1
            self._stats.current_size = self._queue.qsize()
        return job

    def _note_error(self, error: str) -> None:
        """Caller holds self._lock."""
        self._last_error = error[:_MAX_ERROR_CHARS]
        self._last_error_ts = self.wall_clock()

    def qsize(self) -> int:
        return self._queue.qsize()

    def snapshot(self) -> dict[str, Any]:
        """For the edge status file (watchdog/edge_status.py)."""
        with self._lock:
            return {
                "queued": self._queue.qsize(),
                "maxsize": self.maxsize,
                "dispatched": self._dispatched,
                "rejected": self._rejected,
                "failed": self._failed,
                "retries": self._retries,
                "video_paths": self._video_paths,
                "video_path_failures": self._video_path_failures,
                "last_error": self._last_error,
                "last_error_ts": self._last_error_ts,
                "loop_ts": self.loop_ts,
            }
