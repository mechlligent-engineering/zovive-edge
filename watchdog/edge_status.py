"""Detection-process health, shared from edge_main.py to health_main.py.

The two run as separate processes, so health_main cannot see edge_main's
threads or in-memory queues directly. Instead:

    edge_main.py  : its threads update an `EdgeStatus` as they work; the
                    main thread writes `EdgeStatus.snapshot()` to
                    `paths.EDGE_STATUS_PATH` every few seconds
                    (`write_status_file`, atomic replace).
    health_main.py: reads that file each heartbeat (`read_status_file`) and
                    turns it into verdicts (`evaluate_edge_health`).

Each loop records a timestamp on EVERY iteration, including idle ones that
waited for input and got none, so an old timestamp means the thread is
stuck or dead rather than merely idle. That is what lets the base station
tell "no animals" apart from "the pipeline thread died an hour ago".
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

UNKNOWN = "unknown"
STALE = "stale"


class EdgeStatus:
    """Thread-safe recorder, one per edge_main process."""

    def __init__(self, camera_id: str, clock: Callable[[], float] = time.time):
        self.camera_id = camera_id
        self.clock = clock
        self._lock = threading.Lock()
        self._started_ts = clock()
        self._inference_loop_ts: float | None = None
        self._last_frame_ts: float | None = None
        self._last_inference_ts: float | None = None
        self._last_inference_error_ts: float | None = None
        self._inference_error_count = 0
        self._last_detection_ts: float | None = None
        self._pipeline_loop_ts: float | None = None
        self._active_tracks = 0
        self._zoom_state: str | None = None

    # -- inference thread ----------------------------------------------------

    def inference_loop_tick(self) -> None:
        with self._lock:
            self._inference_loop_ts = self.clock()

    def frame_received(self) -> None:
        with self._lock:
            self._last_frame_ts = self.clock()

    def inference_ok(self, detection_count: int) -> None:
        now = self.clock()
        with self._lock:
            self._last_inference_ts = now
            if detection_count > 0:
                self._last_detection_ts = now

    def inference_failed(self) -> None:
        with self._lock:
            self._last_inference_error_ts = self.clock()
            self._inference_error_count += 1

    # -- pipeline thread -----------------------------------------------------

    def pipeline_tick(self, active_tracks: int | None = None, zoom_state: str | None = None) -> None:
        with self._lock:
            self._pipeline_loop_ts = self.clock()
            if active_tracks is not None:
                self._active_tracks = active_tracks
            if zoom_state is not None:
                self._zoom_state = zoom_state

    # -- liveness, read by watchdog/supervisor.py --------------------------------

    def inference_loop_ts(self) -> float | None:
        with self._lock:
            return self._inference_loop_ts

    def pipeline_loop_ts(self) -> float | None:
        with self._lock:
            return self._pipeline_loop_ts

    # -- main thread -----------------------------------------------------------

    def snapshot(
        self,
        stream: dict | None = None,
        queues: dict | None = None,
        supervisor: dict | None = None,
        process: dict | None = None,
    ) -> dict:
        with self._lock:
            return {
                "camera_id": self.camera_id,
                "pid": os.getpid(),
                "written_ts": self.clock(),
                "started_ts": self._started_ts,
                "inference_loop_ts": self._inference_loop_ts,
                "last_frame_ts": self._last_frame_ts,
                "last_inference_ts": self._last_inference_ts,
                "last_inference_error_ts": self._last_inference_error_ts,
                "inference_error_count": self._inference_error_count,
                "last_detection_ts": self._last_detection_ts,
                "pipeline_loop_ts": self._pipeline_loop_ts,
                "active_tracks": self._active_tracks,
                "zoom_state": self._zoom_state,
                "stream": stream or {},
                "queues": queues or {},
                "supervisor": supervisor or {},
                "process": process or {},
            }


def write_status_file(path: Path, data: dict) -> None:
    """Write atomically: readers see the old file or the new one, never half."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, default=str), encoding="utf-8")
    os.replace(tmp, path)


def read_status_file(path: Path) -> dict | None:
    """None if the file is missing or unreadable (e.g. edge_main never started)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _age(now: float, ts: Any) -> float | None:
    return now - ts if isinstance(ts, (int, float)) else None


def evaluate_edge_health(status: dict | None, now: float, stale_after_sec: float = 30.0) -> dict:
    """Turn a raw status snapshot into the heartbeat's `edge` section.

    Statuses:
        inference_status     ok | idle (loop alive, no frames) | error (latest
                             attempt failed) | starting | frozen | stale | unknown
        tracker_status       ok | frozen | starting | stale | unknown
        camera_stream_status ok | stalled | no_frames | stale | unknown
    `stale` = edge_main stopped writing its status file, so nothing in it can be trusted.
    """
    if status is None:
        return {
            "edge_process_alive": False,
            "status_age_sec": None,
            "last_detection_timestamp": None,
            "last_frame_timestamp": None,
            "inference_status": UNKNOWN,
            "tracker_status": UNKNOWN,
            "camera_stream_status": UNKNOWN,
            "queue_depths": {},
            "restarts": summarize_restarts(None),
        }

    status_age = _age(now, status.get("written_ts"))
    alive = status_age is not None and status_age <= stale_after_sec

    def fresh(key: str) -> bool:
        age = _age(now, status.get(key))
        return age is not None and age <= stale_after_sec

    if not alive:
        inference = tracker = camera = STALE
    else:
        if status.get("inference_loop_ts") is None:
            inference = "starting"
        elif not fresh("inference_loop_ts"):
            inference = "frozen"
        elif (status.get("last_inference_error_ts") or 0) > (status.get("last_inference_ts") or 0):
            inference = "error"
        elif fresh("last_inference_ts"):
            inference = "ok"
        else:
            inference = "idle"

        if status.get("pipeline_loop_ts") is None:
            tracker = "starting"
        else:
            tracker = "ok" if fresh("pipeline_loop_ts") else "frozen"

        if status.get("last_frame_ts") is None:
            camera = "no_frames"
        elif fresh("last_frame_ts") and not (status.get("stream") or {}).get("is_stalled"):
            camera = "ok"
        else:
            camera = "stalled"

    queues = status.get("queues") or {}
    return {
        "edge_process_alive": alive,
        "status_age_sec": round(status_age, 1) if status_age is not None else None,
        "last_detection_timestamp": status.get("last_detection_ts"),
        "last_frame_timestamp": status.get("last_frame_ts"),
        "inference_status": inference,
        "inference_error_count": status.get("inference_error_count", 0),
        "tracker_status": tracker,
        "active_tracks": status.get("active_tracks", 0),
        "zoom_state": status.get("zoom_state"),
        "camera_stream_status": camera,
        "stream_reconnect_count": (status.get("stream") or {}).get("reconnect_count"),
        "queue_depths": {
            name: {
                "depth": q.get("current_size"),
                "maxsize": q.get("maxsize"),
                "dropped": q.get("drop_count"),
            }
            for name, q in queues.items()
            if isinstance(q, dict)
        },
        "restarts": summarize_restarts(status),
    }


def summarize_restarts(status: dict | None) -> dict:
    """Heartbeat view of watchdog/supervisor.py and watchdog/restart_history.py.

    process_starts counts every edge_main start since the history file was
    created; thread_restarts counts in-process worker restarts this run.
    Reported even when the status file is stale, since a crash loop is
    exactly when these numbers matter.
    """
    status = status or {}
    process = status.get("process") or {}
    workers = status.get("supervisor") or {}
    return {
        "process_starts": process.get("process_starts"),
        "last_exit_reason": process.get("last_exit_reason"),
        "last_exit_code": process.get("last_exit_code"),
        "last_exit_ts": process.get("last_exit_ts"),
        "thread_restarts": sum(int(w.get("restarts") or 0) for w in workers.values() if isinstance(w, dict)),
        "threads": workers,
    }
