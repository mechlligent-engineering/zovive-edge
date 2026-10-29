"""pipeline/dispatch_worker.py: alert disk writes off the pipeline thread.

Real SQLite outbox and real evidence store in a temp dir, like
test_pipeline.py. Most tests run the worker's loop on the test thread
with the stop event already set, which processes the queue
deterministically (the shutdown drain); the ordering and slow-disk tests
run it on a real thread.
"""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

import paths
from camera_control.camera_api import FakeLensCamera
from camera_control.mode_manager import ModeManager
from camera_control.zoom_controller import AnimalZoomController
from db.init_db import get_connection, init_db
from db.outbox import get_pending, set_video_path
from edge_main import _pipeline_loop
from inference.bytetrack_wrapper import IouTracker
from pipeline.alert_dispatcher import AlertDispatcher
from pipeline.best_snapshot import BestSnapshotStore
from pipeline.dispatch_worker import QUEUE_NAME, DispatchWorker
from pipeline.motion_gate import MotionGate
from pipeline.stage1_gate import Stage1Config, Stage1Gate
from pipeline.track_state_machine import TrackStateMachine
from pipeline.zone_filter import ZoneFilter
from queues.alert_queue import AlertQueue
from queues.detection_queue import DetectionBatch, DetectionQueue
from queues.queue_monitor import snapshot as queues_snapshot
from storage.evidence_store import EvidenceStore, SnapshotWriteError
from tests.helpers import blank_frame, make_detection

NODE_CFG = {
    "node": {"camera_id": "cam01", "camera_name": "Test Cam", "forest_name": "Test Forest", "gps": {}}
}
POLICY = {"alert_policy": {"high_priority_species": ["tiger"], "send_unknown_animal_alerts": False}}


def _drained(worker: DispatchWorker) -> None:
    """Run the worker's loop on this thread with the stop event already
    set: it processes everything queued (the shutdown drain) and returns."""
    stop = threading.Event()
    stop.set()
    worker.run(stop)


def _video_path(event_id: str) -> str | None:
    row = get_connection().execute("SELECT video_path FROM events WHERE event_id = ?", (event_id,)).fetchone()
    return None if row is None else row["video_path"]


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="zovive_dispatch_test_"))
        self._orig = (paths.DB_PATH, paths.SNAPSHOT_DIR)
        paths.DB_PATH = self.tmpdir / "test.sqlite3"
        paths.SNAPSHOT_DIR = self.tmpdir / "snapshots"
        paths.SNAPSHOT_DIR.mkdir(parents=True)
        self.addCleanup(self._restore)
        init_db()
        self.alert_queue = AlertQueue()
        self.evidence = EvidenceStore(snapshot_dir=paths.SNAPSHOT_DIR)
        self.dispatcher = AlertDispatcher(
            self.alert_queue, evidence_store=self.evidence, node_config=NODE_CFG, alert_policy=POLICY
        )
        self.sleeps: list[float] = []

    def _restore(self):
        paths.DB_PATH, paths.SNAPSHOT_DIR = self._orig

    def _worker(self, **kw) -> DispatchWorker:
        kw.setdefault("sleep", self.sleeps.append)
        return DispatchWorker(self.dispatcher, **kw)

    def _snapshots(self) -> list[Path]:
        return sorted(paths.SNAPSHOT_DIR.glob("*.jpg"))


class TestDispatchWorker(_Base):
    def test_dispatch_only_queues_the_event(self):
        worker = self._worker()
        event_id = worker.dispatch("t-1", "tiger", 0.9, blank_frame())
        self.assertIsNotNone(event_id)
        # Nothing touched the disk on the calling (pipeline) thread.
        self.assertEqual(get_pending(), [])
        self.assertEqual(self._snapshots(), [])
        self.assertEqual(worker.qsize(), 1)

    def test_worker_saves_snapshot_then_row_then_announces(self):
        calls: list[str] = []
        for step in ("save_snapshot", "insert", "announce"):
            original = getattr(self.dispatcher, step)

            def recorded(*args, _original=original, _step=step):
                calls.append(_step)
                return _original(*args)

            setattr(self.dispatcher, step, recorded)
        worker = self._worker()
        event_id = worker.dispatch("t-1", "tiger", 0.9, blank_frame(), preset_name="wide", model_version="m1")
        _drained(worker)

        self.assertEqual(calls, ["save_snapshot", "insert", "announce"])
        (row,) = get_pending()
        self.assertEqual(row["event_id"], event_id)
        self.assertEqual((row["animal_name"], row["priority"], row["preset_name"]), ("tiger", "high", "wide"))
        self.assertTrue(Path(row["snapshot_path"]).exists())
        payload = self.alert_queue.get(timeout=0.1)
        self.assertEqual((payload.event_id, payload.snapshot_path), (event_id, row["snapshot_path"]))
        self.assertEqual(len(self._snapshots()), 1, "one snapshot per event")
        self.assertEqual(worker.snapshot()["dispatched"], 1)

    def test_policy_suppressed_alert_is_not_queued(self):
        worker = self._worker()
        self.assertIsNone(worker.dispatch("t-1", "unknown_animal", 0.5, blank_frame()))
        self.assertEqual(worker.qsize(), 0)
        self.assertEqual(worker.snapshot()["rejected"], 0)

    def test_full_queue_rejects_loudly_and_returns_none(self):
        worker = self._worker(maxsize=1)
        first = worker.dispatch("t-1", "tiger", 0.9, blank_frame())
        with self.assertLogs("pipeline.dispatch_worker", level="ERROR") as logs:
            second = worker.dispatch("t-2", "leopard", 0.8, blank_frame())
        self.assertIsNone(second, "the pipeline must not mark a track alerted that has no event")
        self.assertIn("NOT recorded", logs.output[0])
        self.assertEqual(worker.snapshot()["rejected"], 1)
        _drained(worker)
        self.assertEqual([r["event_id"] for r in get_pending()], [first])

    def test_failed_step_is_retried_from_where_it_failed(self):
        original_insert = self.dispatcher.insert
        failures = [OSError("database is locked")]

        def flaky_insert(record):
            if failures:
                raise failures.pop()
            original_insert(record)

        self.dispatcher.insert = flaky_insert
        worker = self._worker()
        event_id = worker.dispatch("t-1", "tiger", 0.9, blank_frame())
        _drained(worker)

        self.assertEqual([r["event_id"] for r in get_pending()], [event_id])
        # The snapshot was saved once; a second save would have been refused
        # (the evidence store never overwrites) and failed the event.
        self.assertEqual(len(self._snapshots()), 1)
        self.assertEqual(self.sleeps, [0.5])
        snap = worker.snapshot()
        self.assertEqual((snap["dispatched"], snap["retries"], snap["failed"]), (1, 1, 0))

    def test_persistent_failure_is_reported_not_silent(self):
        def broken_disk(record, image):
            raise SnapshotWriteError("No space left on device")

        self.dispatcher.save_snapshot = broken_disk
        worker = self._worker(max_attempts=3)
        worker.dispatch("t-1", "tiger", 0.9, blank_frame())
        with self.assertLogs("pipeline.dispatch_worker", level="WARNING") as logs:
            _drained(worker)

        self.assertEqual(get_pending(), [], "no row without its snapshot")
        self.assertIsNone(self.alert_queue.get(timeout=0.05))
        self.assertEqual(self.sleeps, [0.5, 1.0])
        self.assertTrue(any("ERROR" in line and "NOT recorded" in line for line in logs.output))
        snap = worker.snapshot()
        self.assertEqual((snap["failed"], snap["dispatched"]), (1, 0))
        self.assertIn("No space left", snap["last_error"])

    def test_shutdown_drains_every_queued_alert(self):
        worker = self._worker()
        ids = [worker.dispatch(f"t-{n}", "tiger", 0.9, blank_frame()) for n in range(3)]
        _drained(worker)
        self.assertEqual(sorted(r["event_id"] for r in get_pending()), sorted(ids))

    def test_alerts_left_after_the_drain_timeout_are_reported(self):
        worker = self._worker(drain_timeout_sec=0.0)
        worker.dispatch("t-1", "tiger", 0.9, blank_frame())
        worker.dispatch("t-2", "tiger", 0.9, blank_frame())
        with self.assertLogs("pipeline.dispatch_worker", level="ERROR") as logs:
            _drained(worker)
        self.assertEqual(get_pending(), [])
        self.assertEqual(worker.snapshot()["failed"], 2)
        self.assertEqual(sum("NOT recorded" in line for line in logs.output), 2)

    def test_video_path_is_recorded_directly_when_the_worker_is_not_running(self):
        seen: list[tuple[str, str]] = []
        worker = self._worker(on_video_path=lambda e, p: seen.append((e, p)))
        worker.video_path_ready("ev-1", "/clips/ev-1.mp4")
        self.assertEqual(seen, [("ev-1", "/clips/ev-1.mp4")])
        self.assertEqual(worker.qsize(), 0)

    def test_queue_appears_in_the_queue_monitor(self):
        worker = self._worker()
        before = queues_snapshot().get(QUEUE_NAME, {}).get("put_count", 0)
        worker.dispatch("t-1", "tiger", 0.9, blank_frame())
        stats = queues_snapshot()[QUEUE_NAME]
        self.assertEqual(stats["put_count"], before + 1)
        self.assertIn("drop_count", stats)  # rejections, as for every queue


class TestDispatchWorkerOnAThread(_Base):
    def _start(self, worker: DispatchWorker) -> threading.Event:
        stop = threading.Event()
        thread = threading.Thread(target=worker.run, args=(stop,), daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5.0)
        self.addCleanup(stop.set)
        deadline = time.monotonic() + 5.0
        while worker.loop_ts is None and time.monotonic() < deadline:
            time.sleep(0.01)
        return stop

    def _slow_disk(self) -> threading.Event:
        """save_snapshot blocks until the returned event is set."""
        release = threading.Event()
        original = self.dispatcher.save_snapshot

        def blocked(record, image):
            release.wait(5.0)
            return original(record, image)

        self.dispatcher.save_snapshot = blocked
        self.addCleanup(release.set)
        return release

    def _wait_for(self, condition, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while not condition():
            if time.monotonic() > deadline:
                self.fail("condition not reached")
            time.sleep(0.01)

    def test_pipeline_thread_never_waits_for_a_stalled_disk(self):
        release = self._slow_disk()
        worker = self._worker(sleep=time.sleep)
        self._start(worker)
        started = time.monotonic()
        ids = [worker.dispatch(f"t-{n}", "tiger", 0.9, blank_frame()) for n in range(5)]
        self.assertLess(time.monotonic() - started, 0.5, "dispatch() must not wait for the disk")
        self.assertTrue(all(ids))
        self.assertEqual(get_pending(), [])
        release.set()
        self._wait_for(lambda: len(get_pending()) == 5)

    def test_clip_path_is_recorded_after_its_event_row(self):
        release = self._slow_disk()
        worker = self._worker(sleep=time.sleep)
        self._start(worker)
        event_id = worker.dispatch("t-1", "tiger", 0.9, blank_frame())
        # The clip finishes while the event's own write is still stuck: the
        # path must wait behind it instead of updating a row that doesn't exist.
        worker.video_path_ready(event_id, "/clips/t-1.mp4")
        self.assertEqual(get_pending(), [])
        release.set()
        self._wait_for(lambda: _video_path(event_id) == "/clips/t-1.mp4")
        self.assertEqual(worker.snapshot()["video_paths"], 1)


class TestPipelineWithDispatchWorker(_Base):
    def test_pipeline_loop_produces_the_same_single_event(self):
        worker = self._worker()
        detection_queue = DetectionQueue(maxsize=20)
        frame = blank_frame(320, 240)
        det = make_detection()
        for i in range(5):
            detection_queue.put(
                DetectionBatch(seq=i, timestamp=float(i), frame=frame, detections=[det], camera_id="cam01")
            )
        _pipeline_loop(
            detection_queue,
            AnimalZoomController(FakeLensCamera(), mode_manager=ModeManager()),
            IouTracker(iou_match_threshold=0.3, max_age_frames=15, min_hits_to_confirm=1, presence_window=5),
            TrackStateMachine(Stage1Gate(Stage1Config(min_hits_in_window=3, window_size=5))),
            ZoneFilter({"zones": {}}),
            MotionGate(),
            BestSnapshotStore(),
            worker,
            inf_cfg={"stage2": {"min_confidence": 0.5, "voting_strategy": "average"}},
            model_version="test-v1",
            stop_event=threading.Event(),
            max_cycles=5,
        )
        self.assertEqual(get_pending(), [], "the pipeline thread wrote nothing itself")
        self.assertEqual(worker.qsize(), 1)
        _drained(worker)
        (row,) = get_pending()
        self.assertEqual((row["animal_name"], row["model_version"]), ("tiger", "test-v1"))
        self.assertTrue(Path(row["snapshot_path"]).exists())

    def test_set_video_path_default_records_into_the_outbox(self):
        worker = self._worker()
        self.assertIs(worker.on_video_path, set_video_path)


if __name__ == "__main__":
    unittest.main()
