"""Worker supervision, restart history, systemd notify, and the pipeline
crash -> reset -> restart path running the real edge_main pipeline loop."""

import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import edge_main
import paths
from camera_control.camera_api import FakeLensCamera
from camera_control.mode_manager import ModeManager
from camera_control.zoom_controller import AnimalZoomController, ZoomState
from db.init_db import init_db
from db.outbox import get_pending
from inference.base import Detection
from inference.bytetrack_wrapper import IouTracker
from pipeline.alert_dispatcher import AlertDispatcher
from pipeline.best_snapshot import BestSnapshotStore
from pipeline.motion_gate import MotionGate
from pipeline.stage1_gate import Stage1Config, Stage1Gate
from pipeline.track_state_machine import TrackStateMachine
from pipeline.zone_filter import ZoneFilter
from queues.alert_queue import AlertQueue
from queues.detection_queue import DetectionBatch, DetectionQueue
from storage.evidence_store import EvidenceStore
from tests.helpers import AlwaysClassifier, blank_frame, camera_credentials
from watchdog import sd_notify
from watchdog.edge_status import EdgeStatus, evaluate_edge_health, read_status_file
from watchdog.restart_history import UNCLEAN_EXIT, record_exit, record_start
from watchdog.supervisor import BACKOFF, RUNNING, STOPPED, RestartPolicy, SupervisedThread, Supervisor


class Clock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


def _crash():
    raise RuntimeError("boom")


def _finish(worker: SupervisedThread) -> None:
    worker.join(2.0)
    if worker.is_alive():
        raise AssertionError(f"{worker.name} did not exit")


class TestSupervisor(unittest.TestCase):
    def setUp(self):
        self.mono, self.wall = Clock(), Clock(5000.0)
        self.policy = RestartPolicy(max_restarts=3, window_sec=600, initial_backoff_sec=1, max_backoff_sec=60)
        self.sup = Supervisor(self.policy, clock=self.mono, wall_clock=self.wall)

    def test_crashed_worker_is_reset_and_restarted_after_backoff(self):
        reset = MagicMock()
        w = self.sup.add(SupervisedThread("pipeline", _crash, on_restart=reset))
        self.sup.start_all()
        _finish(w)

        self.assertIsNone(self.sup.check())
        self.assertEqual(w.state, BACKOFF)
        self.assertIn("RuntimeError: boom", w.last_error)
        reset.assert_not_called()  # still inside the 1 s backoff

        self.mono.now += 1.0
        self.assertIsNone(self.sup.check())
        reset.assert_called_once()
        self.assertEqual(w.restart_count, 1)
        self.assertEqual(w.state, RUNNING)

    def test_backoff_doubles(self):
        self.assertEqual([self.policy.backoff(n) for n in range(8)], [1, 2, 4, 8, 16, 32, 60, 60])

    def test_restart_budget_exhaustion_escalates(self):
        w = self.sup.add(SupervisedThread("pipeline", _crash))
        self.sup.start_all()
        for _ in range(3):
            _finish(w)
            self.sup.check()
            self.mono.now += 60
            self.sup.check()
        _finish(w)
        reason = self.sup.check()
        self.assertIn("restart budget exhausted", reason)
        self.assertIn("RuntimeError: boom", reason)

    def test_old_restarts_leave_the_window(self):
        w = self.sup.add(SupervisedThread("pipeline", _crash))
        self.sup.start_all()
        for _ in range(3):
            _finish(w)
            self.sup.check()
            self.mono.now += 60
            self.sup.check()
        self.mono.now += 601  # all three restarts are now older than the window
        _finish(w)
        self.assertIsNone(self.sup.check())
        self.assertEqual(w.state, BACKOFF)

    def test_hung_worker_escalates(self):
        release = threading.Event()
        w = self.sup.add(SupervisedThread("inference", lambda: release.wait(5), liveness=lambda: 5000.0))
        self.sup.start_all()
        try:
            self.wall.now += 59
            self.assertIsNone(self.sup.check())
            self.wall.now += 2
            self.assertIn("inference hung", self.sup.check())
        finally:
            release.set()
            w.join(2)

    def test_tick_from_before_a_restart_is_not_a_hang(self):
        stale_tick = 5000.0
        calls = {"n": 0}
        release = threading.Event()

        def crash_then_block():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("first run")
            release.wait(5)

        w = self.sup.add(SupervisedThread("pipeline", crash_then_block, liveness=lambda: stale_tick))
        self.sup.start_all()
        try:
            _finish(w)
            self.sup.check()
            self.wall.now += 120  # backoff took a while; the last tick is 120 s old
            self.mono.now += 1
            self.assertIsNone(self.sup.check())  # restarts now
            self.assertIsNone(self.sup.check())  # new thread judged from its own start
        finally:
            release.set()
            w.join(2)

    def test_clean_exit_is_stopped_not_restarted(self):
        w = self.sup.add(SupervisedThread("capture", lambda: None))
        self.sup.start_all()
        _finish(w)
        self.assertIsNone(self.sup.check())
        self.assertEqual(w.state, STOPPED)
        self.assertTrue(self.sup.stopped("capture"))
        self.assertEqual(w.restart_count, 0)

    def test_failing_reset_hook_escalates(self):
        w = self.sup.add(SupervisedThread("pipeline", _crash, on_restart=_crash))
        self.sup.start_all()
        _finish(w)
        self.sup.check()
        self.mono.now += 1
        self.assertIn("reset before restart failed", self.sup.check())

    def test_not_restartable_escalates_immediately(self):
        w = self.sup.add(SupervisedThread("x", _crash, restartable=False))
        self.sup.start_all()
        _finish(w)
        self.assertIn("not restartable", self.sup.check())

    def test_policy_from_config_clamps(self):
        p = RestartPolicy.from_config({"supervisor": {"max_restarts": -1, "hang_threshold_sec": 1}})
        self.assertEqual(p.max_restarts, 0)
        self.assertEqual(p.hang_threshold_sec, 5.0)
        self.assertEqual(RestartPolicy.from_config({}).hang_threshold_sec, 60.0)


class TestRestartHistory(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp(prefix="zovive_hist_")) / "restart_history.json"

    def test_clean_exit_reason_is_reported_on_next_start(self):
        first = record_start(self.path, 1.0)
        self.assertEqual(first["process_starts"], 1)
        self.assertIsNone(first["last_exit_reason"])
        record_exit(self.path, first, "pipeline hung", 70, 2.0)
        second = record_start(self.path, 3.0)
        self.assertEqual(second["process_starts"], 2)
        self.assertEqual(second["last_exit_reason"], "pipeline hung")
        self.assertEqual(second["last_exit_code"], 70)

    def test_run_that_never_recorded_an_exit_is_unclean(self):
        record_start(self.path, 1.0)  # e.g. killed by the systemd watchdog
        second = record_start(self.path, 2.0)
        self.assertEqual(second["last_exit_reason"], UNCLEAN_EXIT)
        self.assertIsNone(second["last_exit_code"])


class TestSdNotify(unittest.TestCase):
    def test_noop_without_socket(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(sd_notify.ready())
            self.assertIsNone(sd_notify.watchdog_interval_sec())

    def test_sends_datagram_to_abstract_socket(self):
        fake_sock = MagicMock()
        fake_sock.__enter__.return_value = fake_sock
        with patch.dict("os.environ", {"NOTIFY_SOCKET": "@/org/systemd/notify"}), patch.object(
            sd_notify.socket, "AF_UNIX", 1, create=True
        ), patch.object(sd_notify.socket, "socket", return_value=fake_sock):
            self.assertTrue(sd_notify.watchdog_ping())
        fake_sock.connect.assert_called_once_with("\0/org/systemd/notify")
        fake_sock.sendall.assert_called_once_with(b"WATCHDOG=1")

    def test_status_cannot_inject_extra_lines(self):
        with patch.object(sd_notify, "notify", return_value=True) as notify:
            sd_notify.status("ok\nREADY=1")
        notify.assert_called_once_with("STATUS=ok READY=1")

    def test_watchdog_interval_only_for_this_pid(self):
        with patch.dict("os.environ", {"WATCHDOG_USEC": "30000000", "WATCHDOG_PID": "1"}):
            self.assertIsNone(sd_notify.watchdog_interval_sec())
        with patch.dict("os.environ", {"WATCHDOG_USEC": "30000000"}, clear=True):
            self.assertEqual(sd_notify.watchdog_interval_sec(), 30.0)


class TestZoomAbort(unittest.TestCase):
    def test_abort_while_zoomed_returns_to_wide_without_alert(self):
        cam = FakeLensCamera()
        clock = Clock()
        zoom = AnimalZoomController(cam, zoom_settle_sec=0, focus_settle_sec=0, clock=clock)
        zoom.engage("t-1", "tiger", (300, 170, 340, 190), 4.0)
        zoom.abort()
        self.assertIsNone(zoom.session)
        self.assertEqual(zoom.state, ZoomState.RETURNING)
        self.assertEqual(cam.get_zoom(), 0.0)
        zoom.tick()
        zoom.tick()
        self.assertEqual(zoom.state, ZoomState.WIDE)
        self.assertIsNone(zoom.match_known("tiger", (300, 170, 340, 190)))  # sighting dropped

    def test_abort_in_wide_view_does_not_touch_the_lens(self):
        cam = FakeLensCamera()
        AnimalZoomController(cam).abort()
        self.assertEqual(cam.call_log, [])


class TestCaptureRestartable(unittest.TestCase):
    def _reader(self):
        from capture.rtsp_reader import RtspReader
        from queues.frame_queue import LatestFrameSlot

        # Default configs/rtsp_config.yaml (with fake credentials);
        # _open_capture is patched, so nothing ever connects to a camera.
        with camera_credentials():
            return RtspReader(LatestFrameSlot(name="test_sup_slot"))

    def test_crash_releases_capture_and_records_liveness(self):
        reader = self._reader()
        cap = MagicMock()
        cap.read.side_effect = RuntimeError("decoder exploded")
        with patch.object(reader, "_open_capture", return_value=cap), self.assertRaises(RuntimeError):
            reader.run()
        cap.release.assert_called_once()
        self.assertIsNotNone(reader.loop_ts)


class TestEdgeHealthRestarts(unittest.TestCase):
    def test_restarts_reported_even_when_status_is_stale(self):
        status = EdgeStatus("cam01", clock=lambda: 100.0).snapshot(
            supervisor={"pipeline": {"state": "running", "restarts": 2}, "capture": {"restarts": 1}},
            process={"process_starts": 4, "last_exit_reason": "pipeline hung", "last_exit_code": 70},
        )
        edge = evaluate_edge_health(status, now=10_000.0)
        self.assertFalse(edge["edge_process_alive"])
        self.assertEqual(edge["restarts"]["thread_restarts"], 3)
        self.assertEqual(edge["restarts"]["process_starts"], 4)
        self.assertEqual(edge["restarts"]["last_exit_code"], 70)
        self.assertEqual(evaluate_edge_health(None, 0)["restarts"]["thread_restarts"], 0)


class _CrashOnceDispatcher:
    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def dispatch(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise OSError("No space left on device")
        return self.inner.dispatch(**kwargs)


class _FakeCapture:
    def __init__(self):
        self.health = MagicMock()
        self.health.snapshot.return_value = {"is_stalled": False}


class TestPipelineCrashRecovery(unittest.TestCase):
    """The real edge_main pipeline loop and supervisor loop: the first
    dispatch raises (disk full), the supervisor resets state and restarts
    the thread, and the next confirmed tiger is alerted normally."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="zovive_sup_test_")
        self._orig = (paths.DB_PATH, paths.SNAPSHOT_DIR, paths.EDGE_STATUS_PATH)
        paths.DB_PATH = Path(self.tmpdir) / "test.sqlite3"
        paths.SNAPSHOT_DIR = Path(self.tmpdir) / "snapshots"
        paths.EDGE_STATUS_PATH = Path(self.tmpdir) / "edge_status.json"
        paths.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        init_db()

    def tearDown(self):
        paths.DB_PATH, paths.SNAPSHOT_DIR, paths.EDGE_STATUS_PATH = self._orig

    def test_crash_restart_then_alert(self):
        queue = DetectionQueue(maxsize=20)
        frame = blank_frame(640, 360)
        det = Detection(box=(100, 60, 400, 300), score=0.9, class_id=5, class_name="tiger")
        for i in range(6):
            queue.put(DetectionBatch(seq=i, timestamp=float(i), frame=frame, detections=[det], camera_id="cam01"))

        tracker = IouTracker(iou_match_threshold=0.3, max_age_frames=15, min_hits_to_confirm=1, presence_window=5)
        track_sm = TrackStateMachine(Stage1Gate(Stage1Config(min_hits_in_window=3, window_size=5)))
        snapshot_store = BestSnapshotStore()
        zoom = AnimalZoomController(FakeLensCamera(), mode_manager=ModeManager())
        dispatcher = _CrashOnceDispatcher(
            AlertDispatcher(
                AlertQueue(),
                evidence_store=EvidenceStore(snapshot_dir=paths.SNAPSHOT_DIR),
                node_config={"node": {"camera_id": "cam01"}},
                alert_policy={},
            )
        )
        stop = threading.Event()
        status = EdgeStatus("cam01")

        def reset():
            edge_main._reset_tracking(tracker, track_sm, snapshot_store)
            zoom.abort()

        sup = Supervisor(RestartPolicy(initial_backoff_sec=0, max_restarts=5))
        pipeline = sup.add(
            SupervisedThread(
                "pipeline",
                lambda: edge_main._pipeline_loop(
                    queue, AlwaysClassifier("tiger", 0.92), zoom, tracker, track_sm, ZoneFilter({"zones": {}}),
                    MotionGate(), snapshot_store, dispatcher,
                    {"classifier": {"max_crops_per_track": 5, "min_confidence": 0.5}},
                    "test-v1", stop, 3, None, status,
                ),
                liveness=status.pipeline_loop_ts,
                on_restart=reset,
            )
        )
        sup.start_all()

        safety = threading.Timer(10.0, stop.set)  # a regression fails the test, not hangs it
        safety.start()
        try:
            # Returns once the restarted pipeline finishes its 3 cycles.
            reason, code = edge_main._supervise(sup, status, _FakeCapture(), {"process_starts": 1}, 0.2, stop)
        finally:
            safety.cancel()
            stop.set()
            sup.join_all(2)

        self.assertEqual((reason, code), ("pipeline finished (--max-cycles)", edge_main.EXIT_OK))
        self.assertEqual(pipeline.restart_count, 1)
        self.assertIn("OSError: No space left on device", pipeline.last_error)
        pending = get_pending(limit=10)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["animal_name"], "tiger")

        # Written while the restarted thread ran (write interval 0.2 s).
        time.sleep(0.3)
        edge_main._write_edge_status(status, _FakeCapture(), sup, {"process_starts": 1})
        written = read_status_file(paths.EDGE_STATUS_PATH)
        self.assertEqual(written["supervisor"]["pipeline"]["restarts"], 1)
        self.assertEqual(evaluate_edge_health(written, time.time())["restarts"]["thread_restarts"], 1)

    def test_hung_worker_makes_supervise_exit_for_systemd(self):
        release = threading.Event()
        sup = Supervisor(RestartPolicy(hang_threshold_sec=0.2))
        sup.add(SupervisedThread("inference", lambda: release.wait(5), liveness=lambda: None))
        sup.start_all()
        try:
            reason, code = edge_main._supervise(
                sup, EdgeStatus("cam01"), _FakeCapture(), {}, 1.0, threading.Event()
            )
        finally:
            release.set()
            sup.join_all(2)
        self.assertEqual(code, edge_main.EXIT_SUPERVISOR)
        self.assertIn("inference hung", reason)


if __name__ == "__main__":
    unittest.main()
