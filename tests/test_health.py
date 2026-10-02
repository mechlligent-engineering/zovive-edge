"""health_main.py and watchdog/edge_status.py: every failure path must turn
into a heartbeat field, never a crash, and edge_main's health must show up
as frozen/stale when its threads stop."""

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

import health_main
import paths
from db.init_db import get_connection, init_db
from edge_main import _inference_loop
from inference.base import Detection
from network.base_station_client import BaseStationClient, BaseStationError, build_client
from queues.detection_queue import DetectionQueue
from queues.frame_queue import LatestFrameSlot
from watchdog.edge_status import EdgeStatus, evaluate_edge_health, read_status_file, write_status_file


class FakeClock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


class TestThrottle(unittest.TestCase):
    def test_decode_flags(self):
        flags = health_main._decode_throttled("throttled=0x50005")
        self.assertTrue(flags["under_voltage_now"])
        self.assertTrue(flags["throttled_now"])
        self.assertTrue(flags["under_voltage_occurred"])
        self.assertFalse(flags["soft_temp_limit_now"])
        self.assertIsNone(health_main._decode_throttled("garbage"))

    def test_not_available_off_pi(self):
        with patch.object(health_main, "_find_vcgencmd", return_value=None):
            self.assertEqual(health_main._pi_throttle_status()["error"], "not_available")

    def test_timeout_does_not_raise(self):
        with patch.object(health_main, "_find_vcgencmd", return_value="/usr/bin/vcgencmd"), patch.object(
            health_main.subprocess, "run", side_effect=health_main.subprocess.TimeoutExpired("vcgencmd", 2)
        ):
            self.assertEqual(health_main._pi_throttle_status()["error"], "timeout")

    def test_nonzero_exit_is_failed(self):
        result = MagicMock(returncode=255, stdout="", stderr="VCHI init failed")
        with patch.object(health_main, "_find_vcgencmd", return_value="/usr/bin/vcgencmd"), patch.object(
            health_main.subprocess, "run", return_value=result
        ):
            self.assertEqual(health_main._pi_throttle_status()["error"], "failed")

    def test_success_uses_absolute_path_and_no_shell(self):
        result = MagicMock(returncode=0, stdout="throttled=0x0\n", stderr="")
        with patch.object(health_main, "_find_vcgencmd", return_value="/usr/bin/vcgencmd"), patch.object(
            health_main.subprocess, "run", return_value=result
        ) as run:
            status = health_main._pi_throttle_status()
        self.assertEqual(status["raw"], "throttled=0x0")
        self.assertIsNone(status["error"])
        args, kwargs = run.call_args
        self.assertEqual(args[0], ["/usr/bin/vcgencmd", "get_throttled"])
        self.assertNotIn("shell", kwargs)


class TestHeartbeatStorage(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="zovive_health_test_")
        self._orig_db = paths.DB_PATH
        self._orig_status = paths.EDGE_STATUS_PATH
        paths.DB_PATH = Path(self.tmpdir) / "test.sqlite3"
        paths.EDGE_STATUS_PATH = Path(self.tmpdir) / "edge_status.json"
        self.conn = init_db()

    def tearDown(self):
        paths.DB_PATH = self._orig_db
        paths.EDGE_STATUS_PATH = self._orig_status

    def _count(self):
        return get_connection().execute("SELECT COUNT(*) FROM heartbeats").fetchone()[0]

    def test_prune_keeps_recent_rows_only(self):
        now = 10_000_000.0
        with self.conn:
            for age_days in (1, 6, 8, 30):
                self.conn.execute(
                    "INSERT INTO heartbeats (camera_id, payload_json, created_ts) VALUES ('cam01', '{}', ?)",
                    (now - age_days * 86400,),
                )
        self.assertEqual(health_main.prune_heartbeats(self.conn, 7, now=now), 2)
        self.assertEqual(self._count(), 2)

    def test_db_error_still_sends(self):
        broken = MagicMock()
        broken.__enter__.side_effect = health_main.sqlite3.OperationalError("database or disk is full")
        client = MagicMock()
        payload = health_main.run_cycle(broken, client, "cam01", 30.0, health_main._SendLog())
        self.assertFalse(payload["db_write_ok"])
        client.send_heartbeat.assert_called_once()

    def test_send_failure_does_not_raise_and_row_is_stored(self):
        client = MagicMock()
        client.send_heartbeat.side_effect = BaseStationError("connection refused")
        send_log = health_main._SendLog()
        payload = health_main.run_cycle(self.conn, client, "cam01", 30.0, send_log)
        self.assertTrue(payload["db_write_ok"])
        self.assertEqual(send_log.consecutive_failures, 1)
        self.assertEqual(self._count(), 1)

    def test_each_cycle_sends_only_the_current_heartbeat(self):
        client = MagicMock()
        client.send_heartbeat.side_effect = [BaseStationError("down"), BaseStationError("down"), None]
        send_log = health_main._SendLog()
        for _ in range(3):
            health_main.run_cycle(self.conn, client, "cam01", 30.0, send_log)
        # 3 cycles -> 3 sends, one payload each: nothing from the outage is replayed.
        self.assertEqual(client.send_heartbeat.call_count, 3)
        self.assertEqual(send_log.consecutive_failures, 0)
        self.assertEqual(self._count(), 3)

    def test_missing_edge_status_reports_not_alive(self):
        payload = health_main.collect_heartbeat("cam01")
        self.assertFalse(payload["edge"]["edge_process_alive"])
        self.assertEqual(payload["edge"]["inference_status"], "unknown")


class TestConfigHandling(unittest.TestCase):
    def test_interval_is_clamped(self):
        self.assertEqual(health_main._clamp_interval(0), health_main.MIN_INTERVAL_SEC)
        self.assertEqual(health_main._clamp_interval(-5), health_main.MIN_INTERVAL_SEC)
        self.assertEqual(health_main._clamp_interval(15), 15)
        self.assertEqual(health_main._clamp_interval(1e9), health_main.MAX_INTERVAL_SEC)

    def test_send_failures_are_rate_limited(self):
        send_log = health_main._SendLog(every=20)
        with patch.object(health_main.log, "warning") as warning:
            for _ in range(40):
                send_log.failed(BaseStationError("down"))
        self.assertEqual(warning.call_count, 3)  # failures 1, 20, 40

    def test_build_client_passes_ca_bundle(self):
        client = build_client({"base_station": {"url": "https://bs.local", "ca_bundle": "/etc/zovive/ca.pem"}})
        self.assertIsInstance(client, BaseStationClient)
        self.assertEqual(client.verify, "/etc/zovive/ca.pem")


class TestEdgeHealth(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.status = EdgeStatus("cam01", clock=self.clock)
        self.status.inference_loop_tick()
        self.status.frame_received()
        self.status.inference_ok(detection_count=1)
        self.status.pipeline_tick(active_tracks=2, zoom_state="WIDE")

    def _eval(self, advance: float = 0.0):
        snap = self.status.snapshot(stream={"is_stalled": False}, queues={"detection_queue": {"current_size": 1}})
        self.clock.now += advance
        return evaluate_edge_health(snap, self.clock.now, stale_after_sec=30)

    def test_healthy(self):
        edge = self._eval()
        self.assertTrue(edge["edge_process_alive"])
        self.assertEqual(
            (edge["inference_status"], edge["tracker_status"], edge["camera_stream_status"]), ("ok", "ok", "ok")
        )
        self.assertEqual(edge["last_detection_timestamp"], 1000.0)
        self.assertEqual(edge["queue_depths"]["detection_queue"]["depth"], 1)

    def test_status_file_not_updated_is_stale(self):
        edge = self._eval(advance=60)
        self.assertFalse(edge["edge_process_alive"])
        self.assertEqual(edge["tracker_status"], "stale")

    def test_dead_pipeline_thread_is_frozen_while_process_alive(self):
        self.clock.now += 60
        self.status.inference_loop_tick()
        self.status.frame_received()
        self.status.inference_ok(0)
        edge = self._eval()
        self.assertTrue(edge["edge_process_alive"])
        self.assertEqual(edge["tracker_status"], "frozen")
        self.assertEqual(edge["inference_status"], "ok")

    def test_camera_down_shows_idle_inference_and_stalled_stream(self):
        self.clock.now += 60
        self.status.inference_loop_tick()
        self.status.pipeline_tick()
        edge = self._eval()
        self.assertEqual(edge["inference_status"], "idle")
        self.assertEqual(edge["camera_stream_status"], "stalled")

    def test_latest_inference_failed_is_error(self):
        self.clock.now += 1
        self.status.inference_loop_tick()
        self.status.inference_failed()
        self.assertEqual(self._eval()["inference_status"], "error")

    def test_file_roundtrip(self):
        path = Path(tempfile.mkdtemp(prefix="zovive_status_")) / "edge_status.json"
        self.assertIsNone(read_status_file(path))
        write_status_file(path, self.status.snapshot())
        self.assertEqual(read_status_file(path)["camera_id"], "cam01")
        path.write_text("{not json", encoding="utf-8")
        self.assertIsNone(read_status_file(path))


class _OneTigerDetector:
    def infer(self, frame):
        return [Detection(box=(0, 0, 10, 10), score=0.9, class_id=5, class_name="tiger")]


class TestEdgeMainRecordsStatus(unittest.TestCase):
    def test_inference_loop_updates_status(self):
        slot = LatestFrameSlot(name="test_health_slot")
        slot.put(np.zeros((10, 10, 3), dtype=np.uint8))
        status = EdgeStatus("cam01")
        _inference_loop(_OneTigerDetector(), slot, DetectionQueue(maxsize=2), "cam01", threading.Event(), 1, status)
        snap = status.snapshot()
        self.assertIsNotNone(snap["inference_loop_ts"])
        self.assertIsNotNone(snap["last_frame_ts"])
        self.assertIsNotNone(snap["last_detection_ts"])


if __name__ == "__main__":
    unittest.main()
