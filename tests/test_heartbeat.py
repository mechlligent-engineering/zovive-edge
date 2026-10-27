"""network/heartbeat.py and its use in health_main.collect_heartbeat().

Node identity, disk, sensors and /proc are mocked or faked, so no camera,
Hailo, network or Raspberry Pi is needed and Windows and Linux behave alike.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import health_main
from network import heartbeat
from network.heartbeat import build_heartbeat_payload, validate_heartbeat_payload
from watchdog.edge_status import evaluate_edge_health

NOW = 1_760_012_400.0
LAST_DETECTION = 1_760_012_345.2

VALID_PAYLOAD = {
    "node_id": "Node-001",
    "status": "ok",
    "uptime_sec": 12345,
    "storage_free_gb": 180.2,
    "temperature_c": 54.1,
    "last_detection_ts": LAST_DETECTION,
}

HEALTHY_EDGE = {
    "edge_process_alive": True,
    "inference_status": "ok",
    "tracker_status": "ok",
    "camera_stream_status": "ok",
    "last_detection_timestamp": LAST_DETECTION,
}

DISK = {"total_gb": 250.0, "used_gb": 69.8, "free_gb": 180.2, "used_percent": 27.9}


def payload(**changes) -> dict:
    return {**VALID_PAYLOAD, **changes}


def edge(**changes) -> dict:
    return {**HEALTHY_EDGE, **changes}


class HeartbeatTestCase(unittest.TestCase):
    def setUp(self):
        identity = mock.patch.object(heartbeat, "get_identity", return_value={"node_id": "Node-001"})
        self.get_identity = identity.start()
        self.addCleanup(identity.stop)


class ValidationTests(HeartbeatTestCase):
    def test_valid_payload(self):
        self.assertEqual(validate_heartbeat_payload(payload()), [])

    def test_payload_none(self):
        self.assertEqual(validate_heartbeat_payload(None), ["payload must be a dictionary"])

    def test_payload_not_dict(self):
        for value in ("ok", 42, [VALID_PAYLOAD], ("node_id", "Node-001")):
            self.assertEqual(validate_heartbeat_payload(value), ["payload must be a dictionary"], value)

    def test_missing_node_id(self):
        missing = {k: v for k, v in VALID_PAYLOAD.items() if k != "node_id"}
        self.assertEqual(validate_heartbeat_payload(missing), ["missing node_id"])

    def test_invalid_node_id(self):
        for value in ("", "   ", None, 1, ["Node-001"]):
            self.assertEqual(validate_heartbeat_payload(payload(node_id=value)), ["invalid node_id"], value)

    def test_invalid_status(self):
        for value in ("OK", "unknown", "", None, 1):
            self.assertEqual(validate_heartbeat_payload(payload(status=value)), ["invalid status"], value)

    def test_every_status_value_is_valid(self):
        for value in ("ok", "degraded", "starting", "down"):
            self.assertEqual(validate_heartbeat_payload(payload(status=value)), [], value)

    def test_missing_uptime(self):
        missing = {k: v for k, v in VALID_PAYLOAD.items() if k != "uptime_sec"}
        self.assertEqual(validate_heartbeat_payload(missing), ["missing uptime_sec"])

    def test_uptime_none(self):
        self.assertEqual(validate_heartbeat_payload(payload(uptime_sec=None)), [])

    def test_uptime_must_be_integer(self):
        for value in (12345.5, "12345"):
            self.assertEqual(
                validate_heartbeat_payload(payload(uptime_sec=value)), ["invalid uptime_sec"], value
            )

    def test_temperature_none(self):
        self.assertEqual(validate_heartbeat_payload(payload(temperature_c=None)), [])

    def test_temperature_may_be_below_zero(self):
        self.assertEqual(validate_heartbeat_payload(payload(temperature_c=-4.5)), [])

    def test_last_detection_none(self):
        self.assertEqual(validate_heartbeat_payload(payload(last_detection_ts=None)), [])

    def test_invalid_storage(self):
        for value in ("180.2", [180.2], {"free_gb": 180.2}):
            errors = validate_heartbeat_payload(payload(storage_free_gb=value))
            self.assertEqual(errors, ["invalid storage_free_gb"], value)

    def test_invalid_timestamp(self):
        for value in ("2026-10-09T10:00:00", [LAST_DETECTION], {"ts": LAST_DETECTION}):
            errors = validate_heartbeat_payload(payload(last_detection_ts=value))
            self.assertEqual(errors, ["invalid last_detection_ts"], value)

    def test_bool_as_uptime(self):
        for value in (True, False):
            self.assertEqual(
                validate_heartbeat_payload(payload(uptime_sec=value)), ["invalid uptime_sec"], value
            )

    def test_bool_as_other_numbers(self):
        errors = validate_heartbeat_payload(
            payload(storage_free_gb=True, temperature_c=False, last_detection_ts=True)
        )
        self.assertEqual(
            errors, ["invalid storage_free_gb", "invalid temperature_c", "invalid last_detection_ts"]
        )

    def test_nan_values(self):
        nan = float("nan")
        errors = validate_heartbeat_payload(
            payload(storage_free_gb=nan, temperature_c=nan, last_detection_ts=nan)
        )
        self.assertEqual(
            errors, ["invalid storage_free_gb", "invalid temperature_c", "invalid last_detection_ts"]
        )

    def test_infinity_values(self):
        inf = float("inf")
        errors = validate_heartbeat_payload(
            payload(storage_free_gb=inf, temperature_c=-inf, last_detection_ts=inf)
        )
        self.assertEqual(
            errors, ["invalid storage_free_gb", "invalid temperature_c", "invalid last_detection_ts"]
        )

    def test_negative_uptime(self):
        self.assertEqual(validate_heartbeat_payload(payload(uptime_sec=-1)), ["invalid uptime_sec"])

    def test_negative_storage(self):
        self.assertEqual(
            validate_heartbeat_payload(payload(storage_free_gb=-0.5)), ["invalid storage_free_gb"]
        )

    def test_zero_is_valid(self):
        self.assertEqual(validate_heartbeat_payload(payload(uptime_sec=0, storage_free_gb=0)), [])

    def test_multiple_validation_errors(self):
        errors = validate_heartbeat_payload(
            {"status": "broken", "uptime_sec": -5, "storage_free_gb": None, "temperature_c": "hot"}
        )
        self.assertEqual(
            sorted(errors),
            sorted(
                [
                    "missing node_id",
                    "missing last_detection_ts",
                    "invalid status",
                    "invalid uptime_sec",
                    "invalid temperature_c",
                ]
            ),
        )

    def test_linux_style_payload(self):
        # Raspberry Pi: every sensor present.
        linux = payload(uptime_sec=864_000, storage_free_gb=412.7, temperature_c=61.3)
        self.assertEqual(validate_heartbeat_payload(linux), [])

    def test_windows_style_payload(self):
        # Development laptop: no /proc/uptime, no SoC sensor, no detection yet.
        windows = payload(uptime_sec=None, temperature_c=None, last_detection_ts=None, status="down")
        self.assertEqual(validate_heartbeat_payload(windows), [])

    def test_extra_keys_are_ignored(self):
        full = payload(camera_id="cam01", ts=NOW, disk=DISK, edge=HEALTHY_EDGE, db_write_ok=True)
        self.assertEqual(validate_heartbeat_payload(full), [])

    def test_unexpected_error_is_contained(self):
        with (
            mock.patch.object(heartbeat, "_validate", side_effect=RuntimeError("boom")),
            self.assertLogs("network.heartbeat", "ERROR"),
        ):
            self.assertEqual(validate_heartbeat_payload(payload()), ["internal validation error"])


class BuildPayloadTests(HeartbeatTestCase):
    def test_build_payload_success(self):
        built = build_heartbeat_payload(HEALTHY_EDGE, DISK, uptime_sec=12345, temperature_c=54.1)
        self.assertEqual(built, VALID_PAYLOAD)
        self.assertEqual(list(built), list(VALID_PAYLOAD))
        self.assertEqual(validate_heartbeat_payload(built), [])

    def test_node_id_comes_from_identity(self):
        self.get_identity.return_value = {"node_id": "Node-042", "mac": "88:a2:9e:c1:68:6f"}
        self.assertEqual(build_heartbeat_payload(HEALTHY_EDGE, DISK)["node_id"], "Node-042")
        self.get_identity.assert_called_once_with()

    def test_identity_failure_does_not_lose_the_heartbeat(self):
        self.get_identity.side_effect = RuntimeError("boom")
        with self.assertLogs("network.heartbeat", "ERROR"):
            built = build_heartbeat_payload(HEALTHY_EDGE, DISK)
        self.assertIsNone(built["node_id"])
        self.assertEqual(built["status"], "ok")

    def test_storage_from_disk_status(self):
        self.assertEqual(build_heartbeat_payload(HEALTHY_EDGE, DISK)["storage_free_gb"], 180.2)

    def test_storage_none(self):
        # health_main._disk_usage() returns None when the disk can't be read.
        built = build_heartbeat_payload(HEALTHY_EDGE, None)
        self.assertIsNone(built["storage_free_gb"])
        self.assertEqual(validate_heartbeat_payload(built), [])

    def test_last_detection_explicit_argument_wins(self):
        built = build_heartbeat_payload(HEALTHY_EDGE, DISK, last_detection_ts=1_760_000_000.0)
        self.assertEqual(built["last_detection_ts"], 1_760_000_000.0)

    def test_last_detection_from_edge_health(self):
        self.assertEqual(build_heartbeat_payload(HEALTHY_EDGE, DISK)["last_detection_ts"], LAST_DETECTION)

    def test_last_detection_none_when_unknown(self):
        built = build_heartbeat_payload(edge(last_detection_timestamp=None), DISK)
        self.assertIsNone(built["last_detection_ts"])

    def test_values_passed_through_unchanged(self):
        built = build_heartbeat_payload(HEALTHY_EDGE, DISK, uptime_sec=None, temperature_c=None)
        self.assertIsNone(built["uptime_sec"])
        self.assertIsNone(built["temperature_c"])

    def test_bad_inputs_do_not_raise(self):
        for edge_health, disk in ((None, None), ("x", 42), ([], [])):
            built = build_heartbeat_payload(edge_health, disk)
            self.assertEqual(built["status"], "down")
            self.assertIsNone(built["storage_free_gb"])


class StatusTests(HeartbeatTestCase):
    def status(self, edge_health) -> str:
        return build_heartbeat_payload(edge_health, DISK)["status"]

    def test_ok(self):
        self.assertEqual(self.status(HEALTHY_EDGE), "ok")

    def test_down_when_process_not_alive(self):
        self.assertEqual(self.status(edge(edge_process_alive=False)), "down")

    def test_down_without_edge_health(self):
        self.assertEqual(self.status(None), "down")
        self.assertEqual(self.status({}), "down")

    def test_starting(self):
        self.assertEqual(self.status(edge(inference_status="starting")), "starting")
        self.assertEqual(self.status(edge(tracker_status="starting")), "starting")

    def test_degraded(self):
        for changes in (
            {"camera_stream_status": "stalled"},
            {"camera_stream_status": "no_frames", "inference_status": "idle"},
            {"tracker_status": "frozen"},
            {"inference_status": "error"},
            {"inference_status": "frozen"},
        ):
            self.assertEqual(self.status(edge(**changes)), "degraded", changes)

    def test_status_from_real_edge_health(self):
        fresh = {
            "written_ts": NOW,
            "inference_loop_ts": NOW,
            "last_inference_ts": NOW,
            "pipeline_loop_ts": NOW,
            "last_frame_ts": NOW,
            "stream": {"is_stalled": False},
        }
        self.assertEqual(self.status(evaluate_edge_health(fresh, NOW)), "ok")
        self.assertEqual(
            self.status(evaluate_edge_health({**fresh, "pipeline_loop_ts": None}, NOW)), "starting"
        )
        self.assertEqual(self.status(evaluate_edge_health(fresh, NOW + 120)), "down")  # status file stale
        self.assertEqual(self.status(evaluate_edge_health(None, NOW)), "down")  # no status file


class CollectHeartbeatTests(HeartbeatTestCase):
    """health_main.collect_heartbeat(): existing fields kept, six appended."""

    EXISTING_KEYS = ("camera_id", "ts", "queues", "disk", "throttled", "throttle", "edge", "retention")

    def setUp(self):
        super().setUp()
        for name, value in (
            ("read_status_file", None),
            ("_pi_throttle_status", {"raw": None, "flags": None, "error": "not_available"}),
            ("_disk_usage", DISK),
            ("_system_uptime_sec", 12345),
            ("read_soc_temp_c", 54.1),
        ):
            patcher = mock.patch.object(health_main, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_summary_fields_appended(self):
        result = health_main.collect_heartbeat("cam01", now=NOW)
        self.assertEqual(list(result), [*self.EXISTING_KEYS, *VALID_PAYLOAD])
        self.assertEqual(result["node_id"], "Node-001")
        self.assertEqual(result["status"], "down")  # no edge_status.json
        self.assertEqual(result["uptime_sec"], 12345)
        self.assertEqual(result["storage_free_gb"], DISK["free_gb"])
        self.assertEqual(result["temperature_c"], 54.1)
        self.assertIsNone(result["last_detection_ts"])
        self.assertEqual(validate_heartbeat_payload(result), [])

    def test_existing_fields_unchanged(self):
        result = health_main.collect_heartbeat("cam01", now=NOW)
        self.assertEqual(result["camera_id"], "cam01")
        self.assertEqual(result["ts"], NOW)
        self.assertEqual(result["disk"], DISK)
        self.assertFalse(result["edge"]["edge_process_alive"])
        self.assertEqual(result["throttle"], {"flags": None, "error": "not_available"})

    def test_disk_is_read_once(self):
        health_main.collect_heartbeat("cam01", now=NOW)
        health_main._disk_usage.assert_called_once()

    def test_running_edge_reports_ok_and_last_detection(self):
        status_file = {
            "written_ts": NOW,
            "inference_loop_ts": NOW,
            "last_inference_ts": NOW,
            "pipeline_loop_ts": NOW,
            "last_frame_ts": NOW,
            "last_detection_ts": LAST_DETECTION,
            "stream": {"is_stalled": False},
        }
        with mock.patch.object(health_main, "read_status_file", return_value=status_file):
            result = health_main.collect_heartbeat("cam01", now=NOW)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["last_detection_ts"], LAST_DETECTION)
        self.assertEqual(result["edge"]["last_detection_timestamp"], LAST_DETECTION)


class SystemUptimeTests(unittest.TestCase):
    """health_main._system_uptime_sec(), with a fake /proc/uptime."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.proc_uptime = Path(tmp.name) / "uptime"
        patcher = mock.patch.object(health_main, "_PROC_UPTIME", str(self.proc_uptime))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_linux_environment(self):
        self.proc_uptime.write_text("864000.73 3400000.12\n", encoding="ascii")
        self.assertEqual(health_main._system_uptime_sec(), 864000)

    def test_windows_environment(self):
        # No /proc on Windows: uptime is unknown, not an error.
        self.assertIsNone(health_main._system_uptime_sec())

    def test_unreadable_contents(self):
        for text in ("", "garbage\n"):
            self.proc_uptime.write_text(text, encoding="ascii")
            self.assertIsNone(health_main._system_uptime_sec(), text)


if __name__ == "__main__":
    unittest.main()
