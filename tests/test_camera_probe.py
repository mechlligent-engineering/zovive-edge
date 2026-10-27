"""tools/camera_probe.py: the Phase 0 camera check. Everything runs against
fakes; the real camera is only ever touched by a person running the tool."""

import itertools
import unittest
from unittest.mock import MagicMock

import cv2
import numpy as np

from tests.helpers import TEST_CAMERA_ENV, blank_frame
from tools.camera_probe import (
    _wait_lens_settled,
    compare_expected,
    largest_profile_token,
    lens_travel_test,
    measure_stream,
    phase0_checklist,
    run_probe,
    with_credentials,
)

PASSWORD = TEST_CAMERA_ENV["ZOVIVE_CAMERA_PASSWORD"]
SETTLE_KEYS = {"settled", "settle_sec", "final_position", "move_status_reported", "saw_moving"}


def _profile(token, width=None, height=None):
    return dict(token=token, width=width, height=height)


def _counter_clock(step=0.01):
    ticks = itertools.count()
    return lambda: next(ticks) * step


def _fake_cap(frame, fourcc="h264", fps=10.0, fail_after=None):
    cap = MagicMock()
    reads = {"n": 0}

    def read():
        reads["n"] += 1
        if fail_after is not None and reads["n"] > fail_after:
            return False, None
        return True, frame

    cap.read.side_effect = read
    code = sum(ord(c) << (8 * i) for i, c in enumerate(fourcc))
    cap.get.side_effect = lambda prop: {cv2.CAP_PROP_FOURCC: code, cv2.CAP_PROP_FPS: fps}.get(prop, 0.0)
    return cap


def _rtsp_cfg():
    return {
        "camera": {
            "host": "10.0.0.5",
            "username": "admin",
            "password": PASSWORD,
            "sub_stream": {
                "url": "rtsp://{user}:{pass}@{host}:554/live?subtype=1",
                "width": 1280,
                "height": 720,
                "fps": 10,
            },
            "main_stream": {
                "url": "rtsp://{user}:{pass}@{host}:554/live?subtype=0",
                "width": 2688,
                "height": 1520,
                "fps": 25,
            },
        }
    }


class FakeProbeCamera:
    """Duck-typed OnvifLensCamera with a lens that takes a few polls to move."""

    def __init__(self, move_status_supported=True):
        self.position = 0.0
        self.target = 0.0
        self.move_status_supported = move_status_supported
        self.calls = []

    def describe_media_profiles(self):
        return [
            _profile("P0", 2688, 1520),
            _profile("P1", 1280, 720),
            _profile("P2", 704, 576),
        ]

    def get_stream_uri(self, token):
        return {"P0": "rtsp://10.0.0.5:554/live?subtype=0", "P1": "rtsp://10.0.0.5:554/live?subtype=1"}.get(
            token, "rtsp://10.0.0.5:554/live?subtype=2"
        )

    def get_snapshot_uri(self, token):
        return f"http://admin:{PASSWORD}@10.0.0.5/snap/{token}"

    def get_device_info(self):
        return {"Model": "CP-UNC-TE41ZL6C-VMDS-Q"}

    def get_zoom_limits(self):
        return (0.0, 1.0)

    def get_zoom(self):
        # Moves 0.25 per poll, so a full wide-to-tele travel takes several polls.
        delta = self.target - self.position
        self.position = self.target if abs(delta) <= 0.25 else self.position + (0.25 if delta > 0 else -0.25)
        return self.position

    def get_move_status(self):
        if not self.move_status_supported:
            return None
        return "MOVING" if abs(self.target - self.position) > 0.001 else "IDLE"

    def get_focus_status(self):
        return {"position": 0.5, "move_status": "IDLE", "error": ""}

    def get_imaging_options(self):
        return {"autofocus_modes": ["AUTO"]}

    def get_event_topics(self):
        return ["RuleEngine/CellMotionDetector/Motion", "VideoSource/MotionAlarm", "VideoSource/ImageTooDark"]

    def get_snapshot_jpeg(self, token):
        ok, data = cv2.imencode(".jpg", np.zeros((1520, 2688, 3), dtype=np.uint8))
        return data.tobytes()

    def zoom_to(self, level):
        self.calls.append(("zoom_to", level))
        self.target = level
        return True

    def trigger_autofocus(self):
        self.calls.append(("trigger_autofocus",))
        return True


class CpPlusLens:
    """The CP-UNC-TE41ZL6C as seen on the bench: MoveStatus is IDLE even
    while the motor runs, the motor only starts some polls after the command
    (8 here, longer than the 5-poll stability window), and a full-tele command
    stops the optical zoom at ~0.76."""

    def __init__(self, start_delay_polls=8, step=0.1, stop_at=0.7609):
        self.position = 0.0
        self.target = 0.0
        self.start_delay_polls = start_delay_polls
        self.step = step
        self.stop_at = stop_at
        self.polls = 0

    def zoom_to(self, level):
        self.target = level
        self.polls = 0
        return True

    def get_move_status(self):
        return "IDLE"

    def get_zoom(self):
        self.polls += 1
        if self.polls > self.start_delay_polls:
            goal = min(self.target, self.stop_at)
            delta = goal - self.position
            self.position = (
                goal if abs(delta) <= self.step else self.position + (self.step if delta > 0 else -self.step)
            )
        return self.position


class TestMeasureStream(unittest.TestCase):
    def test_measures_resolution_codec_and_fps(self):
        cap = _fake_cap(blank_frame(1280, 720))
        result = measure_stream(
            f"rtsp://admin:{PASSWORD}@cam/live",
            1.0,
            opener=lambda url: cap,
            clock=_counter_clock(),
            cpu_clock=_counter_clock(0.0),
        )
        self.assertTrue(result["opened"])
        self.assertEqual((result["width"], result["height"]), (1280, 720))
        self.assertEqual(result["codec"], "h264")
        self.assertGreater(result["measured_fps"], 0)
        self.assertNotIn(PASSWORD, result["url"])
        cap.release.assert_called_once()

    def test_unopenable_stream(self):
        result = measure_stream("rtsp://cam/x", 1.0, opener=lambda url: None)
        self.assertEqual(result["opened"], False)

    def test_stops_after_repeated_read_failures(self):
        cap = _fake_cap(blank_frame(), fail_after=5)
        result = measure_stream("rtsp://cam/x", 100.0, opener=lambda url: cap, clock=_counter_clock())
        self.assertEqual(result["read_failures"], 3)
        self.assertEqual(result["frames"], 4)  # 5 good reads, the first one timed separately


class TestHelpers(unittest.TestCase):
    def test_with_credentials_quotes_special_characters(self):
        url = with_credentials("rtsp://10.0.0.5:554/live?subtype=2", "admin", "p@ss:word")
        self.assertEqual(url, "rtsp://admin:p%40ss%3Aword@10.0.0.5:554/live?subtype=2")
        self.assertEqual(with_credentials("rtsp://u:p@h/x", "a", "b"), "rtsp://u:p@h/x")

    def test_largest_profile(self):
        self.assertEqual(largest_profile_token(FakeProbeCamera().describe_media_profiles()), "P0")
        self.assertEqual(largest_profile_token([_profile("only")]), "only")
        self.assertIsNone(largest_profile_token([]))

    def test_compare_expected_reports_mismatches(self):
        streams = {
            "sub_stream": {
                "opened": True,
                "first_frame_sec": 0.5,
                "width": 704,
                "height": 576,
                "measured_fps": 25.0,
            },
            "main_stream": {
                "opened": True,
                "first_frame_sec": 0.8,
                "width": 2688,
                "height": 1520,
                "measured_fps": 24.6,
            },
        }
        issues = compare_expected(_rtsp_cfg(), streams)
        self.assertEqual(len(issues), 2)
        self.assertIn("sub_stream: resolution 704x576", issues[0])
        self.assertIn("measured 25.0 fps, expected 10", issues[1])

    def test_compare_expected_unopened(self):
        issues = compare_expected(_rtsp_cfg(), {"sub_stream": {"opened": False}})
        self.assertEqual(
            issues, ["sub_stream: could not open the stream", "main_stream: could not open the stream"]
        )


class TestLensTravel(unittest.TestCase):
    def test_measures_both_directions_and_ends_wide(self):
        cam = FakeProbeCamera()
        result = lens_travel_test(cam, (0.0, 1.0), clock=_counter_clock(0.2), sleep=lambda s: None)
        self.assertTrue(result["wide_to_tele"]["settled"])
        self.assertTrue(result["wide_to_tele"]["saw_moving"])
        self.assertTrue(result["tele_to_wide"]["settled"])
        self.assertEqual(cam.calls[-2], ("zoom_to", 0.0))  # back to the patrol (wide) position
        self.assertEqual(cam.calls[-1], ("trigger_autofocus",))
        self.assertTrue(result["autofocus"]["settled"])

    def test_falls_back_to_position_when_move_status_unsupported(self):
        cam = FakeProbeCamera(move_status_supported=False)
        result = lens_travel_test(cam, (0.0, 1.0), clock=_counter_clock(0.2), sleep=lambda s: None)
        self.assertTrue(result["wide_to_tele"]["settled"])
        self.assertFalse(result["wide_to_tele"]["move_status_reported"])

    def test_never_pans_or_tilts(self):
        cam = FakeProbeCamera()
        lens_travel_test(cam, None, clock=_counter_clock(0.2), sleep=lambda s: None)
        self.assertEqual({c[0] for c in cam.calls}, {"zoom_to", "trigger_autofocus"})


class TestWaitLensSettled(unittest.TestCase):
    """Regression: the CP Plus reports MoveStatus IDLE during zoom travel, and
    the probe used to report "settled at 0.0" before its motor had started."""

    def _wait(self, camera, start, target, timeout_sec=20.0):
        return _wait_lens_settled(
            camera, start, target, timeout_sec, 0.2, _counter_clock(0.2), lambda s: None
        )

    def test_always_idle_camera_waits_for_late_motor(self):
        lens = CpPlusLens()
        lens.zoom_to(1.0)
        result = self._wait(lens, 0.0, 1.0)
        self.assertEqual(set(result), SETTLE_KEYS)
        self.assertTrue(result["settled"])
        self.assertAlmostEqual(result["final_position"], 0.7609, places=4)
        self.assertTrue(result["move_status_reported"])
        self.assertFalse(result["saw_moving"])

    def test_lens_that_never_moves_exits_with_unchanged_position(self):
        cam = MagicMock()
        cam.get_move_status.return_value = "IDLE"
        cam.get_zoom.return_value = 0.0
        result = self._wait(cam, 0.0, 1.0, timeout_sec=2.0)
        self.assertEqual(set(result), SETTLE_KEYS)
        self.assertFalse(result["settled"])
        self.assertIsNone(result["settle_sec"])
        self.assertEqual(result["final_position"], 0.0)
        self.assertGreater(cam.get_zoom.call_count, 5)  # kept polling past the stability window

    def test_lens_already_at_target_settles_without_waiting(self):
        cam = MagicMock()
        cam.get_move_status.return_value = "IDLE"
        cam.get_zoom.return_value = 0.0
        result = self._wait(cam, 0.0, 0.0)
        self.assertTrue(result["settled"])
        self.assertEqual(result["final_position"], 0.0)
        self.assertEqual(cam.get_zoom.call_count, 6)  # first read + 5 stable polls


class TestRunProbe(unittest.TestCase):
    def _measurer(self):
        seen = []

        def measure(url, seconds):
            seen.append(url)
            ok = {"opened": True, "first_frame_sec": 0.4, "measured_fps": 10.0, "codec": "h264"}
            if "subtype=0" in url:
                return {**ok, "width": 2688, "height": 1520, "measured_fps": 25.0, "codec": "hevc"}
            if "subtype=1" in url:
                return {**ok, "width": 1280, "height": 720}
            return {**ok, "width": 704, "height": 576}

        return measure, seen

    def test_full_probe_with_fakes(self):
        measure, seen = self._measurer()
        report = run_probe(
            _rtsp_cfg(),
            1.0,
            snapshot_count=2,
            lens_test=False,
            camera_factory=lambda cfg: FakeProbeCamera(),
            stream_measurer=measure,
        )
        # Configured sub + main, plus the ONVIF-only third profile.
        self.assertEqual(set(report["streams"]), {"sub_stream", "main_stream", "onvif:P2"})
        self.assertIn(f"admin:{PASSWORD}@", seen[-1])  # credentials injected for the extra stream...
        self.assertNotIn(PASSWORD, str(report["onvif"]["profiles"]))  # ...but never in the report
        checklist = report["checklist"]
        self.assertTrue(checklist["sub_stream_ok"] and checklist["main_stream_ok"])
        self.assertEqual(checklist["main_stream_codec"], "hevc")
        self.assertEqual(checklist["extra_streams_working"], ["onvif:P2"])
        self.assertEqual(checklist["snapshot_resolution"], "2688x1520")
        self.assertEqual(
            checklist["motion_event_topics"],
            ["RuleEngine/CellMotionDetector/Motion", "VideoSource/MotionAlarm"],
        )
        self.assertEqual(checklist["mismatches"], [])
        self.assertNotIn("lens_test", report)

    def test_onvif_failure_still_runs_rtsp_checks(self):
        measure, _ = self._measurer()

        def broken(cfg):
            raise RuntimeError("onvif-zeep is not installed")

        report = run_probe(_rtsp_cfg(), 1.0, 2, False, camera_factory=broken, stream_measurer=measure)
        self.assertIsNone(report["onvif"])
        self.assertIn("onvif_error", report)
        self.assertEqual(set(report["streams"]), {"sub_stream", "main_stream"})
        self.assertFalse(report["checklist"]["onvif_ok"])

    def test_lens_test_only_when_asked(self):
        measure, _ = self._measurer()
        report = run_probe(
            _rtsp_cfg(),
            1.0,
            0,
            lens_test=True,
            camera_factory=lambda cfg: FakeProbeCamera(),
            stream_measurer=measure,
        )
        self.assertIn("lens_test", report)
        self.assertNotIn("snapshots", report)

    def test_checklist_on_empty_report(self):
        checklist = phase0_checklist({})
        self.assertFalse(checklist["sub_stream_ok"])
        self.assertIsNone(checklist["snapshot_resolution"])


if __name__ == "__main__":
    unittest.main()
