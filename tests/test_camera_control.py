import unittest
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import requests
from requests.auth import HTTPDigestAuth

from camera_control import camera_api
from camera_control.camera_api import FakeLensCamera, OnvifLensCamera, event_topic_paths, fetch_snapshot

_WSTOP = "{http://docs.oasis-open.org/wsn/t-1}topic"
_TNS1 = "{http://www.onvif.org/ver10/topics}"
# ONVIF profile / PTZ configuration identifiers (not secrets).
MAIN_PROFILE, SUB_PROFILE, PTZ_CONFIG = "Profile000", "Profile001", "PTZ000"


class _Node:
    """Minimal stand-in for the XML elements zeep returns in a TopicSet."""

    def __init__(self, tag, topic=False, children=()):
        self.tag = tag
        self._attrs = {_WSTOP: "true"} if topic else {}
        self._children = list(children)

    def get(self, key, default=None):
        return self._attrs.get(key, default)

    def __iter__(self):
        return iter(self._children)


def _topic_set():
    return [
        _Node(
            f"{_TNS1}RuleEngine",
            children=[
                _Node(
                    f"{_TNS1}CellMotionDetector",
                    children=[_Node(f"{_TNS1}Motion", topic=True, children=[_Node("{x}MessageDescription")])],
                ),
            ],
        ),
        _Node(f"{_TNS1}VideoSource", children=[_Node(f"{_TNS1}MotionAlarm", topic=True)]),
        NS(tag=None),  # an XML comment: no string tag
    ]


def _fake_onvif_class(events_fail=False):
    main = NS(
        token=MAIN_PROFILE,
        Name="MainStream",
        VideoSourceConfiguration=NS(SourceToken="VideoSource000"),
        VideoEncoderConfiguration=NS(
            Encoding="H264",
            Resolution=NS(Width=2688, Height=1520),
            RateControl=NS(FrameRateLimit=25, BitrateLimit=6144),
            H264=NS(GovLength=25),
        ),
        PTZConfiguration=NS(token=PTZ_CONFIG),
    )
    sub = NS(
        token=SUB_PROFILE,
        Name="SubStream",
        VideoSourceConfiguration=NS(SourceToken="VideoSource000"),
        VideoEncoderConfiguration=NS(
            Encoding="H264", Resolution=NS(Width=1280, Height=720), RateControl=NS(FrameRateLimit=10)
        ),
        PTZConfiguration=None,
    )
    media = MagicMock()
    media.GetProfiles.return_value = [main, sub]
    media.GetStreamUri.side_effect = lambda req: NS(Uri=f"rtsp://10.0.0.5:554/{req['ProfileToken']}")
    media.GetSnapshotUri.side_effect = lambda req: NS(Uri=f"http://10.0.0.5/snap/{req['ProfileToken']}")
    ptz = MagicMock()
    ptz.GetStatus.return_value = NS(MoveStatus=NS(Zoom="IDLE"), Position=NS(Zoom=NS(x=0.0)))
    ptz.GetConfigurationOptions.return_value = NS(
        Spaces=NS(AbsoluteZoomPositionSpace=[NS(XRange=NS(Min=0.0, Max=1.0))])
    )
    imaging = MagicMock()
    imaging.GetStatus.return_value = NS(FocusStatus20=NS(Position=0.42, MoveStatus="IDLE", Error=""))
    imaging.GetOptions.return_value = NS(
        Focus=NS(AutoFocusModes=["AUTO", "MANUAL"]), IrCutFilterModes=["AUTO"]
    )
    imaging.GetImagingSettings.return_value = NS(Focus=NS(AutoFocusMode="AUTO"), IrCutFilter="AUTO")
    events = MagicMock()
    if events_fail:
        events.GetEventProperties.side_effect = RuntimeError("no events service")
    else:
        events.GetEventProperties.return_value = NS(TopicSet=NS(_value_1=_topic_set()))

    class FakeONVIFCamera:
        def __init__(self, host, port, user, password, transport=None):
            self.devicemgmt = NS(
                GetDeviceInformation=lambda: NS(
                    Manufacturer="CP PLUS",
                    Model="CP-UNC-TE41ZL6C-VMDS-Q",
                    FirmwareVersion="2.840",
                    SerialNumber="SN1",
                    HardwareId="1.00",
                )
            )
            self.media, self.ptz, self.imaging, self.events = media, ptz, imaging, events

        def create_media_service(self):
            return self.media

        def create_ptz_service(self):
            return self.ptz

        def create_imaging_service(self):
            return self.imaging

        def create_events_service(self):
            return self.events

    return FakeONVIFCamera


def _onvif_camera(**kwargs) -> OnvifLensCamera:
    with (
        patch.object(camera_api, "_import_onvif", return_value=_fake_onvif_class(**kwargs)),
        patch.object(camera_api, "_onvif_transport", return_value=None),
    ):
        return OnvifLensCamera("10.0.0.5", 80, "admin", "secret")


class TestFakeLensCamera(unittest.TestCase):
    def test_zoom_to_sets_level(self):
        cam = FakeLensCamera()
        self.assertTrue(cam.zoom_to(0.5))
        self.assertEqual(cam.get_zoom(), 0.5)
        self.assertEqual(cam.call_log[-1][0], "zoom_to")

    def test_trigger_autofocus_records_call(self):
        cam = FakeLensCamera()
        self.assertTrue(cam.trigger_autofocus())
        self.assertEqual(cam.focus_count, 1)
        self.assertEqual(cam.call_log[-1][0], "trigger_autofocus")


class TestNoPanTilt(unittest.TestCase):
    def test_lens_interface_has_no_pan_tilt_or_presets(self):
        forbidden = ("pan", "tilt", "preset", "relative_move", "continuous_move")
        for cls in (camera_api.LensCamera, camera_api.FakeLensCamera, camera_api.OnvifLensCamera):
            for name in dir(cls):
                if name.startswith("_"):
                    continue
                self.assertFalse(any(f in name.lower() for f in forbidden), f"{cls.__name__}.{name}")


class TestOnvifReadOnlyQueries(unittest.TestCase):
    def test_device_info(self):
        info = _onvif_camera().get_device_info()
        self.assertEqual(info["Model"], "CP-UNC-TE41ZL6C-VMDS-Q")

    def test_media_profiles_describe_encoders_and_tolerate_missing_fields(self):
        main, sub = _onvif_camera().describe_media_profiles()
        self.assertEqual((main["width"], main["height"], main["fps_limit"]), (2688, 1520, 25))
        self.assertEqual(main["gov_length"], 25)
        self.assertTrue(main["has_ptz"])
        self.assertIsNone(sub["gov_length"])  # no H264 block reported
        self.assertFalse(sub["has_ptz"])

    def test_stream_and_snapshot_uris_per_profile(self):
        cam = _onvif_camera()
        self.assertEqual(cam.get_stream_uri("Profile001"), "rtsp://10.0.0.5:554/Profile001")
        self.assertEqual(cam.get_stream_uri(), "rtsp://10.0.0.5:554/Profile000")  # default profile
        self.assertEqual(cam.get_snapshot_uri("Profile000"), "http://10.0.0.5/snap/Profile000")

    def test_snapshot_uri_is_cached(self):
        cam = _onvif_camera()
        cam.get_snapshot_uri("Profile000")
        cam.get_snapshot_uri("Profile000")
        self.assertEqual(cam._media.GetSnapshotUri.call_count, 1)

    def test_status_and_limits(self):
        cam = _onvif_camera()
        self.assertEqual(cam.get_move_status(), "IDLE")
        self.assertEqual(cam.get_zoom_limits(), (0.0, 1.0))
        self.assertEqual(cam.get_focus_status(), {"position": 0.42, "move_status": "IDLE", "error": ""})
        imaging = cam.get_imaging_options()
        self.assertEqual(imaging["autofocus_modes"], ["AUTO", "MANUAL"])
        self.assertEqual(imaging["ircut_current"], "AUTO")

    def test_move_status_none_when_not_reported(self):
        cam = _onvif_camera()
        cam._ptz.GetStatus.return_value = NS(MoveStatus=None, Position=NS(Zoom=NS(x=0.0)))
        self.assertIsNone(cam.get_move_status())

    def test_failures_return_none_instead_of_raising(self):
        cam = _onvif_camera()
        cam._ptz.GetStatus.side_effect = RuntimeError("soap fault")
        cam._imaging.GetStatus.side_effect = RuntimeError("soap fault")
        cam._media.GetStreamUri.side_effect = RuntimeError("soap fault")
        with self.assertLogs(camera_api.log, "ERROR"):
            self.assertIsNone(cam.get_move_status())
            self.assertIsNone(cam.get_focus_status())
            self.assertIsNone(cam.get_stream_uri())

    def test_event_topics(self):
        self.assertEqual(
            _onvif_camera().get_event_topics(),
            ["RuleEngine/CellMotionDetector/Motion", "VideoSource/MotionAlarm"],
        )

    def test_event_topics_none_on_failure(self):
        with self.assertLogs(camera_api.log, "ERROR"):
            self.assertIsNone(_onvif_camera(events_fail=True).get_event_topics())

    def test_event_topic_paths_skips_message_descriptions(self):
        self.assertNotIn("MessageDescription", " ".join(event_topic_paths(_topic_set())))
        self.assertEqual(event_topic_paths(None), [])


class TestFetchSnapshot(unittest.TestCase):
    def test_returns_jpeg_using_digest_auth_only(self):
        resp = NS(status_code=200, content=b"\xff\xd8\xff\xe0jpeg")
        with patch.object(camera_api.requests, "get", return_value=resp) as get:
            self.assertEqual(fetch_snapshot("http://cam/snap", "u", "p"), resp.content)
        self.assertIsInstance(get.call_args.kwargs["auth"], HTTPDigestAuth)
        self.assertIn("timeout", get.call_args.kwargs)

    def test_rejected_credentials_do_not_fall_back_to_basic_auth(self):
        resp = NS(status_code=401, content=b"")
        with self.assertLogs(camera_api.log, "WARNING"):
            with patch.object(camera_api.requests, "get", return_value=resp) as get:
                self.assertIsNone(fetch_snapshot("http://cam/snap", "u", "p"))
        self.assertEqual(get.call_count, 1)

    def test_non_jpeg_and_network_errors_return_none(self):
        with self.assertLogs(camera_api.log, "WARNING"):
            with patch.object(
                camera_api.requests, "get", return_value=NS(status_code=200, content=b"<html>")
            ):
                self.assertIsNone(fetch_snapshot("http://cam/snap", "u", "p"))
            with patch.object(camera_api.requests, "get", side_effect=requests.ConnectionError("down")):
                self.assertIsNone(fetch_snapshot("http://cam/snap", "u", "p"))

    def test_camera_snapshot_goes_through_snapshot_uri(self):
        cam = _onvif_camera()
        resp = NS(status_code=200, content=b"\xff\xd8data")
        with patch.object(camera_api.requests, "get", return_value=resp) as get:
            self.assertEqual(cam.get_snapshot_jpeg("Profile000"), b"\xff\xd8data")
        self.assertEqual(get.call_args.args[0], "http://10.0.0.5/snap/Profile000")


if __name__ == "__main__":
    unittest.main()
