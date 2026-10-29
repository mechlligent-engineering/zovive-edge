"""Production capture streams from the RTSP addresses in configs/rtsp_config.yaml,
never from the one the camera's ONVIF service reports. The CP Plus appends
unicast=true&proto=Onvif to its ONVIF address; proto=Onvif switches the camera
to its ONVIF streaming mode, which refuses the RTSP account with 401 (verified
on the camera), so a capture path built on it would never get a frame."""

import unittest
from unittest.mock import patch
from urllib.parse import parse_qsl, urlsplit

from capture import rtsp_reader
from capture.rtsp_reader import RtspReader
from queues.frame_queue import LatestFrameSlot
from tests.helpers import camera_credentials


class TestProductionCaptureUrls(unittest.TestCase):
    def setUp(self):
        with camera_credentials():
            self.reader = RtspReader(LatestFrameSlot(name="test_capture_urls_slot"))

    def test_uses_the_verified_configured_addresses(self):
        # The addresses opened on the camera: no ONVIF session parameters.
        for url, subtype in ((self.reader.sub_url, "1"), (self.reader.main_url, "0")):
            with self.subTest(subtype=subtype):
                parts = urlsplit(url)
                self.assertEqual(parts.path, "/video/live")
                self.assertEqual(parse_qsl(parts.query), [("channel", "1"), ("subtype", subtype)])

    def test_opens_exactly_those_addresses(self):
        # Sub-stream for continuous capture, main stream for the evidence grab.
        with patch.object(rtsp_reader, "open_rtsp_capture", return_value=None) as opener:
            self.reader._open_capture()
            self.reader.grab_main_stream_frame()
        self.assertEqual(
            [c.args[0] for c in opener.call_args_list], [self.reader.sub_url, self.reader.main_url]
        )


if __name__ == "__main__":
    unittest.main()
