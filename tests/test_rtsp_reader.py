"""RtspReader reads at the camera encoder rate: it must never sleep between
frames (a throttled reader falls further behind real time every second,
because H.264 frames can't be skipped without decoding them)."""

import unittest
from unittest.mock import MagicMock, patch

from capture import rtsp_reader
from capture.rtsp_reader import RtspReader, build_rtsp_url, open_rtsp_capture
from queues.frame_queue import LatestFrameSlot
from tests.helpers import blank_frame, camera_credentials


def _reader() -> RtspReader:
    with camera_credentials():
        return RtspReader(LatestFrameSlot(name="test_rtsp_reader_slot"))


class TestNoThrottle(unittest.TestCase):
    def test_reads_every_frame_without_sleeping(self):
        reader = _reader()
        frames_to_send = 25
        sent = {"n": 0}

        def read():
            sent["n"] += 1
            if sent["n"] >= frames_to_send:
                reader._stop_event.set()
            return True, blank_frame()

        cap = MagicMock()
        cap.read.side_effect = read
        with (
            patch.object(reader, "_open_capture", return_value=cap),
            patch.object(rtsp_reader.time, "sleep") as sleep,
        ):
            reader.run()

        sleep.assert_not_called()
        self.assertEqual(reader.frame_slot.last_seq, frames_to_send)
        cap.release.assert_called_once()


class TestOpenCapture(unittest.TestCase):
    def test_returns_none_and_releases_when_not_opened(self):
        cap = MagicMock()
        cap.isOpened.return_value = False
        with patch.object(rtsp_reader.cv2, "VideoCapture", return_value=cap) as ctor:
            self.assertIsNone(open_rtsp_capture("rtsp://cam/stream"))
        cap.release.assert_called_once()
        self.assertEqual(ctor.call_args.args[0], "rtsp://cam/stream")

    def test_returns_capture_when_opened(self):
        cap = MagicMock()
        cap.isOpened.return_value = True
        with patch.object(rtsp_reader.cv2, "VideoCapture", return_value=cap):
            self.assertIs(open_rtsp_capture("rtsp://cam/stream"), cap)

    def test_build_rtsp_url_fills_template(self):
        url = build_rtsp_url("rtsp://{user}:{pass}@{host}:554/x", "10.0.0.5", "u", "p")
        self.assertEqual(url, "rtsp://u:p@10.0.0.5:554/x")


if __name__ == "__main__":
    unittest.main()
