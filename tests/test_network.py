import unittest
from unittest.mock import MagicMock, patch

import requests

from network.base_station_client import (
    BaseStationClient,
    BaseStationError,
    BaseStationRejected,
    BaseStationUnreachable,
    NullBaseStationClient,
)


class TestBaseStationClient(unittest.TestCase):
    def setUp(self):
        self.client = BaseStationClient("http://base-station.local:8000", api_key="secret", timeout_sec=1.0)

    @patch("network.base_station_client.requests.post")
    def test_send_event_success(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200, raise_for_status=lambda: None)
        self.client.send_event({"event_id": "e1"})
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "http://base-station.local:8000/api/events")

    @patch("network.base_station_client.requests.post")
    def test_send_event_raises_base_station_error_on_failure(self, mock_post):
        import requests

        mock_post.side_effect = requests.ConnectionError("refused")
        with self.assertRaises(BaseStationError):
            self.client.send_event({"event_id": "e1"})

    @patch("network.base_station_client.requests.post")
    def test_send_heartbeat_success(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200, raise_for_status=lambda: None)
        self.client.send_heartbeat({"camera_id": "cam01"})
        mock_post.assert_called_once()

    @patch("network.base_station_client.requests.post")
    def test_send_video_success_posts_to_event_video_endpoint(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200, raise_for_status=lambda: None)
        self.client.send_video("evt-1", b"fake-mp4-bytes")
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "http://base-station.local:8000/api/events/evt-1/video")
        self.assertIn("video", kwargs["files"])

    @patch("network.base_station_client.requests.post")
    def test_send_video_uses_override_timeout(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200, raise_for_status=lambda: None)
        self.client.send_video("evt-1", b"x", timeout_sec=60.0)
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["timeout"], 60.0)

    @patch("network.base_station_client.requests.post")
    def test_send_video_raises_base_station_error_on_failure(self, mock_post):
        import requests

        mock_post.side_effect = requests.ConnectionError("refused")
        with self.assertRaises(BaseStationError):
            self.client.send_video("evt-1", b"x")


class TestNullBaseStationClient(unittest.TestCase):
    def test_never_raises(self):
        client = NullBaseStationClient()
        client.send_event({"event_id": "e1"})  # should just log
        client.send_video("e1", b"fake-mp4-bytes")
        client.send_heartbeat({"camera_id": "cam01"})


def _response(status_code: int, retry_after: str | None = None) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status_code
    if retry_after is not None:
        resp.headers["Retry-After"] = retry_after
    return resp


class TestFailureKinds(unittest.TestCase):
    """What a failed send is (network/retry_policy.py decides what it costs):
    only a 4xx other than 429 is a rejection; no connection is unreachable;
    everything else is a busy base station."""

    def setUp(self):
        self.client = BaseStationClient("http://base-station.local:8000", timeout_sec=1.0)

    def _raised(self, *, response=None, side_effect=None) -> BaseStationError:
        with patch(
            "network.base_station_client.requests.post", return_value=response, side_effect=side_effect
        ):
            with self.assertRaises(BaseStationError) as cm:
                self.client.send_event({"event_id": "e1"}, snapshot_bytes=b"jpeg")
        return cm.exception

    def test_4xx_other_than_429_is_a_rejection(self):
        for code in (400, 401, 404, 409, 413, 422):
            with self.subTest(code=code):
                exc = self._raised(response=_response(code))
                self.assertIsInstance(exc, BaseStationRejected)
                self.assertEqual(exc.status_code, code)

    def test_no_connection_is_unreachable(self):
        for err in (
            requests.ConnectionError("refused"),
            requests.ConnectTimeout("connect timed out"),
            requests.exceptions.SSLError("handshake failed"),
        ):
            with self.subTest(err=type(err).__name__):
                self.assertIsInstance(self._raised(side_effect=err), BaseStationUnreachable)

    def test_busy_answers_are_neither(self):
        cases = {
            "429": _response(429, retry_after="120"),
            "500": _response(500),
            "503": _response(503, retry_after="Wed, 21 Oct 2026 07:28:00 GMT"),
            "302": _response(302),
        }
        for name, resp in cases.items():
            with self.subTest(code=name):
                exc = self._raised(response=resp)
                self.assertNotIsInstance(exc, (BaseStationRejected, BaseStationUnreachable))
        self.assertEqual(self._raised(response=cases["429"]).retry_after, 120.0)
        self.assertIsNone(self._raised(response=cases["503"]).retry_after, "HTTP-date form is ignored")
        self.assertIn("redirects are not followed", str(self._raised(response=cases["302"])))

    def test_read_timeout_is_busy_not_unreachable(self):
        exc = self._raised(side_effect=requests.ReadTimeout("read timed out"))
        self.assertNotIsInstance(exc, (BaseStationRejected, BaseStationUnreachable))


if __name__ == "__main__":
    unittest.main()
