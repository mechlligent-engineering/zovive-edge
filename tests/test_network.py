import unittest
from unittest.mock import MagicMock, patch

from network.base_station_client import BaseStationClient, BaseStationError, NullBaseStationClient
from network.transfer_state import TransferState


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


class TestTransferState(unittest.TestCase):
    def test_new_event_is_immediately_ready(self):
        state = TransferState()
        self.assertTrue(state.is_ready("e1", now=1000.0))

    def test_failure_schedules_backoff(self):
        state = TransferState(initial_backoff_sec=10.0, max_backoff_sec=100.0, multiplier=2.0)
        state.record_failure("e1", now=1000.0)
        self.assertFalse(state.is_ready("e1", now=1005.0))
        self.assertTrue(state.is_ready("e1", now=1011.0))

    def test_repeated_failures_increase_backoff(self):
        state = TransferState(initial_backoff_sec=10.0, max_backoff_sec=1000.0, multiplier=2.0)
        b1 = state.record_failure("e1", now=0.0)
        b2 = state.record_failure("e1", now=0.0)
        self.assertGreater(b2, b1)

    def test_backoff_caps_at_max(self):
        state = TransferState(initial_backoff_sec=10.0, max_backoff_sec=15.0, multiplier=10.0)
        state.record_failure("e1", now=0.0)
        b2 = state.record_failure("e1", now=0.0)
        self.assertLessEqual(b2, 15.0)

    def test_clear_resets_state(self):
        state = TransferState()
        state.record_failure("e1", now=0.0)
        self.assertFalse(state.is_ready("e1", now=0.0))
        state.clear("e1")
        self.assertTrue(state.is_ready("e1", now=0.0))


if __name__ == "__main__":
    unittest.main()
