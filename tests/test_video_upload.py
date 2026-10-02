"""End-to-end test of the video-upload path added on top of batch 1's
store-and-forward outbox: an event gets its video_path set once
pipeline/clip_extractor.py finishes a clip (independently of whether
its image alert already sent), transfer_main.py's video-send loop
picks it up, and a failed upload is retried without touching the
image alert's own status/backoff at all.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import paths
from db.init_db import init_db
from db.outbox import (
    EventRecord,
    get_pending_videos,
    insert_event,
    mark_sent,
    mark_video_failed,
    mark_video_sent,
    set_video_path,
)
from network.base_station_client import BaseStationError
from network.transfer_state import TransferState
from transfer_main import _send_pending_videos


class _FakeClient:
    def __init__(self, fail_first_n: int = 0):
        self.fail_first_n = fail_first_n
        self.calls: list[tuple[str, int]] = []

    def send_video(self, event_id: str, video_bytes: bytes, timeout_sec: float | None = None) -> None:
        self.calls.append((event_id, len(video_bytes)))
        if len(self.calls) <= self.fail_first_n:
            raise BaseStationError("simulated failure")


class TestVideoOutboxHelpers(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="zovive_video_test_")
        self._orig_db_path = paths.DB_PATH
        paths.DB_PATH = Path(self.tmpdir) / "test.sqlite3"
        init_db()

        self.video_path = Path(self.tmpdir) / "clip.mp4"
        self.video_path.write_bytes(b"fake-mp4-bytes")

        self.event_id = insert_event(
            EventRecord(
                track_id="t1",
                camera_id="cam01",
                animal_name="tiger",
                confidence=0.9,
                snapshot_path="snapshots/x.jpg",
            )
        )

    def tearDown(self):
        paths.DB_PATH = self._orig_db_path

    def test_new_event_has_no_video_pending(self):
        self.assertEqual(get_pending_videos(), [])

    def test_set_video_path_marks_pending(self):
        set_video_path(self.event_id, str(self.video_path))
        pending = get_pending_videos()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["event_id"], self.event_id)
        self.assertEqual(pending[0]["video_status"], "pending")

    def test_video_status_independent_of_image_status(self):
        # Image alert already sent...
        mark_sent(self.event_id)
        # ...video clip only becomes ready afterwards. Both must be
        # trackable at once without one clobbering the other.
        set_video_path(self.event_id, str(self.video_path))

        pending_videos = get_pending_videos()
        self.assertEqual(len(pending_videos), 1)
        self.assertEqual(pending_videos[0]["status"], "sent")
        self.assertEqual(pending_videos[0]["video_status"], "pending")

    def test_mark_video_sent_removes_from_pending(self):
        set_video_path(self.event_id, str(self.video_path))
        mark_video_sent(self.event_id)
        self.assertEqual(get_pending_videos(), [])

    def test_mark_video_failed_keeps_it_out_of_pending_until_reset(self):
        set_video_path(self.event_id, str(self.video_path))
        mark_video_failed(self.event_id, "connection refused")
        # video_status is now 'failed', not 'pending' -> not returned by
        # get_pending_videos() until something resets it, mirroring the
        # image alert's failed/pending split.
        self.assertEqual(get_pending_videos(), [])


class TestSendPendingVideosLoop(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="zovive_video_loop_test_")
        self._orig_db_path = paths.DB_PATH
        paths.DB_PATH = Path(self.tmpdir) / "test.sqlite3"
        init_db()

        self.video_path = Path(self.tmpdir) / "clip.mp4"
        self.video_path.write_bytes(b"0123456789")

        self.event_id = insert_event(
            EventRecord(
                track_id="t1",
                camera_id="cam01",
                animal_name="tiger",
                confidence=0.9,
                snapshot_path="snapshots/x.jpg",
            )
        )
        set_video_path(self.event_id, str(self.video_path))

    def tearDown(self):
        paths.DB_PATH = self._orig_db_path

    def test_successful_upload_marks_sent(self):
        client = _FakeClient(fail_first_n=0)
        state = TransferState()

        _send_pending_videos(client, state, batch_size=10, timeout_sec=30.0)

        self.assertEqual(client.calls, [(self.event_id, 10)])
        self.assertEqual(get_pending_videos(), [])

    def test_missing_video_file_marks_failed_permanently_without_retry(self):
        self.video_path.unlink()  # simulate disk cleanup / lost clip
        client = _FakeClient()
        state = TransferState()

        _send_pending_videos(client, state, batch_size=10, timeout_sec=30.0)

        self.assertEqual(client.calls, [], "should never call send_video for a missing file")
        self.assertEqual(get_pending_videos(), [])

    def test_failed_upload_is_retried_after_backoff(self):
        client = _FakeClient(fail_first_n=1)
        state = TransferState(
            initial_backoff_sec=100.0
        )  # long backoff so the 2nd call below is deterministic

        _send_pending_videos(client, state, batch_size=10, timeout_sec=30.0)
        self.assertEqual(len(client.calls), 1)
        # video_status is 'failed' now, so it's not in get_pending_videos()
        # until something resets it back to pending (same as image alerts).
        self.assertEqual(get_pending_videos(), [])

        # Simulate an operator/retry mechanism resetting it, then confirm
        # backoff (not yet elapsed) holds it back from a second attempt.
        from db.outbox import reset_video_to_pending

        reset_video_to_pending(self.event_id)
        _send_pending_videos(client, state, batch_size=10, timeout_sec=30.0)
        self.assertEqual(len(client.calls), 1, "backoff should have prevented an immediate retry")


if __name__ == "__main__":
    unittest.main()
