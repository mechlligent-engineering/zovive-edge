"""Exercises the store-and-forward resume path: an event that fails to
send stays 'failed' in the outbox with an incrementing attempt_count,
network.transfer_state.TransferState keeps it out of the retry pool
until its backoff elapses, and once it's reset to pending it can be
picked up and marked sent — the sequence transfer_main.py's loop
performs, tested here without needing an actual base station.
"""

import tempfile
import unittest
from pathlib import Path

import paths
from db.init_db import init_db
from db.outbox import (
    EventRecord,
    count_by_status,
    get_pending,
    insert_event,
    mark_failed,
    mark_sent,
    reset_to_pending,
)
from network.transfer_state import TransferState


class TestTransferResume(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="zovive_transfer_test_")
        self._orig_db_path = paths.DB_PATH
        paths.DB_PATH = Path(self.tmpdir) / "test.sqlite3"
        init_db()

    def tearDown(self):
        paths.DB_PATH = self._orig_db_path

    def test_failed_event_is_retried_after_backoff_and_marked_sent(self):
        event_id = insert_event(
            EventRecord(
                track_id="t1", camera_id="cam01", animal_name="tiger", confidence=0.9, snapshot_path="snapshots/x.jpg"
            )
        )
        self.assertEqual(count_by_status(), {"pending": 1})

        state = TransferState(initial_backoff_sec=5.0)

        # First attempt "fails" (simulating a base station send error).
        mark_failed(event_id, "connection refused")
        state.record_failure(event_id, now=1000.0)
        self.assertEqual(count_by_status(), {"failed": 1})
        self.assertFalse(state.is_ready(event_id, now=1002.0))

        # Not ready yet -> transfer_main's loop would skip it.
        self.assertTrue(state.is_ready(event_id, now=1010.0))

        # Operator/transfer loop puts it back into the pending pool and retries.
        reset_to_pending(event_id)
        self.assertEqual(get_pending()[0]["event_id"], event_id)

        # Second attempt succeeds.
        mark_sent(event_id)
        state.clear(event_id)
        self.assertEqual(count_by_status(), {"sent": 1})
        self.assertEqual(get_pending(), [])

    def test_multiple_events_tracked_independently(self):
        e1 = insert_event(EventRecord(track_id="t1", camera_id="cam01", animal_name="tiger", confidence=0.9, snapshot_path="snapshots/a.jpg"))
        e2 = insert_event(EventRecord(track_id="t2", camera_id="cam01", animal_name="gaur", confidence=0.8, snapshot_path="snapshots/b.jpg"))

        mark_sent(e1)
        mark_failed(e2, "timeout")

        counts = count_by_status()
        self.assertEqual(counts.get("sent"), 1)
        self.assertEqual(counts.get("failed"), 1)

        remaining_pending = get_pending()
        self.assertEqual(remaining_pending, [])


if __name__ == "__main__":
    unittest.main()
