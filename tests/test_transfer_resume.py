"""Exercises the store-and-forward resume path: an event that fails to
send stays 'failed' in the outbox with its retry time on the row
(db/outbox.py claim_due / record_send_failure), is not picked up before
that time, and is picked up and marked sent after it — the sequence
transfer_main.py's loop performs, tested here without needing an actual
base station. tests/test_upload_retry.py covers the full retry rules.
"""

import tempfile
import unittest
from pathlib import Path

import paths
from db.init_db import init_db
from db.outbox import (
    EventRecord,
    claim_due,
    count_by_status,
    insert_event,
    mark_sent,
    record_send_failure,
)


def _claim(now: float) -> tuple[str, list[str]]:
    claim, rows = claim_due("image", now=now, limit=10, lease_sec=50.0, max_retries=5, max_wait_sec=1e6)
    return claim, [r["event_id"] for r in rows]


class TestTransferResume(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="zovive_transfer_test_")
        self._orig_db_path = paths.DB_PATH
        paths.DB_PATH = Path(self.tmpdir) / "test.sqlite3"
        init_db()

    def tearDown(self):
        paths.DB_PATH = self._orig_db_path

    def _event(self, track_id: str, snapshot: str) -> str:
        return insert_event(
            EventRecord(
                track_id=track_id,
                camera_id="cam01",
                animal_name="tiger",
                confidence=0.9,
                snapshot_path=snapshot,
            )
        )

    def test_failed_event_is_retried_after_its_retry_time_and_marked_sent(self):
        event_id = self._event("t1", "snapshots/x.jpg")
        self.assertEqual(count_by_status(), {"pending": 1})

        # First attempt fails with a rejection: retry in 30 s.
        claim, ids = _claim(now=1000.0)
        self.assertEqual(ids, [event_id])
        record_send_failure(
            "image", event_id, claim, now=1000.0, error="HTTP 422", retry_at=1030.0, charged=True
        )
        self.assertEqual(count_by_status(), {"failed": 1})

        # Not due yet -> transfer_main's loop would skip it.
        self.assertEqual(_claim(now=1010.0)[1], [])

        # Due -> claimed again and this time sent.
        claim, ids = _claim(now=1030.0)
        self.assertEqual(ids, [event_id])
        self.assertTrue(mark_sent(event_id, claim=claim))
        self.assertEqual(count_by_status(), {"sent": 1})
        self.assertEqual(_claim(now=5000.0)[1], [])

    def test_multiple_events_tracked_independently(self):
        e1 = self._event("t1", "snapshots/a.jpg")
        e2 = self._event("t2", "snapshots/b.jpg")

        claim, _ = _claim(now=1000.0)
        mark_sent(e1, claim=claim)
        record_send_failure("image", e2, claim, now=1000.0, error="timeout", retry_at=1060.0, charged=False)

        counts = count_by_status()
        self.assertEqual(counts.get("sent"), 1)
        self.assertEqual(counts.get("failed"), 1)
        self.assertEqual(_claim(now=1059.0)[1], [])
        self.assertEqual(_claim(now=1060.0)[1], [e2])


if __name__ == "__main__":
    unittest.main()
