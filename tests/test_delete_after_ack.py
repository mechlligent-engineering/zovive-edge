"""Tests the ACK-triggered local-file-deletion behavior added to
transfer_main.py's send loops: Pi storage is temporary, the base
station is the permanent copy, so a successful send (an ACK) deletes
the local file, while a failed send keeps it and retries — matching
the storage/network model described for this node (temporary Pi
files + SQLite metadata; permanent base-station copies; ACK -> delete,
no ACK -> keep + retry).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import paths
from db.init_db import init_db
from db.outbox import (
    EventRecord,
    get_pending,
    get_pending_videos,
    insert_event,
    set_video_path,
)
from network.base_station_client import BaseStationError
from network.transfer_state import TransferState
from transfer_main import _send_pending_images, _send_pending_videos, run
from utils import config_loader


class _FakeImageClient:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = 0

    def send_event(self, event, snapshot_bytes=None):
        self.calls += 1
        if self.fail:
            raise BaseStationError("simulated failure")


class _FakeVideoClient:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = 0

    def send_video(self, event_id, video_bytes, timeout_sec=None):
        self.calls += 1
        if self.fail:
            raise BaseStationError("simulated failure")


class TestDeleteImageAfterAck(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="zovive_delete_ack_test_")
        self._orig_db_path = paths.DB_PATH
        paths.DB_PATH = Path(self.tmpdir) / "test.sqlite3"
        init_db()

        self.snapshot_path = Path(self.tmpdir) / "snap.jpg"
        self.snapshot_path.write_bytes(b"fake-jpeg-bytes")
        self.event_id = insert_event(
            EventRecord(
                track_id="t1",
                camera_id="cam01",
                animal_name="tiger",
                confidence=0.9,
                snapshot_path=str(self.snapshot_path),
            )
        )

    def tearDown(self):
        paths.DB_PATH = self._orig_db_path

    def test_ack_deletes_local_snapshot_when_enabled(self):
        client = _FakeImageClient(fail=False)
        _send_pending_images(client, TransferState(), batch_size=10, delete_after_ack=True)

        self.assertFalse(self.snapshot_path.exists(), "local snapshot should be deleted after ack")
        row = get_pending(limit=10)
        self.assertEqual(row, [], "event should no longer be pending (status='sent')")

    def test_ack_keeps_local_snapshot_when_disabled(self):
        client = _FakeImageClient(fail=False)
        _send_pending_images(client, TransferState(), batch_size=10, delete_after_ack=False)

        self.assertTrue(self.snapshot_path.exists(), "snapshot must be kept when the flag is off")

    def test_no_ack_keeps_local_snapshot_and_retries(self):
        client = _FakeImageClient(fail=True)
        _send_pending_images(client, TransferState(), batch_size=10, delete_after_ack=True)

        self.assertTrue(self.snapshot_path.exists(), "a failed send must never delete the local file")
        # Still shows up for retry (status='failed', not deleted from the
        # pending pool by delete-after-ack logic — that's the outbox's
        # own retry mechanism, unaffected by this feature).
        from db.outbox import count_by_status

        self.assertEqual(count_by_status(), {"failed": 1})

    def test_default_delete_after_ack_is_off(self):
        # Callers that don't pass delete_after_ack explicitly (e.g. an
        # older direct caller) must keep the pre-existing, safe behavior.
        client = _FakeImageClient(fail=False)
        _send_pending_images(client, TransferState(), batch_size=10)

        self.assertTrue(self.snapshot_path.exists())


class TestDeleteVideoAfterAck(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="zovive_delete_ack_video_test_")
        self._orig_db_path = paths.DB_PATH
        paths.DB_PATH = Path(self.tmpdir) / "test.sqlite3"
        init_db()

        self.video_path = Path(self.tmpdir) / "clip.mp4"
        self.video_path.write_bytes(b"fake-mp4-bytes")
        self.event_id = insert_event(
            EventRecord(
                track_id="t1", camera_id="cam01", animal_name="tiger", confidence=0.9, snapshot_path="snapshots/x.jpg"
            )
        )
        set_video_path(self.event_id, str(self.video_path))

    def tearDown(self):
        paths.DB_PATH = self._orig_db_path

    def test_ack_deletes_local_video_when_enabled(self):
        client = _FakeVideoClient(fail=False)
        _send_pending_videos(client, TransferState(), batch_size=10, timeout_sec=30.0, delete_after_ack=True)

        self.assertFalse(self.video_path.exists(), "local clip should be deleted after ack")
        self.assertEqual(get_pending_videos(), [])

    def test_no_ack_keeps_local_video_and_retries(self):
        client = _FakeVideoClient(fail=True)
        _send_pending_videos(client, TransferState(), batch_size=10, timeout_sec=30.0, delete_after_ack=True)

        self.assertTrue(self.video_path.exists(), "a failed send must never delete the local clip")


class TestNoBaseStationNeverDeletes(unittest.TestCase):
    """transfer_main.run() itself must force delete_local_files_after_ack
    off when base_station.url is empty, regardless of the config value —
    NullBaseStationClient "succeeds" unconditionally, so without this
    guard a dev/test setup with no real base station would delete every
    snapshot right after creating it, with nothing backing it up
    anywhere. This exercises run() end-to-end, not just the already-
    resolved boolean the other tests in this file pass directly.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="zovive_no_base_station_test_")
        self._orig_db_path = paths.DB_PATH
        self._orig_transfer_config = paths.TRANSFER_CONFIG
        paths.DB_PATH = Path(self.tmpdir) / "test.sqlite3"

        transfer_config_path = Path(self.tmpdir) / "transfer_config.yaml"
        transfer_config_path.write_text(
            "base_station:\n"
            '  url: ""\n'
            "transfer:\n"
            "  poll_interval_sec: 0.01\n"
            "  video_upload_enabled: false\n"
            "  delete_local_files_after_ack: true\n"  # on in config...
        )
        paths.TRANSFER_CONFIG = transfer_config_path
        config_loader.clear_cache()

        init_db()
        self.snapshot_path = Path(self.tmpdir) / "snap.jpg"
        self.snapshot_path.write_bytes(b"fake-jpeg-bytes")
        insert_event(
            EventRecord(
                track_id="t1",
                camera_id="cam01",
                animal_name="tiger",
                confidence=0.9,
                snapshot_path=str(self.snapshot_path),
            )
        )

    def tearDown(self):
        paths.DB_PATH = self._orig_db_path
        paths.TRANSFER_CONFIG = self._orig_transfer_config
        config_loader.clear_cache()

    @patch("transfer_main.migrate")
    @patch("transfer_main.init_db")
    def test_empty_base_station_url_forces_deletion_off(self, mock_init_db, mock_migrate):
        # run() calls init_db()/migrate() again internally; the DB is
        # already set up by setUp() against the same (monkeypatched)
        # paths.DB_PATH, so these just need to not blow up the real
        # runtime dirs — no-op them.
        run(max_cycles=1)  # ...but url is empty -> NullBaseStationClient -> must NOT delete

        self.assertTrue(
            self.snapshot_path.exists(),
            "with no base_station.url configured, the local file must never be deleted "
            "even though delete_local_files_after_ack: true is set",
        )


if __name__ == "__main__":
    unittest.main()
