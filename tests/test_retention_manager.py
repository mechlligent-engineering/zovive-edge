"""storage/retention_manager.py and the ACK recording it depends on.

Every test runs against its own temporary SQLite database, retention policy,
media directories, status file and lock file; a guard in setUp fails the
test if any of those points at the real deployment locations. The Base
Station is never contacted: transfers use fake clients, a mocked
requests.post or the NullBaseStationClient, and disk usage comes from a fake
filesystem that counts the media files actually left in the temp directory.
"""

from __future__ import annotations

import configparser
import logging
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import NamedTuple
from unittest import mock

import requests

import health_main
import paths
import transfer_main
from db import init_db as init_db_module
from db.init_db import get_connection, init_db
from db.outbox import EventRecord, get_event, insert_event
from network.base_station_client import BaseStationClient, BaseStationError
from network.transfer_state import TransferState
from storage import retention_manager
from storage.retention_manager import (
    EXIT_CONFIG,
    EXIT_OK,
    EXIT_TEMPFAIL,
    apply_retention,
    apply_storage_pressure_cleanup,
    get_sent_snapshots_ready_for_delete,
    get_sent_videos_ready_for_delete,
    get_storage_pressure_candidates,
    get_unacknowledged_backlog,
    purge_snapshot_after_ack,
    purge_video_after_ack,
    run_job,
)
from utils import config_loader

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGGER = "storage.retention_manager"
DAY = 86400.0
NOW = time.time()

# The real deployment locations, captured before any test patches them.
REAL_PATHS = {
    name: getattr(paths, name)
    for name in (
        "DB_PATH",
        "SNAPSHOT_DIR",
        "CLIP_DIR",
        "RETENTION_POLICY_CONFIG",
        "RETENTION_STATUS_PATH",
        "RETENTION_LOCK_PATH",
        "TRANSFER_CONFIG",
    )
}

POLICY = """\
enabled: true
retention_days: 14
storage_pressure:
  trigger_percent: 82
  target_percent: 79
overdue_unacknowledged_days: 14
"""

FILE_BYTES = 10  # every fake media file


class Event(NamedTuple):
    event_id: str
    snapshot: Path
    clip: Path | None


class FakeFilesystem:
    """Disk usage of the media filesystem: a fixed baseline plus the media
    files really present under ``root``, so deleting one lowers usage."""

    def __init__(self, root: Path, capacity: int = 1000, base_used: int = 0, releases_space: bool = True):
        self.root = root
        self.capacity = capacity
        self.base_used = base_used
        self.releases_space = releases_space
        self.calls: list[Path] = []
        self._first: int | None = None

    def used(self) -> int:
        media = sum(
            p.stat().st_size for p in self.root.rglob("*") if p.is_file() and p.suffix in (".jpg", ".mp4")
        )
        used = self.base_used + media
        if not self.releases_space:  # e.g. files still held open by another process
            self._first = used if self._first is None else self._first
            return self._first
        return used

    def __call__(self, path: Path) -> tuple[float, int]:
        self.calls.append(Path(path))
        used = self.used()
        return 100.0 * used / self.capacity, used


class RetentionTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.snapshot_dir = self.root / "snapshots"
        self.clip_dir = self.root / "clips"
        self.snapshot_dir.mkdir()
        self.clip_dir.mkdir()
        self.policy_file = self.root / "retention_policy.yaml"
        self.policy_file.write_text(POLICY, encoding="utf-8")
        self.transfer_config = self.root / "transfer_config.yaml"
        self.transfer_config.write_text(
            'base_station:\n  url: "http://192.168.1.110:8000"\n', encoding="utf-8"
        )
        self.db_path = self.root / "zovive.sqlite3"

        self.patch(paths, "DB_PATH", self.db_path)
        self.patch(paths, "SNAPSHOT_DIR", self.snapshot_dir)
        self.patch(paths, "CLIP_DIR", self.clip_dir)
        self.patch(paths, "RETENTION_POLICY_CONFIG", self.policy_file)
        self.patch(paths, "RETENTION_STATUS_PATH", self.root / "retention_status.json")
        self.patch(paths, "RETENTION_LOCK_PATH", self.root / "retention.lock")
        self.patch(paths, "TRANSFER_CONFIG", self.transfer_config)
        self.assert_isolated()

        config_loader.clear_cache()
        self.addCleanup(config_loader.clear_cache)
        init_db()
        self.addCleanup(self.close_db)

        for name in ("storage", "transfer_main", "network"):
            logger = logging.getLogger(name)
            handler = logging.NullHandler()
            logger.addHandler(handler)
            self.addCleanup(logger.removeHandler, handler)
            self.patch(logger, "propagate", False)

    def assert_isolated(self) -> None:
        """No test may reach the real database, evidence or config."""
        for name, real in REAL_PATHS.items():
            current = Path(getattr(paths, name)).resolve()
            self.assertTrue(current.is_relative_to(self.root), f"{name} is not inside the test directory")
            self.assertNotEqual(current, Path(real).resolve(), f"{name} points at real deployment data")

    def patch(self, target, attribute, *args, **kwargs):
        patcher = mock.patch.object(target, attribute, *args, **kwargs)
        mocked = patcher.start()
        self.addCleanup(patcher.stop)
        return mocked

    def close_db(self):
        get_connection(self.db_path).close()
        delattr(init_db_module._local, f"_conn_{self.db_path}")

    def sql(self, statement: str, params: tuple = ()) -> None:
        conn = get_connection()
        with conn:
            conn.execute(statement, params)

    def write_policy(self, text: str) -> None:
        self.policy_file.write_text(text, encoding="utf-8")
        config_loader.clear_cache()

    def add_event(
        self,
        name: str = "e1",
        *,
        age_days: float = 20.0,
        status: str = "sent",
        snapshot_acked: bool = True,
        ack_age_days: float | None = None,
        snapshot_file: bool = True,
        snapshot_path: Path | None = None,
        video: bool = False,
        video_status: str = "sent",
        video_acked: bool = True,
        video_file: bool = True,
    ) -> Event:
        """An event recorded ``age_days`` ago; ACKs default to the same time."""
        created = NOW - age_days * DAY
        acked = NOW - (age_days if ack_age_days is None else ack_age_days) * DAY
        snapshot = snapshot_path or self.snapshot_dir / f"{name}.jpg"
        if snapshot_file and snapshot_path is None:
            snapshot.write_bytes(b"j" * FILE_BYTES)
        event_id = insert_event(
            EventRecord(
                track_id=f"t-{name}",
                camera_id="cam01",
                animal_name="elephant",
                confidence=0.9,
                snapshot_path=str(snapshot),
                created_ts=created,
            )
        )
        self.sql(
            "UPDATE events SET status = ?, snapshot_acked_ts = ? WHERE event_id = ?",
            (status, acked if snapshot_acked else None, event_id),
        )
        clip = None
        if video:
            clip = self.clip_dir / f"{name}.mp4"
            if video_file:
                clip.write_bytes(b"m" * FILE_BYTES)
            self.sql(
                "UPDATE events SET video_path = ?, video_status = ?, video_acked_ts = ? WHERE event_id = ?",
                (str(clip), video_status, acked if video_acked else None, event_id),
            )
        return Event(event_id, snapshot, clip)

    def fake_disk(self, **kwargs) -> FakeFilesystem:
        disk = FakeFilesystem(self.root, **kwargs)
        self.patch(retention_manager, "_disk_usage", side_effect=disk)
        return disk


# ---------------------------------------------------------------------------
# Daily 14-day retention
# ---------------------------------------------------------------------------


class DailyRetentionTests(RetentionTestCase):
    def test_acked_snapshot_older_than_14_days_is_deleted(self):
        ev = self.add_event(age_days=15)
        summary = apply_retention(now=NOW)
        self.assertEqual(summary["snapshots_deleted"], 1)
        self.assertFalse(ev.snapshot.exists())
        self.assertIsNotNone(get_event(ev.event_id)["snapshot_deleted_ts"])

    def test_acked_video_older_than_14_days_is_deleted(self):
        ev = self.add_event(age_days=15, video=True)
        summary = apply_retention(now=NOW)
        self.assertEqual(summary["videos_deleted"], 1)
        self.assertFalse(ev.clip.exists())
        self.assertIsNotNone(get_event(ev.event_id)["video_deleted_ts"])

    def test_unacked_snapshot_older_than_14_days_is_preserved(self):
        ev = self.add_event(age_days=60, snapshot_acked=False)
        summary = apply_retention(now=NOW)
        self.assertEqual(summary["snapshots_deleted"], 0)
        self.assertTrue(ev.snapshot.exists())
        self.assertIsNone(get_event(ev.event_id)["snapshot_deleted_ts"])

    def test_unacked_video_older_than_14_days_is_preserved(self):
        ev = self.add_event(age_days=60, video=True, video_acked=False)
        summary = apply_retention(now=NOW)
        self.assertEqual(summary["videos_deleted"], 0)
        self.assertTrue(ev.clip.exists())
        self.assertFalse(ev.snapshot.exists())  # the snapshot had its own ACK

    def test_acked_media_younger_than_14_days_is_preserved(self):
        ev = self.add_event(age_days=13.9, video=True)
        summary = apply_retention(now=NOW)
        self.assertEqual((summary["snapshots_deleted"], summary["videos_deleted"]), (0, 0))
        self.assertTrue(ev.snapshot.exists())
        self.assertTrue(ev.clip.exists())

    def test_exactly_14_days_old_is_deleted(self):
        ev = self.add_event(age_days=14)
        apply_retention(now=NOW)
        self.assertFalse(ev.snapshot.exists())

    def test_recent_ack_of_backdated_recording_is_preserved(self):
        # Recorded "20 days ago" by a clock that was wrong before the first
        # time sync, ACKed 2 days ago: the ACK time proves it is young.
        ev = self.add_event(age_days=20, ack_age_days=2)
        apply_retention(now=NOW)
        self.assertTrue(ev.snapshot.exists())

    def test_future_dated_recording_is_preserved(self):
        ev = self.add_event(age_days=-30, ack_age_days=20)  # clock ran fast when recorded
        apply_retention(now=NOW)
        self.assertTrue(ev.snapshot.exists())

    def test_snapshot_ack_does_not_authorize_video_deletion(self):
        ev = self.add_event(age_days=30, video=True, video_acked=False)
        apply_retention(now=NOW)
        self.assertFalse(ev.snapshot.exists())
        self.assertTrue(ev.clip.exists())
        self.assertIn("no Base Station ACK for this video", purge_video_after_ack(ev.event_id, now=NOW))

    def test_video_ack_does_not_authorize_snapshot_deletion(self):
        ev = self.add_event(age_days=30, snapshot_acked=False, video=True)
        apply_retention(now=NOW)
        self.assertTrue(ev.snapshot.exists())
        self.assertFalse(ev.clip.exists())
        self.assertIn("no Base Station ACK for this snapshot", purge_snapshot_after_ack(ev.event_id, now=NOW))

    def test_historic_sent_rows_without_ack_remain_protected(self):
        # 'sent' long ago (NullBaseStationClient era or pre-v4), never ACKed.
        ev = self.add_event(age_days=90, snapshot_acked=False, video=True, video_acked=False)
        self.sql(
            "UPDATE events SET sent_ts = ?, video_sent_ts = ? WHERE event_id = ?",
            (NOW - 89 * DAY,) * 2 + (ev.event_id,),
        )
        for _ in range(2):
            apply_retention(now=NOW)
            self.fake_disk(base_used=990)  # and even under storage pressure
            apply_storage_pressure_cleanup(now=NOW)
        self.assertTrue(ev.snapshot.exists())
        self.assertTrue(ev.clip.exists())
        row = get_event(ev.event_id)
        self.assertIsNone(row["snapshot_deleted_ts"])
        self.assertIsNone(row["video_deleted_ts"])

    def test_status_must_permit_deletion(self):
        # An ACK alone is not enough if the row was reset for a re-send.
        pending = self.add_event("pending", age_days=30, status="pending")
        failed = self.add_event("failed", age_days=30, status="failed")
        apply_retention(now=NOW)
        self.assertTrue(pending.snapshot.exists())
        self.assertTrue(failed.snapshot.exists())

    def test_repeated_cleanup_is_idempotent(self):
        self.add_event("a", age_days=20, video=True)
        self.add_event("b", age_days=20, snapshot_file=False)
        first = apply_retention(now=NOW)
        second = apply_retention(now=NOW)
        third = apply_retention(now=NOW)
        self.assertEqual(
            (first["snapshots_deleted"], first["snapshots_missing"], first["videos_deleted"]), (1, 1, 1)
        )
        for later in (second, third):
            self.assertEqual(
                (
                    later["snapshots_deleted"],
                    later["videos_deleted"],
                    later["snapshots_missing"],
                    later["error_count"],
                ),
                (0, 0, 0, 0),
            )

    def test_empty_database(self):
        summary = apply_retention(now=NOW)
        self.assertTrue(summary["enabled"])
        self.assertEqual(
            (summary["snapshots_deleted"], summary["videos_deleted"], summary["errors"]), (0, 0, [])
        )

    def test_eligibility_queries(self):
        old = self.add_event("old", age_days=20, video=True)
        self.add_event("young", age_days=2, video=True)
        self.add_event("unacked", age_days=20, snapshot_acked=False, video=True, video_acked=False)
        self.assertEqual(
            [r["event_id"] for r in get_sent_snapshots_ready_for_delete(now=NOW)], [old.event_id]
        )
        self.assertEqual([r["event_id"] for r in get_sent_videos_ready_for_delete(now=NOW)], [old.event_id])

    def test_per_event_purge_applies_retention_rules(self):
        young = self.add_event("young", age_days=3)
        old = self.add_event("old", age_days=30)
        self.assertIn(
            "recorded less than retention_days ago", purge_snapshot_after_ack(young.event_id, now=NOW)
        )
        self.assertIsNone(purge_snapshot_after_ack(old.event_id, now=NOW))
        self.assertIsNone(purge_snapshot_after_ack(old.event_id, now=NOW))  # already done: still success
        self.assertTrue(young.snapshot.exists())
        self.assertFalse(old.snapshot.exists())

    def test_disabled_policy_deletes_nothing(self):
        self.write_policy(POLICY.replace("enabled: true", "enabled: false"))
        ev = self.add_event(age_days=30)
        summary = apply_retention(now=NOW)
        self.assertFalse(summary["enabled"])
        self.assertTrue(ev.snapshot.exists())

    def test_invalid_policy_deletes_nothing(self):
        for bad in (
            POLICY.replace("target_percent: 79", "target_percent: 85"),
            POLICY.replace("retention_days: 14", "retention_days: 0"),
            POLICY.replace("trigger_percent: 82", "trigger_percent: true"),
            "storage_pressure: [unclosed\n",
        ):
            with self.subTest(policy=bad):
                self.write_policy(bad)
                ev = self.add_event(f"e{len(bad)}", age_days=30)
                for summary in (apply_retention(now=NOW), apply_storage_pressure_cleanup(now=NOW)):
                    self.assertFalse(summary["enabled"])
                    self.assertIn("invalid retention policy", summary["disabled_reason"])
                self.assertTrue(ev.snapshot.exists())


# ---------------------------------------------------------------------------
# What creates an ACK (transfer_main.py + base_station_client.py)
# ---------------------------------------------------------------------------


class _FakeClient:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def send_event(self, event, snapshot_bytes=None):
        if self.fail:
            raise BaseStationError("simulated failure")

    def send_video(self, event_id, video_bytes, timeout_sec=None):
        if self.fail:
            raise BaseStationError("simulated failure")


def _response(status_code: int) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status_code
    resp.url = "http://192.168.1.110:8000/api/events"
    return resp


class AckRecordingTests(RetentionTestCase):
    def pending_event(self, name: str = "p", *, video: bool = False) -> Event:
        ev = self.add_event(
            name, age_days=30, status="pending", snapshot_acked=False, video=video, video_acked=False
        )
        if video:
            self.sql("UPDATE events SET video_status = 'pending' WHERE event_id = ?", (ev.event_id,))
        return ev

    def send_images(self, client, **kwargs) -> None:
        transfer_main._send_pending_images(client, TransferState(), batch_size=10, **kwargs)

    def send_videos(self, client, **kwargs) -> None:
        transfer_main._send_pending_videos(client, TransferState(), batch_size=10, timeout_sec=5.0, **kwargs)

    def test_valid_ack_is_recorded(self):
        ev = self.pending_event(video=True)
        client = BaseStationClient("http://192.168.1.110:8000")
        before = time.time()
        with mock.patch("network.base_station_client.requests.post", return_value=_response(201)):
            self.send_images(client, record_ack=True)
            self.send_videos(client, record_ack=True)
        row = get_event(ev.event_id)
        self.assertEqual((row["status"], row["video_status"]), ("sent", "sent"))
        for column in ("snapshot_acked_ts", "video_acked_ts"):
            self.assertGreaterEqual(row[column], before, column)
        self.assertTrue(ev.snapshot.exists())  # delete-right-after-ACK is off by default
        # ACK + 14 days later, the daily run deletes it.
        apply_retention(now=time.time() + 15 * DAY)
        self.assertFalse(ev.snapshot.exists())
        self.assertFalse(ev.clip.exists())

    def test_configured_url_alone_creates_no_ack(self):
        ev = self.pending_event()
        with mock.patch(
            "network.base_station_client.requests.post", side_effect=requests.ConnectionError("unreachable")
        ):
            self.send_images(BaseStationClient("http://192.168.1.110:8000"), record_ack=True)
        row = get_event(ev.event_id)
        self.assertEqual(row["status"], "failed")
        self.assertIsNone(row["snapshot_acked_ts"])
        # Even a row marked 'sent' with a URL configured stays protected.
        self.sql("UPDATE events SET status = 'sent' WHERE event_id = ?", (ev.event_id,))
        apply_retention(now=NOW + 30 * DAY)
        self.assertTrue(ev.snapshot.exists())

    def test_null_client_success_creates_no_ack(self):
        ev = self.pending_event()
        self.transfer_config.write_text(
            'base_station:\n  url: ""\ntransfer:\n  video_upload_enabled: false\n  delete_local_files_after_ack: true\n',
            encoding="utf-8",
        )
        config_loader.clear_cache()
        with (
            mock.patch.object(transfer_main, "configure_root"),
            mock.patch.object(transfer_main, "init_db"),
            mock.patch.object(transfer_main, "migrate"),
            mock.patch("network.base_station_client.requests.post") as post,
        ):
            transfer_main.run(max_cycles=1)
        post.assert_not_called()
        row = get_event(ev.event_id)
        self.assertEqual(row["status"], "sent")  # the Null client "succeeded"...
        self.assertIsNone(row["snapshot_acked_ts"])  # ...but that is no ACK
        apply_retention(now=NOW + 30 * DAY)
        self.assertTrue(ev.snapshot.exists())

    def test_metadata_only_send_creates_no_snapshot_ack(self):
        ev = self.add_event(age_days=30, status="pending", snapshot_acked=False, snapshot_file=False)
        self.send_images(_FakeClient(), record_ack=True, delete_after_ack=True)
        row = get_event(ev.event_id)
        self.assertEqual(row["status"], "sent")
        self.assertIsNone(row["snapshot_acked_ts"])
        ev.snapshot.write_bytes(b"j" * FILE_BYTES)  # the image becomes readable again later
        apply_retention(now=NOW + 30 * DAY)
        self.assertTrue(ev.snapshot.exists())

    def test_failed_or_partial_transmission_creates_no_ack(self):
        client = BaseStationClient("http://192.168.1.110:8000")
        for i, (case, effect) in enumerate(
            (
                ("partial upload", requests.exceptions.ChunkedEncodingError("connection dropped mid-upload")),
                ("server error", _response(500)),
                ("redirect", _response(302)),
                ("not modified", _response(304)),
            )
        ):
            with self.subTest(case):
                ev = self.pending_event(f"p{i}", video=True)
                kwargs = (
                    {"side_effect": effect} if isinstance(effect, Exception) else {"return_value": effect}
                )
                with mock.patch("network.base_station_client.requests.post", **kwargs):
                    self.send_images(client, record_ack=True, delete_after_ack=True)
                    self.send_videos(client, record_ack=True, delete_after_ack=True)
                row = get_event(ev.event_id)
                self.assertEqual((row["status"], row["video_status"]), ("failed", "failed"))
                self.assertIsNone(row["snapshot_acked_ts"])
                self.assertIsNone(row["video_acked_ts"])
                self.assertTrue(ev.snapshot.exists())
                self.assertTrue(ev.clip.exists())

    def test_uploads_do_not_follow_redirects(self):
        with mock.patch("network.base_station_client.requests.post", return_value=_response(201)) as post:
            BaseStationClient("http://192.168.1.110:8000").send_video("e1", b"x")
        self.assertIs(post.call_args.kwargs["allow_redirects"], False)

    def test_ack_database_failure_prevents_deletion(self):
        ev = self.pending_event()
        with (
            mock.patch.object(
                transfer_main, "mark_sent", side_effect=sqlite3.OperationalError("disk I/O error")
            ),
            self.assertRaises(sqlite3.OperationalError),
        ):
            self.send_images(_FakeClient(), record_ack=True, delete_after_ack=True)
        self.assertTrue(ev.snapshot.exists())
        row = get_event(ev.event_id)
        self.assertEqual(row["status"], "pending")  # retried later; nothing recorded
        self.assertIsNone(row["snapshot_acked_ts"])
        apply_retention(now=NOW + 30 * DAY)
        self.assertTrue(ev.snapshot.exists())

    def test_immediate_delete_after_ack_still_works_when_enabled(self):
        ev = self.pending_event(video=True)
        self.send_images(_FakeClient(), delete_after_ack=True)
        self.send_videos(_FakeClient(), delete_after_ack=True)
        row = get_event(ev.event_id)
        self.assertFalse(ev.snapshot.exists())
        self.assertFalse(ev.clip.exists())
        self.assertIsNotNone(row["snapshot_acked_ts"])
        self.assertIsNotNone(row["snapshot_deleted_ts"])
        self.assertIsNotNone(row["video_deleted_ts"])

    def test_shipped_config_keeps_acked_media_for_retention(self):
        config_loader.clear_cache()
        shipped = config_loader.load(REPO_ROOT / "configs" / "transfer_config.yaml")
        self.assertIs(shipped["transfer"]["delete_local_files_after_ack"], False)


# ---------------------------------------------------------------------------
# Storage-pressure cleanup (82 % -> 79 %)
# ---------------------------------------------------------------------------


class StoragePressureTests(RetentionTestCase):
    def test_below_82_percent_does_not_trigger(self):
        self.add_event(age_days=30)
        disk = self.fake_disk(base_used=800)  # 81 %
        summary = apply_storage_pressure_cleanup(now=NOW)
        self.assertFalse(summary["triggered"])
        self.assertEqual(summary["snapshots_deleted"], 0)
        self.assertEqual(len(disk.calls), 1)

    def test_exactly_82_percent_triggers(self):
        ev = self.add_event(age_days=1)
        self.fake_disk(base_used=810)  # 810 + 10 = 82 %
        summary = apply_storage_pressure_cleanup(now=NOW)
        self.assertTrue(summary["triggered"])
        self.assertEqual(summary["usage_before_percent"], 82.0)
        self.assertFalse(ev.snapshot.exists())

    def test_above_82_percent_triggers(self):
        self.add_event(age_days=1)
        self.fake_disk(base_used=850)
        self.assertTrue(apply_storage_pressure_cleanup(now=NOW)["triggered"])

    def test_oldest_eligible_media_deleted_first(self):
        middle = self.add_event("middle", age_days=10)
        oldest = self.add_event("oldest", age_days=12, video=True)
        newest = self.add_event("newest", age_days=1)
        # 4 files x 10 bytes; 790 + 40 = 83 %; two deletions reach 81 %, three 80 %, four 79 %.
        self.fake_disk(base_used=790)
        self.assertEqual(
            [(r["event_id"], r["media_kind"]) for r in get_storage_pressure_candidates()],
            [
                (oldest.event_id, "snapshot"),
                (oldest.event_id, "video"),
                (middle.event_id, "snapshot"),
                (newest.event_id, "snapshot"),
            ],
        )
        summary = apply_storage_pressure_cleanup(now=NOW)
        self.assertEqual(summary["usage_after_percent"], 79.0)
        self.assertFalse(newest.snapshot.exists())  # all four were needed
        self.assertFalse(oldest.clip.exists())

    def test_cleanup_stops_at_79_percent(self):
        events = [self.add_event(f"e{i}", age_days=10 - i) for i in range(5)]  # e0 oldest
        self.fake_disk(base_used=770)  # 770 + 50 = 82 %; three deletions -> 79 %
        summary = apply_storage_pressure_cleanup(now=NOW)
        self.assertEqual(summary["snapshots_deleted"], 3)
        self.assertEqual(summary["usage_after_percent"], 79.0)
        self.assertTrue(summary["target_reached"])
        self.assertFalse(summary["critical"])
        self.assertEqual([e.snapshot.exists() for e in events], [False, False, False, True, True])

    def test_young_acked_media_may_be_deleted_under_pressure(self):
        ev = self.add_event(age_days=0.1)
        self.fake_disk(base_used=815)
        apply_storage_pressure_cleanup(now=NOW)
        self.assertFalse(ev.snapshot.exists())

    def test_unacked_media_preserved_under_pressure(self):
        unacked = self.add_event("unacked", age_days=90, snapshot_acked=False, video=True, video_acked=False)
        acked = self.add_event("acked", age_days=1)
        self.fake_disk(base_used=900)
        summary = apply_storage_pressure_cleanup(now=NOW)
        self.assertTrue(unacked.snapshot.exists())
        self.assertTrue(unacked.clip.exists())
        self.assertFalse(acked.snapshot.exists())
        self.assertTrue(summary["critical"])

    def test_critical_when_no_eligible_media_remains(self):
        self.add_event(age_days=90, snapshot_acked=False)
        self.fake_disk(base_used=900)
        with self.assertLogs(LOGGER, "ERROR") as logs:
            summary = apply_storage_pressure_cleanup(now=NOW)
        self.assertTrue(summary["critical"])
        self.assertFalse(summary["target_reached"])
        self.assertEqual(summary["eligible_remaining"], 0)
        self.assertTrue(summary["blocked_by_unacknowledged"])
        self.assertEqual(summary["unacknowledged"]["unacked_snapshots"], 1)
        self.assertTrue(any("CRITICAL storage pressure" in line for line in logs.output))

    def test_measures_the_media_filesystem(self):
        self.add_event(age_days=1)
        disk = self.fake_disk(base_used=100)
        summary = apply_storage_pressure_cleanup(now=NOW)
        self.assertEqual(disk.calls, [self.snapshot_dir])  # one filesystem holds both media dirs
        self.assertEqual(summary["volumes"][0]["media"], ["snapshot", "video"])
        self.assertNotEqual(disk.calls[0], Path(Path.cwd().anchor))  # not the OS root

    def test_media_directories_on_separate_filesystems_are_measured_separately(self):
        devices = {str(self.snapshot_dir): 1, str(self.clip_dir): 2}
        real_stat = os.stat

        def fake_stat(path, *args, **kwargs):
            result = real_stat(path, *args, **kwargs)
            device = devices.get(str(path))
            return os.stat_result((*result[:2], device, *result[3:])) if device else result

        disk = self.fake_disk(base_used=100)
        with mock.patch.object(retention_manager.os, "stat", side_effect=fake_stat):
            summary = apply_storage_pressure_cleanup(now=NOW)
        self.assertEqual(sorted(map(str, disk.calls)), sorted(devices))
        self.assertEqual(sorted(v["media"][0] for v in summary["volumes"]), ["snapshot", "video"])

    def test_usage_rechecked_after_every_deletion(self):
        for i in range(3):
            self.add_event(f"e{i}", age_days=5 + i)
        disk = self.fake_disk(base_used=800)  # 83 %; needs all three deletions
        summary = apply_storage_pressure_cleanup(now=NOW)
        self.assertEqual(summary["snapshots_deleted"], 3)
        self.assertEqual(len(disk.calls), 1 + 3)
        self.assertEqual(summary["volumes"][0]["bytes_released"], 3 * FILE_BYTES)

    def test_reports_failure_when_target_not_reached(self):
        self.add_event(age_days=5)
        self.fake_disk(base_used=850)  # one 10-byte file cannot bring 86 % down to 79 %
        summary = apply_storage_pressure_cleanup(now=NOW)
        self.assertEqual(summary["snapshots_deleted"], 1)
        self.assertFalse(summary["target_reached"])
        self.assertTrue(summary["critical"])
        self.assertEqual(summary["usage_after_percent"], 85.0)

    def test_space_not_released_is_reported(self):
        self.add_event("a", age_days=5)
        self.add_event("b", age_days=4)
        self.fake_disk(base_used=820, releases_space=False)
        summary = apply_storage_pressure_cleanup(now=NOW)
        self.assertTrue(summary["critical"])
        self.assertTrue(any("released only 0" in e for e in summary["errors"]), summary["errors"])


# ---------------------------------------------------------------------------
# Filesystem and database safety
# ---------------------------------------------------------------------------


class SafetyTests(RetentionTestCase):
    def test_missing_eligible_acked_file_is_recorded(self):
        ev = self.add_event(age_days=30, snapshot_file=False)
        with self.assertLogs(LOGGER, "WARNING") as logs:
            summary = apply_retention(now=NOW)
        self.assertEqual((summary["snapshots_missing"], summary["snapshots_deleted"]), (1, 0))
        self.assertIn("already missing", logs.output[0])
        self.assertIsNotNone(get_event(ev.event_id)["snapshot_deleted_ts"])

    def test_missing_unacked_file_is_not_recorded(self):
        ev = self.add_event(age_days=30, snapshot_acked=False, snapshot_file=False)
        self.fake_disk(base_used=900)
        apply_retention(now=NOW)
        apply_storage_pressure_cleanup(now=NOW)
        self.assertIsNone(get_event(ev.event_id)["snapshot_deleted_ts"])

    def test_failed_file_deletion_stays_retryable(self):
        ev = self.add_event(age_days=30)
        with mock.patch.object(retention_manager, "delete_local_file", return_value=False):
            summary = apply_retention(now=NOW)
        self.assertEqual((summary["failed"], summary["snapshots_deleted"]), (1, 0))
        self.assertIn("could not delete", summary["errors"][0])
        self.assertIsNone(get_event(ev.event_id)["snapshot_deleted_ts"])
        self.assertEqual(apply_retention(now=NOW)["snapshots_deleted"], 1)
        self.assertFalse(ev.snapshot.exists())

    def test_database_update_failure_is_not_a_successful_deletion(self):
        ev = self.add_event(age_days=30)
        with (
            mock.patch.object(
                retention_manager,
                "mark_snapshot_deleted",
                side_effect=sqlite3.OperationalError("database is locked"),
            ),
            self.assertLogs(LOGGER, "ERROR"),
        ):
            summary = apply_retention(now=NOW)
        self.assertEqual((summary["snapshots_deleted"], summary["failed"]), (0, 1))
        self.assertIn("database not updated", summary["errors"][0])
        self.assertIsNone(get_event(ev.event_id)["snapshot_deleted_ts"])
        retry = apply_retention(now=NOW)  # file is gone now; this run records it
        self.assertEqual(retry["snapshots_missing"], 1)
        self.assertIsNotNone(get_event(ev.event_id)["snapshot_deleted_ts"])

    def test_database_listing_failure_is_reported(self):
        self.add_event(age_days=30, video=True)
        with (
            mock.patch.object(
                retention_manager,
                "get_acked_snapshots_not_deleted",
                side_effect=sqlite3.OperationalError("locked"),
            ),
            self.assertLogs(LOGGER, "ERROR"),
        ):
            summary = apply_retention(now=NOW)
        self.assertEqual((summary["snapshots_deleted"], summary["videos_deleted"]), (0, 1))
        self.assertIn("cannot list ACKed media", summary["errors"][0])

    def test_unexpected_errors_are_not_swallowed(self):
        self.add_event(age_days=30)
        with (
            mock.patch.object(retention_manager, "_unsafe_reason", side_effect=KeyError("bug")),
            self.assertRaises(KeyError),
        ):
            apply_retention(now=NOW)

    def assert_refused(self, target: Path, *, write: bool = True) -> None:
        if write:
            target.write_bytes(b"precious")
        ev = self.add_event(f"r{abs(hash(str(target)))}", age_days=30, snapshot_path=target)
        summary = apply_retention(now=NOW)
        self.assertTrue(
            any(ev.event_id in e and "refused" in e for e in summary["errors"]), summary["errors"]
        )
        self.assertIsNone(get_event(ev.event_id)["snapshot_deleted_ts"])
        if write:
            self.assertTrue(target.exists())

    def test_refuses_non_media_files(self):
        for name in (
            "zovive.sqlite3.bak",
            "edge.log",
            "model.hef",
            "model.onnx",
            "node_config.yaml",
            "manifest.json",
            "edge_main.py",
            ".env",
        ):
            with self.subTest(name=name):
                self.assert_refused(self.root / name)

    def test_refuses_wrong_extension_inside_media_directory(self):
        self.assert_refused(self.snapshot_dir / "camera.yaml")

    def test_refuses_clip_recorded_as_snapshot(self):
        self.assert_refused(self.clip_dir / "a.mp4")

    def test_refuses_path_traversal(self):
        (self.root / "outside.jpg").write_bytes(b"precious")
        self.assert_refused(self.snapshot_dir / ".." / "outside.jpg", write=False)
        self.assertTrue((self.root / "outside.jpg").exists())

    def test_refuses_relative_path(self):
        self.assert_refused(Path("snapshots") / "a.jpg", write=False)

    def test_refuses_directory_named_like_media(self):
        directory = self.snapshot_dir / "folder.jpg"
        directory.mkdir()
        self.assert_refused(directory, write=False)
        self.assertTrue(directory.is_dir())

    def test_refuses_symlinked_file(self):
        target = self.root / "outside.jpg"
        target.write_bytes(b"precious")
        link = self.snapshot_dir / "link.jpg"
        try:
            os.symlink(target, link)
        except (OSError, NotImplementedError):
            self.skipTest("this account cannot create symlinks")
        self.assert_refused(link, write=False)
        self.assertTrue(target.exists())
        self.assertTrue(link.is_symlink())

    def test_refuses_symlinked_directory_escape(self):
        outside = self.root / "outside_dir"
        outside.mkdir()
        (outside / "a.jpg").write_bytes(b"precious")
        try:
            os.symlink(outside, self.snapshot_dir / "sub", target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("this account cannot create symlinks")
        self.assert_refused(self.snapshot_dir / "sub" / "a.jpg", write=False)
        self.assertTrue((outside / "a.jpg").exists())

    def test_real_deployment_data_untouched(self):
        def snapshot_of(path: Path):
            if path.is_dir():
                return sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in path.iterdir())
            return (path.stat().st_size, path.stat().st_mtime_ns) if path.exists() else None

        real = {
            name: REAL_PATHS[name]
            for name in ("DB_PATH", "SNAPSHOT_DIR", "CLIP_DIR", "RETENTION_STATUS_PATH")
        }
        before = {name: snapshot_of(Path(p)) for name, p in real.items()}
        self.add_event("a", age_days=30, video=True)
        self.add_event("b", age_days=1)
        self.fake_disk(base_used=900)
        apply_retention(now=NOW)
        apply_storage_pressure_cleanup(now=NOW)
        self.assertEqual(run_job("daily", lock_wait_sec=0), EXIT_OK)
        self.assertEqual({name: snapshot_of(Path(p)) for name, p in real.items()}, before)


# ---------------------------------------------------------------------------
# Shared files: one event's ACK never deletes a file another event still needs
# ---------------------------------------------------------------------------


class SharedFileTests(RetentionTestCase):
    def shared_pair(self) -> tuple[Path, Event, Event]:
        """An ACKed, eligible event and a pending, unACKed one storing the same snapshot path."""
        shared = self.snapshot_dir / "shared.jpg"
        shared.write_bytes(b"s" * FILE_BYTES)
        acked = self.add_event("acked", snapshot_path=shared)
        pending = self.add_event("pending", status="pending", snapshot_acked=False, snapshot_path=shared)
        return shared, acked, pending

    def assert_shared_file_kept(self, summary: dict, shared: Path, acked: Event, pending: Event) -> None:
        self.assertTrue(shared.exists())
        for ev in (acked, pending):
            self.assertIsNone(get_event(ev.event_id)["snapshot_deleted_ts"])
        self.assertEqual((summary["snapshots_deleted"], summary["refused"]), (0, 1))
        self.assertIn(f"still referenced by {pending.event_id} (snapshot)", summary["errors"][0])

    def test_acked_event_cannot_delete_snapshot_shared_with_pending_event(self):
        shared, acked, pending = self.shared_pair()
        summary = apply_retention(now=NOW)
        self.assert_shared_file_kept(summary, shared, acked, pending)

    def test_storage_pressure_cannot_delete_snapshot_shared_with_pending_event(self):
        shared, acked, pending = self.shared_pair()
        self.fake_disk(base_used=900)
        summary = apply_storage_pressure_cleanup(now=NOW)
        self.assert_shared_file_kept(summary, shared, acked, pending)
        self.assertTrue(summary["critical"])

    def test_failed_reference_lookup_prevents_deletion(self):
        ev = self.add_event("e1")  # ACKed, eligible, unshared
        for run in ("daily", "pressure"):
            with (
                self.subTest(run=run),
                mock.patch.object(
                    retention_manager,
                    "get_active_references_by_file_name",
                    side_effect=sqlite3.OperationalError("database is locked"),
                ),
                self.assertLogs(LOGGER, "ERROR"),
            ):
                if run == "daily":
                    summary = apply_retention(now=NOW)
                else:
                    self.fake_disk(base_used=900)
                    summary = apply_storage_pressure_cleanup(now=NOW)
                self.assertTrue(ev.snapshot.exists())
                self.assertIsNone(get_event(ev.event_id)["snapshot_deleted_ts"])
                self.assertEqual((summary["failed"], summary["snapshots_deleted"]), (1, 0))
                self.assertIn("cannot check other references", summary["errors"][0])

    def test_unshared_eligible_snapshot_is_still_deleted(self):
        # A file with the same name in another directory is a different file.
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        for run in ("daily", "pressure"):
            with self.subTest(run=run):
                ev = self.add_event(run)
                decoy = elsewhere / ev.snapshot.name
                decoy.write_bytes(b"d" * FILE_BYTES)
                self.add_event(f"{run}-decoy", status="pending", snapshot_acked=False, snapshot_path=decoy)
                if run == "daily":
                    summary = apply_retention(now=NOW)
                else:
                    self.fake_disk(base_used=900)
                    summary = apply_storage_pressure_cleanup(now=NOW)
                self.assertFalse(ev.snapshot.exists())
                self.assertIsNotNone(get_event(ev.event_id)["snapshot_deleted_ts"])
                self.assertEqual((summary["snapshots_deleted"], summary["refused"]), (1, 0))
                self.assertTrue(decoy.exists())

    def test_relative_reference_with_the_same_file_name_keeps_the_file(self):
        # Where a relative path points depends on the writer's working
        # directory, so it cannot be ruled out: keep the file, do not guess.
        ev = self.add_event("e1")
        other = self.add_event(
            "other", status="pending", snapshot_acked=False, snapshot_path=Path("snapshots") / "e1.jpg"
        )
        summary = apply_retention(now=NOW)
        self.assertTrue(ev.snapshot.exists())
        self.assertEqual((summary["snapshots_deleted"], summary["refused"]), (0, 1))
        self.assertIn(f"{other.event_id} (snapshot, relative path", summary["errors"][0])


# ---------------------------------------------------------------------------
# Scheduling, locking and reporting
# ---------------------------------------------------------------------------


def _unit(name: str) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str  # systemd keys are case-sensitive
    parser.read(REPO_ROOT / "systemd" / name, encoding="utf-8")
    return parser


class SchedulingTests(RetentionTestCase):
    def test_daily_and_pressure_timers_are_configured(self):
        daily = _unit("zovive-retention.timer")
        self.assertEqual(daily["Timer"]["OnCalendar"], "*-*-* 03:30:00")  # once a day
        self.assertEqual(daily["Timer"]["Persistent"], "true")  # catches up after downtime
        self.assertEqual(daily["Timer"]["Unit"], "zovive-retention@daily.service")
        self.assertEqual(daily["Install"]["WantedBy"], "timers.target")

        pressure = _unit("zovive-storage-pressure.timer")
        self.assertEqual(pressure["Timer"]["OnUnitActiveSec"], "5min")
        self.assertEqual(pressure["Timer"]["OnBootSec"], "2min")
        self.assertEqual(pressure["Timer"]["Unit"], "zovive-retention@pressure.service")

        service = _unit("zovive-retention@.service")["Service"]
        self.assertEqual(service["Type"], "oneshot")
        self.assertTrue(service["ExecStart"].endswith("-m storage.retention_manager %i"))
        self.assertEqual(service["RestartPreventExitStatus"], str(EXIT_CONFIG))
        self.assertIn(
            "zovive-retention.timer zovive-storage-pressure.timer",
            (REPO_ROOT / "provisioning" / "setup_pi.sh").read_text(encoding="utf-8"),
        )

    def test_job_runs_and_writes_distinct_summaries(self):
        self.add_event("old", age_days=20)
        self.add_event("young", age_days=1)
        self.fake_disk(base_used=900)
        with self.assertLogs(LOGGER, "INFO") as logs:
            self.assertEqual(run_job("daily", lock_wait_sec=0), EXIT_OK)
            self.assertEqual(run_job("pressure", lock_wait_sec=0), EXIT_OK)
        status = health_main.read_status_file(paths.RETENTION_STATUS_PATH)
        daily, pressure = status["last_scheduled_retention"], status["last_storage_pressure"]
        self.assertEqual((daily["run"], pressure["run"]), ("scheduled_retention", "storage_pressure"))
        self.assertEqual((daily["snapshots_deleted"], pressure["snapshots_deleted"]), (1, 1))
        self.assertNotIn("triggered", daily)
        self.assertIn("usage_before_percent", pressure)
        messages = "\n".join(logs.output)
        self.assertIn("scheduled retention finished", messages)
        self.assertIn("storage pressure finished", messages)

    def test_overlapping_runs_are_prevented(self):
        ev = self.add_event(age_days=30)
        held = threading.Event()
        release = threading.Event()

        def hold_lock():
            with retention_manager._run_lock(paths.RETENTION_LOCK_PATH, 0) as acquired:
                self.assertTrue(acquired)
                held.set()
                release.wait(10)

        holder = threading.Thread(target=hold_lock)
        holder.start()
        try:
            self.assertTrue(held.wait(5))
            self.assertEqual(run_job("daily", lock_wait_sec=0), EXIT_TEMPFAIL)
            self.assertTrue(ev.snapshot.exists())
        finally:
            release.set()
            holder.join(5)
        self.assertEqual(run_job("daily", lock_wait_sec=0), EXIT_OK)
        self.assertFalse(ev.snapshot.exists())

    def test_runs_safely_after_restart(self):
        # A crash left the lock file behind and stopped between deleting a
        # file and recording it; the next run must neither block nor redo.
        paths.RETENTION_LOCK_PATH.write_bytes(b"")
        ev = self.add_event(age_days=30)
        with mock.patch.object(
            retention_manager, "mark_snapshot_deleted", side_effect=sqlite3.OperationalError("power lost")
        ):
            run_job("daily", lock_wait_sec=0)
        self.assertFalse(ev.snapshot.exists())
        self.assertIsNone(get_event(ev.event_id)["snapshot_deleted_ts"])
        self.assertEqual(run_job("daily", lock_wait_sec=0), EXIT_OK)
        self.assertIsNotNone(get_event(ev.event_id)["snapshot_deleted_ts"])

    def test_database_unavailable_exits_safely(self):
        ev = self.add_event(age_days=30)
        with (
            mock.patch.object(
                retention_manager, "init_db", side_effect=sqlite3.OperationalError("unable to open")
            ),
            self.assertLogs(LOGGER, "ERROR"),
        ):
            self.assertEqual(run_job("daily", lock_wait_sec=0), EXIT_TEMPFAIL)
        self.assertTrue(ev.snapshot.exists())

    def test_invalid_policy_exit_code(self):
        self.write_policy(POLICY.replace("target_percent: 79", "target_percent: 90"))
        self.assertEqual(run_job("pressure", lock_wait_sec=0), EXIT_CONFIG)

    def test_overdue_unacknowledged_files_are_counted(self):
        self.add_event("overdue", age_days=20, snapshot_acked=False, video=True, video_acked=False)
        self.add_event("recent", age_days=2, snapshot_acked=False)
        self.add_event("acked", age_days=30)
        self.add_event("failed", age_days=1, status="failed", snapshot_acked=False)
        backlog = get_unacknowledged_backlog(now=NOW)
        self.assertEqual((backlog["unacked_snapshots"], backlog["unacked_videos"]), (3, 1))
        self.assertEqual(
            (backlog["overdue_snapshots"], backlog["overdue_videos"], backlog["overdue_total"]), (1, 1, 2)
        )
        self.assertEqual(backlog["failed_snapshots"], 1)
        self.assertAlmostEqual(backlog["oldest_unacked_age_days"], 20, places=1)
        summary = apply_retention(now=NOW)
        self.assertEqual(summary["unacknowledged"]["overdue_total"], 2)

    def test_critical_pressure_reaches_the_heartbeat(self):
        self.add_event(age_days=40, snapshot_acked=False)
        self.fake_disk(base_used=950)
        self.assertEqual(run_job("pressure", lock_wait_sec=0), EXIT_OK)
        self.patch(paths, "EDGE_STATUS_PATH", self.root / "edge_status.json")
        with (
            mock.patch.object(
                health_main, "_pi_throttle_status", return_value={"raw": None, "flags": None, "error": "x"}
            ),
            mock.patch.object(health_main, "_disk_usage", return_value=None),
            mock.patch("network.heartbeat.get_identity", return_value={"node_id": "Node-001"}),
        ):
            heartbeat = health_main.collect_heartbeat("cam01", now=NOW)
        pressure = heartbeat["retention"]["last_storage_pressure"]
        self.assertTrue(pressure["critical"])
        self.assertTrue(pressure["blocked_by_unacknowledged"])
        self.assertEqual(heartbeat["retention"]["unacknowledged"]["overdue_snapshots"], 1)


if __name__ == "__main__":
    unittest.main()
