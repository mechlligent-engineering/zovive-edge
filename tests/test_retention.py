"""storage/retention_manager.py: one event's ACK never authorizes deleting a
file another event still references (shared paths), and storage pressure
never overrides status or ACK protection.

Runs on the isolated harness from tests/test_retention_manager.py: its own
temporary database, media directories, policy, status and lock files, with a
guard that fails the test if any of them points at deployment data. Disk
usage comes from a fake filesystem; the Base Station is never contacted.
"""

from __future__ import annotations

import errno
import os
import sqlite3
from pathlib import Path
from unittest import mock

from db.outbox import get_event
from storage import retention_manager
from storage.retention_manager import (
    apply_retention,
    apply_storage_pressure_cleanup,
    purge_snapshot_after_ack,
    purge_video_after_ack,
)
from tests.test_retention_manager import FILE_BYTES, LOGGER, NOW, RetentionTestCase

# The *other* event sharing the file: (label, add_event keyword arguments).
OTHER_STATES = (
    ("pending, no ACK", {"status": "pending", "snapshot_acked": False}),
    ("failed, no ACK", {"status": "failed", "snapshot_acked": False}),
    ("failed, with ACK", {"status": "failed"}),
    ("sent, no ACK", {"snapshot_acked": False}),
    ("sent, ACKed, old", {}),
    ("sent, ACKed, young", {"age_days": 1}),
)


class SharedPathTestCase(RetentionTestCase):
    def media_file(self, directory: Path, name: str) -> Path:
        path = directory / name
        path.write_bytes(b"s" * FILE_BYTES)
        return path

    _POINT_AT = {
        "snapshot_path": "UPDATE events SET snapshot_path = ? WHERE event_id = ?",
        "video_path": "UPDATE events SET video_path = ? WHERE event_id = ?",
    }

    def point_at(self, event_id: str, column: str, raw_path: str) -> None:
        """Store an exact path string (pathlib would normalise "." away)."""
        self.sql(self._POINT_AT[column], (raw_path, event_id))

    def run_cleanup(self, run: str) -> dict:
        if run == "daily":
            return apply_retention(now=NOW)
        self.fake_disk(base_used=900)  # 90 %+: delete everything eligible
        return apply_storage_pressure_cleanup(now=NOW)

    def assert_kept(self, path: Path, *event_ids: str, column: str = "snapshot_deleted_ts") -> None:
        self.assertTrue(path.exists(), f"{path.name} must be kept")
        for event_id in event_ids:
            self.assertIsNone(get_event(event_id)[column], f"{event_id}: no deletion may be recorded")

    def assert_reported(self, summary: dict, sharer_id: str, kind: str = "snapshot") -> None:
        expected = f"still referenced by {sharer_id} ({kind})"
        self.assertTrue(
            any(expected in e for e in summary["errors"]), f"{expected!r} not in {summary['errors']}"
        )


class SharedSnapshotTests(SharedPathTestCase):
    def check_every_other_state(self, run: str) -> None:
        cases = []
        for i, (label, other_state) in enumerate(OTHER_STATES):
            shared = self.media_file(self.snapshot_dir, f"shared{i}.jpg")
            acked = self.add_event(f"acked{i}", snapshot_path=shared)  # eligible on its own
            other = self.add_event(f"other{i}", snapshot_path=shared, **other_state)
            cases.append((label, shared, acked, other))
        control = self.add_event("control")  # unshared, ACKed, 20 days old

        summary = self.run_cleanup(run)

        for label, shared, acked, other in cases:
            with self.subTest(run=run, other=label):
                self.assert_kept(shared, acked.event_id, other.event_id)
                self.assert_reported(summary, other.event_id)
        self.assertFalse(control.snapshot.exists(), "an unshared ACKed file is still deleted")
        self.assertEqual(summary["snapshots_deleted"], 1)
        self.assertGreaterEqual(summary["refused"], len(cases))

    def test_daily_retention_keeps_shared_file_whatever_the_other_event_state(self):
        self.check_every_other_state("daily")

    def test_storage_pressure_keeps_shared_file_whatever_the_other_event_state(self):
        self.check_every_other_state("pressure")

    def test_reference_through_the_other_events_clip_column_also_blocks(self):
        shared = self.media_file(self.snapshot_dir, "shared.jpg")
        acked = self.add_event("acked", snapshot_path=shared)
        other = self.add_event(
            "other",
            snapshot_acked=False,
            video=True,
            video_status="pending",
            video_acked=False,
            video_file=False,
        )
        self.point_at(other.event_id, "video_path", str(shared))
        for run in ("daily", "pressure"):
            with self.subTest(run=run):
                summary = self.run_cleanup(run)
                self.assert_kept(shared, acked.event_id)
                self.assert_reported(summary, other.event_id, "video")

    def test_other_row_with_its_deletion_recorded_does_not_block(self):
        shared = self.media_file(self.snapshot_dir, "shared.jpg")
        gone = self.add_event("gone", snapshot_path=shared, status="pending", snapshot_acked=False)
        recorded_ts = NOW - 100.0
        self.sql("UPDATE events SET snapshot_deleted_ts = ? WHERE event_id = ?", (recorded_ts, gone.event_id))
        acked = self.add_event("acked", snapshot_path=shared)

        summary = apply_retention(now=NOW)

        self.assertFalse(shared.exists())
        self.assertEqual(summary["snapshots_deleted"], 1)
        self.assertIsNotNone(get_event(acked.event_id)["snapshot_deleted_ts"])
        self.assertEqual(
            get_event(gone.event_id)["snapshot_deleted_ts"], recorded_ts, "audit record unchanged"
        )

    def test_same_file_spelled_differently_is_detected(self):
        spellings = {
            "parent hop": lambda p: f"{p.parent}{os.sep}..{os.sep}{p.parent.name}{os.sep}{p.name}",
            "dot segment": lambda p: f"{p.parent}{os.sep}.{os.sep}{p.name}",
        }
        if os.name == "nt":  # case-insensitive filesystem
            spellings["upper case"] = lambda p: str(p).upper()
        cases = []
        for i, (label, spell) in enumerate(spellings.items()):
            shared = self.media_file(self.snapshot_dir, f"spelled{i}.jpg")
            acked = self.add_event(f"acked{i}", snapshot_path=shared)
            other = self.add_event(f"other{i}", status="pending", snapshot_acked=False, snapshot_file=False)
            self.point_at(other.event_id, "snapshot_path", spell(shared))
            cases.append((label, shared, acked, other))

        summary = apply_retention(now=NOW)

        for label, shared, acked, other in cases:
            with self.subTest(label):
                self.assert_kept(shared, acked.event_id)
                self.assert_reported(summary, other.event_id)

    def test_same_file_name_in_another_directory_does_not_block(self):
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        unrelated = self.media_file(elsewhere, "e1.jpg")
        self.add_event("other", snapshot_path=unrelated, status="pending", snapshot_acked=False)
        acked = self.add_event("e1")  # snapshots/e1.jpg: same name, different file

        apply_retention(now=NOW)

        self.assertFalse(acked.snapshot.exists())
        self.assertTrue(unrelated.exists())

    def test_per_event_purge_also_refuses(self):
        shared = self.media_file(self.snapshot_dir, "shared.jpg")
        acked = self.add_event("acked", snapshot_path=shared)
        other = self.add_event("other", snapshot_path=shared, status="pending", snapshot_acked=False)

        reason = purge_snapshot_after_ack(acked.event_id, now=NOW)

        self.assertIn(f"still referenced by {other.event_id} (snapshot)", reason)
        self.assert_kept(shared, acked.event_id, other.event_id)

    def test_already_missing_shared_file_is_recorded_only_for_the_acked_row(self):
        # Nothing is deleted; the ACKed row truthfully records its file as gone.
        missing = self.snapshot_dir / "missing.jpg"
        acked = self.add_event("acked", snapshot_path=missing)
        other = self.add_event("other", snapshot_path=missing, status="pending", snapshot_acked=False)

        summary = apply_retention(now=NOW)

        self.assertEqual(summary["snapshots_missing"], 1)
        self.assertIsNotNone(get_event(acked.event_id)["snapshot_deleted_ts"])
        self.assertIsNone(get_event(other.event_id)["snapshot_deleted_ts"])


class SharedClipTests(SharedPathTestCase):
    def test_shared_clip_kept_by_daily_pressure_and_per_event_purge(self):
        shared = self.media_file(self.clip_dir, "shared.mp4")
        acked = self.add_event("acked", video=True, video_file=False)
        other = self.add_event(
            "other",
            status="pending",
            snapshot_acked=False,
            video=True,
            video_status="pending",
            video_acked=False,
            video_file=False,
        )
        for event in (acked, other):
            self.point_at(event.event_id, "video_path", str(shared))

        for run in ("daily", "pressure"):
            with self.subTest(run=run):
                summary = self.run_cleanup(run)
                self.assert_kept(shared, acked.event_id, other.event_id, column="video_deleted_ts")
                self.assert_reported(summary, other.event_id, "video")
        reason = purge_video_after_ack(acked.event_id, now=NOW)
        self.assertIn(f"still referenced by {other.event_id} (video)", reason)
        self.assertTrue(shared.exists())
        self.assertFalse(acked.snapshot.exists(), "its own unshared ACKed snapshot is still deleted")


class ReferenceCheckFailureTests(SharedPathTestCase):
    def test_database_error_keeps_the_file_and_is_retried(self):
        ev = self.add_event("e1")
        with (
            mock.patch.object(
                retention_manager,
                "get_active_references_by_file_name",
                side_effect=sqlite3.OperationalError("database is locked"),
            ),
            self.assertLogs(LOGGER, "ERROR"),
        ):
            summary = apply_retention(now=NOW)
        self.assertTrue(ev.snapshot.exists())
        self.assertEqual((summary["failed"], summary["snapshots_deleted"]), (1, 0))
        self.assertIn("cannot check other references", summary["errors"][0])
        self.assertIsNone(get_event(ev.event_id)["snapshot_deleted_ts"])
        self.assertEqual(apply_retention(now=NOW)["snapshots_deleted"], 1, "retried on the next run")

    def test_path_resolution_errors_keep_the_file(self):
        ev = self.add_event("e1")
        errors = (
            RuntimeError("Symlink loop"),
            OSError(errno.ENAMETOOLONG, "File name too long"),
            ValueError("embedded null byte"),
        )
        for exc in errors:
            with (
                self.subTest(type(exc).__name__),
                mock.patch.object(retention_manager, "_resolved", side_effect=exc),
                self.assertLogs(LOGGER, "ERROR"),
            ):
                summary = apply_retention(now=NOW)
                self.assertEqual(summary["failed"], 1)
                self.assert_kept(ev.snapshot, ev.event_id)

    def test_failure_under_pressure_keeps_the_file_and_counts_as_remaining(self):
        ev = self.add_event("e1")
        self.fake_disk(base_used=900)
        with (
            mock.patch.object(
                retention_manager,
                "get_active_references_by_file_name",
                side_effect=sqlite3.OperationalError("database is locked"),
            ),
            self.assertLogs(LOGGER, "ERROR"),
        ):
            summary = apply_storage_pressure_cleanup(now=NOW)
        self.assert_kept(ev.snapshot, ev.event_id)
        self.assertTrue(summary["critical"])
        self.assertEqual(summary["eligible_remaining"], 1, "still eligible: retried next run")


class PressureStatusTests(RetentionTestCase):
    """Storage pressure ignores age, never status or ACK."""

    def test_pending_or_failed_rows_with_an_ack_are_kept_under_pressure(self):
        kept = [
            self.add_event("snap_pending_acked", status="pending"),
            self.add_event("snap_failed_acked", status="failed"),
            self.add_event("video_pending_acked", snapshot_acked=False, video=True, video_status="pending"),
            self.add_event("video_failed_acked", snapshot_acked=False, video=True, video_status="failed"),
        ]
        control = self.add_event("control", age_days=1)
        self.fake_disk(base_used=900)

        summary = apply_storage_pressure_cleanup(now=NOW)

        self.assertFalse(control.snapshot.exists(), "young but ACKed and sent: may go under pressure")
        for ev in kept:
            with self.subTest(ev.snapshot.name):
                self.assertTrue(ev.snapshot.exists())
                row = get_event(ev.event_id)
                self.assertIsNone(row["snapshot_deleted_ts"])
                if ev.clip is not None:
                    self.assertTrue(ev.clip.exists())
                    self.assertIsNone(row["video_deleted_ts"])
        self.assertEqual((summary["snapshots_deleted"], summary["videos_deleted"]), (1, 0))
        self.assertTrue(summary["critical"])
