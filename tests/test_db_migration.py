"""Tests db/migrations.py's forward migrations: version 1 -> 2 (adds
video_path/video_status, from the video-upload enhancement),
2 -> 3 (adds snapshot_deleted_ts/video_deleted_ts, from the
ACK-triggered local-file-cleanup enhancement), 3 -> 4 (adds the
per-artifact snapshot_acked_ts/video_acked_ts, never backfilled) and
4 -> 5 (the upload retry queue columns, plus their indexes, which
migrate() creates for every database).

The important case here isn't the fresh-install path (that's covered
implicitly by every other test that calls init_db() — schema.sql
already creates every column and seeds schema_version=5 directly).
It's the *upgrade* path: a database that already exists on a deployed
node, at an older schema_version, without the newer columns. That's
simulated here by hand-building older-shaped schemas, so this test
would fail loudly if db/migrations.py's ALTER TABLE statements were
ever wrong or missing — a bug there would surface for the first time
on a real Pi's already-deployed database, which is exactly what this
guards against.
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from db.migrations import current_version, migrate

LATEST = 5
_RETRY_COLUMNS = ("next_retry_ts", "send_claim", "video_next_retry_ts", "video_send_claim")
_RETRY_INDEXES = {"idx_events_send_queue", "idx_events_video_queue"}
_SCHEMA_SQL = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


def _schema_at(version: int) -> str:
    """db/schema.sql as it was at an older version (3 or 4): without the
    retry-queue columns and, below 4, without the ACK columns. (Built from the
    text: SQLite's DROP COLUMN cannot cope with the comments in the table.)"""
    text = _SCHEMA_SQL.read_text(encoding="utf-8")
    start = text.index("    video_acked_ts      REAL,\n")
    end = text.index("    video_send_claim    TEXT\n") + len("    video_send_claim    TEXT\n")
    text = text[:start] + "    video_acked_ts      REAL\n" + text[end:]
    if version < 4:
        text = text.replace("    video_deleted_ts     REAL,", "    video_deleted_ts     REAL").replace(
            "    snapshot_acked_ts   REAL,\n    video_acked_ts      REAL\n", ""
        )
    return text.replace(f"('schema_version', '{LATEST}')", f"('schema_version', '{version}')")


def _columns(conn: sqlite3.Connection) -> list[str]:
    return [r[1] for r in conn.execute("PRAGMA table_info(events)").fetchall()]


def _indexes(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'").fetchall()}


_BATCH1_SCHEMA = """
CREATE TABLE events (
    event_id        TEXT PRIMARY KEY,
    track_id        TEXT NOT NULL,
    camera_id       TEXT NOT NULL,
    camera_name     TEXT,
    forest_name     TEXT,
    gps_lat         REAL,
    gps_lon         REAL,
    animal_name     TEXT NOT NULL,
    confidence      REAL NOT NULL,
    snapshot_path   TEXT NOT NULL,
    model_version   TEXT,
    preset_name     TEXT,
    priority        TEXT NOT NULL DEFAULT 'normal',
    status          TEXT NOT NULL DEFAULT 'pending',
    attempt_count   INTEGER NOT NULL DEFAULT 0,
    last_attempt_ts REAL,
    error_message   TEXT,
    created_ts      REAL NOT NULL,
    sent_ts         REAL
);
CREATE TABLE schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
INSERT INTO schema_meta (key, value) VALUES ('schema_version', '1');
"""


class TestVideoColumnMigration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="zovive_migration_test_")
        self.db_path = Path(self.tmpdir) / "batch1.sqlite3"
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_BATCH1_SCHEMA)
        self.conn.execute(
            """
            INSERT INTO events (
                event_id, track_id, camera_id, animal_name, confidence,
                snapshot_path, created_ts
            ) VALUES ('evt-old', 'trk-old', 'cam01', 'tiger', 0.9, '/snap.jpg', 1000.0)
            """
        )
        self.conn.commit()

    def tearDown(self):
        self.conn.close()

    def test_starts_at_version_1(self):
        self.assertEqual(current_version(self.conn), 1)

    def test_migrate_from_v1_lands_on_latest_with_all_columns_and_no_data_loss(self):
        # A v1 database should walk every step (v1->v2->...->v5) in one
        # migrate() call, ending at the latest schema.
        migrate(self.conn)

        self.assertEqual(current_version(self.conn), LATEST)

        cols = set(_columns(self.conn))
        for expected in (
            "video_path",
            "video_status",
            "video_attempt_count",
            "video_last_attempt_ts",
            "video_error_message",
            "video_sent_ts",
            "snapshot_deleted_ts",
            "video_deleted_ts",
            "snapshot_acked_ts",
            "video_acked_ts",
            *_RETRY_COLUMNS,
        ):
            self.assertIn(expected, cols)
        self.assertTrue(_RETRY_INDEXES <= _indexes(self.conn))

        row = self.conn.execute("SELECT * FROM events WHERE event_id = 'evt-old'").fetchone()
        self.assertIsNotNone(row, "pre-existing row must survive the migration")
        self.assertEqual(row["animal_name"], "tiger")
        self.assertEqual(row["video_status"], "none", "new column should default to 'none' on old rows")
        self.assertIsNone(row["video_path"])
        self.assertIsNone(row["snapshot_deleted_ts"])
        self.assertIsNone(row["video_deleted_ts"])
        self.assertIsNone(row["snapshot_acked_ts"])
        self.assertIsNone(row["video_acked_ts"])

    def test_migrate_is_idempotent(self):
        migrate(self.conn)
        migrate(self.conn)  # must not raise (e.g. "duplicate column") on a second call
        self.assertEqual(current_version(self.conn), LATEST)


_BATCH1_V2_SCHEMA = """
CREATE TABLE events (
    event_id        TEXT PRIMARY KEY,
    track_id        TEXT NOT NULL,
    camera_id       TEXT NOT NULL,
    animal_name     TEXT NOT NULL,
    confidence      REAL NOT NULL,
    snapshot_path   TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    created_ts      REAL NOT NULL,
    video_path          TEXT,
    video_status         TEXT NOT NULL DEFAULT 'none',
    video_attempt_count  INTEGER NOT NULL DEFAULT 0,
    video_last_attempt_ts REAL,
    video_error_message  TEXT,
    video_sent_ts         REAL
);
CREATE TABLE schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
INSERT INTO schema_meta (key, value) VALUES ('schema_version', '2');
"""


class TestDeletedTsColumnMigration(unittest.TestCase):
    """Specifically exercises the 2 -> 3 step in isolation, starting
    from a database that already has the video columns (i.e. one that
    already picked up the video-upload enhancement) but not yet the
    ACK-cleanup columns."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="zovive_migration_v2_test_")
        self.conn = sqlite3.connect(str(Path(self.tmpdir) / "batch1_v2.sqlite3"))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_BATCH1_V2_SCHEMA)
        self.conn.execute(
            "INSERT INTO events (event_id, track_id, camera_id, animal_name, confidence, snapshot_path, "
            "video_path, video_status, created_ts) VALUES "
            "('evt-v2', 'trk-v2', 'cam01', 'leopard', 0.8, '/snap2.jpg', '/clip2.mp4', 'sent', 2000.0)"
        )
        self.conn.commit()

    def tearDown(self):
        self.conn.close()

    def test_migrate_adds_deleted_ts_columns_without_touching_video_state(self):
        self.assertEqual(current_version(self.conn), 2)

        migrate(self.conn)

        self.assertEqual(current_version(self.conn), LATEST)
        row = self.conn.execute("SELECT * FROM events WHERE event_id = 'evt-v2'").fetchone()
        self.assertEqual(row["video_status"], "sent", "existing video_status must be untouched by this step")
        self.assertIsNone(row["snapshot_deleted_ts"])
        self.assertIsNone(row["video_deleted_ts"])


class TestAckColumnMigration(unittest.TestCase):
    """The 3 -> 4 step on a deployed v3 database: rows already marked
    'sent' (possibly by the NullBaseStationClient or as metadata only) must
    not gain an ACK, and nothing else may change."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.conn = sqlite3.connect(str(Path(self.tmpdir.name) / "v3.sqlite3"))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_schema_at(3))
        self.conn.execute(
            "INSERT INTO events (event_id, track_id, camera_id, animal_name, confidence, snapshot_path, "
            "status, sent_ts, video_path, video_status, video_sent_ts, snapshot_deleted_ts, created_ts) VALUES "
            "('evt-sent', 't', 'cam01', 'elephant', 0.9, '/data/snapshots/a.jpg', 'sent', 1500.0, "
            "'/data/clips/a.mp4', 'sent', 1600.0, NULL, 1000.0)"
        )
        self.conn.commit()

    def tearDown(self):
        self.conn.close()
        self.tmpdir.cleanup()

    def test_starts_without_ack_columns(self):
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(events)").fetchall()}
        self.assertEqual(current_version(self.conn), 3)
        self.assertNotIn("snapshot_acked_ts", cols)

    def test_sent_rows_keep_no_ack_and_data_is_preserved(self):
        before = dict(self.conn.execute("SELECT * FROM events WHERE event_id = 'evt-sent'").fetchone())
        self.assertEqual(migrate(self.conn), LATEST)
        after = dict(self.conn.execute("SELECT * FROM events WHERE event_id = 'evt-sent'").fetchone())
        self.assertIsNone(after.pop("snapshot_acked_ts"), "historic 'sent' must not become ACKed")
        self.assertIsNone(after.pop("video_acked_ts"), "historic 'sent' must not become ACKed")
        for col in _RETRY_COLUMNS:
            self.assertIsNone(after.pop(col))
        self.assertEqual(after, before)

    def test_fresh_schema_is_already_latest(self):
        fresh = sqlite3.connect(":memory:")
        fresh.row_factory = sqlite3.Row
        fresh.executescript(_SCHEMA_SQL.read_text(encoding="utf-8"))
        self.assertEqual(current_version(fresh), LATEST)
        self.assertEqual(migrate(fresh), LATEST)
        self.assertTrue({"snapshot_acked_ts", "video_acked_ts", *_RETRY_COLUMNS} <= set(_columns(fresh)))
        self.assertTrue(
            _RETRY_INDEXES <= _indexes(fresh), "migrate() adds the queue indexes to a fresh DB too"
        )
        fresh.close()


class TestRetryQueueMigration(unittest.TestCase):
    """The 4 -> 5 step: an upgraded database ends up exactly like a fresh one,
    and rows stuck as 'failed' (never retried before version 5) are due now."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.conn = sqlite3.connect(str(Path(self.tmpdir.name) / "v4.sqlite3"))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_schema_at(4))
        self.assertFalse(set(_RETRY_COLUMNS) & set(_columns(self.conn)))
        self.conn.execute(
            "INSERT INTO events (event_id, track_id, camera_id, animal_name, confidence, snapshot_path, "
            "status, attempt_count, error_message, created_ts) VALUES "
            "('evt-failed', 't', 'cam01', 'tiger', 0.9, '/data/snapshots/f.jpg', 'failed', 1, 'refused', 1000.0)"
        )
        self.conn.commit()

    def tearDown(self):
        self.conn.close()
        self.tmpdir.cleanup()

    def test_upgrade_matches_a_fresh_database(self):
        self.assertEqual(current_version(self.conn), 4)
        self.assertEqual(migrate(self.conn), LATEST)
        fresh = sqlite3.connect(":memory:")
        fresh.row_factory = sqlite3.Row
        fresh.executescript(_SCHEMA_SQL.read_text(encoding="utf-8"))
        migrate(fresh)
        self.assertEqual(_columns(self.conn), _columns(fresh))
        self.assertEqual(_indexes(self.conn), _indexes(fresh))
        fresh.close()

    def test_stuck_failed_rows_become_due_with_their_history(self):
        migrate(self.conn)
        row = self.conn.execute("SELECT * FROM events WHERE event_id = 'evt-failed'").fetchone()
        self.assertEqual(
            (row["status"], row["attempt_count"], row["error_message"]), ("failed", 1, "refused")
        )
        self.assertIsNone(row["next_retry_ts"], "NULL next_retry_ts = due now")
        self.assertIsNone(row["send_claim"])


if __name__ == "__main__":
    unittest.main()
