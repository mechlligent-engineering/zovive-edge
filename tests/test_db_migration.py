"""Tests db/migrations.py's forward migrations: version 1 -> 2 (adds
video_path/video_status, from the video-upload enhancement) and
2 -> 3 (adds snapshot_deleted_ts/video_deleted_ts, from the
ACK-triggered local-file-cleanup enhancement).

The important case here isn't the fresh-install path (that's covered
implicitly by every other test that calls init_db() — schema.sql
already creates every column and seeds schema_version=3 directly).
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
        # A v1 database should walk both the v1->v2 and v2->v3 steps in
        # one migrate() call, ending at the latest schema.
        migrate(self.conn)

        self.assertEqual(current_version(self.conn), 3)

        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(events)").fetchall()}
        for expected in (
            "video_path",
            "video_status",
            "video_attempt_count",
            "video_last_attempt_ts",
            "video_error_message",
            "video_sent_ts",
            "snapshot_deleted_ts",
            "video_deleted_ts",
        ):
            self.assertIn(expected, cols)

        row = self.conn.execute("SELECT * FROM events WHERE event_id = 'evt-old'").fetchone()
        self.assertIsNotNone(row, "pre-existing row must survive the migration")
        self.assertEqual(row["animal_name"], "tiger")
        self.assertEqual(row["video_status"], "none", "new column should default to 'none' on old rows")
        self.assertIsNone(row["video_path"])
        self.assertIsNone(row["snapshot_deleted_ts"])
        self.assertIsNone(row["video_deleted_ts"])

    def test_migrate_is_idempotent(self):
        migrate(self.conn)
        migrate(self.conn)  # must not raise (e.g. "duplicate column") on a second call
        self.assertEqual(current_version(self.conn), 3)


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

        self.assertEqual(current_version(self.conn), 3)
        row = self.conn.execute("SELECT * FROM events WHERE event_id = 'evt-v2'").fetchone()
        self.assertEqual(row["video_status"], "sent", "existing video_status must be untouched by this step")
        self.assertIsNone(row["snapshot_deleted_ts"])
        self.assertIsNone(row["video_deleted_ts"])


if __name__ == "__main__":
    unittest.main()
