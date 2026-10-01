"""Minimal forward-only migration runner.

schema.sql already handles the common case (new install: CREATE TABLE
IF NOT EXISTS covers everything). This module exists for the day a
column needs adding to an existing deployed DB without losing data —
each entry in `MIGRATIONS` is (target_version, sql_statements),
applied in order, each inside its own transaction.
"""

from __future__ import annotations

import sqlite3

MIGRATIONS: list[tuple[int, list[str]]] = [
    # Adds event-video-clip tracking (pipeline/clip_extractor.py,
    # network/base_station_client.py's send_video) to a database created
    # by batch 1, which predates this schema. A fresh install never runs
    # this — db/schema.sql already creates these columns directly and
    # seeds schema_version at 2 — this path exists only for a database
    # that was already deployed with the batch-1 schema (version 1) and
    # needs to gain the columns without losing its existing rows.
    (
        2,
        [
            "ALTER TABLE events ADD COLUMN video_path TEXT",
            "ALTER TABLE events ADD COLUMN video_status TEXT NOT NULL DEFAULT 'none'",
            "ALTER TABLE events ADD COLUMN video_attempt_count INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE events ADD COLUMN video_last_attempt_ts REAL",
            "ALTER TABLE events ADD COLUMN video_error_message TEXT",
            "ALTER TABLE events ADD COLUMN video_sent_ts REAL",
            "CREATE INDEX IF NOT EXISTS idx_events_video_status ON events(video_status)",
        ],
    ),
    # Adds ACK-triggered local-file-deletion tracking (transfer_main.py's
    # send loops + storage/local_file_cleanup.py): Pi storage is meant to
    # be temporary, the base station the permanent copy, so once an
    # artifact is ACKed (HTTP 200) its local file is deleted and that's
    # recorded here — separately from status/video_status, which track
    # the *send*, not the *local cleanup*. A fresh install is already at
    # version 3 (schema.sql), so this only fires for a database upgrading
    # from version 1 or 2.
    (
        3,
        [
            "ALTER TABLE events ADD COLUMN snapshot_deleted_ts REAL",
            "ALTER TABLE events ADD COLUMN video_deleted_ts REAL",
        ],
    ),
]


def current_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()
    return int(row["value"]) if row else 0


def migrate(conn: sqlite3.Connection) -> int:
    """Apply any pending migrations. Returns the resulting schema version."""
    version = current_version(conn)
    for target_version, statements in sorted(MIGRATIONS, key=lambda m: m[0]):
        if target_version <= version:
            continue
        with conn:
            for stmt in statements:
                conn.execute(stmt)
            conn.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(target_version),),
            )
        version = target_version
    return version
