-- ZOVIVE edge node SQLite schema. WAL mode (set by db/init_db.py) so a
-- writer (pipeline thread inserting events) and a reader (transfer_main.py
-- scanning for pending rows) don't block each other.

PRAGMA foreign_keys = ON;

-- The outbox: one row per detection event. `status` drives
-- transfer_main.py's send loop; nothing is deleted on send success,
-- only marked, so the DB is also the permanent local record.
CREATE TABLE IF NOT EXISTS events (
    event_id        TEXT PRIMARY KEY,      -- uuid4
    track_id        TEXT NOT NULL,
    camera_id       TEXT NOT NULL,
    camera_name     TEXT,
    forest_name     TEXT,
    gps_lat         REAL,
    gps_lon         REAL,
    animal_name     TEXT NOT NULL,         -- species, or 'unknown_animal'
    confidence      REAL NOT NULL,
    snapshot_path   TEXT NOT NULL,
    model_version   TEXT,
    preset_name     TEXT,
    priority        TEXT NOT NULL DEFAULT 'normal',   -- normal | high
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending | failed | sent | dead  (image alert)
    attempt_count   INTEGER NOT NULL DEFAULT 0,
    last_attempt_ts REAL,
    error_message   TEXT,
    created_ts      REAL NOT NULL,
    sent_ts         REAL,
    -- Event video clip (pipeline/clip_extractor.py), attached to an
    -- already-dispatched event asynchronously — video_path starts NULL
    -- and video_status 'none' until pipeline/clip_extractor.py finishes
    -- writing the MP4, independently of whether the image alert above
    -- has already been sent. transfer_main.py polls video_status
    -- separately from status so a slow/failed video upload never blocks
    -- or gets confused with the (already-sent) image alert.
    video_path          TEXT,
    video_status         TEXT NOT NULL DEFAULT 'none',  -- none | pending | failed | sent | dead
    video_attempt_count  INTEGER NOT NULL DEFAULT 0,
    video_last_attempt_ts REAL,
    video_error_message  TEXT,
    video_sent_ts         REAL,
    -- Set once transfer_main.py has deleted the corresponding local file
    -- after the base station ACKed it (HTTP 200 on send_event/send_video)
    -- — Pi storage is meant to be temporary; the base station holds the
    -- permanent copy. NULL means the local file (if any) still exists.
    -- This row itself is never deleted either way — it's the permanent
    -- local *metadata* record, only the heavy binary files get purged.
    snapshot_deleted_ts REAL,
    video_deleted_ts     REAL,
    -- Per-artifact proof of receipt (storage/retention_manager.py deletes
    -- only media whose own column is set). Written in the same UPDATE that
    -- marks the artifact sent, and only when a real base station answered
    -- 2xx, without a redirect, to a request that carried the complete file
    -- (transfer_main.py). NULL = no proof: a metadata-only send, the
    -- NullBaseStationClient, or a row sent before this column existed.
    snapshot_acked_ts   REAL,
    video_acked_ts      REAL,
    -- Upload retry queue, one pair per artifact (transfer_main.py,
    -- network/retry_policy.py). A sender claims due rows by moving
    -- *next_retry_ts forward by a lease and stamping *send_claim; only that
    -- claim may record the result, and an expired lease makes the row due
    -- again. NULL next_retry_ts = due now. 'dead' (status/video_status):
    -- retries exhausted after HTTP 4xx rejections or an unrecoverable local
    -- problem; never retried automatically (transfer_main.py --requeue-dead).
    -- attempt_count/video_attempt_count count only those rejections.
    next_retry_ts       REAL,
    send_claim          TEXT,
    video_next_retry_ts REAL,
    video_send_claim    TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_status ON events(status);
CREATE INDEX IF NOT EXISTS idx_events_track ON events(track_id);
CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_ts);
CREATE INDEX IF NOT EXISTS idx_events_video_status ON events(video_status);

-- Periodic health heartbeats, so a run of missed heartbeats is visible
-- locally even before the base station gets one.
CREATE TABLE IF NOT EXISTS heartbeats (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id       TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    created_ts      REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_heartbeats_created ON heartbeats(created_ts);

-- Schema version, so migrations.py knows what to apply. A fresh install
-- runs this file (which already has every column above) and starts at
-- the latest version directly; an existing deployment picks up
-- whichever db/migrations.py entries are still ahead of its own
-- version, applied in order, so every path converges on the same
-- schema regardless of which version it started from.
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('schema_version', '5');
-- Indexes on columns added by a migration live in db/migrations.py
-- (POST_MIGRATION_INDEXES): this file also runs against an existing,
-- older database before migrate() has added those columns.
