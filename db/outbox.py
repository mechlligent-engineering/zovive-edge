"""Read/write helpers for the `events` outbox table.

Not in the original skeleton listing — added because db/ had
init_db.py, migrations.py and schema.sql but nothing that actually
inserts or queries a row; alert_dispatcher.py (pipeline/) and, later,
the transfer/sync code in network/ both need this. Keeping the SQL in
one place means the column list only has to match schema.sql once.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from db.init_db import get_connection


@dataclass
class EventRecord:
    track_id: str
    camera_id: str
    animal_name: str
    confidence: float
    snapshot_path: str
    camera_name: str = ""
    forest_name: str = ""
    gps_lat: float | None = None
    gps_lon: float | None = None
    model_version: str = ""
    preset_name: str = ""
    priority: str = "normal"
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_ts: float = field(default_factory=time.time)


def insert_event(record: EventRecord) -> str:
    """Insert a new pending event. Returns its event_id."""
    conn = get_connection()
    with conn:
        conn.execute(
            """
            INSERT INTO events (
                event_id, track_id, camera_id, camera_name, forest_name,
                gps_lat, gps_lon, animal_name, confidence, snapshot_path,
                model_version, preset_name, priority, status, created_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                record.event_id,
                record.track_id,
                record.camera_id,
                record.camera_name,
                record.forest_name,
                record.gps_lat,
                record.gps_lon,
                record.animal_name,
                record.confidence,
                record.snapshot_path,
                record.model_version,
                record.preset_name,
                record.priority,
                record.created_ts,
            ),
        )
    return record.event_id


def get_pending(limit: int = 50) -> list[dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM events WHERE status = 'pending' ORDER BY created_ts ASC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def mark_sent(event_id: str) -> None:
    conn = get_connection()
    with conn:
        conn.execute(
            "UPDATE events SET status = 'sent', sent_ts = ?, last_attempt_ts = ? WHERE event_id = ?",
            (time.time(), time.time(), event_id),
        )


def mark_failed(event_id: str, error_message: str) -> None:
    conn = get_connection()
    with conn:
        conn.execute(
            """
            UPDATE events
            SET status = 'failed', attempt_count = attempt_count + 1,
                last_attempt_ts = ?, error_message = ?
            WHERE event_id = ?
            """,
            (time.time(), error_message, event_id),
        )


def reset_to_pending(event_id: str) -> None:
    """Used to retry a previously-failed event."""
    conn = get_connection()
    with conn:
        conn.execute("UPDATE events SET status = 'pending' WHERE event_id = ?", (event_id,))


def count_by_status() -> dict[str, int]:
    conn = get_connection()
    rows = conn.execute("SELECT status, COUNT(*) AS n FROM events GROUP BY status").fetchall()
    return {r["status"]: r["n"] for r in rows}


# -- Event video clip (video_path/video_status columns) ----------------------
#
# Deliberately separate from status/mark_sent/mark_failed above: an
# event's image alert and its video clip travel on independent
# timelines (pipeline/clip_extractor.py finishes the MP4 well after
# the image alert row was already inserted and possibly already sent),
# so they get their own status column and their own functions rather
# than overloading `status` with a third "has a video too" meaning.


def set_video_path(event_id: str, video_path: str) -> None:
    """Called by pipeline/clip_extractor.py's on_clip_ready callback once
    the MP4 finishes writing — marks the video ready for transfer_main.py
    to pick up (video_status: none -> pending)."""
    conn = get_connection()
    with conn:
        conn.execute(
            "UPDATE events SET video_path = ?, video_status = 'pending' WHERE event_id = ?",
            (video_path, event_id),
        )


def get_pending_videos(limit: int = 50) -> list[dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM events WHERE video_status = 'pending' AND video_path IS NOT NULL "
        "ORDER BY created_ts ASC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def mark_video_sent(event_id: str) -> None:
    conn = get_connection()
    with conn:
        conn.execute(
            "UPDATE events SET video_status = 'sent', video_sent_ts = ?, video_last_attempt_ts = ? "
            "WHERE event_id = ?",
            (time.time(), time.time(), event_id),
        )


def mark_video_failed(event_id: str, error_message: str) -> None:
    conn = get_connection()
    with conn:
        conn.execute(
            """
            UPDATE events
            SET video_status = 'failed', video_attempt_count = video_attempt_count + 1,
                video_last_attempt_ts = ?, video_error_message = ?
            WHERE event_id = ?
            """,
            (time.time(), error_message, event_id),
        )


def reset_video_to_pending(event_id: str) -> None:
    """Used to retry a previously-failed video upload."""
    conn = get_connection()
    with conn:
        conn.execute("UPDATE events SET video_status = 'pending' WHERE event_id = ?", (event_id,))


# -- Local-file cleanup after ACK (Pi storage is temporary) -------------------
#
# Separate from status/video_status: those track whether the base
# station has the artifact, these track whether the *local* copy has
# been removed. An event can be status='sent' with snapshot_deleted_ts
# still NULL for a while (deletion happens right after the ACK, in the
# same transfer_main.py call, but is still its own recorded step) or
# forever NULL if transfer_config.yaml's delete_local_files_after_ack
# is off.


def mark_snapshot_deleted(event_id: str) -> None:
    conn = get_connection()
    with conn:
        conn.execute("UPDATE events SET snapshot_deleted_ts = ? WHERE event_id = ?", (time.time(), event_id))


def mark_video_deleted(event_id: str) -> None:
    conn = get_connection()
    with conn:
        conn.execute("UPDATE events SET video_deleted_ts = ? WHERE event_id = ?", (time.time(), event_id))
