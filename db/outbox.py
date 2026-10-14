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


def mark_sent(event_id: str, *, snapshot_acked: bool = False) -> None:
    """Mark the image alert sent. ``snapshot_acked=True`` also records the
    snapshot's Base Station ACK in the same UPDATE, so the ACK can never be
    persisted without the status change or the other way round. Pass it only
    when a real base station confirmed a request that carried the snapshot."""
    now = time.time()
    conn = get_connection()
    with conn:
        conn.execute(
            "UPDATE events SET status = 'sent', sent_ts = ?, last_attempt_ts = ?, "
            "snapshot_acked_ts = CASE WHEN ? THEN ? ELSE snapshot_acked_ts END "
            "WHERE event_id = ?",
            (now, now, snapshot_acked, now, event_id),
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


def mark_video_sent(event_id: str, *, acked: bool = False) -> None:
    """Mark the video sent; ``acked=True`` also records the clip's Base
    Station ACK in the same UPDATE (see mark_sent)."""
    now = time.time()
    conn = get_connection()
    with conn:
        conn.execute(
            "UPDATE events SET video_status = 'sent', video_sent_ts = ?, video_last_attempt_ts = ?, "
            "video_acked_ts = CASE WHEN ? THEN ? ELSE video_acked_ts END "
            "WHERE event_id = ?",
            (now, now, acked, now, event_id),
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


# -- Local-file cleanup (Pi storage is temporary) -----------------------------
#
# Separate from status/video_status: those track the *send*, these track
# whether the base station confirmed receipt of the file (*_acked_ts) and
# whether the *local* copy has been removed (*_deleted_ts). Only media whose
# own *_acked_ts is set may ever be deleted (storage/retention_manager.py,
# and transfer_main.py's optional delete-right-after-ACK).


def get_event(event_id: str) -> dict[str, Any] | None:
    """One event row by id, or None if there is no such event."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM events WHERE event_id = ?", (event_id,)).fetchone()
    return dict(row) if row is not None else None


def get_acked_snapshots_not_deleted(cutoff_ts: float | None = None) -> list[dict[str, Any]]:
    """Sent events whose snapshot has its own ACK and is not yet recorded as
    deleted, oldest first. With ``cutoff_ts``, only rows both recorded and
    ACKed at or before that time."""
    sql = (
        "SELECT * FROM events WHERE status = 'sent' AND snapshot_acked_ts IS NOT NULL "
        "AND snapshot_deleted_ts IS NULL AND snapshot_path IS NOT NULL"
    )
    params: tuple[Any, ...] = ()
    if cutoff_ts is not None:
        sql += " AND created_ts <= ? AND snapshot_acked_ts <= ?"
        params = (cutoff_ts, cutoff_ts)
    conn = get_connection()
    return [dict(r) for r in conn.execute(sql + " ORDER BY created_ts ASC", params).fetchall()]


def get_acked_videos_not_deleted(cutoff_ts: float | None = None) -> list[dict[str, Any]]:
    """Like get_acked_snapshots_not_deleted, for the event video clips."""
    sql = (
        "SELECT * FROM events WHERE video_status = 'sent' AND video_acked_ts IS NOT NULL "
        "AND video_deleted_ts IS NULL AND video_path IS NOT NULL"
    )
    params: tuple[Any, ...] = ()
    if cutoff_ts is not None:
        sql += " AND created_ts <= ? AND video_acked_ts <= ?"
        params = (cutoff_ts, cutoff_ts)
    conn = get_connection()
    return [dict(r) for r in conn.execute(sql + " ORDER BY created_ts ASC", params).fetchall()]


def get_active_references_by_file_name(file_name: str, *, exclude_event_id: str) -> list[dict[str, Any]]:
    """Rows other than ``exclude_event_id`` whose snapshot or clip path ends
    in ``file_name`` (ASCII case-insensitive) and has no deletion recorded.
    A superset: storage/retention_manager.py compares the resolved paths."""
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT event_id, snapshot_path, snapshot_deleted_ts, video_path, video_deleted_ts
        FROM events
        WHERE event_id != :exclude AND (
            (snapshot_deleted_ts IS NULL AND lower(substr(snapshot_path, -:n)) = lower(:name))
            OR (video_deleted_ts IS NULL AND lower(substr(video_path, -:n)) = lower(:name))
        )
        """,
        {"exclude": exclude_event_id, "n": len(file_name), "name": file_name},
    ).fetchall()
    return [dict(r) for r in rows]


def mark_snapshot_deleted(event_id: str, *, require_ack: bool = False) -> bool:
    """Record the local snapshot as deleted. Keeps the first timestamp if it
    is already recorded. ``require_ack=True`` updates only a row whose
    snapshot has an ACK. Returns True if this call recorded it."""
    sql = "UPDATE events SET snapshot_deleted_ts = ? WHERE event_id = ? AND snapshot_deleted_ts IS NULL"
    if require_ack:
        sql += " AND snapshot_acked_ts IS NOT NULL"
    conn = get_connection()
    with conn:
        return conn.execute(sql, (time.time(), event_id)).rowcount == 1


def mark_video_deleted(event_id: str, *, require_ack: bool = False) -> bool:
    """Record the local clip as deleted; see mark_snapshot_deleted."""
    sql = "UPDATE events SET video_deleted_ts = ? WHERE event_id = ? AND video_deleted_ts IS NULL"
    if require_ack:
        sql += " AND video_acked_ts IS NOT NULL"
    conn = get_connection()
    with conn:
        return conn.execute(sql, (time.time(), event_id)).rowcount == 1


def media_backlog(overdue_before_ts: float) -> dict[str, Any]:
    """Unacknowledged evidence still on the Pi, for cleanup reports and the
    heartbeat. A snapshot or clip is unacknowledged while it has no ACK and
    has not been recorded as deleted; overdue if its event was recorded at or
    before ``overdue_before_ts``."""
    conn = get_connection()
    row = conn.execute(
        """
        SELECT
            COALESCE(SUM(snap_open), 0) AS unacked_snapshots,
            COALESCE(SUM(video_open), 0) AS unacked_videos,
            COALESCE(SUM(snap_open AND created_ts <= :overdue), 0) AS overdue_snapshots,
            COALESCE(SUM(video_open AND created_ts <= :overdue), 0) AS overdue_videos,
            MIN(CASE WHEN snap_open OR video_open THEN created_ts END) AS oldest_unacked_created_ts,
            COALESCE(SUM(status = 'failed'), 0) AS failed_snapshots,
            COALESCE(SUM(video_status = 'failed'), 0) AS failed_videos,
            MAX(snapshot_acked_ts) AS last_snapshot_ack_ts,
            MAX(video_acked_ts) AS last_video_ack_ts
        FROM (
            SELECT *,
                (snapshot_acked_ts IS NULL AND snapshot_deleted_ts IS NULL) AS snap_open,
                (video_path IS NOT NULL AND video_acked_ts IS NULL AND video_deleted_ts IS NULL) AS video_open
            FROM events
        )
        """,
        {"overdue": overdue_before_ts},
    ).fetchone()
    return dict(row)
