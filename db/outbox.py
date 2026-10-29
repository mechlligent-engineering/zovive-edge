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


def mark_sent(event_id: str, *, snapshot_acked: bool = False, claim: str | None = None) -> bool:
    """Mark the image alert sent. ``snapshot_acked=True`` also records the
    snapshot's Base Station ACK in the same UPDATE, so the ACK can never be
    persisted without the status change or the other way round. Pass it only
    when a real base station confirmed a request that carried the snapshot.
    With ``claim`` (transfer_main.py) the row changes only while it still
    holds that claim (see claim_due). Returns True if the row was updated."""
    now = time.time()
    sql = (
        "UPDATE events SET status = 'sent', sent_ts = ?, last_attempt_ts = ?, "
        "snapshot_acked_ts = CASE WHEN ? THEN ? ELSE snapshot_acked_ts END, "
        "next_retry_ts = NULL, send_claim = NULL WHERE event_id = ?"
    )
    params: tuple[Any, ...] = (now, now, snapshot_acked, now, event_id)
    if claim is not None:
        sql += " AND send_claim = ?"
        params += (claim,)
    conn = get_connection()
    with conn:
        return conn.execute(sql, params).rowcount == 1


def count_by_status() -> dict[str, int]:
    conn = get_connection()
    rows = conn.execute("SELECT status, COUNT(*) AS n FROM events GROUP BY status").fetchall()
    return {r["status"]: r["n"] for r in rows}


# -- Event video clip (video_path/video_status columns) ----------------------
#
# Deliberately separate from status/mark_sent above: an
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


def mark_video_sent(event_id: str, *, acked: bool = False, claim: str | None = None) -> bool:
    """Mark the video sent; ``acked=True`` also records the clip's Base
    Station ACK in the same UPDATE, ``claim`` as in mark_sent."""
    now = time.time()
    sql = (
        "UPDATE events SET video_status = 'sent', video_sent_ts = ?, video_last_attempt_ts = ?, "
        "video_acked_ts = CASE WHEN ? THEN ? ELSE video_acked_ts END, "
        "video_next_retry_ts = NULL, video_send_claim = NULL WHERE event_id = ?"
    )
    params: tuple[Any, ...] = (now, now, acked, now, event_id)
    if claim is not None:
        sql += " AND video_send_claim = ?"
        params += (claim,)
    conn = get_connection()
    with conn:
        return conn.execute(sql, params).rowcount == 1


# -- Upload retry queue (status/video_status, *next_retry_ts, *send_claim) ------
#
# One implementation for both artifacts: _QUEUES maps "image" / "video" to
# their columns. transfer_main.py claims due rows (claim_due), sends each, and
# records the result under the same claim token, so a sender whose lease ran
# out can never overwrite a newer result. Times are epoch seconds like every
# other *_ts column (time.time(), not SQLite's CURRENT_TIMESTAMP text), passed
# in so tests control the clock.
#
# Bandit B608: the queries below interpolate only column names from the fixed
# _QUEUES table; every value is a bound parameter.


@dataclass(frozen=True)
class _Queue:
    status: str
    attempts: str
    next_retry: str
    claim: str
    last_attempt: str
    error: str
    member: str  # extra condition for a row to be in this queue at all


_QUEUES = {
    "image": _Queue(
        "status", "attempt_count", "next_retry_ts", "send_claim", "last_attempt_ts", "error_message", "1"
    ),
    "video": _Queue(
        "video_status",
        "video_attempt_count",
        "video_next_retry_ts",
        "video_send_claim",
        "video_last_attempt_ts",
        "video_error_message",
        "video_path IS NOT NULL",
    ),
}
QUEUE_NAMES = tuple(_QUEUES)
_ERROR_MAX_LEN = 500


def _due(q: _Queue) -> str:
    """WHERE clause: rows of queue ``q`` a sender may take at :now. NULL
    next_retry_ts is due; a value beyond :now + :max_wait can only come from a
    clock that jumped back, so it is due too rather than stuck for hours."""
    return (
        f"{q.member} AND ({q.status} = 'pending' OR ({q.status} = 'failed' AND {q.attempts} <= :max_retries)) "
        f"AND ({q.next_retry} IS NULL OR {q.next_retry} <= :now OR {q.next_retry} > :now + :max_wait)"
    )


def claim_due(
    queue: str, *, now: float, limit: int, lease_sec: float, max_retries: int, max_wait_sec: float
) -> tuple[str, list[dict[str, Any]]]:
    """Claim up to ``limit`` due rows of ``queue`` ("image" / "video") for one
    send attempt each, oldest due first. Returns (claim token, rows).

    Due: 'pending', or 'failed' with attempt_count <= max_retries, whose
    next_retry_ts is unset or reached (see _due). One BEGIN IMMEDIATE
    transaction, so concurrent senders queue on SQLite's write lock
    (busy_timeout) and never claim the same row. Each claimed row's
    next_retry_ts moves to now + lease_sec, hiding it from other senders until
    this one records a result or the lease expires (a sender that died).
    'failed' rows already over the budget (max_retries was lowered) become
    'dead' in the same transaction."""
    q = _QUEUES[queue]
    token = uuid.uuid4().hex
    params = {"now": now, "max_retries": max_retries, "max_wait": max_wait_sec, "limit": limit}
    conn = get_connection()
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            f"UPDATE events SET {q.status} = 'dead', {q.next_retry} = NULL "  # nosec B608
            f"WHERE {q.status} = 'failed' AND {q.attempts} > :max_retries",
            params,
        )
        ids = [
            row[0]
            for row in conn.execute(
                f"SELECT event_id FROM events WHERE {_due(q)} "  # nosec B608
                f"ORDER BY COALESCE({q.next_retry}, created_ts), created_ts LIMIT :limit",
                params,
            )
        ]
        conn.executemany(
            f"UPDATE events SET {q.next_retry} = ?, {q.claim} = ?, {q.last_attempt} = ? "  # nosec B608
            "WHERE event_id = ?",
            [(now + lease_sec, token, now, event_id) for event_id in ids],
        )
        rows = {
            r["event_id"]: dict(r)
            for r in conn.execute(f"SELECT * FROM events WHERE {q.claim} = ?", (token,))  # nosec B608
        }
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    return token, [rows[event_id] for event_id in ids]


def record_send_failure(
    queue: str, event_id: str, claim: str, *, now: float, error: str, retry_at: float | None, charged: bool
) -> bool:
    """Record a failed send of a claimed row. ``charged`` (an application-level
    rejection) adds one to attempt_count; an uncharged failure (Base Station
    unreachable or busy) leaves it. The row is 'failed' and due again at
    ``retry_at``, or 'dead' when ``retry_at`` is None. Only while the row still
    holds ``claim``; returns True if it was updated."""
    q = _QUEUES[queue]
    status = "dead" if retry_at is None else "failed"
    attempts = f"{q.attempts} + 1" if charged else q.attempts
    conn = get_connection()
    with conn:
        return (
            conn.execute(
                f"UPDATE events SET {q.status} = ?, {q.attempts} = {attempts}, {q.next_retry} = ?, "  # nosec B608
                f"{q.claim} = NULL, {q.last_attempt} = ?, {q.error} = ? "
                f"WHERE event_id = ? AND {q.claim} = ?",
                (status, retry_at, now, error[:_ERROR_MAX_LEN], event_id, claim),
            ).rowcount
            == 1
        )


def release_claim(queue: str, claim: str, *, until: float) -> int:
    """Hand back rows claimed but not attempted (the batch stopped early):
    they are due again at ``until``."""
    q = _QUEUES[queue]
    conn = get_connection()
    with conn:
        return conn.execute(
            f"UPDATE events SET {q.next_retry} = ?, {q.claim} = NULL WHERE {q.claim} = ?",  # nosec B608
            (until, claim),
        ).rowcount


def defer_due(queue: str, *, until: float, max_retries: int) -> int:
    """Pause a queue: every queued row that would be due before ``until``
    waits until then, so a Base Station outage costs one probe per pause
    instead of a timeout per row. Includes the item a previous pause put just
    behind its end, so it cannot slip in between probes."""
    q = _QUEUES[queue]
    conn = get_connection()
    with conn:
        return conn.execute(
            f"UPDATE events SET {q.next_retry} = :until WHERE {q.member} "  # nosec B608
            f"AND ({q.status} = 'pending' OR ({q.status} = 'failed' AND {q.attempts} <= :max_retries)) "
            f"AND ({q.next_retry} IS NULL OR {q.next_retry} < :until)",
            {"until": until, "max_retries": max_retries},
        ).rowcount


def requeue_dead(queue: str) -> int:
    """Put every dead row of ``queue`` back in the retry queue with a fresh
    budget, due now (transfer_main.py --requeue-dead). Returns how many."""
    q = _QUEUES[queue]
    conn = get_connection()
    with conn:
        return conn.execute(
            f"UPDATE events SET {q.status} = 'failed', {q.attempts} = 0, {q.next_retry} = NULL, "  # nosec B608
            f"{q.claim} = NULL WHERE {q.status} = 'dead'"
        ).rowcount


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
            COALESCE(SUM(status = 'dead'), 0) AS dead_snapshots,
            COALESCE(SUM(video_status = 'dead'), 0) AS dead_videos,
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
