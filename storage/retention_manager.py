"""Deletes local snapshots and clips the base station has ACKed, and nothing else.

Two separate runs, both started by systemd (``python -m storage.retention_manager
daily|pressure``; zovive-retention.timer and zovive-storage-pressure.timer):

* **Scheduled retention** (once a day): delete ACKed media recorded *and*
  ACKed at least ``retention_days`` (14) ago. Both timestamps come from the
  Raspberry Pi's own clock (``time.time()`` when the row was written), not
  from the base station. Requiring both to be that old protects media whose
  timestamps were taken while the clock was ahead; if the clock was behind
  when the ACK was recorded (e.g. before the Pi synchronised its time), the
  media looks older than it is and can be deleted early. Only ACKed media is
  affected by this.
* **Storage pressure** (every few minutes): when the filesystem holding the
  media directories is at or above ``trigger_percent`` (82), delete ACKed
  media oldest first, regardless of age, re-measuring the filesystem after
  every file, until usage is at or below ``target_percent`` (79) or nothing
  eligible is left; then report critical pressure instead of going further.

What makes a file deletable, re-checked on a fresh database row right
before every delete:

1. its own ACK (``snapshot_acked_ts`` / ``video_acked_ts``, written by
   transfer_main.py only for a real base station's 2xx to a request that
   carried the complete file). A snapshot ACK says nothing about the clip and
   vice versa; ``status='sent'`` or a configured base station URL alone is
   never enough.
2. ``status`` / ``video_status`` is ``'sent'`` and no deletion is recorded yet.
3. an absolute path whose real location is a ``.jpg`` inside
   ``paths.SNAPSHOT_DIR`` or an ``.mp4`` inside ``paths.CLIP_DIR``, not a
   symlink, not anything else.
4. no other event row references the same file (same resolved path, as its
   snapshot or its clip) without that artifact's deletion recorded: one
   event's ACK never authorizes deleting a file another event still needs.
   If this cannot be checked, the file is kept and the run reports a failure.

Unacknowledged evidence is never deleted automatically, at any age or disk
usage; it is counted and reported (``unacknowledged`` in every summary,
``retention_status.json`` for the heartbeat). A file that is already gone is
recorded as deleted only when its row passes the same checks. A failed
filesystem delete or database update is reported and retried on the next run.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import paths
from db.init_db import init_db
from db.migrations import migrate
from db.outbox import (
    get_acked_snapshots_not_deleted,
    get_acked_videos_not_deleted,
    get_active_references_by_file_name,
    get_event,
    mark_snapshot_deleted,
    mark_video_deleted,
    media_backlog,
)
from storage.local_file_cleanup import delete_local_file
from utils.config_loader import ConfigError, load
from watchdog.edge_status import read_status_file, write_status_file

log = logging.getLogger(__name__)

DAY_SEC = 86400.0
MAX_REPORTED_ERRORS = 20  # summaries carry error_count plus the first few messages

EXIT_OK = 0
EXIT_TEMPFAIL = 75  # could not run now (another run holds the lock, database unavailable)
EXIT_CONFIG = 78  # retention_policy.yaml is invalid; retrying will not help

DAILY_LOCK_WAIT_SEC = 600.0  # a storage-pressure run is usually done well within this


class RetentionPolicyError(ValueError):
    """configs/retention_policy.yaml is unreadable or inconsistent."""


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    enabled: bool = True
    retention_days: float = 14.0
    trigger_percent: float = 82.0
    target_percent: float = 79.0
    overdue_unacknowledged_days: float = 14.0


@dataclass(frozen=True, slots=True)
class _Media:
    kind: str
    status_column: str
    path_column: str
    acked_column: str
    deleted_column: str
    dir_attr: str  # attribute of `paths`, read at call time
    suffix: str


_SNAPSHOT = _Media(
    "snapshot", "status", "snapshot_path", "snapshot_acked_ts", "snapshot_deleted_ts", "SNAPSHOT_DIR", ".jpg"
)
_VIDEO = _Media(
    "video", "video_status", "video_path", "video_acked_ts", "video_deleted_ts", "CLIP_DIR", ".mp4"
)
_MEDIA = (_SNAPSHOT, _VIDEO)


@dataclass(slots=True)
class _Outcome:
    status: str  # deleted | missing | already_deleted | skipped | refused | failed
    message: str | None = None
    bytes_deleted: int = 0


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def load_policy() -> RetentionPolicy:
    """configs/retention_policy.yaml, validated. A missing file means the
    approved defaults (14 days, 82 % -> 79 %)."""
    try:
        cfg = load(paths.RETENTION_POLICY_CONFIG, required=False)
    except ConfigError as exc:
        raise RetentionPolicyError(f"cannot read retention policy: {exc}") from exc
    pressure = cfg.get("storage_pressure", {})
    if not isinstance(pressure, dict):
        raise RetentionPolicyError("storage_pressure must be a mapping")
    defaults = RetentionPolicy()
    enabled = cfg.get("enabled", defaults.enabled)
    if not isinstance(enabled, bool):
        raise RetentionPolicyError("enabled must be true or false")
    policy = RetentionPolicy(
        enabled=enabled,
        retention_days=_number(cfg, "retention_days", defaults.retention_days),
        trigger_percent=_number(pressure, "trigger_percent", defaults.trigger_percent),
        target_percent=_number(pressure, "target_percent", defaults.target_percent),
        overdue_unacknowledged_days=_number(
            cfg, "overdue_unacknowledged_days", defaults.overdue_unacknowledged_days
        ),
    )
    if policy.retention_days <= 0:
        raise RetentionPolicyError("retention_days must be greater than 0")
    if policy.overdue_unacknowledged_days <= 0:
        raise RetentionPolicyError("overdue_unacknowledged_days must be greater than 0")
    if not 0 < policy.target_percent < policy.trigger_percent <= 100:
        raise RetentionPolicyError(
            "storage_pressure needs 0 < target_percent < trigger_percent <= 100, got "
            f"target {policy.target_percent}, trigger {policy.trigger_percent}"
        )
    return policy


def _number(section: dict[str, Any], key: str, default: float) -> float:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RetentionPolicyError(f"{key} must be a number, got {value!r}")
    return float(value)


# ---------------------------------------------------------------------------
# Eligibility queries
# ---------------------------------------------------------------------------


def get_sent_snapshots_ready_for_delete(now: float | None = None) -> list[dict[str, Any]]:
    """Snapshots the daily run may delete: own ACK, sent, not yet deleted,
    recorded and ACKed at least retention_days ago."""
    return _rows(_SNAPSHOT, _retention_cutoff(now))


def get_sent_videos_ready_for_delete(now: float | None = None) -> list[dict[str, Any]]:
    """Clips the daily run may delete; see get_sent_snapshots_ready_for_delete."""
    return _rows(_VIDEO, _retention_cutoff(now))


def get_storage_pressure_candidates() -> list[dict[str, Any]]:
    """Every ACKed, not yet deleted snapshot and clip, any age, oldest first.
    Each row carries ``media_kind`` ("snapshot" or "video")."""
    candidates = [{**row, "media_kind": media.kind} for media in _MEDIA for row in _rows(media, None)]
    return sorted(candidates, key=lambda r: (r["created_ts"], r["event_id"], r["media_kind"]))


def get_unacknowledged_backlog(now: float | None = None) -> dict[str, Any]:
    """Unacknowledged evidence still on the Pi (never deleted automatically):
    counts, overdue counts, transfer failures, oldest age, last ACK times."""
    now = time.time() if now is None else now
    policy = load_policy()
    backlog = media_backlog(now - policy.overdue_unacknowledged_days * DAY_SEC)
    oldest = backlog.get("oldest_unacked_created_ts")
    backlog["oldest_unacked_age_days"] = round((now - oldest) / DAY_SEC, 2) if oldest is not None else None
    backlog["overdue_after_days"] = policy.overdue_unacknowledged_days
    backlog["overdue_total"] = backlog["overdue_snapshots"] + backlog["overdue_videos"]
    return backlog


def _retention_cutoff(now: float | None) -> float:
    now = time.time() if now is None else now
    return now - load_policy().retention_days * DAY_SEC


def _rows(media: _Media, cutoff_ts: float | None) -> list[dict[str, Any]]:
    if media is _SNAPSHOT:
        return get_acked_snapshots_not_deleted(cutoff_ts)
    return get_acked_videos_not_deleted(cutoff_ts)


# ---------------------------------------------------------------------------
# Deleting one file
# ---------------------------------------------------------------------------


def purge_snapshot_after_ack(event_id: str, now: float | None = None) -> str | None:
    """Delete one event's snapshot under the daily-retention rules (own ACK,
    retention_days). None on success, including already deleted or already
    missing; otherwise why it was kept."""
    return _purge_under_retention(event_id, _SNAPSHOT, now)


def purge_video_after_ack(event_id: str, now: float | None = None) -> str | None:
    """Delete one event's clip under the daily-retention rules; see
    purge_snapshot_after_ack."""
    return _purge_under_retention(event_id, _VIDEO, now)


def _purge_under_retention(event_id: str, media: _Media, now: float | None) -> str | None:
    now = time.time() if now is None else now
    try:
        policy = load_policy()
    except RetentionPolicyError as exc:
        return f"{media.kind} {event_id}: retention policy invalid: {exc}"
    if not policy.enabled:
        return f"{media.kind} {event_id}: retention disabled in retention_policy.yaml"
    outcome = _purge(event_id, media, cutoff_ts=now - policy.retention_days * DAY_SEC)
    if outcome.status in ("deleted", "missing", "already_deleted"):
        return None
    return outcome.message


def _purge(event_id: str, media: _Media, cutoff_ts: float | None) -> _Outcome:
    """Delete one file after re-checking its row. ``cutoff_ts`` applies the
    daily-retention age rule; None (storage pressure) means any age."""
    label = f"{media.kind} {event_id}"
    try:
        row = get_event(event_id)
    except sqlite3.Error as exc:
        log.exception("retention: cannot read event", extra={"event_id": event_id, "kind": media.kind})
        return _Outcome("failed", f"{label}: cannot read event: {exc}")
    if row is None:
        return _Outcome("skipped", f"{label}: no such event")
    if row[media.deleted_column] is not None:
        return _Outcome("already_deleted")
    reason = _ineligible_reason(row, media, cutoff_ts)
    if reason:
        return _Outcome("skipped", f"{label}: {reason}, file kept")

    path = Path(row[media.path_column])
    unsafe = _unsafe_reason(path, media)
    if unsafe:
        log.error(
            "retention: refused to delete",
            extra={"event_id": event_id, "kind": media.kind, "path": str(path)},
        )
        return _Outcome("refused", f"{label}: refused to delete {path}: {unsafe}")

    if os.path.lexists(path):
        try:
            sharers = _other_active_references(event_id, path)
        except (sqlite3.Error, OSError, RuntimeError, ValueError) as exc:
            log.exception(
                "retention: cannot check other references; file kept",
                extra={"event_id": event_id, "kind": media.kind, "path": str(path)},
            )
            return _Outcome("failed", f"{label}: cannot check other references to {path} ({exc}); file kept")
        if sharers:
            log.error(
                "retention: refused to delete a file another event still references",
                extra={
                    "event_id": event_id,
                    "kind": media.kind,
                    "path": str(path),
                    "referenced_by": ", ".join(sharers),
                },
            )
            return _Outcome(
                "refused", f"{label}: refused to delete {path}: still referenced by {', '.join(sharers)}"
            )

    try:
        existed = os.path.lexists(path)
        size = path.stat().st_size if existed else 0
        if existed:
            delete_local_file(path)
            if os.path.lexists(path):
                return _Outcome("failed", f"{label}: could not delete {path}, will retry next run")
    except OSError as exc:
        log.exception("retention: filesystem error", extra={"event_id": event_id, "kind": media.kind})
        return _Outcome("failed", f"{label}: filesystem error: {exc}")

    try:
        recorded = _mark_deleted(media, event_id)
    except sqlite3.Error as exc:
        log.exception(
            "retention: deletion not recorded in database", extra={"event_id": event_id, "kind": media.kind}
        )
        what = "file deleted" if existed else "file already missing"
        return _Outcome(
            "failed", f"{label}: {what} but database not updated ({exc}); the next run records it"
        )

    if not existed:
        if not recorded:
            return _Outcome("already_deleted")  # another process recorded it first
        log.warning(
            "retention: ACKed file already missing, recorded as deleted",
            extra={"event_id": event_id, "kind": media.kind, "path": str(path)},
        )
        return _Outcome("missing", f"{label}: file already missing")
    log.info(
        "retention: deleted ACKed file",
        extra={"event_id": event_id, "kind": media.kind, "path": str(path), "bytes": size},
    )
    return _Outcome("deleted", bytes_deleted=size)


def _ineligible_reason(row: dict[str, Any], media: _Media, cutoff_ts: float | None) -> str | None:
    status = row.get(media.status_column)
    if status != "sent":
        return f"not sent ({media.status_column}={status!r})"
    acked_ts = row.get(media.acked_column)
    if acked_ts is None:
        return f"no Base Station ACK for this {media.kind}"
    if not row.get(media.path_column):
        return f"no {media.path_column} recorded"
    if cutoff_ts is not None:
        if row["created_ts"] > cutoff_ts:
            return "recorded less than retention_days ago"
        if acked_ts > cutoff_ts:
            return "ACKed less than retention_days ago"
    return None


def _unsafe_reason(path: Path, media: _Media) -> str | None:
    """Why ``path`` must not be deleted as ``media``, or None if it is a
    plain media file inside its media directory."""
    if not path.is_absolute():
        return "not an absolute path"
    if path.is_symlink():
        return "is a symlink"
    root = Path(getattr(paths, media.dir_attr)).resolve()
    real = path.resolve()  # follows symlinked parents and ".."
    if real == root or not real.is_relative_to(root):
        return f"not inside {root}"
    if real.suffix.lower() != media.suffix:
        return f"not a {media.suffix} file"
    if os.path.lexists(real) and not real.is_file():
        return "not a regular file"
    return None


def _other_active_references(event_id: str, path: Path) -> list[str]:
    """Other events whose snapshot or clip is this same file with no deletion
    recorded for it, e.g. "<event_id> (video)". Raises when that cannot be
    determined; the caller then keeps the file.

    Candidates are rows whose stored path ends in the same file name (any
    case); a file name alone proves nothing, so each absolute path is compared
    by resolved location. A relative stored path cannot be located (it depends
    on the writer's working directory), so one with the same file name counts
    as a reference rather than being guessed at."""
    target = _resolved(path)
    name = path.name.lower()
    found = []
    for row in get_active_references_by_file_name(path.name, exclude_event_id=event_id):
        for media in _MEDIA:
            other = row[media.path_column]
            if not other or row[media.deleted_column] is not None:
                continue
            other_path = Path(other)
            if not other_path.is_absolute():
                if other_path.name.lower() == name:
                    found.append(f"{row['event_id']} ({media.kind}, relative path {other!r})")
            elif _resolved(other_path) == target:
                found.append(f"{row['event_id']} ({media.kind})")
    return found


def _resolved(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def _mark_deleted(media: _Media, event_id: str) -> bool:
    """Conditional update: only a row with this media's ACK and no deletion
    recorded yet. True if this call recorded it."""
    if media is _SNAPSHOT:
        return mark_snapshot_deleted(event_id, require_ack=True)
    return mark_video_deleted(event_id, require_ack=True)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def apply_retention(now: float | None = None) -> dict[str, Any]:
    """Daily scheduled retention: delete ACKed media at least retention_days
    old. Younger or unACKed media is kept. Returns the run summary."""
    now = time.time() if now is None else now
    summary = _new_summary("scheduled_retention", now)
    policy = _policy_or_disable(summary)
    if policy is None:
        return _finish(summary)

    cutoff = now - policy.retention_days * DAY_SEC
    summary["retention_days"] = policy.retention_days
    summary["cutoff_ts"] = cutoff
    log.info("scheduled retention started", extra={"retention_days": policy.retention_days})
    for media in _MEDIA:
        try:
            rows = _rows(media, cutoff)
        except sqlite3.Error as exc:
            log.exception("scheduled retention: cannot list ACKed media", extra={"kind": media.kind})
            _add_error(summary, f"{media.kind}: cannot list ACKed media: {exc}")
            continue
        for row in rows:
            _tally(summary, media, _purge(row["event_id"], media, cutoff))
    _attach_backlog(summary, now)
    return _finish(summary)


def apply_storage_pressure_cleanup(now: float | None = None) -> dict[str, Any]:
    """Storage-pressure cleanup: at or above trigger_percent on a filesystem
    holding media, delete its ACKed media oldest first (any age) until usage
    is at or below target_percent. Never touches unACKed media."""
    now = time.time() if now is None else now
    summary = _new_summary("storage_pressure", now)
    summary.update(
        {
            "trigger_percent": None,
            "target_percent": None,
            "triggered": False,
            "usage_before_percent": None,
            "usage_after_percent": None,
            "target_reached": None,
            "critical": False,
            "blocked_by_unacknowledged": False,
            "eligible_remaining": 0,
            "volumes": [],
        }
    )
    policy = _policy_or_disable(summary)
    if policy is None:
        return _finish(summary)
    summary["trigger_percent"] = policy.trigger_percent
    summary["target_percent"] = policy.target_percent

    try:
        volumes = _media_volumes()
    except OSError as exc:
        log.exception("storage pressure: cannot locate media filesystems")
        _add_error(summary, f"cannot locate media filesystems: {exc}")
        summary["critical"] = True
        return _finish(summary)

    for root, kinds in volumes:
        summary["volumes"].append(_relieve_volume(root, kinds, policy, summary))

    measured = [v for v in summary["volumes"] if v["usage_before_percent"] is not None]
    if measured:
        summary["usage_before_percent"] = max(v["usage_before_percent"] for v in measured)
        summary["usage_after_percent"] = max(v["usage_after_percent"] for v in measured)
        summary["target_reached"] = all(v["target_reached"] for v in measured)
    summary["triggered"] = any(v["triggered"] for v in summary["volumes"])
    summary["critical"] = any(v["critical"] for v in summary["volumes"])
    summary["eligible_remaining"] = sum(v["eligible_remaining"] for v in summary["volumes"])
    _attach_backlog(summary, now)
    backlog = summary["unacknowledged"] or {}
    summary["blocked_by_unacknowledged"] = bool(
        summary["critical"]
        and summary["eligible_remaining"] == 0
        and (backlog.get("unacked_snapshots", 0) or backlog.get("unacked_videos", 0))
    )
    if summary["critical"]:
        log.error(
            "CRITICAL storage pressure: usage still above target",
            extra={
                "usage_percent": summary["usage_after_percent"],
                "target_percent": policy.target_percent,
                "eligible_remaining": summary["eligible_remaining"],
                "blocked_by_unacknowledged": summary["blocked_by_unacknowledged"],
                "unacked_snapshots": backlog.get("unacked_snapshots"),
                "unacked_videos": backlog.get("unacked_videos"),
            },
        )
    return _finish(summary)


def _relieve_volume(root: Path, kinds: tuple[_Media, ...], policy: RetentionPolicy, summary: dict) -> dict:
    volume: dict[str, Any] = {
        "path": str(root),
        "media": [m.kind for m in kinds],
        "usage_before_percent": None,
        "usage_after_percent": None,
        "triggered": False,
        "target_reached": None,
        "critical": False,
        "eligible_remaining": 0,
        "bytes_deleted": 0,
        "bytes_released": 0,
    }
    try:
        percent, used_before = _disk_usage(root)
    except OSError as exc:
        log.exception("storage pressure: cannot measure filesystem", extra={"path": str(root)})
        _add_error(summary, f"cannot measure {root}: {exc}")
        volume["critical"] = True
        return volume
    used_now = used_before
    volume["usage_before_percent"] = volume["usage_after_percent"] = round(percent, 2)
    volume["target_reached"] = percent <= policy.target_percent
    if percent < policy.trigger_percent:
        return volume

    volume["triggered"] = True
    log.warning(
        "storage pressure: cleanup triggered",
        extra={
            "path": str(root),
            "usage_percent": round(percent, 2),
            "trigger_percent": policy.trigger_percent,
        },
    )
    try:
        candidates = sorted(
            ((row["created_ts"], row["event_id"], media) for media in kinds for row in _rows(media, None)),
            key=lambda c: (c[0], c[1], c[2].kind),
        )
    except sqlite3.Error as exc:
        log.exception("storage pressure: cannot list ACKed media")
        _add_error(summary, f"cannot list ACKed media: {exc}")
        candidates = []

    attempted = failed = 0
    for _, event_id, media in candidates:
        if percent <= policy.target_percent:
            break
        attempted += 1
        outcome = _purge(event_id, media, cutoff_ts=None)
        _tally(summary, media, outcome)
        if outcome.status == "failed":
            failed += 1  # still eligible; retried next run
        if outcome.status != "deleted":
            continue
        volume["bytes_deleted"] += outcome.bytes_deleted
        try:
            percent, used_now = _disk_usage(root)  # actual usage, not database sizes
        except OSError as exc:
            log.exception("storage pressure: cannot re-measure filesystem", extra={"path": str(root)})
            _add_error(summary, f"cannot re-measure {root}: {exc}")
            break

    volume["usage_after_percent"] = round(percent, 2)
    volume["bytes_released"] = max(used_before - used_now, 0)
    volume["target_reached"] = percent <= policy.target_percent
    volume["eligible_remaining"] = len(candidates) - attempted + failed
    if volume["bytes_deleted"] and volume["bytes_released"] < volume["bytes_deleted"] // 2:
        _add_error(
            summary,
            f"{root}: deleted {volume['bytes_deleted']} bytes but the filesystem released only "
            f"{volume['bytes_released']} (files still open, or other writers)",
        )
    if not volume["target_reached"]:
        volume["critical"] = True
    log.info(
        "storage pressure: cleanup finished",
        extra={
            "path": str(root),
            "usage_before_percent": volume["usage_before_percent"],
            "usage_after_percent": volume["usage_after_percent"],
            "target_reached": volume["target_reached"],
            "eligible_remaining": volume["eligible_remaining"],
        },
    )
    return volume


def _media_volumes() -> list[tuple[Path, tuple[_Media, ...]]]:
    """Filesystems holding the media directories, each with the media kinds
    stored on it (normally one filesystem holding both)."""
    groups: dict[int, tuple[Path, list[_Media]]] = {}
    for media in _MEDIA:
        directory = Path(getattr(paths, media.dir_attr))
        device = os.stat(directory).st_dev
        groups.setdefault(device, (directory, []))[1].append(media)
    return [(directory, tuple(kinds)) for directory, kinds in groups.values()]


def _disk_usage(path: Path) -> tuple[float, int]:
    """Usage percent as ``df`` reports it (used / (used + available)), and bytes used."""
    usage = shutil.disk_usage(path)
    capacity = usage.used + usage.free
    return (100.0 * usage.used / capacity if capacity else 100.0), usage.used


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _new_summary(run: str, now: float) -> dict[str, Any]:
    return {
        "run": run,
        "started_ts": now,
        "finished_ts": None,
        "enabled": True,
        "disabled_reason": None,
        "snapshots_deleted": 0,
        "videos_deleted": 0,
        "snapshots_missing": 0,
        "videos_missing": 0,
        "refused": 0,
        "failed": 0,
        "bytes_deleted": 0,
        "error_count": 0,
        "errors": [],
        "unacknowledged": None,
    }


def _policy_or_disable(summary: dict[str, Any]) -> RetentionPolicy | None:
    try:
        policy = load_policy()
    except RetentionPolicyError as exc:
        log.error("retention policy invalid; nothing deleted", extra={"error": str(exc)})
        summary["enabled"] = False
        summary["disabled_reason"] = f"invalid retention policy: {exc}"
        summary["policy_error"] = True
        return None
    if not policy.enabled:
        summary["enabled"] = False
        summary["disabled_reason"] = "retention_policy.yaml enabled: false"
        log.info("retention disabled; nothing deleted", extra={"run": summary["run"]})
        return None
    return policy


def _tally(summary: dict[str, Any], media: _Media, outcome: _Outcome) -> None:
    if outcome.status == "deleted":
        summary[f"{media.kind}s_deleted"] += 1
        summary["bytes_deleted"] += outcome.bytes_deleted
    elif outcome.status == "missing":
        summary[f"{media.kind}s_missing"] += 1
    elif outcome.status in ("refused", "failed"):
        summary[outcome.status] += 1
        _add_error(summary, outcome.message or f"{media.kind}: {outcome.status}")
    elif outcome.status == "skipped":
        log.debug("retention: skipped", extra={"reason": outcome.message})


def _add_error(summary: dict[str, Any], message: str) -> None:
    summary["error_count"] += 1
    if len(summary["errors"]) < MAX_REPORTED_ERRORS:
        summary["errors"].append(message)


def _attach_backlog(summary: dict[str, Any], now: float) -> None:
    try:
        summary["unacknowledged"] = get_unacknowledged_backlog(now)
    except (sqlite3.Error, RetentionPolicyError) as exc:
        log.exception("retention: cannot count unacknowledged media")
        _add_error(summary, f"cannot count unacknowledged media: {exc}")


def _finish(summary: dict[str, Any]) -> dict[str, Any]:
    summary["finished_ts"] = time.time()
    backlog = summary.get("unacknowledged") or {}
    log.info(
        f"{summary['run'].replace('_', ' ')} finished",
        extra={
            "run": summary["run"],
            "enabled": summary["enabled"],
            "snapshots_deleted": summary["snapshots_deleted"],
            "videos_deleted": summary["videos_deleted"],
            "snapshots_missing": summary["snapshots_missing"],
            "videos_missing": summary["videos_missing"],
            "refused": summary["refused"],
            "failed": summary["failed"],
            "error_count": summary["error_count"],
            "overdue_unacknowledged": backlog.get("overdue_total"),
        },
    )
    return summary


# ---------------------------------------------------------------------------
# Scheduled entry point: python -m storage.retention_manager daily|pressure
# ---------------------------------------------------------------------------

_STATUS_KEYS = {"daily": "last_scheduled_retention", "pressure": "last_storage_pressure"}


def run_job(job: str, lock_wait_sec: float | None = None) -> int:
    """One locked run of ``job`` ("daily" or "pressure"). Writes the summary
    to paths.RETENTION_STATUS_PATH for the heartbeat. Returns an exit code."""
    if job not in _STATUS_KEYS:
        raise ValueError(f"unknown retention job {job!r}")
    wait = (DAILY_LOCK_WAIT_SEC if job == "daily" else 0.0) if lock_wait_sec is None else lock_wait_sec
    with _run_lock(paths.RETENTION_LOCK_PATH, wait) as acquired:
        if not acquired:
            log.warning("retention: another cleanup run holds the lock; not running", extra={"job": job})
            return EXIT_TEMPFAIL
        log.info("retention job started", extra={"job": job})
        try:
            migrate(init_db())
        except sqlite3.Error:
            log.exception("retention: database unavailable; nothing deleted", extra={"job": job})
            return EXIT_TEMPFAIL
        summary = apply_retention() if job == "daily" else apply_storage_pressure_cleanup()
        _write_status(job, summary)
    if summary.get("policy_error"):
        return EXIT_CONFIG
    return EXIT_OK


def _write_status(job: str, summary: dict[str, Any]) -> None:
    status = read_status_file(paths.RETENTION_STATUS_PATH) or {}
    status[_STATUS_KEYS[job]] = summary
    if summary.get("unacknowledged") is not None:
        status["unacknowledged"] = summary["unacknowledged"]
    status["updated_ts"] = time.time()
    try:
        write_status_file(paths.RETENTION_STATUS_PATH, status)
    except OSError:
        log.exception("retention: cannot write status file", extra={"path": str(paths.RETENTION_STATUS_PATH)})


@contextmanager
def _run_lock(path: Path, wait_sec: float) -> Iterator[bool]:
    """Exclusive lock shared by both jobs, so a daily run and a storage-pressure
    run never overlap. The OS releases it when the process exits, so a crash or
    reboot never leaves a stale lock. Yields whether it was acquired."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as handle:
        deadline = time.monotonic() + wait_sec
        acquired = _try_lock(handle)
        while not acquired and time.monotonic() < deadline:
            time.sleep(1.0)
            acquired = _try_lock(handle)
        try:
            yield acquired
        finally:
            if acquired:
                _unlock(handle)


def _try_lock(handle: IO[bytes]) -> bool:
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _unlock(handle: IO[bytes]) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main(argv: list[str] | None = None) -> int:
    from logger_setup import configure_root

    parser = argparse.ArgumentParser(description="ZOVIVE ACK-based media retention")
    parser.add_argument(
        "job", choices=sorted(_STATUS_KEYS), help="daily: 14-day retention; pressure: 82%% check"
    )
    args = parser.parse_args(argv)
    configure_root("retention")
    return run_job(args.job)


if __name__ == "__main__":
    raise SystemExit(main())
