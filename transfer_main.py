"""Entry point for the transfer process (systemd: zovive-transfer.service).

Sends the SQLite outbox to the base station: image alerts and event clips,
each from its own queue (db/outbox.py claim_due). The retry state lives on
the outbox rows, so it survives restarts and several senders can share the
queues safely; network/retry_policy.py has the rules (an HTTP 4xx rejection
costs the item one of its retries, an unreachable or busy base station pauses
the queue and is probed about once a minute at no cost). Runs as its own
process/systemd unit, deliberately separate from edge_main.py's detection
loop, so a slow or offline base station can never stall capture or inference.

    python transfer_main.py [--max-cycles N]
    python transfer_main.py --requeue-dead   # retry dead-lettered uploads afresh
"""

from __future__ import annotations

import argparse
import random
import signal
import time
from collections.abc import Callable

import paths
from db.init_db import init_db
from db.migrations import migrate
from db.outbox import (
    QUEUE_NAMES,
    claim_due,
    defer_due,
    mark_sent,
    mark_snapshot_deleted,
    mark_video_deleted,
    mark_video_sent,
    record_send_failure,
    release_claim,
    requeue_dead,
)
from logger_setup import configure_root, get_logger
from network.base_station_client import (
    BaseStationError,
    BaseStationRejected,
    BaseStationUnreachable,
    build_client,
)
from network.retry_policy import BEHIND_SEC, RetryPolicy
from storage.local_file_cleanup import delete_local_file
from utils.config_loader import load

log = get_logger(__name__)

_shutdown = False

# Retry jitter only, not security.
_RNG = random.Random()  # nosec B311

# Attempt counter column of each queue (charged rejections so far).
_ATTEMPTS = {"image": "attempt_count", "video": "video_attempt_count"}


def _handle_signal(signum, frame):  # noqa: ARG001
    global _shutdown
    _shutdown = True


def _build_client(cfg: dict):
    if not cfg.get("base_station", {}).get("url", ""):
        log.warning("transfer_config.base_station.url is empty; using NullBaseStationClient (logs only)")
    return build_client(cfg)


def _lease_sec(timeout_sec: float) -> float:
    """How long a claimed row stays hidden from other senders: longer than
    one send can take, so a live sender's claim never runs out mid-send."""
    return 2 * timeout_sec + 30.0


def _event_to_payload(row: dict) -> dict:
    return {
        "event_id": row["event_id"],
        "track_id": row["track_id"],
        "camera_id": row["camera_id"],
        "camera_name": row["camera_name"],
        "forest_name": row["forest_name"],
        "gps_lat": row["gps_lat"],
        "gps_lon": row["gps_lon"],
        "animal_name": row["animal_name"],
        "confidence": row["confidence"],
        "model_version": row["model_version"],
        "priority": row["priority"],
        "created_ts": row["created_ts"],
    }


def _record_failure(
    queue: str,
    row: dict,
    claim: str,
    exc: BaseStationError,
    policy: RetryPolicy,
    now: float,
    rng: random.Random,
) -> float | None:
    """Record a failed send. Returns when the queue's pause ends if the base
    station is unreachable or busy (the caller stops its batch), else None."""
    event_id = row["event_id"]
    if isinstance(exc, BaseStationRejected):
        rejections = row[_ATTEMPTS[queue]] + 1
        retry_at = policy.retry_at(now, rejections, rng)
        record_send_failure(queue, event_id, claim, now=now, error=str(exc), retry_at=retry_at, charged=True)
        extra = {"queue": queue, "event_id": event_id, "rejections": rejections, "error": str(exc)}
        if retry_at is None:
            log.error("upload rejected too often; dead-lettered", extra=extra)
        else:
            log.warning(
                "upload rejected, will retry", extra={**extra, "retry_in_sec": round(retry_at - now, 1)}
            )
        return None

    until = policy.pause_until(now, rng, retry_after=exc.retry_after)
    # Not charged. This item goes just behind the rest, so the next probe
    # tries a different one and a single always-failing item cannot block
    # the queue.
    record_send_failure(
        queue, event_id, claim, now=now, error=str(exc), retry_at=until + BEHIND_SEC, charged=False
    )
    paused = QUEUE_NAMES if isinstance(exc, BaseStationUnreachable) else (queue,)
    for name in paused:
        defer_due(name, until=until, max_retries=policy.max_retries)
    log.warning(
        "base station unreachable" if isinstance(exc, BaseStationUnreachable) else "base station busy",
        extra={
            "queues": list(paused),
            "pause_sec": round(until - now, 1),
            "event_id": event_id,
            "error": str(exc),
        },
    )
    return until


def _send_pending_images(
    client,
    policy: RetryPolicy,
    batch_size: int,
    delete_after_ack: bool = False,
    record_ack: bool | None = None,
    *,
    lease_sec: float = _lease_sec(10.0),
    clock: Callable[[], float] = time.time,
    rng: random.Random = _RNG,
) -> bool:
    """Send due image alerts. ``record_ack``: the client is a real base
    station, so a successful send of a request carrying the snapshot is
    recorded as that snapshot's ACK (db/outbox.py snapshot_acked_ts).
    Defaults to ``delete_after_ack``, which run() only enables with a real
    base station. Local deletion always requires that ACK. Returns True if
    the queue paused (base station unreachable or busy)."""
    if record_ack is None:
        record_ack = delete_after_ack
    claim, rows = claim_due(
        "image",
        now=clock(),
        limit=batch_size,
        lease_sec=lease_sec,
        max_retries=policy.max_retries,
        max_wait_sec=policy.max_wait_sec,
    )
    for row in rows:
        event_id = row["event_id"]
        snapshot_path = row["snapshot_path"]
        snapshot_bytes = None
        try:
            with open(snapshot_path, "rb") as f:
                snapshot_bytes = f.read()
        except OSError:
            log.warning("snapshot file missing, sending metadata only", extra={"event_id": event_id})

        try:
            client.send_event(_event_to_payload(row), snapshot_bytes=snapshot_bytes)
        except BaseStationError as exc:
            # No ACK -> the file stays (nothing deleted below).
            until = _record_failure("image", row, claim, exc, policy, clock(), rng)
            if until is not None:
                release_claim("image", claim, until=until)
                return True
            continue

        # A metadata-only send (snapshot unreadable) is not an ACK of the
        # snapshot: the base station never got the image.
        snapshot_acked = record_ack and snapshot_bytes is not None
        if not mark_sent(event_id, snapshot_acked=snapshot_acked, claim=claim):
            log.warning(
                "event sent after its claim expired; left to the current claim", extra={"event_id": event_id}
            )
            continue
        log.info("event sent", extra={"event_id": event_id, "snapshot_acked": snapshot_acked})

        # ACK recorded (same UPDATE as 'sent') -> the local copy may go
        # now if delete-right-after-ACK is enabled. If recording failed,
        # mark_sent raised and nothing below runs: the file stays.
        if delete_after_ack and snapshot_acked:
            if delete_local_file(snapshot_path):
                mark_snapshot_deleted(event_id)
                log.info("local snapshot deleted after ack", extra={"event_id": event_id})
    return False


def _send_pending_videos(
    client,
    policy: RetryPolicy,
    batch_size: int,
    timeout_sec: float,
    delete_after_ack: bool = False,
    record_ack: bool | None = None,
    *,
    lease_sec: float | None = None,
    clock: Callable[[], float] = time.time,
    rng: random.Random = _RNG,
) -> bool:
    """Send due clips; ``record_ack`` and the return value as in
    _send_pending_images."""
    if record_ack is None:
        record_ack = delete_after_ack
    claim, rows = claim_due(
        "video",
        now=clock(),
        limit=batch_size,
        lease_sec=lease_sec if lease_sec is not None else _lease_sec(timeout_sec),
        max_retries=policy.max_retries,
        max_wait_sec=policy.max_wait_sec,
    )
    for row in rows:
        event_id = row["event_id"]
        video_path = row["video_path"]
        try:
            with open(video_path, "rb") as f:
                video_bytes = f.read()
        except OSError as exc:
            # No pre-roll/post-roll frames, disk full at write time, etc.
            # — the clip itself is gone, retrying won't produce it, so
            # dead-letter it rather than retry forever.
            record_send_failure(
                "video",
                event_id,
                claim,
                now=clock(),
                error=f"video file missing: {exc}",
                retry_at=None,
                charged=True,
            )
            log.warning("video file missing, dead-lettered", extra={"event_id": event_id})
            continue

        try:
            client.send_video(event_id, video_bytes, timeout_sec=timeout_sec)
        except BaseStationError as exc:
            # No ACK -> the file stays (nothing deleted below).
            until = _record_failure("video", row, claim, exc, policy, clock(), rng)
            if until is not None:
                release_claim("video", claim, until=until)
                return True
            continue

        if not mark_video_sent(event_id, acked=record_ack, claim=claim):
            log.warning(
                "video sent after its claim expired; left to the current claim", extra={"event_id": event_id}
            )
            continue
        log.info(
            "video sent",
            extra={"event_id": event_id, "video_bytes": len(video_bytes), "video_acked": record_ack},
        )

        # ACK recorded -> delete the local clip; see _send_pending_images.
        if delete_after_ack and record_ack:
            if delete_local_file(video_path):
                mark_video_deleted(event_id)
                log.info("local video deleted after ack", extra={"event_id": event_id})
    return False


def run(max_cycles: int | None = None) -> None:
    configure_root("transfer")
    conn = init_db()
    migrate(conn)

    cfg = load(paths.TRANSFER_CONFIG)
    client = _build_client(cfg)
    t = cfg.get("transfer", {})
    poll_interval = float(t.get("poll_interval_sec", 5.0))
    batch_size = int(t.get("batch_size", 10))
    policy = RetryPolicy.from_config(t)
    image_lease_sec = _lease_sec(float(cfg.get("base_station", {}).get("timeout_sec", 10.0)))

    video_upload_enabled = bool(t.get("video_upload_enabled", True))
    video_batch_size = int(t.get("video_batch_size", 3))
    video_upload_timeout_sec = float(t.get("video_upload_timeout_sec", 60.0))

    # Pi storage is meant to be temporary once a real base station has
    # ACKed an artifact — but with no base station configured (empty
    # base_station.url -> NullBaseStationClient, which "succeeds"
    # unconditionally so dev/test setups still exercise the full
    # pipeline), there is no real remote copy to fall back on, so
    # deletion is forced off regardless of the config value. Only a
    # real, configured base station can ever trigger a local delete.
    base_station_configured = bool(cfg.get("base_station", {}).get("url"))
    delete_local_files_after_ack = (
        bool(t.get("delete_local_files_after_ack", True)) and base_station_configured
    )
    if bool(t.get("delete_local_files_after_ack", True)) and not base_station_configured:
        log.warning(
            "delete_local_files_after_ack is on but no base_station.url is configured; "
            "forcing it off so evidence is never deleted without a real remote copy"
        )

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    log.info(
        "transfer_main started",
        extra={
            "poll_interval_sec": poll_interval,
            "video_upload_enabled": video_upload_enabled,
            "delete_local_files_after_ack": delete_local_files_after_ack,
            "retry_schedule_sec": list(policy.schedule_sec),
            "max_retries": policy.max_retries,
        },
    )
    cycles = 0
    while not _shutdown:
        # Only a real base station can ACK media; the Null client never does.
        _send_pending_images(
            client,
            policy,
            batch_size,
            delete_local_files_after_ack,
            record_ack=base_station_configured,
            lease_sec=image_lease_sec,
        )
        if video_upload_enabled:
            _send_pending_videos(
                client,
                policy,
                video_batch_size,
                video_upload_timeout_sec,
                delete_local_files_after_ack,
                record_ack=base_station_configured,
            )

        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            break
        time.sleep(poll_interval)

    log.info("transfer_main stopped")


def requeue_dead_uploads() -> dict[str, int]:
    """Give every dead-lettered upload a fresh retry budget, due now."""
    configure_root("transfer")
    migrate(init_db())
    counts = {queue: requeue_dead(queue) for queue in QUEUE_NAMES}
    log.info("dead-lettered uploads requeued", extra={"requeued": counts})
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="ZOVIVE transfer process")
    parser.add_argument("--max-cycles", type=int, default=None, help="stop after N poll cycles (testing)")
    parser.add_argument(
        "--requeue-dead",
        action="store_true",
        help="retry dead-lettered uploads with a fresh budget, then exit",
    )
    args = parser.parse_args()
    if args.requeue_dead:
        counts = requeue_dead_uploads()
        print(f"requeued: {counts['image']} image alert(s), {counts['video']} video clip(s)")
        return
    run(max_cycles=args.max_cycles)


if __name__ == "__main__":
    main()
