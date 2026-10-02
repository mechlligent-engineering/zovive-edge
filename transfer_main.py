"""Entry point for the transfer process (systemd: zovive-transfer.service).

Polls the SQLite outbox for pending events and sends each one to the
base station, with per-event exponential backoff on failure
(network/transfer_state.py). Runs as its own process/systemd unit,
deliberately separate from edge_main.py's detection loop, so a slow or
offline base station can never stall capture or inference.
"""

from __future__ import annotations

import argparse
import signal
import time

import paths
from config_loader import load
from db.init_db import init_db
from db.migrations import migrate
from db.outbox import (
    get_pending,
    get_pending_videos,
    mark_failed,
    mark_sent,
    mark_snapshot_deleted,
    mark_video_deleted,
    mark_video_failed,
    mark_video_sent,
)
from logger_setup import configure_root, get_logger
from network.base_station_client import BaseStationError, build_client
from network.transfer_state import TransferState
from storage.local_file_cleanup import delete_local_file

log = get_logger(__name__)

_shutdown = False


def _handle_signal(signum, frame):  # noqa: ARG001
    global _shutdown
    _shutdown = True


def _build_client(cfg: dict):
    if not cfg.get("base_station", {}).get("url", ""):
        log.warning("transfer_config.base_station.url is empty; using NullBaseStationClient (logs only)")
    return build_client(cfg)


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


def _send_pending_images(
    client, state: TransferState, batch_size: int, delete_after_ack: bool = False
) -> None:
    pending = get_pending(limit=batch_size)
    for row in pending:
        event_id = row["event_id"]
        if not state.is_ready(event_id):
            continue
        try:
            snapshot_path = row["snapshot_path"]
            snapshot_bytes = None
            try:
                with open(snapshot_path, "rb") as f:
                    snapshot_bytes = f.read()
            except OSError:
                log.warning("snapshot file missing, sending metadata only", extra={"event_id": event_id})

            client.send_event(_event_to_payload(row), snapshot_bytes=snapshot_bytes)
            mark_sent(event_id)
            state.clear(event_id)
            log.info("event sent", extra={"event_id": event_id})

            # ACK received (the send above succeeded) -> Pi storage is
            # temporary, so the local copy can go now that the base
            # station has it. Only if we actually had a file to send in
            # the first place (snapshot_bytes is not None) — nothing to
            # clean up otherwise.
            if delete_after_ack and snapshot_bytes is not None:
                if delete_local_file(snapshot_path):
                    mark_snapshot_deleted(event_id)
                    log.info("local snapshot deleted after ack", extra={"event_id": event_id})
        except BaseStationError as exc:
            # No ACK -> keep the file (nothing deleted above) and retry.
            backoff = state.record_failure(event_id)
            mark_failed(event_id, str(exc))
            log.warning(
                "event send failed, will retry",
                extra={"event_id": event_id, "backoff_sec": round(backoff, 1), "error": str(exc)},
            )


def _send_pending_videos(
    client, video_state: TransferState, batch_size: int, timeout_sec: float, delete_after_ack: bool = False
) -> None:
    # Keyed distinctly from the image alert's TransferState entries
    # (same event_id would otherwise collide two independent retry
    # timelines into one backoff schedule).
    pending = get_pending_videos(limit=batch_size)
    for row in pending:
        event_id = row["event_id"]
        state_key = f"{event_id}:video"
        if not video_state.is_ready(state_key):
            continue
        video_path = row["video_path"]
        try:
            with open(video_path, "rb") as f:
                video_bytes = f.read()
        except OSError as exc:
            # No pre-roll/post-roll frames, disk full at write time, etc.
            # — the clip itself is gone, retrying won't produce it, so
            # fail permanently rather than retry forever.
            mark_video_failed(event_id, f"video file missing: {exc}")
            log.warning("video file missing, giving up on this clip", extra={"event_id": event_id})
            continue

        try:
            client.send_video(event_id, video_bytes, timeout_sec=timeout_sec)
            mark_video_sent(event_id)
            video_state.clear(state_key)
            log.info("video sent", extra={"event_id": event_id, "video_bytes": len(video_bytes)})

            # ACK received -> delete the local clip; see _send_pending_images.
            if delete_after_ack:
                if delete_local_file(video_path):
                    mark_video_deleted(event_id)
                    log.info("local video deleted after ack", extra={"event_id": event_id})
        except BaseStationError as exc:
            # No ACK -> keep the file (nothing deleted above) and retry.
            backoff = video_state.record_failure(state_key)
            mark_video_failed(event_id, str(exc))
            log.warning(
                "video send failed, will retry",
                extra={"event_id": event_id, "backoff_sec": round(backoff, 1), "error": str(exc)},
            )


def run(max_cycles: int | None = None) -> None:
    configure_root("transfer")
    conn = init_db()
    migrate(conn)

    cfg = load(paths.TRANSFER_CONFIG)
    client = _build_client(cfg)
    t = cfg.get("transfer", {})
    poll_interval = float(t.get("poll_interval_sec", 5.0))
    batch_size = int(t.get("batch_size", 10))

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
    delete_local_files_after_ack = bool(t.get("delete_local_files_after_ack", True)) and base_station_configured
    if bool(t.get("delete_local_files_after_ack", True)) and not base_station_configured:
        log.warning(
            "delete_local_files_after_ack is on but no base_station.url is configured; "
            "forcing it off so evidence is never deleted without a real remote copy"
        )

    state = TransferState(
        initial_backoff_sec=float(t.get("retry_initial_backoff_sec", 2.0)),
        max_backoff_sec=float(t.get("retry_max_backoff_sec", 300.0)),
    )
    video_state = TransferState(
        initial_backoff_sec=float(t.get("video_retry_initial_backoff_sec", 5.0)),
        max_backoff_sec=float(t.get("video_retry_max_backoff_sec", 300.0)),
    )

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    log.info(
        "transfer_main started",
        extra={
            "poll_interval_sec": poll_interval,
            "video_upload_enabled": video_upload_enabled,
            "delete_local_files_after_ack": delete_local_files_after_ack,
        },
    )
    cycles = 0
    while not _shutdown:
        _send_pending_images(client, state, batch_size, delete_local_files_after_ack)
        if video_upload_enabled:
            _send_pending_videos(
                client, video_state, video_batch_size, video_upload_timeout_sec, delete_local_files_after_ack
            )

        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            break
        time.sleep(poll_interval)

    log.info("transfer_main stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description="ZOVIVE transfer process")
    parser.add_argument("--max-cycles", type=int, default=None, help="stop after N poll cycles (testing)")
    args = parser.parse_args()
    run(max_cycles=args.max_cycles)


if __name__ == "__main__":
    main()
