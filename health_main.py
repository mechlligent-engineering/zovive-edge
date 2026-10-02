"""Entry point for the health/heartbeat process (systemd: zovive-health.service).

Every `heartbeat_interval_sec` (default 15 s) it collects disk usage, the
Pi's throttle flags and edge_main.py's own health (read from the status
file edge_main writes, watchdog/edge_status.py), stores the heartbeat in
the local `heartbeats` table, and sends it to the base station.

Deliberately its own process: it must keep reporting even if
edge_main.py's detection loop hangs, which is the whole point of a
heartbeat. For the same reason one cycle must never take the process
down: every collector, the DB write and the send each fail on their own,
and a failure becomes a field in the heartbeat rather than a crash.

Outage behaviour: each cycle sends only the current heartbeat. Heartbeats
from an outage stay in the local table (7 days) for diagnostics and are
never replayed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sqlite3

# nosec B404: used only to run vcgencmd by absolute path with constant
# arguments and no shell; see _pi_throttle_status.
import subprocess  # nosec B404
import threading
import time

import paths
from db.init_db import init_db
from logger_setup import configure_root, get_logger
from network.base_station_client import BaseStationError, build_client
from utils.config_loader import load
from watchdog.edge_status import evaluate_edge_health, read_status_file

log = get_logger(__name__)

_shutdown_event = threading.Event()

MIN_INTERVAL_SEC = 5.0
MAX_INTERVAL_SEC = 3600.0
PRUNE_EVERY_SEC = 3600.0
# Log every Nth consecutive send failure (~5 min at 15 s) instead of every one.
SEND_FAILURE_LOG_EVERY = 20

# Fixed locations only (never PATH lookup), so nothing in the environment
# can substitute a different program. Pi OS ships it in /usr/bin.
_VCGENCMD_CANDIDATES = ("/usr/bin/vcgencmd", "/opt/vc/bin/vcgencmd")
_VCGENCMD_TIMEOUT_SEC = 2.0

# `vcgencmd get_throttled` bit meanings (Raspberry Pi firmware documentation).
_THROTTLE_BITS = {
    0: "under_voltage_now",
    1: "freq_capped_now",
    2: "throttled_now",
    3: "soft_temp_limit_now",
    16: "under_voltage_occurred",
    17: "freq_capped_occurred",
    18: "throttled_occurred",
    19: "soft_temp_limit_occurred",
}


def _handle_signal(signum, frame):  # noqa: ARG001
    # Only sets a flag: an in-progress DB transaction always completes.
    _shutdown_event.set()


def _find_vcgencmd() -> str | None:
    for candidate in _VCGENCMD_CANDIDATES:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _decode_throttled(raw: str) -> dict[str, bool] | None:
    """'throttled=0x50005' -> {'under_voltage_now': True, ...}; None if unparseable."""
    _, _, value = raw.partition("=")
    try:
        bits = int(value.strip(), 16)
    except ValueError:
        return None
    return {name: bool(bits & (1 << bit)) for bit, name in _THROTTLE_BITS.items()}


def _pi_throttle_status() -> dict:
    """Under-voltage / thermal-throttle flags from the Pi firmware.

    Returns {"raw", "flags", "error"}; error is None, "not_available"
    (off-Pi), "timeout" or "failed". Never raises: a wedged firmware
    mailbox must not stop the heartbeat that would report it.
    """
    exe = _find_vcgencmd()
    if exe is None:
        return {"raw": None, "flags": None, "error": "not_available"}
    try:
        # nosec B603: absolute executable path from a fixed list, constant
        # arguments, shell=False; no external input reaches this call.
        out = subprocess.run(  # nosec B603
            [exe, "get_throttled"],
            capture_output=True,
            text=True,
            timeout=_VCGENCMD_TIMEOUT_SEC,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"raw": None, "flags": None, "error": "timeout"}
    except OSError:
        return {"raw": None, "flags": None, "error": "failed"}
    if out.returncode != 0:
        log.debug("vcgencmd failed", extra={"returncode": out.returncode, "stderr": out.stderr[:200]})
        return {"raw": None, "flags": None, "error": "failed"}
    raw = out.stdout.strip()
    return {"raw": raw, "flags": _decode_throttled(raw), "error": None}


def _disk_usage(path) -> dict | None:
    try:
        total, used, free = shutil.disk_usage(path)
    except OSError:
        log.warning("disk usage unavailable", extra={"path": str(path)})
        return None
    return {
        "total_gb": round(total / 1e9, 2),
        "used_gb": round(used / 1e9, 2),
        "free_gb": round(free / 1e9, 2),
        "used_percent": round(100.0 * used / total, 1) if total else None,
    }


def collect_heartbeat(camera_id: str, stale_after_sec: float = 30.0, now: float | None = None) -> dict:
    now = now if now is not None else time.time()
    edge_raw = read_status_file(paths.EDGE_STATUS_PATH)
    throttle = _pi_throttle_status()
    return {
        "camera_id": camera_id,
        "ts": now,
        # Detection process's queue stats (this process has none of its own).
        "queues": (edge_raw or {}).get("queues", {}),
        "disk": _disk_usage(paths.DATA_ROOT),
        # Raw string kept under its original key for existing consumers.
        "throttled": throttle["raw"],
        "throttle": {"flags": throttle["flags"], "error": throttle["error"]},
        "edge": evaluate_edge_health(edge_raw, now, stale_after_sec),
    }


def _store_heartbeat(conn: sqlite3.Connection, camera_id: str, payload: dict) -> bool:
    try:
        with conn:
            conn.execute(
                "INSERT INTO heartbeats (camera_id, payload_json, created_ts) VALUES (?, ?, ?)",
                (camera_id, json.dumps(payload, default=str), payload["ts"]),
            )
        return True
    except sqlite3.Error as exc:
        # Locked past busy_timeout, or disk full. Still send the heartbeat:
        # this is exactly the state the base station needs to hear about.
        log.error("heartbeat not stored locally", extra={"error": str(exc)})
        return False


def prune_heartbeats(conn: sqlite3.Connection, retention_days: float, now: float | None = None) -> int:
    """Delete local heartbeat rows older than `retention_days`. Events,
    snapshots and clips live elsewhere and are never touched."""
    cutoff = (now if now is not None else time.time()) - retention_days * 86400.0
    try:
        with conn:
            deleted = conn.execute("DELETE FROM heartbeats WHERE created_ts < ?", (cutoff,)).rowcount
    except sqlite3.Error as exc:
        log.warning("heartbeat pruning failed", extra={"error": str(exc)})
        return 0
    if deleted:
        log.info("pruned old heartbeats", extra={"deleted": deleted, "retention_days": retention_days})
    return deleted


class _SendLog:
    """Logs the first failure, then every Nth, then the recovery, so days
    of outage produce a few hundred lines instead of thousands."""

    def __init__(self, every: int = SEND_FAILURE_LOG_EVERY):
        self.every = every
        self.consecutive_failures = 0

    def failed(self, exc: Exception) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures == 1 or self.consecutive_failures % self.every == 0:
            log.warning(
                "heartbeat send failed",
                extra={"error": str(exc), "consecutive_failures": self.consecutive_failures},
            )

    def succeeded(self) -> None:
        if self.consecutive_failures:
            log.info("heartbeat send recovered", extra={"after_failures": self.consecutive_failures})
        self.consecutive_failures = 0


def _clamp_interval(value: float) -> float:
    clamped = min(max(value, MIN_INTERVAL_SEC), MAX_INTERVAL_SEC)
    if clamped != value:
        log.warning("heartbeat_interval_sec out of range; clamped", extra={"configured": value, "used": clamped})
    return clamped


def run_cycle(conn, client, camera_id: str, stale_after_sec: float, send_log: _SendLog) -> dict:
    payload = collect_heartbeat(camera_id, stale_after_sec)
    payload["db_write_ok"] = _store_heartbeat(conn, camera_id, payload)

    edge = payload["edge"]
    log.info(
        "heartbeat",
        extra={
            "edge_alive": edge["edge_process_alive"],
            "inference": edge["inference_status"],
            "tracker": edge["tracker_status"],
            "camera": edge["camera_stream_status"],
            "thread_restarts": edge["restarts"]["thread_restarts"],
            "process_starts": edge["restarts"]["process_starts"],
            "free_gb": (payload["disk"] or {}).get("free_gb"),
            "throttled": payload["throttled"],
            "db_write_ok": payload["db_write_ok"],
        },
    )
    log.debug("heartbeat payload", extra={"payload": payload})

    try:
        client.send_heartbeat(payload)
        send_log.succeeded()
    except BaseStationError as exc:
        send_log.failed(exc)
    return payload


def run(max_cycles: int | None = None) -> None:
    configure_root("health")
    conn = init_db()

    node_cfg = load(paths.NODE_CONFIG).get("node", {})
    camera_id = node_cfg.get("camera_id", "unknown")

    client = build_client(load(paths.TRANSFER_CONFIG))
    watchdog_cfg = load(paths.WATCHDOG_CONFIG, required=False).get("watchdog", {})
    interval = _clamp_interval(float(watchdog_cfg.get("heartbeat_interval_sec", 15)))
    retention_days = float(watchdog_cfg.get("heartbeat_retention_days", 7))
    stale_after_sec = float(watchdog_cfg.get("edge_status_stale_sec", 30))

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    log.info(
        "health_main started",
        extra={"interval_sec": interval, "retention_days": retention_days, "stale_after_sec": stale_after_sec},
    )
    send_log = _SendLog()
    next_prune = 0.0
    cycles = 0
    while not _shutdown_event.is_set():
        try:
            if time.monotonic() >= next_prune:
                prune_heartbeats(conn, retention_days)
                next_prune = time.monotonic() + PRUNE_EVERY_SEC
            run_cycle(conn, client, camera_id, stale_after_sec, send_log)
        except Exception:
            # Last resort: one bad cycle must not stop the heartbeat process.
            log.exception("heartbeat cycle failed")

        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            break
        _shutdown_event.wait(interval)  # returns at once on SIGTERM

    log.info("health_main stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description="ZOVIVE health/heartbeat process")
    parser.add_argument("--max-cycles", type=int, default=None)
    args = parser.parse_args()
    run(max_cycles=args.max_cycles)


if __name__ == "__main__":
    main()
