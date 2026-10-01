"""Entry point for the health/heartbeat process (systemd: zovive-health.service).

Collects a snapshot of queue depths, stream liveness, NPU latency and
disk usage, logs it, records it locally (db.heartbeats — so a run of
missed heartbeats is visible even before a base station exists), and
sends it out via the same base station client transfer_main.py uses.

Deliberately its own process: it must keep reporting even if
edge_main.py's detection loop hangs, which is the whole point of a
heartbeat.
"""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import time

import paths
from config_loader import load
from db.init_db import init_db
from logger_setup import configure_root, get_logger
from network.base_station_client import BaseStationClient, BaseStationError, NullBaseStationClient
from queues.queue_monitor import snapshot as queues_snapshot

log = get_logger(__name__)

_shutdown = False


def _handle_signal(signum, frame):  # noqa: ARG001
    global _shutdown
    _shutdown = True


def _build_client(cfg: dict):
    bs = cfg.get("base_station", {})
    url = bs.get("url", "")
    if not url:
        return NullBaseStationClient()
    return BaseStationClient(url, api_key=bs.get("api_key", ""), timeout_sec=float(bs.get("timeout_sec", 10.0)))


def _pi_throttle_status() -> str | None:
    """Best-effort read of `vcgencmd get_throttled`, matching the field
    report's undervoltage/thermal-throttle investigation. Returns None
    off-Pi (command not found) rather than raising."""
    import subprocess

    try:
        out = subprocess.run(
            ["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (FileNotFoundError, OSError):
        return None


def _disk_usage(path) -> dict:
    total, used, free = shutil.disk_usage(path)
    return {"total_gb": round(total / 1e9, 2), "used_gb": round(used / 1e9, 2), "free_gb": round(free / 1e9, 2)}


def collect_heartbeat(camera_id: str) -> dict:
    return {
        "camera_id": camera_id,
        "ts": time.time(),
        "queues": queues_snapshot(),
        "disk": _disk_usage(paths.DATA_ROOT),
        "throttled": _pi_throttle_status(),
    }


def run(max_cycles: int | None = None) -> None:
    configure_root("health")
    conn = init_db()

    node_cfg = load(paths.NODE_CONFIG).get("node", {})
    camera_id = node_cfg.get("camera_id", "unknown")

    transfer_cfg = load(paths.TRANSFER_CONFIG)
    client = _build_client(transfer_cfg)
    interval = float(load(paths.WATCHDOG_CONFIG).get("watchdog", {}).get("heartbeat_interval_sec", 15))

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    log.info("health_main started", extra={"interval_sec": interval})
    cycles = 0
    while not _shutdown:
        payload = collect_heartbeat(camera_id)

        with conn:
            conn.execute(
                "INSERT INTO heartbeats (camera_id, payload_json, created_ts) VALUES (?, ?, ?)",
                (camera_id, json.dumps(payload, default=str), payload["ts"]),
            )

        log.info("heartbeat", extra={"payload": payload})
        try:
            client.send_heartbeat(payload)
        except BaseStationError as exc:
            log.warning("heartbeat send failed", extra={"error": str(exc)})

        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            break
        time.sleep(interval)

    log.info("health_main stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description="ZOVIVE health/heartbeat process")
    parser.add_argument("--max-cycles", type=int, default=None)
    args = parser.parse_args()
    run(max_cycles=args.max_cycles)


if __name__ == "__main__":
    main()
