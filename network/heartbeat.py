"""Heartbeat summary fields: build and validate them. Nothing else.

health_main.py owns the heartbeat: it collects disk usage, throttle flags and
edge_main's health, stores the result and sends it. This module only turns
what health_main already collected into six standard summary fields, which
health_main appends to its existing payload::

    from network.heartbeat import build_heartbeat_payload, validate_heartbeat_payload

    summary = build_heartbeat_payload(edge, disk, uptime_sec=..., temperature_c=...)
    # {"node_id": "Node-001", "status": "ok", "uptime_sec": 12345,
    #  "storage_free_gb": 180.2, "temperature_c": 54.1, "last_detection_ts": 1760012345.2}
    errors = validate_heartbeat_payload(summary)   # [] when valid

Nothing here reads disks, sensors, /proc, config or hardware: values come in
as arguments, and the node id from network.node_identity.get_identity().

status:
    down      edge_main is not alive, or no edge health is available
    starting  inference or tracker loop has not run yet (model load, warmup)
    ok        inference, tracker and camera stream are all ok
    degraded  anything else (camera stalled, tracker frozen, inference error, ...)
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any

from network.node_identity import get_identity

log = logging.getLogger(__name__)

_STATUSES = ("ok", "degraded", "starting", "down")

_REQUIRED_KEYS = (
    "node_id",
    "status",
    "uptime_sec",
    "storage_free_gb",
    "temperature_c",
    "last_detection_ts",
)


def build_heartbeat_payload(
    edge_health: Mapping[str, Any] | None,
    disk_status: Mapping[str, Any] | None,
    uptime_sec: int | None = None,
    temperature_c: float | None = None,
    last_detection_ts: float | None = None,
) -> dict[str, Any]:
    """The six summary fields, from data health_main already collected.

    ``edge_health`` is ``watchdog.edge_status.evaluate_edge_health()`` output
    and ``disk_status`` is ``health_main._disk_usage()`` output (either may be
    ``None``). ``last_detection_ts`` defaults to the edge health's
    ``last_detection_timestamp``. Never raises: one bad field must not cost
    the base station the whole heartbeat.
    """
    edge = edge_health if isinstance(edge_health, Mapping) else {}
    disk = disk_status if isinstance(disk_status, Mapping) else {}
    if last_detection_ts is None:
        last_detection_ts = edge.get("last_detection_timestamp")
    return {
        "node_id": _node_id(),
        "status": _status(edge_health),
        "uptime_sec": uptime_sec,
        "storage_free_gb": disk.get("free_gb"),
        "temperature_c": temperature_c,
        "last_detection_ts": last_detection_ts,
    }


def validate_heartbeat_payload(payload: Any) -> list[str]:
    """Problems found in ``payload``; an empty list means valid. Never raises.

    Keys other than the six summary fields are ignored, so the full payload
    health_main sends can be checked as well.
    """
    try:
        return _validate(payload)
    except Exception:  # a validator must never take down the caller
        log.exception("heartbeat payload validation failed unexpectedly")
        return ["internal validation error"]


def _node_id() -> str | None:
    try:
        return get_identity()["node_id"]
    except Exception:  # get_identity() is built not to raise; this is the backstop
        log.exception("heartbeat: node identity unavailable")
        return None


def _status(edge_health: Any) -> str:
    if not isinstance(edge_health, Mapping) or edge_health.get("edge_process_alive") is not True:
        return "down"
    inference = edge_health.get("inference_status")
    tracker = edge_health.get("tracker_status")
    camera = edge_health.get("camera_stream_status")
    if "starting" in (inference, tracker):
        return "starting"
    if inference == tracker == camera == "ok":
        return "ok"
    return "degraded"


def _validate(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return ["payload must be a dictionary"]

    errors = [f"missing {key}" for key in _REQUIRED_KEYS if key not in payload]

    if "node_id" in payload:
        node_id = payload["node_id"]
        if not isinstance(node_id, str) or not node_id.strip():
            errors.append("invalid node_id")
    if "status" in payload and payload["status"] not in _STATUSES:
        errors.append("invalid status")

    numeric_rules = (
        # key, integer only, must be >= 0
        ("uptime_sec", True, True),
        ("storage_free_gb", False, True),
        ("temperature_c", False, False),
        ("last_detection_ts", False, False),
    )
    for key, integer_only, non_negative in numeric_rules:
        if key in payload and not _is_number_or_none(
            payload[key], integer_only=integer_only, non_negative=non_negative
        ):
            errors.append(f"invalid {key}")
    return errors


def _is_number_or_none(value: Any, *, integer_only: bool, non_negative: bool) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):  # bool is an int subclass, but True is not an uptime
        return False
    if not isinstance(value, int if integer_only else (int, float)):
        return False
    if isinstance(value, float) and not math.isfinite(value):  # NaN/Infinity are not valid JSON
        return False
    return not (non_negative and value < 0)
