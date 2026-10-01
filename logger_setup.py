"""Structured JSON logging for every ZOVIVE process.

Design goals (see PATROL_AND_TRIGGER / RUNBOOK discussion):
  * One JSON object per line, so `journalctl -o cat | jq` and simple
    `grep` both work without a parser.
  * Rotates by size so a stuck camera can't fill the SD card / SSD.
  * Every log call can carry structured context (camera_id, track_id,
    event_id) via `extra={...}` without changing the message string,
    so log lines stay greppable and machine-parseable at once.
  * Also mirrors to stdout so `journalctl -u zovive-detect` shows
    the same lines systemd captures.

Usage:
    from logger_setup import get_logger
    log = get_logger(__name__, camera_id="cam01")
    log.info("track confirmed", extra={"track_id": "t-42", "species": "tiger"})
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import time
from pathlib import Path
from typing import Any

import paths

# Attributes that come standard on every LogRecord; anything else set via
# `extra=` is application context and gets folded into the JSON payload.
_STANDARD_ATTRS = frozenset(logging.LogRecord(
    "", 0, "", 0, "", (), None
).__dict__.keys()) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Renders each LogRecord as a single JSON line."""

    def __init__(self, service: str):
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(record.created)
            )
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "service": self.service,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Fold in any structured context passed via extra=.
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        try:
            return json.dumps(payload, default=str)
        except (TypeError, ValueError):
            payload["msg"] = str(payload.get("msg"))
            return json.dumps(payload, default=str)


class ContextAdapter(logging.LoggerAdapter):
    """Merges logger-level default context (e.g. camera_id) into every call."""

    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        merged = {**self.extra, **extra}
        kwargs["extra"] = merged
        return msg, kwargs


_configured_services: set[str] = set()


def configure_root(
    service: str,
    level: int = logging.INFO,
    log_dir: Path | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    to_stdout: bool = True,
) -> None:
    """Attach rotating-file + stdout JSON handlers to the root logger.

    Call this once near process start (edge_main.py, transfer_main.py,
    health_main.py each call it with their own `service` name so log
    lines are attributable even when journald interleaves them).
    Safe to call more than once for the same service (no-op after first).
    """
    if service in _configured_services:
        return
    _configured_services.add(service)

    log_dir = log_dir or paths.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    formatter = JsonFormatter(service)

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / f"{service}.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    if to_stdout:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)


def get_logger(name: str, **context: Any) -> logging.LoggerAdapter:
    """Return a logger that always includes `context` (e.g. camera_id=...)."""
    return ContextAdapter(logging.getLogger(name), context)
