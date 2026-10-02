"""Process-level restart record for edge_main.py, kept across restarts in
`paths.RESTART_HISTORY_PATH`.

On start, edge_main calls `record_start()`: it bumps `process_starts` and
works out why the previous run ended. A run that exited through
`record_exit()` left its reason and exit code; a run that never got there
(killed by the systemd watchdog, SIGKILL, power loss, a native crash) is
still marked `running` and is reported as an unclean exit.
"""

from __future__ import annotations

import logging
from pathlib import Path

from watchdog.edge_status import read_status_file, write_status_file

log = logging.getLogger(__name__)

UNCLEAN_EXIT = "unclean exit (killed, systemd watchdog, power loss or native crash)"


def record_start(path: Path, now: float) -> dict:
    prev = read_status_file(path) or {}
    unclean = bool(prev.get("running"))
    record = {
        "process_starts": int(prev.get("process_starts", 0) or 0) + 1,
        "running": True,
        "started_ts": now,
        "last_exit_reason": UNCLEAN_EXIT if unclean else prev.get("last_exit_reason"),
        "last_exit_code": None if unclean else prev.get("last_exit_code"),
        "last_exit_ts": prev.get("last_exit_ts"),
    }
    _write(path, record)
    if unclean:
        log.warning("previous edge_main run ended uncleanly", extra={"process_starts": record["process_starts"]})
    return record


def record_exit(path: Path, record: dict, reason: str, exit_code: int, now: float) -> None:
    record.update(running=False, last_exit_reason=reason, last_exit_code=exit_code, last_exit_ts=now)
    _write(path, record)


def _write(path: Path, record: dict) -> None:
    try:
        write_status_file(path, record)
    except OSError:
        log.exception("failed to write restart history", extra={"path": str(path)})
