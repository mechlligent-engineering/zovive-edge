"""Minimal systemd notify client (READY / WATCHDOG / STOPPING / STATUS).

Implements the sd_notify(3) wire protocol directly: one datagram to the
unix socket named in $NOTIFY_SOCKET. Standard library only, so no
python-systemd package is needed on the Pi.

Every function is a silent no-op when $NOTIFY_SOCKET is unset (laptop,
CI, tests, or a unit that is not Type=notify), so callers never branch on
"am I under systemd". Messages are fixed strings built here; STATUS text
is stripped of newlines so it cannot inject extra key=value lines.
"""

from __future__ import annotations

import logging
import os
import socket

log = logging.getLogger(__name__)


def _socket_address() -> str | None:
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return None
    if addr.startswith("@"):
        return "\0" + addr[1:]  # Linux abstract namespace
    return addr


def notify(message: str) -> bool:
    """Send one notification. Returns True if it was delivered."""
    addr = _socket_address()
    family = getattr(socket, "AF_UNIX", None)
    if addr is None or family is None:
        return False
    try:
        with socket.socket(family, socket.SOCK_DGRAM) as sock:
            sock.connect(addr)
            sock.sendall(message.encode("utf-8"))
        return True
    except OSError as exc:
        log.debug("sd_notify failed", extra={"error": str(exc)})
        return False


def ready() -> bool:
    return notify("READY=1")


def watchdog_ping() -> bool:
    return notify("WATCHDOG=1")


def stopping() -> bool:
    return notify("STOPPING=1")


def status(text: str) -> bool:
    return notify("STATUS=" + " ".join(text.split())[:200])


def watchdog_interval_sec() -> float | None:
    """WatchdogSec from the unit, if systemd enabled it for this process."""
    usec = os.environ.get("WATCHDOG_USEC")
    pid = os.environ.get("WATCHDOG_PID")
    if not usec or (pid and pid != str(os.getpid())):
        return None
    try:
        return int(usec) / 1_000_000
    except ValueError:
        return None
