"""Deletes a Pi-local evidence file (snapshot JPEG or event MP4) once
the base station has ACKed receiving it.

Pi storage is meant to be temporary; the base station holds the
permanent copy (see docs/ARCHITECTURE.md's "ACK-triggered cleanup"
section). `transfer_main.py`'s send loops call `delete_local_file()`
right after `mark_sent()`/`mark_video_sent()` succeed — i.e. only once
the base station has actually returned success for that artifact,
never speculatively and never on a failed/retrying send (a failed send
already leaves the file in place and retries, unchanged from before
this module existed).

One function reused for both snapshot and video, not two near-identical
copies: "delete this local file now that a remote copy exists" is
exactly the same operation regardless of which kind of file it is.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def delete_local_file(path: str | Path) -> bool:
    """Best-effort delete. Returns True if a file was actually removed.
    Returns False (never raises) if the file was already gone or
    couldn't be removed — a local cleanup problem must never turn an
    already-ACKed, already-durable-on-the-base-station event into a
    retry-worthy failure; it's logged instead."""
    p = Path(path)
    try:
        p.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        log.exception("failed to delete local evidence file after ack", extra={"path": str(p)})
        return False
