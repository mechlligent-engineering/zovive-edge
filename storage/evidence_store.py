"""Saves alert snapshots to disk for alert_dispatcher.py, so that a snapshot
only ever appears under its final name complete and verified.

``save_snapshot()``:

1. checks the image and JPEG-encodes it in memory (start/end markers checked);
2. writes the bytes to ``<name>.jpg.tmp`` (exclusive create), flushes and
   fsyncs it, then reads it back: same bytes, and it decodes to the image's
   size;
3. gives it its final ``<name>.jpg`` atomically, refusing to replace an
   existing file, and fsyncs the directory. Where the platform or filesystem
   cannot refuse atomically, the save fails instead (see _move_no_replace).

A failed or interrupted write therefore leaves at most a ``.jpg.tmp`` file,
never a partial ``.jpg``. On failure the temp file this call created is
removed; if even that fails it is kept (and logged) for later quarantine.
Filenames carry the full event_id, and an existing evidence file is never
overwritten.

Deleting snapshots is storage/retention_manager.py's job (ACKed media only).
``max_bytes`` / ``_enforce_cap`` predate it and are off by default.
"""

from __future__ import annotations

import ctypes
import errno
import functools
import logging
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import paths

log = logging.getLogger(__name__)

TEMP_SUFFIX = ".tmp"  # "<name>.jpg.tmp": not a .jpg, so nothing treats it as evidence

_JPEG_SOI = b"\xff\xd8"
_JPEG_EOI = b"\xff\xd9"
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_-]")
_MAX_ID_CHARS = 64  # camera_id / track_id; the event_id is never shortened

# Windows rename() (MoveFileExW without MOVEFILE_REPLACE_EXISTING) refuses an
# existing target; POSIX rename() silently replaces it.
_RENAME_REFUSES_EXISTING = os.name == "nt"
_AT_FDCWD = -100  # Linux <fcntl.h>
_RENAME_NOREPLACE = 1  # Linux <linux/fs.h>


class SnapshotWriteError(OSError):
    """A snapshot could not be saved complete and verified. No final file
    was created for it."""


def _safe_component(value: object, limit: int | None = None) -> str:
    """Only letters, digits, '_' and '-': no path separators, no '..'."""
    text = _UNSAFE_FILENAME_CHARS.sub("-", str(value))[:limit]
    return text or "-"


def _event_filename(camera_id: str, track_id: str, event_id: str, ts: float) -> str:
    t = time.strftime("%Y%m%dT%H%M%S", time.localtime(ts))
    camera = _safe_component(camera_id, _MAX_ID_CHARS)
    track = _safe_component(track_id, _MAX_ID_CHARS)
    return f"{t}_{camera}_{track}_{_safe_component(event_id)}.jpg"


def _validate_image(image: object) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise SnapshotWriteError(f"snapshot image must be a numpy array, got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise SnapshotWriteError(f"snapshot image must be uint8, got {image.dtype}")
    if image.ndim not in (2, 3) or image.shape[0] == 0 or image.shape[1] == 0:
        raise SnapshotWriteError(f"snapshot image has unusable shape {image.shape}")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise SnapshotWriteError(f"snapshot image has {image.shape[2]} channels, expected 1, 3 or 4")
    return image


def _encode_jpeg(image: np.ndarray, quality: int) -> bytes:
    try:
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    except cv2.error as exc:
        raise SnapshotWriteError(f"JPEG encoding failed: {exc}") from exc
    data = buf.tobytes() if ok else b""
    if not (data.startswith(_JPEG_SOI) and data.endswith(_JPEG_EOI)):
        raise SnapshotWriteError("JPEG encoding produced no complete image")
    return data


def _verify_written(path: Path, data: bytes, shape: tuple[int, ...]) -> None:
    """Re-read the file through the OS: the exact bytes encoded (a truncated
    JPEG can still decode, so decoding alone is not enough), and decodable at
    the image's size. Catches short or failed writes, not later media faults."""
    on_disk = path.read_bytes()
    if on_disk != data:
        raise SnapshotWriteError(
            f"{path}: {len(on_disk)} bytes on disk, expected {len(data)} identical bytes"
        )
    decoded = cv2.imdecode(np.frombuffer(on_disk, np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.shape[:2] != shape[:2]:
        raise SnapshotWriteError(f"{path}: saved snapshot does not decode to a {shape[1]}x{shape[0]} image")


@functools.cache
def _renameat2():
    """glibc's renameat2() (glibc 2.28+), or None where there is none."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        func = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    except OSError:
        return None
    if func is not None:
        func.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        func.restype = ctypes.c_int
    return func


def _rename_no_replace(src: Path, dst: Path) -> None:
    """Rename ``src`` to ``dst`` in one atomic step that raises FileExistsError
    rather than replace an existing ``dst``. Where the platform or filesystem
    cannot promise that, raises another OSError and renames nothing."""
    if _RENAME_REFUSES_EXISTING:
        os.rename(src, dst)
        return
    renameat2 = _renameat2()
    if renameat2 is None:
        raise OSError(errno.ENOTSUP, "no rename that refuses an existing file on this platform", str(dst))
    if renameat2(_AT_FDCWD, os.fsencode(src), _AT_FDCWD, os.fsencode(dst), _RENAME_NOREPLACE) != 0:
        err = ctypes.get_errno()
        reason = os.strerror(err)
        if err in (errno.EINVAL, errno.ENOSYS):
            reason = "RENAME_NOREPLACE not supported by this filesystem or kernel"
        raise OSError(err, reason, str(src), None, str(dst))


def _move_no_replace(src: Path, dst: Path) -> None:
    """Give ``src`` the name ``dst`` atomically, raising FileExistsError
    instead of replacing an existing ``dst``, even one another writer creates
    at the same moment: nothing here checks first and renames after.

    A hard link is atomic and never replaces. Where link() fails (e.g. no
    hard links on vfat/exfat), a rename that refuses an existing target:
    Windows rename(), Linux renameat2(RENAME_NOREPLACE). Where neither works
    (another OS, or a filesystem with neither, e.g. some FUSE or SMB mounts)
    this raises OSError and renames nothing: fail closed rather than risk
    replacing evidence. On network filesystems the refusal is only as atomic
    as the server makes it."""
    try:
        os.link(src, dst)
    except FileExistsError:
        raise
    except OSError:
        _rename_no_replace(src, dst)
        return
    try:
        src.unlink()
    except OSError:
        # dst is complete; the leftover name is a second link to the same file.
        log.warning("could not remove temp name after finalizing", extra={"path": str(src)}, exc_info=True)


def _fsync_dir(directory: Path) -> None:
    """Make new directory entries durable. POSIX only: Windows cannot open a
    directory, and NTFS journals the rename itself."""
    if os.name == "nt":
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _discard_temp(tmp_path: Path) -> None:
    """Remove the temp file this write created. Never raises: if removal
    fails the file is kept, and logged, for later quarantine, so the
    original write error is the one the caller sees."""
    try:
        tmp_path.unlink(missing_ok=True)
    except OSError:
        log.error(
            "could not remove incomplete snapshot temp file; kept for later quarantine",
            extra={"path": str(tmp_path)},
            exc_info=True,
        )


class EvidenceStore:
    def __init__(
        self, snapshot_dir: Path | None = None, max_bytes: int | None = None, jpeg_quality: int = 90
    ):
        self.snapshot_dir = snapshot_dir or paths.SNAPSHOT_DIR
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        # None = no cap enforced here (leave it to storage/disk_guard.py in batch 2).
        self.max_bytes = max_bytes
        self.jpeg_quality = jpeg_quality

    def save_snapshot(
        self, image: np.ndarray, camera_id: str, track_id: str, event_id: str, ts: float | None = None
    ) -> Path:
        """Save one event's snapshot and return its final path. Raises
        SnapshotWriteError (an OSError) if it cannot be saved complete and
        verified; no final file exists for it then."""
        ts = ts if ts is not None else time.time()
        data = _encode_jpeg(_validate_image(image), self.jpeg_quality)
        out_path = self.snapshot_dir / _event_filename(camera_id, track_id, event_id, ts)
        # Early refusal only; _move_no_replace is what guarantees no overwrite.
        if os.path.lexists(out_path):
            raise SnapshotWriteError(f"refusing to overwrite existing evidence file {out_path}")
        tmp_path = out_path.with_name(out_path.name + TEMP_SUFFIX)
        try:
            tmp = open(tmp_path, "xb")  # exclusive: never reuses someone else's temp file
        except OSError as exc:
            raise SnapshotWriteError(f"cannot create temp file {tmp_path}: {exc}") from exc

        # From here on tmp_path is this write's own file: remove it on any failure.
        try:
            with tmp:
                tmp.write(data)
                tmp.flush()
                os.fsync(tmp.fileno())
            _verify_written(tmp_path, data, image.shape)
            _move_no_replace(tmp_path, out_path)
        except FileExistsError as exc:
            _discard_temp(tmp_path)
            raise SnapshotWriteError(f"refusing to overwrite existing evidence file {out_path}") from exc
        except SnapshotWriteError:
            _discard_temp(tmp_path)
            raise
        except OSError as exc:
            _discard_temp(tmp_path)
            raise SnapshotWriteError(f"failed to write snapshot {out_path}: {exc}") from exc
        except BaseException:
            _discard_temp(tmp_path)
            raise

        try:
            _fsync_dir(self.snapshot_dir)
        except OSError:
            # The file is complete and verified under its final name; only the
            # durability of the new directory entry across power loss is unconfirmed.
            log.error(
                "snapshot saved but its directory could not be fsynced",
                extra={"path": str(out_path)},
                exc_info=True,
            )
        if self.max_bytes:
            self._enforce_cap()
        return out_path

    def _enforce_cap(self) -> None:
        files = sorted(
            (p for p in self.snapshot_dir.glob("*.jpg") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
        )
        total = sum(p.stat().st_size for p in files)
        i = 0
        while total > self.max_bytes and i < len(files):
            p = files[i]
            try:
                size = p.stat().st_size
                p.unlink()
                total -= size
                log.info("evicted snapshot to stay under disk cap", extra={"path": str(p)})
            except OSError:
                log.exception("failed to evict snapshot", extra={"path": str(p)})
            i += 1

    def usage_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.snapshot_dir.glob("*.jpg") if p.is_file())
