"""storage/evidence_store.py: snapshots reach their final name only complete
and verified, never overwrite evidence, and fail cleanly.

Every test writes into its own temporary directory, with paths.SNAPSHOT_DIR
patched there as well; setUp fails the test if it points at the real
location. Disk faults, races and crashes are simulated with mocks.
"""

from __future__ import annotations

import contextlib
import ctypes
import errno
import inspect
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

import paths
from storage import evidence_store
from storage.evidence_store import TEMP_SUFFIX, EvidenceStore, SnapshotWriteError

REAL_SNAPSHOT_DIR = Path(paths.SNAPSHOT_DIR).resolve()
LOGGER = "storage.evidence_store"
EVENT_ID = "3f2b8c1e-9d4a-4f6b-8e2a-1c5d7e9f0a1b"
TS = 1_700_000_000.0


def _image(h: int = 120, w: int = 160, channels: int | None = 3) -> np.ndarray:
    shape = (h, w) if channels is None else (h, w, channels)
    return np.random.default_rng(7).integers(0, 256, shape, dtype=np.uint8)


def _eio() -> OSError:
    return OSError(errno.EIO, "Input/output error")


def _no_hard_links() -> OSError:
    return PermissionError(errno.EPERM, "Operation not permitted")  # link() on vfat/exfat


def _failing_renameat2(err: int):
    """Stands in for libc renameat2(): fails with ``err``, renames nothing."""

    def renameat2(*_args):
        ctypes.set_errno(err)
        return -1

    return renameat2


@contextlib.contextmanager
def _another_writer_before_rename(content: bytes = b"other writer"):
    """Another writer creates the destination immediately before the store's
    rename runs, whichever it uses (os.rename, os.replace or Linux
    renameat2()); the real call then runs on the real filesystem."""
    real_rename, real_replace, real_renameat2 = os.rename, os.replace, evidence_store._renameat2()

    def compete(dst) -> None:
        Path(os.fsdecode(dst)).write_bytes(content)

    def rename(src, dst, **kwargs):
        compete(dst)
        return real_rename(src, dst, **kwargs)

    def replace(src, dst, **kwargs):
        compete(dst)
        return real_replace(src, dst, **kwargs)

    def renameat2(olddirfd, old, newdirfd, new, flags):
        compete(new)
        return real_renameat2(olddirfd, old, newdirfd, new, flags)

    with (
        mock.patch("os.rename", side_effect=rename),
        mock.patch("os.replace", side_effect=replace),
        mock.patch.object(evidence_store, "_renameat2", return_value=renameat2 if real_renameat2 else None),
    ):
        yield


# The two ways a snapshot gets its final name: a hard link, or (link() fails)
# this platform's rename that refuses an existing target.
FINALIZE_PATHS = ("hard link", "no hard links")


def _finalize_via(path: str):
    if path == "no hard links":
        return mock.patch("os.link", side_effect=_no_hard_links())
    return contextlib.nullcontext()


class EvidenceStoreTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.snapshot_dir = self.root / "snapshots"
        patcher = mock.patch.object(paths, "SNAPSHOT_DIR", self.snapshot_dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.assertNotEqual(Path(paths.SNAPSHOT_DIR).resolve(), REAL_SNAPSHOT_DIR)
        self.assertTrue(Path(paths.SNAPSHOT_DIR).resolve().is_relative_to(self.root))

        logger = logging.getLogger(LOGGER)
        handler = logging.NullHandler()
        logger.addHandler(handler)
        self.addCleanup(logger.removeHandler, handler)
        quiet = mock.patch.object(logger, "propagate", False)
        quiet.start()
        self.addCleanup(quiet.stop)

        self.store = EvidenceStore(snapshot_dir=self.snapshot_dir)

    def save(self, event_id: str = EVENT_ID, image: np.ndarray | None = None, **ids) -> Path:
        return self.store.save_snapshot(
            _image() if image is None else image,
            camera_id=ids.get("camera_id", "cam01"),
            track_id=ids.get("track_id", "7"),
            event_id=event_id,
            ts=TS,
        )

    def target(self, event_id: str = EVENT_ID) -> Path:
        return self.snapshot_dir / evidence_store._event_filename("cam01", "7", event_id, TS)

    def files(self) -> list[str]:
        return sorted(p.name for p in self.snapshot_dir.iterdir())


class SaveTests(EvidenceStoreTestCase):
    def test_saves_decodable_jpeg_named_with_full_event_id(self):
        path = self.save()
        self.assertEqual(path, self.target())
        self.assertEqual(path.parent, self.snapshot_dir)
        self.assertEqual(path.suffix, ".jpg")
        self.assertIn(EVENT_ID, path.name)
        self.assertEqual(cv2.imread(str(path)).shape, (120, 160, 3))
        self.assertEqual(self.files(), [path.name], "no temp file may remain")

    def test_existing_interface_is_unchanged(self):
        self.assertEqual(
            list(inspect.signature(EvidenceStore.save_snapshot).parameters),
            ["self", "image", "camera_id", "track_id", "event_id", "ts"],
        )
        self.assertEqual(
            list(inspect.signature(EvidenceStore.__init__).parameters),
            ["self", "snapshot_dir", "max_bytes", "jpeg_quality"],
        )
        store = EvidenceStore(self.snapshot_dir, None, 80)
        positional = store.save_snapshot(_image(), "cam01", "7", "event-positional")
        self.assertIsInstance(positional, Path)
        self.assertTrue(positional.is_file())
        self.assertTrue(issubclass(SnapshotWriteError, OSError), "callers catching OSError keep working")

    def test_supported_image_layouts(self):
        for n, channels in enumerate((None, 1, 3, 4)):
            with self.subTest(channels=channels):
                path = self.save(event_id=f"layout-{n}", image=_image(48, 64, channels))
                decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                self.assertEqual(decoded.shape[:2], (48, 64))

    def test_invalid_images_are_rejected_before_anything_is_written(self):
        cases = {
            "None": None,
            "list": [[1, 2], [3, 4]],
            "empty": np.zeros((0, 0, 3), np.uint8),
            "zero width": np.zeros((10, 0, 3), np.uint8),
            "float": np.zeros((10, 10, 3), np.float32),
            "two channels": np.zeros((10, 10, 2), np.uint8),
            "one dimension": np.zeros(10, np.uint8),
            "four dimensions": np.zeros((2, 10, 10, 3), np.uint8),
        }
        for name, bad in cases.items():
            with self.subTest(name), self.assertRaises(SnapshotWriteError):
                self.store.save_snapshot(bad, "cam01", "7", EVENT_ID, ts=TS)
        self.assertEqual(self.files(), [])

    def test_encoder_failure_is_reported(self):
        with (
            mock.patch.object(cv2, "imencode", return_value=(False, None)),
            self.assertRaises(SnapshotWriteError),
        ):
            self.save()
        self.assertEqual(self.files(), [])

    def test_unsafe_ids_cannot_leave_the_directory(self):
        path = self.save(camera_id="../../etc", track_id="a/b\\c:d")
        self.assertEqual(path.parent, self.snapshot_dir)
        for bad in ("/", "\\", "..", ":"):
            self.assertNotIn(bad, path.name)


class CollisionTests(EvidenceStoreTestCase):
    def test_refuses_to_overwrite_existing_evidence(self):
        target = self.target()
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"earlier evidence")
        with self.assertRaises(SnapshotWriteError):
            self.save()
        self.assertEqual(target.read_bytes(), b"earlier evidence")
        self.assertEqual(self.files(), [target.name])

    def test_same_event_cannot_be_saved_twice(self):
        first = self.save()
        original = first.read_bytes()
        with self.assertRaises(SnapshotWriteError):
            self.save(image=_image(channels=None))
        self.assertEqual(first.read_bytes(), original)
        self.assertEqual(self.files(), [first.name])

    def test_events_sharing_an_id_prefix_get_distinct_files(self):
        # Same camera, track and second: the old 8-character names collided here.
        a = self.save(event_id="aaaaaaaa-0000-4000-8000-000000000000", image=_image(channels=3))
        b = self.save(event_id="aaaaaaaa-1111-4000-8000-000000000000", image=_image(channels=None))
        self.assertNotEqual(a, b)
        self.assertEqual(cv2.imread(str(a), cv2.IMREAD_UNCHANGED).ndim, 3)
        self.assertEqual(cv2.imread(str(b), cv2.IMREAD_UNCHANGED).ndim, 2)

    def test_file_appearing_during_finalize_is_not_replaced(self):
        real_link = os.link

        def racing_link(src, dst):
            Path(dst).write_bytes(b"other writer")
            return real_link(src, dst)

        with mock.patch("os.link", side_effect=racing_link), self.assertRaises(SnapshotWriteError):
            self.save()
        self.assertEqual(self.target().read_bytes(), b"other writer")
        self.assertEqual(self.files(), [self.target().name], "this write's temp file is removed")

    def test_temp_file_of_another_write_is_left_alone(self):
        foreign = self.target().with_name(self.target().name + TEMP_SUFFIX)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        foreign.write_bytes(b"not ours")
        with self.assertRaises(SnapshotWriteError):
            self.save()
        self.assertEqual(foreign.read_bytes(), b"not ours")
        self.assertFalse(self.target().exists())


class FailureTests(EvidenceStoreTestCase):
    def test_fsync_failure_leaves_no_file(self):
        with mock.patch("os.fsync", side_effect=_eio()), self.assertRaises(SnapshotWriteError) as cm:
            self.save()
        self.assertIsInstance(cm.exception.__cause__, OSError)
        self.assertEqual(cm.exception.__cause__.errno, errno.EIO)
        self.assertEqual(self.files(), [])

    def test_short_write_is_detected(self):
        with (
            mock.patch.object(Path, "read_bytes", return_value=b"\xff\xd8 truncated"),
            self.assertRaises(SnapshotWriteError),
        ):
            self.save()
        self.assertEqual(self.files(), [])

    def test_undecodable_file_is_rejected(self):
        for decoded in (None, np.zeros((1, 1, 3), np.uint8)):
            with (
                self.subTest(decoded=None if decoded is None else decoded.shape),
                mock.patch.object(cv2, "imdecode", return_value=decoded),
                self.assertRaises(SnapshotWriteError),
            ):
                self.save()
        self.assertEqual(self.files(), [])

    def test_finalize_failure_leaves_no_file(self):
        with (
            mock.patch("os.link", side_effect=_eio()),
            mock.patch("os.rename", side_effect=_eio()),
            mock.patch.object(evidence_store, "_renameat2", return_value=_failing_renameat2(errno.EIO)),
            self.assertRaises(SnapshotWriteError),
        ):
            self.save()
        self.assertEqual(self.files(), [])

    def test_without_hard_links_rename_is_used(self):
        no_links = PermissionError(errno.EPERM, "Operation not permitted")
        with mock.patch("os.link", side_effect=no_links):
            path = self.save()
        self.assertEqual(cv2.imread(str(path)).shape, (120, 160, 3))
        self.assertEqual(self.files(), [path.name])

    def test_without_hard_links_existing_file_is_still_not_replaced(self):
        def no_links_but_racing(src, dst):
            Path(dst).write_bytes(b"other writer")
            raise PermissionError(errno.EPERM, "Operation not permitted")

        with mock.patch("os.link", side_effect=no_links_but_racing), self.assertRaises(SnapshotWriteError):
            self.save()
        self.assertEqual(self.target().read_bytes(), b"other writer")
        self.assertEqual(self.files(), [self.target().name])

    def test_temp_cleanup_failure_keeps_temp_and_reports_original_error(self):
        with (
            mock.patch("os.fsync", side_effect=_eio()),
            mock.patch.object(Path, "unlink", side_effect=PermissionError(errno.EACCES, "denied")),
            self.assertLogs(LOGGER, "ERROR") as logs,
            self.assertRaises(SnapshotWriteError) as cm,
        ):
            self.save()
        self.assertEqual(cm.exception.__cause__.errno, errno.EIO, "the write error, not the cleanup error")
        self.assertEqual(self.files(), [self.target().name + TEMP_SUFFIX], "kept for later quarantine")
        self.assertTrue(any("kept for later quarantine" in line for line in logs.output))

    def test_interruption_removes_temp_and_propagates(self):
        with mock.patch("os.link", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            self.save()
        self.assertEqual(self.files(), [])

    def test_crash_before_finalize_never_leaves_a_final_file(self):
        # A real crash runs no cleanup at all: at most the .tmp is left behind.
        with (
            mock.patch("os.link", side_effect=SystemExit),
            mock.patch.object(evidence_store, "_discard_temp"),
            self.assertRaises(SystemExit),
        ):
            self.save()
        self.assertEqual(self.files(), [self.target().name + TEMP_SUFFIX])
        self.assertFalse(self.target().name.endswith(TEMP_SUFFIX))

    def test_file_is_fsynced_before_it_gets_its_final_name(self):
        calls = mock.Mock()
        with (
            mock.patch("os.fsync", wraps=os.fsync) as fsync,
            mock.patch("os.link", wraps=os.link) as link,
        ):
            calls.attach_mock(fsync, "fsync")
            calls.attach_mock(link, "link")
            self.save()
        names = [c[0] for c in calls.mock_calls]
        self.assertIn("fsync", names)
        self.assertLess(names.index("fsync"), names.index("link"))

    def test_directory_fsync_failure_keeps_the_verified_snapshot(self):
        with (
            mock.patch.object(evidence_store, "_fsync_dir", side_effect=_eio()),
            self.assertLogs(LOGGER, "ERROR"),
        ):
            path = self.save()
        self.assertEqual(cv2.imread(str(path)).shape, (120, 160, 3))


class NoReplaceFinalizeTests(EvidenceStoreTestCase):
    """The final name is taken in one atomic step that never replaces a file,
    on both finalize paths. Races and missing platform support are simulated;
    link(), rename() and renameat2() themselves run on the real filesystem."""

    def temp_and_final(self, final_bytes: bytes | None = b"earlier evidence") -> tuple[Path, Path]:
        src = self.snapshot_dir / ("x.jpg" + TEMP_SUFFIX)
        dst = self.snapshot_dir / "x.jpg"
        src.write_bytes(b"new snapshot")
        if final_bytes is not None:
            dst.write_bytes(final_bytes)
        return src, dst

    def test_finalize_never_replaces_an_existing_file(self):
        for path in FINALIZE_PATHS:
            with self.subTest(path):
                src, dst = self.temp_and_final()
                with _finalize_via(path), self.assertRaises(FileExistsError):
                    evidence_store._move_no_replace(src, dst)
                self.assertEqual(dst.read_bytes(), b"earlier evidence")
                self.assertEqual(src.read_bytes(), b"new snapshot", "left for the caller to discard")
                src.unlink()
                dst.unlink()

    def test_platform_rename_refuses_an_existing_target(self):
        if sys.platform.startswith("linux"):
            self.assertIsNotNone(evidence_store._renameat2(), "the no-hard-link path needs glibc 2.28+")
        src, dst = self.temp_and_final()
        with self.assertRaises(FileExistsError):
            evidence_store._rename_no_replace(src, dst)
        self.assertEqual(dst.read_bytes(), b"earlier evidence")
        dst.unlink()
        evidence_store._rename_no_replace(src, dst)
        self.assertEqual(dst.read_bytes(), b"new snapshot")
        self.assertFalse(src.exists())

    def test_existing_destination_survives_even_if_an_existence_check_misses_it(self):
        # lexists() reporting "absent" is the window a check-then-rename loses.
        target = self.target()
        for path in FINALIZE_PATHS:
            with self.subTest(path):
                target.write_bytes(b"earlier evidence")
                with (
                    _finalize_via(path),
                    mock.patch("os.path.lexists", return_value=False),
                    self.assertRaises(SnapshotWriteError) as cm,
                ):
                    self.save()
                self.assertIsInstance(cm.exception, OSError)
                self.assertIsInstance(cm.exception.__cause__, FileExistsError)
                self.assertIn("refusing to overwrite", str(cm.exception))
                self.assertEqual(target.read_bytes(), b"earlier evidence")
                self.assertEqual(self.files(), [target.name], "this write's temp file is removed")

    def test_destination_created_just_before_the_rename_is_not_replaced(self):
        with (
            mock.patch("os.link", side_effect=_no_hard_links()),
            _another_writer_before_rename(b"other writer"),
            self.assertRaises(SnapshotWriteError) as cm,
        ):
            self.save()
        self.assertIsInstance(cm.exception.__cause__, FileExistsError)
        self.assertEqual(self.target().read_bytes(), b"other writer")
        self.assertEqual(self.files(), [self.target().name], "this write's temp file is removed")

    def test_without_a_no_replace_rename_the_save_fails_closed(self):
        # A POSIX host whose filesystem has no hard links and whose rename
        # cannot refuse an existing target: the save must fail, not rename().
        earlier = self.snapshot_dir / "earlier.jpg"
        earlier.write_bytes(b"earlier evidence")
        unsupported = {
            "libc without renameat2": None,
            "filesystem without RENAME_NOREPLACE": _failing_renameat2(errno.EINVAL),
            "kernel without renameat2": _failing_renameat2(errno.ENOSYS),
        }
        for name, renameat2 in unsupported.items():
            with (
                self.subTest(name),
                mock.patch.object(evidence_store, "_RENAME_REFUSES_EXISTING", False),
                mock.patch.object(evidence_store, "_renameat2", return_value=renameat2),
                mock.patch("os.link", side_effect=_no_hard_links()),
                mock.patch("os.rename", wraps=os.rename) as rename,
                mock.patch("os.replace", wraps=os.replace) as replace,
            ):
                with self.assertRaises(SnapshotWriteError) as cm:
                    self.save()
                rename.assert_not_called()
                replace.assert_not_called()
                self.assertIsInstance(cm.exception, OSError)
                self.assertNotIsInstance(cm.exception.__cause__, FileExistsError)
                self.assertEqual(earlier.read_bytes(), b"earlier evidence")
                self.assertEqual(self.files(), [earlier.name], "no final file, and no temp file left")

    def test_cleanup_removes_only_this_writes_temp_file(self):
        others = {
            "20231114T000000_cam02_9_other-event.jpg.tmp": b"another write in progress",
            "20231114T000000_cam02_9_other-event.jpg": b"other evidence",
        }
        for name, data in others.items():
            (self.snapshot_dir / name).write_bytes(data)
        failures = {
            "fsync": mock.patch("os.fsync", side_effect=_eio()),
            "verification": mock.patch.object(cv2, "imdecode", return_value=None),
            "finalize": mock.patch.object(
                evidence_store, "_move_no_replace", side_effect=FileExistsError(errno.EEXIST, "exists")
            ),
        }
        for name, failure in failures.items():
            with self.subTest(name), failure, self.assertRaises(SnapshotWriteError):
                self.save()
            self.assertEqual(self.files(), sorted(others), name)
            for other, data in others.items():
                self.assertEqual((self.snapshot_dir / other).read_bytes(), data)

    def test_final_file_is_never_partial_and_holds_exactly_the_verified_bytes(self):
        real_encode, real_verify, real_move = (
            evidence_store._encode_jpeg,
            evidence_store._verify_written,
            evidence_store._move_no_replace,
        )
        for n, path in enumerate(FINALIZE_PATHS):
            with self.subTest(path):
                event_id = f"atomic-{n}"
                target = self.target(event_id)
                encoded: list[bytes] = []
                seen: list[tuple[str, bytes | None]] = []

                def look(step: str, target=target, seen=seen) -> None:
                    seen.append((step, target.read_bytes() if target.exists() else None))

                def encode(image, quality, encoded=encoded):
                    encoded.append(real_encode(image, quality))
                    return encoded[-1]

                def verify(*args, look=look):
                    look("before verify")
                    real_verify(*args)
                    look("after verify")

                def move(src, dst, look=look):
                    look("before finalize")
                    real_move(src, dst)
                    look("after finalize")

                with (
                    _finalize_via(path),
                    mock.patch.object(evidence_store, "_encode_jpeg", side_effect=encode),
                    mock.patch.object(evidence_store, "_verify_written", side_effect=verify),
                    mock.patch.object(evidence_store, "_move_no_replace", side_effect=move),
                ):
                    saved = self.save(event_id=event_id)

                self.assertEqual(saved, target)
                self.assertEqual(len(encoded), 1)
                self.assertEqual(
                    seen,
                    [
                        ("before verify", None),
                        ("after verify", None),
                        ("before finalize", None),
                        ("after finalize", encoded[0]),
                    ],
                    "the final name appears only once, complete",
                )
                self.assertEqual(saved.read_bytes(), encoded[0])
                self.assertNotIn(target.name + TEMP_SUFFIX, self.files())


if __name__ == "__main__":
    unittest.main()
