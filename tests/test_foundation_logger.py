"""Tests for utils/logger.py (Phase 1 foundation)."""

from __future__ import annotations

import io
import logging
import re
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from utils import logger as edge_logger
from utils.logger import (
    LoggingSettings,
    LoggingSetupError,
    RotationMode,
    RotationSettings,
    get_logger,
    parse_level,
    setup_logging,
    shutdown_logging,
)

LINE_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} (DEBUG|INFO|WARNING|ERROR|CRITICAL) \S+ .+$")


class LoggerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="zovive-log-"))
        self.log_dir = self.tmp / "nested" / "logs"
        self.stdout = io.StringIO()
        patcher = mock.patch.object(sys, "stdout", self.stdout)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self) -> None:
        shutdown_logging()  # closes file handles so Windows can delete the temp dir
        shutil.rmtree(self.tmp, ignore_errors=True)

    def settings(self, **kw) -> LoggingSettings:
        kw.setdefault("log_dir", self.log_dir)
        kw.setdefault("capture_uncaught", False)
        return LoggingSettings(**kw)

    def file_lines(self) -> list[str]:
        return (self.log_dir / "edge.log").read_text(encoding="utf-8").splitlines()


class FormatTests(LoggerTestCase):
    def test_line_format_matches_spec(self):
        setup_logging(self.settings())
        get_logger("network").info("Node Registered")
        get_logger("camera").error("RTSP Connection Failed")
        lines = self.file_lines()
        self.assertEqual(len(lines), 2)
        for line in lines:
            self.assertRegex(line, LINE_RE)
        self.assertTrue(lines[0].endswith(" INFO network Node Registered"))
        self.assertTrue(lines[1].endswith(" ERROR camera RTSP Connection Failed"))

    def test_console_and_file_receive_same_lines(self):
        setup_logging(self.settings())
        get_logger("storage").warning("disk 85%")
        self.assertEqual(self.stdout.getvalue().splitlines(), self.file_lines())

    def test_dunder_name_and_nested_modules(self):
        setup_logging(self.settings())
        get_logger("storage.disk_guard").info("x")
        get_logger("zovive.ota").info("y")
        lines = self.file_lines()
        self.assertIn(" INFO storage.disk_guard x", lines[0])
        self.assertIn(" INFO ota y", lines[1])

    def test_exception_traceback_is_logged(self):
        setup_logging(self.settings())
        try:
            raise ValueError("bad frame")
        except ValueError:
            get_logger("capture").exception("decode failed")
        text = (self.log_dir / "edge.log").read_text(encoding="utf-8")
        self.assertIn("ERROR capture decode failed", text)
        self.assertIn("ValueError: bad frame", text)


class LevelTests(LoggerTestCase):
    def test_level_filtering(self):
        setup_logging(self.settings(level="WARNING"))
        log = get_logger("inference")
        log.debug("d")
        log.info("i")
        log.warning("w")
        log.critical("c")
        self.assertEqual([line.split(" ")[2] for line in self.file_lines()], ["WARNING", "CRITICAL"])

    def test_runtime_level_change(self):
        setup_logging(self.settings(level="INFO"))
        edge_logger.set_level("DEBUG")
        get_logger("x").debug("now visible")
        self.assertIn("DEBUG x now visible", self.file_lines()[0])

    def test_parse_level(self):
        self.assertEqual(parse_level("info"), logging.INFO)
        self.assertEqual(parse_level("WARN"), logging.WARNING)
        self.assertEqual(parse_level(logging.ERROR), logging.ERROR)
        for bad in ("verbose", 7, True):
            with self.assertRaises(ValueError):
                parse_level(bad)

    def test_invalid_level_rejected_at_setup(self):
        with self.assertRaises(ValueError):
            setup_logging(self.settings(level="LOUD"))


class SetupTests(LoggerTestCase):
    def test_creates_missing_log_directory(self):
        self.assertFalse(self.log_dir.exists())
        setup_logging(self.settings())
        self.assertTrue((self.log_dir / "edge.log").exists())
        self.assertTrue(edge_logger.file_logging_active())

    def test_repeated_setup_does_not_duplicate_handlers(self):
        for _ in range(3):
            setup_logging(self.settings())
        get_logger("net").info("once")
        self.assertEqual(len(self.file_lines()), 1)
        self.assertEqual(len(logging.getLogger("zovive").handlers), 2)

    def test_does_not_touch_root_logger(self):
        root_handlers = list(logging.getLogger().handlers)
        setup_logging(self.settings())
        self.assertEqual(logging.getLogger().handlers, root_handlers)
        self.assertFalse(logging.getLogger("zovive").propagate)

    def test_env_overrides(self):
        s = LoggingSettings.from_env(
            {"ZOVIVE_LOG_DIR": str(self.tmp / "envlogs"), "ZOVIVE_LOG_LEVEL": "debug", "ZOVIVE_LOG_FILE": "n.log"}
        )
        self.assertEqual(s.log_file, self.tmp / "envlogs" / "n.log")
        self.assertEqual(s.level, "debug")

    def test_default_log_file_is_project_logs_edge_log(self):
        s = LoggingSettings.from_env({})
        self.assertEqual(s.log_file, edge_logger.PROJECT_ROOT / "logs" / "edge.log")


class FailureTests(LoggerTestCase):
    def test_directory_creation_failure_falls_back_to_console(self):
        with mock.patch.object(Path, "mkdir", side_effect=PermissionError(13, "denied")):
            setup_logging(self.settings(console=False))  # console forced on: never zero outputs
        self.assertFalse(edge_logger.file_logging_active())
        get_logger("net").info("still visible")
        out = self.stdout.getvalue()
        self.assertRegex(out, r"WARNING \S+ file logging disabled")
        self.assertIn("permission denied", out)
        self.assertIn("INFO net still visible", out)

    def test_log_dir_path_is_a_file(self):
        self.log_dir.parent.mkdir(parents=True)
        self.log_dir.write_text("not a dir")
        setup_logging(self.settings())
        self.assertIn("file logging disabled", self.stdout.getvalue())

    def test_strict_mode_raises(self):
        with mock.patch.object(Path, "mkdir", side_effect=PermissionError(13, "denied")):
            with self.assertRaises(LoggingSetupError):
                setup_logging(self.settings(strict=True))

    def test_logging_before_setup_is_not_silent(self):
        shutdown_logging()
        stderr = io.StringIO()
        # Other tests may attach handlers to the root logger; isolate so the
        # last-resort stderr path is what gets exercised here.
        with mock.patch.object(sys, "stderr", stderr), mock.patch.object(logging.getLogger(), "handlers", []):
            get_logger("early").error("config failed before logging setup")
        self.assertIn("config failed before logging setup", stderr.getvalue())


class RotationTests(LoggerTestCase):
    def test_size_rotation_creates_backups(self):
        rot = RotationSettings(mode=RotationMode.SIZE, max_bytes=500, backup_count=2)
        setup_logging(self.settings(rotation=rot, console=False))
        log = get_logger("load")
        for i in range(100):
            log.info("message number %03d with padding padding padding", i)
        names = sorted(p.name for p in self.log_dir.iterdir())
        self.assertEqual(names, ["edge.log", "edge.log.1", "edge.log.2"])

    def test_time_and_none_modes_build(self):
        for mode, cls in ((RotationMode.TIME, logging.handlers.TimedRotatingFileHandler), (RotationMode.NONE, logging.FileHandler)):
            setup_logging(self.settings(rotation=RotationSettings(mode=mode)))
            handlers = [h for h in logging.getLogger("zovive").handlers if isinstance(h, logging.FileHandler)]
            self.assertEqual(len(handlers), 1)
            self.assertIsInstance(handlers[0], cls)

    def test_invalid_rotation_rejected(self):
        with self.assertRaises(ValueError):
            RotationSettings(max_bytes=0)


class UncaughtExceptionTests(LoggerTestCase):
    def test_uncaught_exception_logged_as_critical(self):
        setup_logging(self.settings(capture_uncaught=True))
        try:
            raise RuntimeError("camera thread died")
        except RuntimeError:
            sys.excepthook(*sys.exc_info())
        text = (self.log_dir / "edge.log").read_text(encoding="utf-8")
        self.assertIn("CRITICAL main Unhandled exception", text)
        self.assertIn("RuntimeError: camera thread died", text)

    def test_thread_exception_logged(self):
        setup_logging(self.settings(capture_uncaught=True))

        def boom():
            raise OSError("usb ssd vanished")

        t = threading.Thread(target=boom, name="storage-worker")
        t.start()
        t.join()
        text = (self.log_dir / "edge.log").read_text(encoding="utf-8")
        self.assertIn("Unhandled exception in thread storage-worker", text)

    def test_hooks_restored_on_shutdown(self):
        original = sys.excepthook
        setup_logging(self.settings(capture_uncaught=True))
        self.assertIsNot(sys.excepthook, original)
        shutdown_logging()
        self.assertIs(sys.excepthook, original)


if __name__ == "__main__":
    unittest.main()
