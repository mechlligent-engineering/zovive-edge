"""Centralized logging for the ZOVIVE edge node.

One call at process start, then ``get_logger`` everywhere else::

    from utils.logger import setup_logging, get_logger

    setup_logging()                       # console + logs/edge.log
    log = get_logger("network")
    log.info("Node Registered")
    # 2026-09-26 10:12:33 INFO network Node Registered

Design notes
------------
* **Own namespace.** Every logger lives under ``zovive.*`` and handlers are
  attached to the ``zovive`` logger, not the root logger. Third-party
  libraries stay quiet, and this module can coexist with other logging
  setups in the same process. The ``zovive.`` prefix is stripped when
  printed, so the line shows ``network``, not ``zovive.network``.
* **Usable before config.** Logging must work *before* configuration is
  loaded so config errors can be logged, so it only reads ``LoggingSettings``
  (defaults plus ``ZOVIVE_LOG_*`` environment overrides), never the config loader.
* **Never silent.** If the log directory or file can't be created, logging
  falls back to console-only and says so at WARNING (or raises
  ``LoggingSetupError`` with ``strict=True``). Uncaught exceptions in any
  thread are logged at CRITICAL before the process exits. A logger used
  before ``setup_logging`` still reaches stderr via Python's last-resort
  handler.
* **Rotation.** ``RotationSettings`` picks ``size`` (default), ``time`` or
  ``none``; ``none`` is for when an external tool such as logrotate
  handles rotation. Adding a mode means adding one branch in
  ``_build_file_handler``.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ROOT_LOGGER_NAME = "zovive"
LOG_FORMAT = "%(asctime)s %(levelname)s %(module_name)s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

LOG_DIR_ENV = "ZOVIVE_LOG_DIR"
LOG_LEVEL_ENV = "ZOVIVE_LOG_LEVEL"
LOG_FILE_ENV = "ZOVIVE_LOG_FILE"

LEVELS: Mapping[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class LoggingSetupError(RuntimeError):
    """Raised in strict mode when file logging cannot be initialised."""


def parse_level(level: str | int) -> int:
    """Convert ``"info"``/``"INFO"``/``logging.INFO`` to a level number; reject anything else."""
    if isinstance(level, bool):
        raise ValueError(f"invalid log level {level!r}")
    if isinstance(level, int):
        if level in LEVELS.values():
            return level
        raise ValueError(f"invalid log level {level!r}; expected one of {', '.join(LEVELS)}")
    name = str(level).strip().upper()
    if name == "WARN":
        name = "WARNING"
    if name not in LEVELS:
        raise ValueError(f"invalid log level {level!r}; expected one of {', '.join(LEVELS)}")
    return LEVELS[name]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class RotationMode(StrEnum):
    NONE = "none"  # single growing file; use when logrotate manages it
    SIZE = "size"  # rotate at max_bytes, keep backup_count files
    TIME = "time"  # rotate on a schedule (when/interval), keep backup_count files


@dataclass(frozen=True, slots=True)
class RotationSettings:
    mode: RotationMode = RotationMode.SIZE
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5
    when: str = "midnight"
    interval: int = 1

    def __post_init__(self) -> None:
        if self.max_bytes <= 0:
            raise ValueError("rotation max_bytes must be > 0")
        if self.backup_count < 0:
            raise ValueError("rotation backup_count must be >= 0")
        if self.interval <= 0:
            raise ValueError("rotation interval must be > 0")


@dataclass(frozen=True, slots=True)
class LoggingSettings:
    """Everything ``setup_logging`` needs. All fields have safe defaults."""

    level: str | int = "INFO"
    log_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")
    file_name: str = "edge.log"
    console: bool = True
    file: bool = True
    rotation: RotationSettings = field(default_factory=RotationSettings)
    capture_uncaught: bool = True
    strict: bool = False  # True: raise LoggingSetupError instead of degrading to console-only

    @property
    def log_file(self) -> Path:
        return self.log_dir / self.file_name

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, **overrides: Any) -> LoggingSettings:
        """Defaults, then ``ZOVIVE_LOG_DIR`` / ``ZOVIVE_LOG_LEVEL`` / ``ZOVIVE_LOG_FILE``, then ``overrides``."""
        env = os.environ if env is None else env
        values: dict[str, Any] = {}
        if env.get(LOG_DIR_ENV, "").strip():
            values["log_dir"] = Path(env[LOG_DIR_ENV].strip()).expanduser()
        if env.get(LOG_LEVEL_ENV, "").strip():
            values["level"] = env[LOG_LEVEL_ENV].strip()
        if env.get(LOG_FILE_ENV, "").strip():
            values["file_name"] = env[LOG_FILE_ENV].strip()
        values.update(overrides)
        return cls(**values)


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


class EdgeFormatter(logging.Formatter):
    """``<date> <time> <LEVEL> <module> <message>`` with the ``zovive.`` prefix removed."""

    def __init__(self) -> None:
        super().__init__(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        record.module_name = module_display_name(record.name)
        return super().format(record)


def module_display_name(logger_name: str) -> str:
    prefix = ROOT_LOGGER_NAME + "."
    if logger_name.startswith(prefix):
        return logger_name[len(prefix) :]
    return logger_name


# ---------------------------------------------------------------------------
# Setup / teardown
# ---------------------------------------------------------------------------

_lock = threading.RLock()
_handlers: list[logging.Handler] = []
_active: LoggingSettings | None = None
_file_logging_active = False
_previous_hooks: tuple[Any, Any] | None = None


def get_logger(name: str) -> logging.Logger:
    """Return the logger for a module, e.g. ``get_logger("network")`` or ``get_logger(__name__)``."""
    name = name.strip() or "app"
    if name == ROOT_LOGGER_NAME or name.startswith(ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def setup_logging(settings: LoggingSettings | None = None) -> logging.Logger:
    """Configure console and file logging. Safe to call again; later calls reconfigure.

    Returns the ``zovive`` root logger.
    """
    settings = settings or LoggingSettings.from_env()
    level = parse_level(settings.level)

    with _lock:
        _teardown_handlers()
        root = logging.getLogger(ROOT_LOGGER_NAME)
        root.setLevel(level)

        formatter = EdgeFormatter()
        problems: list[str] = []

        file_handler: logging.Handler | None = None
        if settings.file:
            try:
                file_handler = _build_file_handler(settings)
            except OSError as exc:
                message = f"file logging disabled: cannot write {settings.log_file}: {_describe(exc)}"
                if settings.strict:
                    raise LoggingSetupError(message) from exc
                problems.append(message)

        console = settings.console or file_handler is None  # never end up with zero outputs
        if console:
            _install(root, logging.StreamHandler(sys.stdout), formatter)
        if file_handler is not None:
            _install(root, file_handler, formatter)

        # Handlers live on "zovive"; don't also hand records to the root logger.
        root.propagate = False

        global _active, _file_logging_active
        _active = settings
        _file_logging_active = file_handler is not None

        if settings.capture_uncaught:
            _install_exception_hooks()

    for message in problems:
        root.warning(message)
    return root


def _build_file_handler(settings: LoggingSettings) -> logging.Handler:
    """Create the log directory and file handler. Raises ``OSError`` on failure."""
    log_dir = Path(settings.log_dir)
    if log_dir.exists() and not log_dir.is_dir():
        raise NotADirectoryError(f"{log_dir} exists and is not a directory")
    log_dir.mkdir(parents=True, exist_ok=True)

    rot = settings.rotation
    path = settings.log_file
    if rot.mode is RotationMode.SIZE:
        return logging.handlers.RotatingFileHandler(
            path, maxBytes=rot.max_bytes, backupCount=rot.backup_count, encoding="utf-8"
        )
    if rot.mode is RotationMode.TIME:
        return logging.handlers.TimedRotatingFileHandler(
            path, when=rot.when, interval=rot.interval, backupCount=rot.backup_count, encoding="utf-8"
        )
    # WatchedFileHandler reopens the file if logrotate moves it (POSIX); plain FileHandler elsewhere.
    if os.name == "posix":
        return logging.handlers.WatchedFileHandler(path, encoding="utf-8")
    return logging.FileHandler(path, encoding="utf-8")


def _install(root: logging.Logger, handler: logging.Handler, formatter: logging.Formatter) -> None:
    handler.setFormatter(formatter)
    root.addHandler(handler)
    _handlers.append(handler)


def _teardown_handlers() -> None:
    root = logging.getLogger(ROOT_LOGGER_NAME)
    for handler in _handlers:
        root.removeHandler(handler)
        try:
            handler.flush()
            handler.close()
        except OSError:
            pass
    _handlers.clear()


def _describe(exc: OSError) -> str:
    if isinstance(exc, PermissionError):
        return "permission denied"
    return exc.strerror or str(exc)


def shutdown_logging() -> None:
    """Flush and close all handlers and restore exception hooks (process exit, tests)."""
    global _active, _file_logging_active
    with _lock:
        _teardown_handlers()
        _restore_exception_hooks()
        root = logging.getLogger(ROOT_LOGGER_NAME)
        root.propagate = True  # fall back to Python's last-resort stderr handler
        root.setLevel(logging.NOTSET)
        _active = None
        _file_logging_active = False


def set_level(level: str | int) -> None:
    """Change verbosity at runtime (e.g. from a health/debug command)."""
    logging.getLogger(ROOT_LOGGER_NAME).setLevel(parse_level(level))


def is_configured() -> bool:
    return _active is not None


def file_logging_active() -> bool:
    """False if file logging was requested but fell back to console-only."""
    return _file_logging_active


def active_settings() -> LoggingSettings | None:
    return _active


# ---------------------------------------------------------------------------
# Uncaught exceptions
# ---------------------------------------------------------------------------


def _log_uncaught(exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None) -> None:
    get_logger("main").critical("Unhandled exception, process will exit", exc_info=(exc_type, exc, tb))


def _install_exception_hooks() -> None:
    global _previous_hooks
    if _previous_hooks is None:
        _previous_hooks = (sys.excepthook, threading.excepthook)
    prev_sys, prev_thread = _previous_hooks

    def sys_hook(exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            prev_sys(exc_type, exc, tb)
        else:
            _log_uncaught(exc_type, exc, tb)

    def thread_hook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is not SystemExit and args.exc_value is not None:
            thread_name = args.thread.name if args.thread else "unknown"
            get_logger("main").critical(
                "Unhandled exception in thread %s",
                thread_name,
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
        else:
            prev_thread(args)

    sys.excepthook = sys_hook
    threading.excepthook = thread_hook


def _restore_exception_hooks() -> None:
    global _previous_hooks
    if _previous_hooks is not None:
        sys.excepthook, threading.excepthook = _previous_hooks
        _previous_hooks = None
