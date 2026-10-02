"""Typed, validated configuration for the ZOVIVE edge node.

Loads the five core YAML files from the config directory and turns them
into immutable, typed objects::

    from utils.config_loader import load_config

    cfg = load_config()              # raises ConfigError if anything is wrong
    cfg.node.node_id                 # str
    cfg.camera.resolution.width      # int
    cfg.inference.confidence_threshold
    cfg.storage.image_dir            # absolute pathlib.Path
    cfg.network.base_url             # "http://<host>:<port>"

Design notes
------------
* **Fail loudly, all at once.** Every file is read and every field is
  checked before anything is raised, so one ``ConfigError`` lists *all*
  problems (missing files, bad YAML, missing keys, out-of-range values).
  An operator in the field fixes everything in one pass instead of one
  reboot per typo.
* **No hardcoded paths or addresses.** The config directory comes from
  the ``config_dir`` argument, then ``$ZOVIVE_CONFIG_DIR``, then
  ``<project>/configs``. Any string value in YAML may use
  ``${VAR}`` / ``${VAR:-default}`` so IPs and credentials can live in the
  environment (``$$`` is a literal ``$``).
* **Open for extension.** Future modules (db/, watchdog/, ota/ ...) add a
  file without touching this module: write a parser against
  ``ConfigReader`` and register a ``SectionSpec`` (see
  ``ConfigLoader.register``). Unknown keys in existing files are
  reported as warnings, not errors, so a newer YAML still boots older code,
  and every section keeps its full ``raw`` mapping for keys this module
  doesn't model yet.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import ipaddress
import os
import re
import sys
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Generic, TypeVar
from urllib.parse import urlsplit, urlunsplit

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR_ENV = "ZOVIVE_CONFIG_DIR"
SUPPORTED_SCHEMA_VERSION = 1

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class IssueKind(StrEnum):
    """Category of a configuration problem (stable, safe to match on in tests/tools)."""

    MISSING_DIR = "missing_dir"
    MISSING_FILE = "missing_file"
    PERMISSION = "permission"
    FILESYSTEM = "filesystem"
    PARSE = "parse"
    MISSING_KEY = "missing_key"
    INVALID_TYPE = "invalid_type"
    INVALID_VALUE = "invalid_value"
    MISSING_ENV = "missing_env"
    UNKNOWN_KEY = "unknown_key"
    INTERNAL = "internal"


@dataclass(frozen=True, slots=True)
class ConfigIssue:
    """One problem found while loading configuration."""

    source: str
    key: str
    message: str
    kind: IssueKind = IssueKind.INVALID_VALUE

    def __str__(self) -> str:
        where = f"{self.source}: {self.key}" if self.key else self.source
        return f"{where}: {self.message}"


class ConfigError(Exception):
    """Raised when configuration cannot be loaded. ``issues`` lists every problem found."""

    def __init__(self, issues: Iterable[ConfigIssue]):
        self.issues: tuple[ConfigIssue, ...] = tuple(issues)
        lines = "\n".join(f"  - {issue}" for issue in self.issues)
        super().__init__(f"invalid configuration ({len(self.issues)} problem(s)):\n{lines}")

    def kinds(self) -> set[IssueKind]:
        return {issue.kind for issue in self.issues}


class DirectoryCreationError(OSError):
    """Raised when a configured directory cannot be created or is not a directory."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _freeze(value: Any) -> Any:
    """Recursively convert dicts/lists into read-only mappings/tuples."""
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(v) for v in value)
    return value


def redact_url(url: str) -> str:
    """Replace the password in a URL with ``***`` so it is safe to log."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<unparseable url>"
    if parts.password is None:
        return url
    netloc = f"{parts.username or ''}:***@{parts.hostname or ''}"
    if parts.port is not None:
        netloc += f":{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


_HOSTNAME_RE = re.compile(
    r"^(?=.{1,253}$)(?!-)[A-Za-z0-9-]{1,63}(?<!-)(\.(?!-)[A-Za-z0-9-]{1,63}(?<!-))*\.?$"
)


def is_valid_host(host: str) -> bool:
    """True for an IPv4/IPv6 address or an RFC 1123 hostname."""
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return bool(_HOSTNAME_RE.match(host))


# --- ${VAR} / ${VAR:-default} interpolation -------------------------------

_ENV_RE = re.compile(r"\$\$|\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _interpolate(
    value: Any, env: Mapping[str, str], key_path: str, on_missing: Callable[[str, str], None]
) -> Any:
    if isinstance(value, str):

        def _sub(m: re.Match[str]) -> str:
            if m.group(0) == "$$":
                return "$"
            name, default = m.group(1), m.group(2)
            if name in env:
                return env[name]
            if default is not None:
                return default
            on_missing(key_path, name)
            return ""

        return _ENV_RE.sub(_sub, value)
    if isinstance(value, dict):
        return {
            k: _interpolate(v, env, f"{key_path}.{k}" if key_path else str(k), on_missing)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_interpolate(v, env, f"{key_path}[{i}]", on_missing) for i, v in enumerate(value)]
    return value


# --- YAML reading ------------------------------------------------------------


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate keys (PyYAML silently keeps the last one)."""


def _construct_unique_mapping(loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False) -> Any:
    loader.flatten_mapping(node)
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if isinstance(key, Hashable):
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            seen.add(key)
    return loader.construct_mapping(node, deep=deep)


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def read_yaml_file(path: Path, source: str | None = None) -> tuple[dict[str, Any] | None, list[ConfigIssue]]:
    """Read one YAML file into a dict.

    Never raises for expected failures; returns ``(None, issues)`` instead so
    the caller can keep checking the remaining files.
    """
    source = source or path.name
    try:
        if path.is_dir():
            return None, [ConfigIssue(source, "", f"expected a file but found a directory: {path}", IssueKind.FILESYSTEM)]
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, [ConfigIssue(source, "", f"file not found: {path}", IssueKind.MISSING_FILE)]
    except PermissionError:
        return None, [ConfigIssue(source, "", f"permission denied reading {path}", IssueKind.PERMISSION)]
    except UnicodeDecodeError as exc:
        return None, [ConfigIssue(source, "", f"file is not valid UTF-8 ({exc.reason} at byte {exc.start})", IssueKind.PARSE)]
    except OSError as exc:
        return None, [ConfigIssue(source, "", f"cannot read {path}: {exc.strerror or exc}", IssueKind.FILESYSTEM)]

    try:
        data = yaml.load(text, Loader=_StrictLoader)  # nosec B506  # noqa: S506 - _StrictLoader derives from SafeLoader
    except yaml.MarkedYAMLError as exc:
        mark = exc.problem_mark or exc.context_mark
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        problem = exc.problem or exc.context or "syntax error"
        return None, [ConfigIssue(source, "", f"malformed YAML{where}: {problem}", IssueKind.PARSE)]
    except yaml.YAMLError as exc:
        return None, [ConfigIssue(source, "", f"malformed YAML: {exc}", IssueKind.PARSE)]

    if data is None:
        return None, [ConfigIssue(source, "", "file is empty", IssueKind.PARSE)]
    if not isinstance(data, dict):
        return None, [
            ConfigIssue(source, "", f"top level must be a mapping of 'key: value' pairs, got {type(data).__name__}", IssueKind.PARSE)
        ]
    bad_keys = [k for k in data if not isinstance(k, str)]
    if bad_keys:
        return None, [ConfigIssue(source, "", f"top-level keys must be strings, got {bad_keys!r}", IssueKind.PARSE)]
    return data, []


# ---------------------------------------------------------------------------
# ConfigReader: typed field access that records problems instead of raising
# ---------------------------------------------------------------------------

_REQUIRED: Any = object()  # sentinel: "no default, field is required"
_ABSENT: Any = object()  # sentinel: "key not present"


class ConfigReader:
    """Typed accessor over one YAML mapping.

    Every getter validates type and range; on failure it records a
    ``ConfigIssue`` and returns a harmless placeholder so parsing continues
    and all problems are collected. The loader raises once parsing is done,
    so placeholders never reach callers.
    """

    def __init__(
        self,
        data: Mapping[str, Any],
        *,
        source: str,
        issues: list[ConfigIssue],
        prefix: str = "",
        suppressed: bool = False,
    ):
        self._data = data
        self._source = source
        self._issues = issues
        self._prefix = prefix
        self._suppressed = suppressed
        self._consumed: set[str] = set()
        self._children: list[ConfigReader] = []

    # --- bookkeeping --------------------------------------------------------

    @property
    def source(self) -> str:
        return self._source

    @property
    def raw(self) -> Mapping[str, Any]:
        """Read-only deep copy of this mapping, including keys not modelled yet."""
        return _freeze(self._data)

    def key_path(self, key: str) -> str:
        return f"{self._prefix}.{key}" if self._prefix else key

    def error(self, key: str, message: str, kind: IssueKind = IssueKind.INVALID_VALUE) -> None:
        if not self._suppressed:
            self._issues.append(ConfigIssue(self._source, self.key_path(key) if key else self._prefix, message, kind))

    def has(self, key: str) -> bool:
        return self._data.get(key) is not None

    def unknown_keys(self) -> list[str]:
        """Dotted paths of keys present in YAML that no getter consumed."""
        unknown = [self.key_path(k) for k in self._data if k not in self._consumed]
        for child in self._children:
            unknown.extend(child.unknown_keys())
        return unknown

    def _take(self, key: str, required: bool) -> Any:
        self._consumed.add(key)
        value = self._data.get(key)
        if value is None:
            if required:
                self.error(key, "required value is missing", IssueKind.MISSING_KEY)
            return _ABSENT
        return value

    # --- scalar getters -------------------------------------------------------

    def _scalar(
        self,
        key: str,
        default: Any,
        convert: Callable[[Any], Any],
        expected: str,
        placeholder: Any,
    ) -> tuple[Any, bool]:
        """Return ``(value, ok)``; ``ok`` is True only when the key was present and converted."""
        value = self._take(key, default is _REQUIRED)
        if value is _ABSENT:
            return (placeholder if default is _REQUIRED else default), False
        try:
            return convert(value), True
        except (TypeError, ValueError):
            self.error(key, f"expected {expected}, got {value!r} ({type(value).__name__})", IssueKind.INVALID_TYPE)
            return placeholder, False

    def _check_range(
        self,
        key: str,
        value: float | None,
        min_value: float | None,
        max_value: float | None,
        gt: float | None,
    ) -> None:
        if value is None:
            return
        if min_value is not None and value < min_value:
            self.error(key, f"must be >= {min_value}, got {value}")
        if max_value is not None and value > max_value:
            self.error(key, f"must be <= {max_value}, got {value}")
        if gt is not None and value <= gt:
            self.error(key, f"must be > {gt}, got {value}")

    def string(
        self,
        key: str,
        default: Any = _REQUIRED,
        *,
        choices: Iterable[str] | None = None,
        pattern: re.Pattern[str] | None = None,
        pattern_hint: str = "",
        allow_empty: bool = False,
    ) -> Any:
        """Text value. Returns ``default`` (which may be ``None``) when absent."""
        value, ok = self._scalar(key, default, _to_str, "a string (wrap it in quotes)", "")
        if not ok:
            return value
        if not value.strip() and not allow_empty:
            self.error(key, "must not be empty")
            return value
        options = tuple(choices) if choices is not None else None
        if options is not None and value not in options:
            self.error(key, f"must be one of {', '.join(options)}; got {value!r}")
        if pattern is not None and not pattern.match(value):
            self.error(key, f"invalid value {value!r}" + (f": {pattern_hint}" if pattern_hint else ""))
        return value

    def integer(
        self,
        key: str,
        default: Any = _REQUIRED,
        *,
        min_value: int | None = None,
        max_value: int | None = None,
    ) -> Any:
        value, ok = self._scalar(key, default, _to_int, "an integer", 0)
        if ok:
            self._check_range(key, value, min_value, max_value, None)
        return value

    def number(
        self,
        key: str,
        default: Any = _REQUIRED,
        *,
        min_value: float | None = None,
        max_value: float | None = None,
        gt: float | None = None,
    ) -> Any:
        value, ok = self._scalar(key, default, _to_float, "a number", 0.0)
        if ok:
            self._check_range(key, value, min_value, max_value, gt)
        return value

    def boolean(self, key: str, default: Any = _REQUIRED) -> Any:
        return self._scalar(key, default, _to_bool, "true or false", False)[0]

    def date(self, key: str, default: Any = _REQUIRED) -> Any:
        return self._scalar(key, default, _to_date, "a date in YYYY-MM-DD form", None)[0]

    def path(self, key: str, default: Any = _REQUIRED, *, base: Path) -> Any:
        """Filesystem path; relative values are resolved against ``base``."""
        if not self.has(key):
            return self.string(key, default)  # records "missing" if required; else returns default
        text = self.string(key)
        if not text:
            return base  # error already recorded; placeholder
        p = Path(text).expanduser()
        return (p if p.is_absolute() else base / p).resolve()

    def string_list(self, key: str, default: Any = _REQUIRED, *, allow_empty: bool = False) -> Any:
        value = self._take(key, default is _REQUIRED)
        if value is _ABSENT:
            return () if default is _REQUIRED else default
        if not isinstance(value, list):
            self.error(key, f"expected a list, got {type(value).__name__}", IssueKind.INVALID_TYPE)
            return ()
        items: list[str] = []
        for i, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                self.error(f"{key}[{i}]", f"expected a non-empty string, got {item!r}", IssueKind.INVALID_TYPE)
                continue
            items.append(item.strip())
        dupes = sorted({x for x in items if items.count(x) > 1})
        if dupes:
            self.error(key, f"duplicate entries: {', '.join(dupes)}")
        if not items and not allow_empty:
            self.error(key, "must contain at least one entry")
        return tuple(items)

    # --- nested structures ----------------------------------------------------

    def section(self, key: str, *, required: bool = True) -> ConfigReader:
        """Reader for a nested mapping. A missing optional section reads as ``{}``."""
        value = self._take(key, required)
        prefix = self.key_path(key)
        if value is _ABSENT:
            # Required-and-missing was already reported; don't also report every child.
            return ConfigReader({}, source=self._source, issues=self._issues, prefix=prefix, suppressed=required or self._suppressed)
        if not isinstance(value, dict):
            self.error(key, f"expected a mapping (indented 'key: value' block), got {type(value).__name__}", IssueKind.INVALID_TYPE)
            return ConfigReader({}, source=self._source, issues=self._issues, prefix=prefix, suppressed=True)
        child = ConfigReader(value, source=self._source, issues=self._issues, prefix=prefix, suppressed=self._suppressed)
        self._children.append(child)
        return child

    def free_mapping(self, key: str) -> Mapping[str, Any]:
        """Unvalidated read-only mapping (e.g. vendor-specific options)."""
        value = self._take(key, required=False)
        if value is _ABSENT:
            return MappingProxyType({})
        if not isinstance(value, dict):
            self.error(key, f"expected a mapping, got {type(value).__name__}", IssueKind.INVALID_TYPE)
            return MappingProxyType({})
        return _freeze(value)


def _to_str(v: Any) -> str:
    if not isinstance(v, str):
        raise TypeError
    return v.strip()


def _to_int(v: Any) -> int:
    if isinstance(v, bool):
        raise TypeError
    if isinstance(v, int):
        return v
    if isinstance(v, str):  # e.g. from ${ENV_VAR} interpolation
        return int(v.strip())
    raise TypeError


def _to_float(v: Any) -> float:
    if isinstance(v, bool):
        raise TypeError
    if isinstance(v, int | float):
        return float(v)
    if isinstance(v, str):
        return float(v.strip())
    raise TypeError


_TRUE = {"true", "yes", "on", "1"}
_FALSE = {"false", "no", "off", "0"}


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in _TRUE | _FALSE:
        return v.strip().lower() in _TRUE
    raise ValueError


def _to_date(v: Any) -> dt.date:
    if isinstance(v, dt.datetime):
        return v.date()
    if isinstance(v, dt.date):
        return v
    if isinstance(v, str):
        return dt.date.fromisoformat(v.strip())
    raise TypeError


# ---------------------------------------------------------------------------
# Section models
# ---------------------------------------------------------------------------

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$")
_ID_HINT = "use 3-64 letters, digits, '-' or '_', starting with a letter or digit"


def _empty_mapping() -> Mapping[str, Any]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class ParseContext:
    """Values parsers need for resolving relative paths."""

    base_dir: Path
    config_dir: Path


# --- node.yaml --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GpsCoordinates:
    latitude: float
    longitude: float
    altitude_m: float | None = None


@dataclass(frozen=True, slots=True)
class InstallationInfo:
    installed_on: dt.date | None = None
    installed_by: str = ""
    mount_height_m: float | None = None
    bearing_deg: float | None = None
    notes: str = ""


@dataclass(frozen=True, slots=True)
class NodeConfig:
    node_id: str
    forest_name: str
    camera_name: str
    gps: GpsCoordinates
    installation: InstallationInfo
    timezone: str = "UTC"
    tags: tuple[str, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=_empty_mapping, repr=False, compare=False)


def parse_node(r: ConfigReader, ctx: ParseContext) -> NodeConfig:
    gps = r.section("gps")
    inst = r.section("installation", required=False)
    return NodeConfig(
        node_id=r.string("node_id", pattern=_ID_RE, pattern_hint=_ID_HINT),
        forest_name=r.string("forest_name"),
        camera_name=r.string("camera_name"),
        gps=GpsCoordinates(
            latitude=gps.number("latitude", min_value=-90, max_value=90),
            longitude=gps.number("longitude", min_value=-180, max_value=180),
            altitude_m=gps.number("altitude_m", None),
        ),
        installation=InstallationInfo(
            installed_on=inst.date("installed_on", None),
            installed_by=inst.string("installed_by", "", allow_empty=True),
            mount_height_m=inst.number("mount_height_m", None, min_value=0),
            bearing_deg=inst.number("bearing_deg", None, min_value=0, max_value=360),
            notes=inst.string("notes", "", allow_empty=True),
        ),
        timezone=r.string("timezone", "UTC"),
        tags=r.string_list("tags", (), allow_empty=True),
        raw=r.raw,
    )


# --- camera.yaml ------------------------------------------------------------


class RtspTransport(StrEnum):
    TCP = "tcp"
    UDP = "udp"


class PtzProtocol(StrEnum):
    ONVIF = "onvif"
    HTTP_CGI = "http_cgi"


@dataclass(frozen=True, slots=True)
class Resolution:
    width: int
    height: int

    def __str__(self) -> str:
        return f"{self.width}x{self.height}"


@dataclass(frozen=True, slots=True)
class ReconnectPolicy:
    enabled: bool = True
    initial_delay_sec: float = 1.0
    max_delay_sec: float = 30.0
    backoff_multiplier: float = 2.0
    max_attempts: int = 0  # 0 = retry forever

    def delay_for_attempt(self, attempt: int) -> float:
        """Backoff delay before reconnect ``attempt`` (1-based), capped at ``max_delay_sec``."""
        return min(self.max_delay_sec, self.initial_delay_sec * self.backoff_multiplier ** max(0, attempt - 1))


@dataclass(frozen=True, slots=True)
class PtzConfig:
    enabled: bool = False
    protocol: PtzProtocol = PtzProtocol.ONVIF
    host: str | None = None
    port: int = 80
    home_preset: str | None = None
    options: Mapping[str, Any] = field(default_factory=_empty_mapping, repr=False)


@dataclass(frozen=True, slots=True)
class CameraConfig:
    camera_id: str
    rtsp_url: str = field(repr=False)  # may contain credentials; use rtsp_url_redacted for logs
    resolution: Resolution
    fps: int
    timeout_sec: float
    transport: RtspTransport
    reconnect: ReconnectPolicy
    ptz: PtzConfig
    raw: Mapping[str, Any] = field(default_factory=_empty_mapping, repr=False, compare=False)

    @property
    def rtsp_url_redacted(self) -> str:
        return redact_url(self.rtsp_url)


def parse_camera(r: ConfigReader, ctx: ParseContext) -> CameraConfig:
    rtsp_url = r.string("rtsp_url")
    if rtsp_url:
        try:
            parts = urlsplit(rtsp_url)
            if parts.scheme not in ("rtsp", "rtsps"):
                r.error("rtsp_url", f"must start with rtsp:// or rtsps://, got {redact_url(rtsp_url)!r}")
            elif not parts.hostname:
                r.error("rtsp_url", f"has no host: {redact_url(rtsp_url)!r}")
            else:
                parts.port  # noqa: B018 - raises ValueError for a non-numeric port
        except ValueError as exc:
            r.error("rtsp_url", f"is not a valid URL ({exc})")

    res = r.section("resolution")
    rc = r.section("reconnect", required=False)
    reconnect = ReconnectPolicy(
        enabled=rc.boolean("enabled", True),
        initial_delay_sec=rc.number("initial_delay_sec", 1.0, gt=0),
        max_delay_sec=rc.number("max_delay_sec", 30.0, gt=0),
        backoff_multiplier=rc.number("backoff_multiplier", 2.0, min_value=1.0),
        max_attempts=rc.integer("max_attempts", 0, min_value=0),
    )
    if reconnect.max_delay_sec < reconnect.initial_delay_sec:
        rc.error("max_delay_sec", f"must be >= initial_delay_sec ({reconnect.initial_delay_sec})")

    pz = r.section("ptz", required=False)
    ptz_enabled = pz.boolean("enabled", False)
    ptz_host = pz.string("host", None)
    if ptz_host and not is_valid_host(ptz_host):
        pz.error("host", f"is not a valid IP address or hostname: {ptz_host!r}")
    if ptz_enabled and not ptz_host:
        pz.error("host", "is required when ptz.enabled is true", IssueKind.MISSING_KEY)
    protocols = [p.value for p in PtzProtocol]
    protocol = pz.string("protocol", PtzProtocol.ONVIF.value, choices=protocols)
    ptz = PtzConfig(
        enabled=ptz_enabled,
        protocol=PtzProtocol(protocol) if protocol in protocols else PtzProtocol.ONVIF,
        host=ptz_host,
        port=pz.integer("port", 80, min_value=1, max_value=65535),
        home_preset=pz.string("home_preset", None),
        options=pz.free_mapping("options"),
    )

    transport = r.string("transport", RtspTransport.TCP.value, choices=[t.value for t in RtspTransport])
    return CameraConfig(
        camera_id=r.string("camera_id", pattern=_ID_RE, pattern_hint=_ID_HINT),
        rtsp_url=rtsp_url,
        resolution=Resolution(
            width=res.integer("width", min_value=16, max_value=7680),
            height=res.integer("height", min_value=16, max_value=4320),
        ),
        fps=r.integer("fps", min_value=1, max_value=120),
        timeout_sec=r.number("timeout_sec", 10.0, gt=0),
        transport=RtspTransport(transport) if transport in {t.value for t in RtspTransport} else RtspTransport.TCP,
        reconnect=reconnect,
        ptz=ptz,
        raw=r.raw,
    )


# --- inference.yaml ---------------------------------------------------------


class InferenceBackend(StrEnum):
    ONNX = "onnx"  # CPU; laptop, CI and Pi without accelerator
    HAILO = "hailo"  # Hailo-8/8L NPU on the Pi 5 AI HAT


@dataclass(frozen=True, slots=True)
class HailoSettings:
    hef_path: Path | None = None
    device_id: str | None = None
    batch_size: int = 1
    options: Mapping[str, Any] = field(default_factory=_empty_mapping, repr=False)


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    backend: InferenceBackend
    model_path: Path
    input_size: int
    confidence_threshold: float
    iou_threshold: float
    max_detections: int
    classes: tuple[str, ...]
    hailo: HailoSettings
    raw: Mapping[str, Any] = field(default_factory=_empty_mapping, repr=False, compare=False)

    @property
    def active_model_path(self) -> Path:
        """Model file the selected backend will actually load."""
        if self.backend is InferenceBackend.HAILO and self.hailo.hef_path is not None:
            return self.hailo.hef_path
        return self.model_path


def parse_inference(r: ConfigReader, ctx: ParseContext) -> InferenceConfig:
    backends = [b.value for b in InferenceBackend]
    backend_name = r.string("backend", InferenceBackend.ONNX.value, choices=backends)
    backend = InferenceBackend(backend_name) if backend_name in backends else InferenceBackend.ONNX

    model_path = r.path("model_path", base=ctx.base_dir)
    if isinstance(model_path, Path) and r.has("model_path") and model_path.suffix.lower() != ".onnx":
        r.error("model_path", f"expected an .onnx file, got {model_path.name!r}")

    h = r.section("hailo", required=False)
    hef_path = h.path("hef_path", None, base=ctx.base_dir)
    if backend is InferenceBackend.HAILO and hef_path is None:
        h.error("hef_path", "is required when backend is 'hailo'", IssueKind.MISSING_KEY)
    if isinstance(hef_path, Path) and hef_path.suffix.lower() != ".hef":
        h.error("hef_path", f"expected a .hef file, got {hef_path.name!r}")

    return InferenceConfig(
        backend=backend,
        model_path=model_path,
        input_size=r.integer("input_size", 640, min_value=32, max_value=4096),
        confidence_threshold=r.number("confidence_threshold", gt=0.0, max_value=1.0),
        iou_threshold=r.number("iou_threshold", gt=0.0, max_value=1.0),
        max_detections=r.integer("max_detections", min_value=1, max_value=10_000),
        classes=r.string_list("classes"),
        hailo=HailoSettings(
            hef_path=hef_path,
            device_id=h.string("device_id", None),
            batch_size=h.integer("batch_size", 1, min_value=1, max_value=64),
            options=h.free_mapping("options"),
        ),
        raw=r.raw,
    )


# --- storage.yaml -----------------------------------------------------------


class CleanupStrategy(StrEnum):
    OLDEST_FIRST = "oldest_first"  # delete oldest files first
    UPLOADED_FIRST = "uploaded_first"  # delete files already acked by the base station first


@dataclass(frozen=True, slots=True)
class CleanupPolicy:
    strategy: CleanupStrategy = CleanupStrategy.UPLOADED_FIRST
    interval_minutes: int = 60
    delete_only_uploaded: bool = True
    purge_temp_on_startup: bool = True


@dataclass(frozen=True, slots=True)
class DiskThresholds:
    warning_percent: float = 80.0
    critical_percent: float = 90.0


@dataclass(frozen=True, slots=True)
class StorageConfig:
    root: Path
    image_dir: Path
    video_dir: Path
    temp_dir: Path
    database_dir: Path
    retention_days: int
    cleanup: CleanupPolicy
    disk: DiskThresholds
    raw: Mapping[str, Any] = field(default_factory=_empty_mapping, repr=False, compare=False)

    @property
    def directories(self) -> Mapping[str, Path]:
        return MappingProxyType(
            {
                "image_dir": self.image_dir,
                "video_dir": self.video_dir,
                "temp_dir": self.temp_dir,
                "database_dir": self.database_dir,
            }
        )

    def ensure_directories(self) -> None:
        """Create every configured directory. Raises ``DirectoryCreationError`` with the failing path."""
        for name, path in self.directories.items():
            try:
                path.mkdir(parents=True, exist_ok=True)
            except FileExistsError as exc:
                raise DirectoryCreationError(f"storage.{name}: {path} exists but is not a directory") from exc
            except PermissionError as exc:
                raise DirectoryCreationError(f"storage.{name}: permission denied creating {path}") from exc
            except OSError as exc:
                raise DirectoryCreationError(f"storage.{name}: cannot create {path}: {exc.strerror or exc}") from exc
            if not os.access(path, os.W_OK):
                raise DirectoryCreationError(f"storage.{name}: {path} is not writable by this user")


def parse_storage(r: ConfigReader, ctx: ParseContext) -> StorageConfig:
    root = r.path("root", base=ctx.base_dir)
    root = root if isinstance(root, Path) else ctx.base_dir

    dirs: dict[str, Path] = {}
    for key in ("image_dir", "video_dir", "temp_dir", "database_dir"):
        p = r.path(key, base=root)
        dirs[key] = p if isinstance(p, Path) else root
        if r.has(key) and dirs[key].exists() and not dirs[key].is_dir():
            r.error(key, f"{dirs[key]} exists but is not a directory", IssueKind.FILESYSTEM)
    seen: dict[Path, str] = {}
    for key, p in dirs.items():
        if r.has(key) and p in seen:
            r.error(key, f"must differ from {seen[p]} (both resolve to {p})")
        seen.setdefault(p, key)

    c = r.section("cleanup", required=False)
    strategies = [s.value for s in CleanupStrategy]
    strategy = c.string("strategy", CleanupStrategy.UPLOADED_FIRST.value, choices=strategies)
    cleanup = CleanupPolicy(
        strategy=CleanupStrategy(strategy) if strategy in strategies else CleanupStrategy.UPLOADED_FIRST,
        interval_minutes=c.integer("interval_minutes", 60, min_value=1, max_value=7 * 24 * 60),
        delete_only_uploaded=c.boolean("delete_only_uploaded", True),
        purge_temp_on_startup=c.boolean("purge_temp_on_startup", True),
    )

    d = r.section("disk")
    disk = DiskThresholds(
        warning_percent=d.number("usage_threshold_percent", gt=0, max_value=100),
        critical_percent=d.number("critical_percent", 95.0, gt=0, max_value=100),
    )
    if d.has("usage_threshold_percent") and disk.critical_percent <= disk.warning_percent:
        d.error("critical_percent", f"must be greater than usage_threshold_percent ({disk.warning_percent})")

    return StorageConfig(
        root=root,
        image_dir=dirs["image_dir"],
        video_dir=dirs["video_dir"],
        temp_dir=dirs["temp_dir"],
        database_dir=dirs["database_dir"],
        retention_days=r.integer("retention_days", min_value=1, max_value=3650),
        cleanup=cleanup,
        disk=disk,
        raw=r.raw,
    )


# --- network.yaml -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    base_station_host: str
    base_station_port: int
    scheme: str
    heartbeat_interval_sec: float
    upload_retry_count: int
    connection_timeout_sec: float
    registration_timeout_sec: float
    raw: Mapping[str, Any] = field(default_factory=_empty_mapping, repr=False, compare=False)

    @property
    def base_url(self) -> str:
        host = self.base_station_host
        if ":" in host:  # bare IPv6 literal
            host = f"[{host}]"
        return f"{self.scheme}://{host}:{self.base_station_port}"


def parse_network(r: ConfigReader, ctx: ParseContext) -> NetworkConfig:
    bs = r.section("base_station")
    host = bs.string("host")
    if host and not is_valid_host(host):
        bs.error("host", f"is not a valid IP address or hostname: {host!r}")
    return NetworkConfig(
        base_station_host=host,
        base_station_port=bs.integer("port", min_value=1, max_value=65535),
        scheme=bs.string("scheme", "http", choices=("http", "https")),
        heartbeat_interval_sec=r.number("heartbeat_interval_sec", min_value=1),
        upload_retry_count=r.integer("upload_retry_count", min_value=0, max_value=100),
        connection_timeout_sec=r.number("connection_timeout_sec", gt=0, max_value=600),
        registration_timeout_sec=r.number("registration_timeout_sec", gt=0, max_value=3600),
        raw=r.raw,
    )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SectionSpec(Generic[T]):
    """How to load one config file: which file, how to parse it, whether it must exist."""

    name: str
    filename: str
    parser: Callable[[ConfigReader, ParseContext], T]
    required: bool = True


CORE_SECTIONS: tuple[SectionSpec[Any], ...] = (
    SectionSpec("node", "node.yaml", parse_node),
    SectionSpec("camera", "camera.yaml", parse_camera),
    SectionSpec("inference", "inference.yaml", parse_inference),
    SectionSpec("storage", "storage.yaml", parse_storage),
    SectionSpec("network", "network.yaml", parse_network),
)
_CORE_NAMES = frozenset(s.name for s in CORE_SECTIONS)


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Fully validated configuration for one edge node process."""

    node: NodeConfig
    camera: CameraConfig
    inference: InferenceConfig
    storage: StorageConfig
    network: NetworkConfig
    config_dir: Path
    extensions: Mapping[str, Any] = field(default_factory=_empty_mapping)
    warnings: tuple[ConfigIssue, ...] = ()

    def section(self, name: str) -> Any:
        """Access a core or registered extension section by name."""
        if name in _CORE_NAMES:
            return getattr(self, name)
        try:
            return self.extensions[name]
        except KeyError:
            raise KeyError(f"no config section named {name!r}; registered: {sorted(_CORE_NAMES | set(self.extensions))}") from None

    def summary(self) -> dict[str, Any]:
        """Short, credential-free description suitable for a startup log line."""
        return {
            "node_id": self.node.node_id,
            "forest": self.node.forest_name,
            "camera": self.camera.camera_id,
            "stream": self.camera.rtsp_url_redacted,
            "resolution": str(self.camera.resolution),
            "fps": self.camera.fps,
            "backend": self.inference.backend.value,
            "model": self.inference.active_model_path.name,
            "storage_root": str(self.storage.root),
            "base_station": self.network.base_url,
            "config_dir": str(self.config_dir),
        }


def default_config_dir(env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    override = env.get(CONFIG_DIR_ENV, "").strip()
    return Path(override).expanduser().resolve() if override else PROJECT_ROOT / "configs"


class ConfigLoader:
    """Loads, interpolates and validates every registered config file.

    Args:
        config_dir: directory holding the YAML files (default: ``$ZOVIVE_CONFIG_DIR``
            or ``<project>/configs``).
        base_dir: directory relative paths in YAML are resolved against
            (default: project root).
        env: environment used for ``${VAR}`` interpolation (default: ``os.environ``).
        extra_sections: additional ``SectionSpec`` objects for future modules.
    """

    def __init__(
        self,
        config_dir: str | Path | None = None,
        *,
        base_dir: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        extra_sections: Iterable[SectionSpec[Any]] = (),
    ):
        self._env: Mapping[str, str] = dict(os.environ if env is None else env)
        self.config_dir = Path(config_dir).expanduser().resolve() if config_dir else default_config_dir(self._env)
        self.base_dir = Path(base_dir).expanduser().resolve() if base_dir else PROJECT_ROOT
        self._specs: dict[str, SectionSpec[Any]] = {s.name: s for s in CORE_SECTIONS}
        for spec in extra_sections:
            self.register(spec)

    def register(self, spec: SectionSpec[Any]) -> None:
        """Add a config file for a new module (e.g. ``watchdog.yaml``)."""
        if spec.name in self._specs:
            raise ValueError(f"config section {spec.name!r} is already registered")
        if any(s.filename == spec.filename for s in self._specs.values()):
            raise ValueError(f"config file {spec.filename!r} is already registered")
        self._specs[spec.name] = spec

    def load(self) -> AppConfig:
        """Load and validate everything. Raises ``ConfigError`` listing every problem."""
        if not self.config_dir.is_dir():
            what = "is not a directory" if self.config_dir.exists() else "does not exist"
            raise ConfigError([ConfigIssue(str(self.config_dir), "", f"configuration directory {what}", IssueKind.MISSING_DIR)])

        ctx = ParseContext(base_dir=self.base_dir, config_dir=self.config_dir)
        issues: list[ConfigIssue] = []
        warnings: list[ConfigIssue] = []
        sections: dict[str, Any] = {}

        for spec in self._specs.values():
            value, spec_warnings = self._load_section(spec, ctx, issues)
            warnings.extend(spec_warnings)
            if value is not None:
                sections[spec.name] = value

        if issues:
            raise ConfigError(issues)

        return AppConfig(
            node=sections["node"],
            camera=sections["camera"],
            inference=sections["inference"],
            storage=sections["storage"],
            network=sections["network"],
            config_dir=self.config_dir,
            extensions=MappingProxyType({k: v for k, v in sections.items() if k not in _CORE_NAMES}),
            warnings=tuple(warnings),
        )

    def _load_section(
        self, spec: SectionSpec[Any], ctx: ParseContext, issues: list[ConfigIssue]
    ) -> tuple[Any, list[ConfigIssue]]:
        path = self.config_dir / spec.filename
        if not spec.required and not path.exists():
            data: dict[str, Any] | None = {}
        else:
            data, file_issues = read_yaml_file(path, spec.filename)
            issues.extend(file_issues)
            if data is None:
                return None, []

        def on_missing_env(key: str, var: str) -> None:
            issues.append(
                ConfigIssue(spec.filename, key, f"environment variable {var} is not set and no default given "
                            f"(use ${{{var}:-default}} to provide one)", IssueKind.MISSING_ENV)
            )

        data = _interpolate(copy.deepcopy(data), self._env, "", on_missing_env)
        reader = ConfigReader(data, source=spec.filename, issues=issues)
        reader.integer("schema_version", SUPPORTED_SCHEMA_VERSION, min_value=1, max_value=SUPPORTED_SCHEMA_VERSION)
        try:
            value = spec.parser(reader, ctx)
        except Exception as exc:  # a buggy parser must not crash startup without context
            issues.append(ConfigIssue(spec.filename, "", f"parser failed: {type(exc).__name__}: {exc}", IssueKind.INTERNAL))
            return None, []
        warnings = [
            ConfigIssue(spec.filename, key, "unknown key ignored (typo, or newer config than this code?)", IssueKind.UNKNOWN_KEY)
            for key in reader.unknown_keys()
        ]
        return value, warnings


def load_config(config_dir: str | Path | None = None, **kwargs: Any) -> AppConfig:
    """Convenience wrapper: ``ConfigLoader(config_dir, **kwargs).load()``."""
    return ConfigLoader(config_dir, **kwargs).load()


# ---------------------------------------------------------------------------
# CLI: python -m utils.config_loader [--config-dir DIR]
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Validate configuration without starting the node. Exit code 0 = valid."""
    parser = argparse.ArgumentParser(description="Validate ZOVIVE edge configuration.")
    parser.add_argument("--config-dir", type=Path, default=None, help="directory containing node.yaml etc.")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config_dir)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for warning in cfg.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    print(f"OK: configuration in {cfg.config_dir} is valid")
    for key, value in cfg.summary().items():
        print(f"  {key:<14} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
