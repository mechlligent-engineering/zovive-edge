"""Permanent identity of this edge node: who is sending data, from where.

Identity is NOT the IP address. The base station tells nodes apart by
``node_id`` (configs/node_config.yaml, normally ``ZOVIVE_NODE_ID`` from
/opt/zovive/.env) and by the hardware it runs on (MAC, CPU serial). The IP is
reported too, but only as the current return address: it may change with
DHCP or a re-addressed link without the node becoming a different node.

Usage::

    from network.node_identity import get_identity, validate_identity

    identity = get_identity()        # dict, see below
    problems = validate_identity()   # [] when the identity is usable

    {
      "identity_version": 1,
      "node_id": "Node-001",          # the Pi        (node_config.yaml)
      "camera_id": "cam01",           # the sensor    (node_config.yaml)
      "forest_name": "Bandipur",
      "camera_name": "Tower-01",
      "mac": "88:a2:9e:c1:68:6f",
      "cpu_serial": "10000000abcdef", # None off a Raspberry Pi
      "ip": "192.168.1.111",          # None while the network is down
      "hostname": "zovive-node-001"
    }

Design notes
------------
* **Built once, reused everywhere.** Config values, MAC, CPU serial and
  hostname are read on the first ``get_identity()`` call and cached for the
  life of the process. Registration, heartbeat, alerts and uploads all call
  ``get_identity()``; nothing else should read these values itself.
* **IP is re-checked every call.** The lookup asks the kernel which local
  address routes to the base station (``transfer_config.yaml``
  ``base_station.url``). That is the address the base station actually sees,
  even on a Pi with several interfaces and no default gateway. No packet is
  sent. Falls back to the default route, then the hostname's address.
* **Never raises.** Missing config or hardware that can't be read is logged
  and the field is ``None``; ``validate_identity()`` decides whether the
  result is usable. A node that can't identify itself must still record
  detections locally.
* **No fake hardware values.** A development laptop has no CPU serial and
  reports ``None`` rather than a made-up stand-in.
* ``identity_version`` lets the base station recognise when the set of
  fields changes. Bump ``IDENTITY_VERSION`` when adding or renaming a field.
"""

from __future__ import annotations

import ipaddress
import logging
import platform
import re
import socket
import sys
import threading
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import paths
from utils.config_loader import _ID_HINT, _ID_RE, ConfigError, load

log = logging.getLogger(__name__)

IDENTITY_VERSION = 1

# node_config.yaml `node:` keys that must be present; validate_identity()
# also checks node_id/camera_id against config_loader's id rule.
REQUIRED_CONFIG_FIELDS = ("node_id", "camera_id", "forest_name", "camera_name")

# Linux sources. Module attributes so tests can point them at a fake tree.
_SYS_NET = Path("/sys/class/net")
_DT_SERIAL = Path("/sys/firmware/devicetree/base/serial-number")
_CPUINFO = Path("/proc/cpuinfo")

# Used only to ask "which interface has the default route?" when no base
# station is configured. 192.0.2.0/24 is reserved for documentation
# (RFC 5737), so this can never be a real peer; connect() on a UDP socket
# sends nothing.
_DEFAULT_ROUTE_PROBE = "192.0.2.1"
_PROBE_PORT = 80  # any port: a UDP connect() only selects a route

_MAC_RE = re.compile(r"^[0-9a-f]{2}(?::[0-9a-f]{2}){5}$")
_WIRED_PREFIXES = ("eth", "en")
_WIRELESS_PREFIXES = ("wl",)

_lock = threading.Lock()
_cached: dict[str, Any] | None = None  # every field except "ip"
_probe_host: str | None = None  # base station host the IP lookup routes toward
_ip_missing_logged = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_identity(*, refresh: bool = False) -> dict[str, Any]:
    """Return this node's identity. Built once per process; ``refresh=True``
    re-reads the hardware (YAML values come from ``utils.config_loader``'s
    per-process cache). The IP is looked up again on every call. Returns a
    new dict each time, so callers may modify it."""
    global _cached, _probe_host
    with _lock:
        if _cached is None or refresh:
            _cached, _probe_host = _build()
        cached, probe_host = _cached, _probe_host
    return {
        "identity_version": cached["identity_version"],
        "node_id": cached["node_id"],
        "camera_id": cached["camera_id"],
        "forest_name": cached["forest_name"],
        "camera_name": cached["camera_name"],
        "mac": cached["mac"],
        "cpu_serial": cached["cpu_serial"],
        "ip": get_ip_address(probe_host),
        "hostname": cached["hostname"],
    }


def validate_identity(identity: Mapping[str, Any] | None = None) -> list[str]:
    """Check that ``identity`` (default: ``get_identity()``) can be sent to
    the base station. Returns the problems found; an empty list means valid.

    Required: node_id, camera_id, forest_name, camera_name and a unicast MAC.
    A missing CPU serial is only a warning on Linux (where a Pi should have
    one) and expected on Windows development machines.
    """
    if identity is None:
        identity = get_identity()

    problems: list[str] = []
    for key in REQUIRED_CONFIG_FIELDS:
        if _text(identity.get(key)) is None:
            problems.append(f"{key} is missing (configs/node_config.yaml node.{key})")
    for key in ("node_id", "camera_id"):
        value = _text(identity.get(key))
        if value is not None and not _ID_RE.match(value):
            problems.append(f"{key} {value!r} is invalid: {_ID_HINT}")

    mac = identity.get("mac")
    if not mac:
        problems.append("mac address is not available")
    elif normalize_mac(str(mac)) != mac:
        problems.append(f"mac {mac!r} is not a lowercase, colon-separated unicast MAC address")

    if problems:
        log.error("node identity validation failed", extra={"problems": problems})
        return problems

    if identity.get("cpu_serial") is None and _on_linux():
        log.warning(
            "node identity has no cpu serial; the base station can only match this node by MAC",
            extra={"node_id": identity.get("node_id")},
        )
    log.info(
        "node identity validation passed",
        extra={"node_id": identity.get("node_id"), "camera_id": identity.get("camera_id")},
    )
    return problems


def get_mac_address(interface: str | None = None) -> str | None:
    """MAC of ``interface`` if given and readable, otherwise of the first
    physical interface (wired before wireless, then by name, so the choice is
    the same every boot). Falls back to ``uuid.getnode()`` where /sys is not
    available (Windows). Returns ``None`` if nothing usable is found."""
    if interface:
        mac = _interface_mac(interface)
        if mac:
            return mac
        log.warning(
            "node identity: configured interface has no usable MAC address, choosing one automatically",
            extra={"interface": interface},
        )

    for name in _physical_interfaces():
        mac = _interface_mac(name)
        if mac:
            return mac

    mac = _uuid_mac()
    if mac is None:
        log.error("node identity: hardware discovery failed: no usable MAC address found")
    return mac


def get_cpu_serial() -> str | None:
    """Raspberry Pi board serial from the device tree, else the ``Serial``
    line of /proc/cpuinfo. ``None`` where neither exists (Windows, most
    x86 Linux) -- never a substitute value."""
    serial = _clean_serial(_read_text(_DT_SERIAL))
    if serial is None:
        serial = _serial_from_cpuinfo(_read_text(_CPUINFO))
    if serial is None:
        if _on_linux():
            log.warning("node identity: hardware discovery failed: cpu serial not found")
        else:
            log.info("node identity: cpu serial not available on this platform (development machine)")
    return serial


def get_ip_address(target_host: str | None = None) -> str | None:
    """Local IP address used to reach ``target_host`` (normally the base
    station), else the default-route address, else the hostname's address.
    Loopback and unspecified addresses are never returned. ``None`` while no
    network is reachable; that is logged once per outage, not every call."""
    global _ip_missing_logged
    ip = None
    for host in (target_host, _DEFAULT_ROUTE_PROBE):
        if host and (ip := _route_source_ip(host)):
            break
    else:
        ip = _hostname_ip()
    if ip is None:
        if not _ip_missing_logged:
            log.warning(
                "node identity: hardware discovery failed: no usable IP address (network down?)",
                extra={"target_host": target_host},
            )
            _ip_missing_logged = True
    elif _ip_missing_logged:
        log.info("node identity: IP address available again", extra={"ip": ip})
        _ip_missing_logged = False
    return ip


def get_hostname() -> str | None:
    """System hostname, or ``None`` if it can't be read."""
    for source in (socket.gethostname, platform.node):
        try:
            name = source().strip()
        except OSError:
            continue
        if name:
            return name
    log.warning("node identity: hardware discovery failed: hostname not available")
    return None


def normalize_mac(raw: str | None) -> str | None:
    """``"88-A2-9E-C1-68-6F"`` -> ``"88:a2:9e:c1:68:6f"``. ``None`` for
    anything that is not a usable unicast address (all-zero, multicast)."""
    if not raw:
        return None
    mac = raw.strip().lower().replace("-", ":")
    if not _MAC_RE.match(mac) or mac == "00:00:00:00:00:00":
        return None
    if int(mac[:2], 16) & 0x01:  # multicast/broadcast bit: not an interface address
        return None
    return mac


def reset_identity_cache() -> None:
    """Forget the cached identity and outage-log state (tests)."""
    global _cached, _probe_host, _ip_missing_logged
    with _lock:
        _cached = None
        _probe_host = None
        _ip_missing_logged = False


# ---------------------------------------------------------------------------
# Building the cached part
# ---------------------------------------------------------------------------


def _build() -> tuple[dict[str, Any], str | None]:
    node = _node_section()
    missing = [key for key in REQUIRED_CONFIG_FIELDS if _text(node.get(key)) is None]
    if missing:
        log.error(
            "node identity: missing configuration",
            extra={"config_path": str(paths.NODE_CONFIG), "missing_keys": [f"node.{k}" for k in missing]},
        )

    identity = {
        "identity_version": IDENTITY_VERSION,
        "node_id": _text(node.get("node_id")),
        "camera_id": _text(node.get("camera_id")),
        "forest_name": _text(node.get("forest_name")),
        "camera_name": _text(node.get("camera_name")),
        "mac": get_mac_address(_text(node.get("identity_interface"))),
        "cpu_serial": get_cpu_serial(),
        "hostname": get_hostname(),
    }
    probe_host = _base_station_host()
    log.info(
        "node identity loaded",
        extra={
            "node_id": identity["node_id"],
            "camera_id": identity["camera_id"],
            "mac": identity["mac"],
            "cpu_serial": identity["cpu_serial"],
            "hostname": identity["hostname"],
            "base_station_host": probe_host,
        },
    )
    return identity, probe_host


def _node_section() -> Mapping[str, Any]:
    """The ``node:`` mapping of node_config.yaml, or ``{}`` (logged) if the
    file is missing, unreadable or has no such section."""
    try:
        cfg = load(paths.NODE_CONFIG, required=False)
    except ConfigError as exc:
        log.error(
            "node identity: cannot read node configuration",
            extra={"config_path": str(paths.NODE_CONFIG), "error": str(exc)},
        )
        return {}
    node = cfg.get("node")
    if not isinstance(node, Mapping):
        log.error(
            "node identity: missing configuration: no 'node' section",
            extra={"config_path": str(paths.NODE_CONFIG)},
        )
        return {}
    return node


def _base_station_host() -> str | None:
    """Host part of transfer_config.yaml ``base_station.url``; ``None`` if
    unset (the IP lookup then uses the default route)."""
    try:
        cfg = load(paths.TRANSFER_CONFIG, required=False)
    except ConfigError as exc:
        log.warning(
            "node identity: cannot read transfer configuration, using default route for IP lookup",
            extra={"config_path": str(paths.TRANSFER_CONFIG), "error": str(exc)},
        )
        return None
    base_station = cfg.get("base_station")
    url = _text(base_station.get("url")) if isinstance(base_station, Mapping) else None
    if url is None:
        return None
    try:
        host = urlsplit(url).hostname
    except ValueError:
        host = None
    if not host:
        log.warning(
            "node identity: base_station.url has no host, using default route for IP lookup",
            extra={"url": url},
        )
    return host or None


# ---------------------------------------------------------------------------
# Hardware helpers
# ---------------------------------------------------------------------------


def _physical_interfaces() -> list[str]:
    """Network interfaces backed by a device (skips lo, docker, bridges,
    VPNs), ordered wired, wireless, other, then by name."""
    try:
        entries = [p for p in _SYS_NET.iterdir() if (p / "device").exists()]
    except OSError:
        return []

    def rank(name: str) -> tuple[int, str]:
        if name.startswith(_WIRED_PREFIXES):
            return 0, name
        if name.startswith(_WIRELESS_PREFIXES):
            return 1, name
        return 2, name

    return sorted((p.name for p in entries), key=rank)


def _interface_mac(name: str) -> str | None:
    if Path(name).name != name:  # config value must be a bare interface name
        return None
    return normalize_mac(_read_text(_SYS_NET / name / "address"))


def _uuid_mac() -> str | None:
    node = uuid.getnode()
    if (node >> 40) & 0x01:  # uuid.getnode() found no hardware address and made one up
        return None
    return normalize_mac(":".join(f"{(node >> shift) & 0xFF:02x}" for shift in range(40, -1, -8)))


def _serial_from_cpuinfo(text: str | None) -> str | None:
    for line in (text or "").splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip().lower() == "serial":
            return _clean_serial(value)
    return None


def _clean_serial(raw: str | None) -> str | None:
    serial = (raw or "").replace("\x00", "").strip().lower()
    if not serial or not serial.isalnum() or set(serial) == {"0"}:
        return None
    return serial


def _route_source_ip(host: str) -> str | None:
    """Local address the kernel would use to send to ``host``, without
    sending anything."""
    try:
        infos = socket.getaddrinfo(host, _PROBE_PORT, type=socket.SOCK_DGRAM)
    except (OSError, UnicodeError):
        return None
    for family, sock_type, proto, _, sockaddr in infos:
        try:
            with socket.socket(family, sock_type, proto) as sock:
                sock.connect(sockaddr)
                ip = _usable_ip(sock.getsockname()[0])
        except OSError:
            continue
        if ip:
            return ip
    return None


def _hostname_ip() -> str | None:
    try:
        _, _, addresses = socket.gethostbyname_ex(socket.gethostname())
    except (OSError, UnicodeError):
        return None
    return next((ip for ip in map(_usable_ip, addresses) if ip), None)


def _usable_ip(value: str) -> str | None:
    """``value`` without any IPv6 zone suffix, or ``None`` if it is not a
    reachable unicast address (loopback, 0.0.0.0, link-local)."""
    text = str(value).split("%", 1)[0]
    try:
        addr = ipaddress.ip_address(text)
    except ValueError:
        return None
    if addr.is_loopback or addr.is_unspecified or addr.is_link_local or addr.is_multicast:
        return None
    return text


def _read_text(path: Path) -> str | None:
    try:
        return path.read_bytes().decode("ascii", errors="replace")
    except OSError:
        return None


def _text(value: Any) -> str | None:
    """Config scalar as a stripped string; ``None`` for missing/empty/non-scalar."""
    if value is None or isinstance(value, (Mapping, list)):
        return None
    text = str(value).strip()
    return text or None


def _on_linux() -> bool:
    return sys.platform.startswith("linux")
