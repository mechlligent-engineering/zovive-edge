"""MAC + CPU serial + static IP registration on boot: the payload and its checks.

This module only builds and validates the registration payload. Sending it,
retrying and scheduling belong to the caller. The payload IS the node
identity, unchanged::

    from network.boot_registration import build_registration_payload, validate_registration_payload

    payload = build_registration_payload()
    errors = validate_registration_payload(payload)   # [] when valid

Identity comes only from network.node_identity.get_identity(); nothing here
reads node_config.yaml, .env or the hardware. Field rules shared with the
identity (node_id, camera_id, forest_name, camera_name, mac) are checked by
validate_identity(), not repeated here.
"""

from __future__ import annotations

import copy
import ipaddress
import logging
from typing import Any

from network.node_identity import IDENTITY_VERSION, get_identity, validate_identity

log = logging.getLogger(__name__)

# Every key get_identity() returns; the base station needs all of them.
_REQUIRED_KEYS = (
    "identity_version",
    "node_id",
    "camera_id",
    "forest_name",
    "camera_name",
    "mac",
    "cpu_serial",
    "ip",
    "hostname",
)


def build_registration_payload() -> dict[str, Any]:
    """The node identity as a registration payload: same fields, same
    values, as an independent copy."""
    return copy.deepcopy(get_identity())


def validate_registration_payload(payload: Any) -> list[str]:
    """Problems found in ``payload``; an empty list means valid. Never raises."""
    try:
        return _validate(payload)
    except Exception:  # a validator must never take down the caller
        log.exception("registration payload validation failed unexpectedly")
        return ["internal validation error"]


def _validate(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return ["payload must be a dictionary"]

    errors = [f"missing {key}" for key in _REQUIRED_KEYS if key not in payload]
    errors.extend(validate_identity(payload))

    if "identity_version" in payload and not _is_identity_version(payload["identity_version"]):
        errors.append("invalid identity_version")
    if not _is_ip_or_none(payload.get("ip")):
        errors.append("invalid ip")
    if not _is_str_or_none(payload.get("hostname")):
        errors.append("invalid hostname")
    if not _is_str_or_none(payload.get("cpu_serial")):
        errors.append("invalid cpu_serial")
    return errors


def _is_identity_version(value: Any) -> bool:
    # type() rather than isinstance(): True == 1 must not pass as version 1.
    return type(value) is int and value == IDENTITY_VERSION


def _is_ip_or_none(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):  # ip_address() would also accept an int
        return False
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return False
    return True


def _is_str_or_none(value: Any) -> bool:
    return value is None or isinstance(value, str)
