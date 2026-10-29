"""network/boot_registration.py: payload building and validation.

get_identity() is mocked, so no camera, network, Raspberry Pi or base
station is needed; validate_identity() is the real one, since registration
relies on it for the shared identity rules.
"""

from __future__ import annotations

import logging
import unittest
from unittest import mock

from network import boot_registration
from network.boot_registration import build_registration_payload, validate_registration_payload
from network.node_identity import IDENTITY_VERSION

VALID_PAYLOAD = {
    "identity_version": IDENTITY_VERSION,
    "node_id": "Node-001",
    "camera_id": "cam01",
    "forest_name": "Bandipur",
    "camera_name": "Tower-01",
    "mac": "88:a2:9e:c1:68:6f",
    "cpu_serial": "10000000abcdef12",
    "ip": "192.168.1.111",
    "hostname": "zovive-node-001",
}


def payload(**changes) -> dict:
    return {**VALID_PAYLOAD, **changes}


def without(*keys: str) -> dict:
    return {k: v for k, v in VALID_PAYLOAD.items() if k not in keys}


class RegistrationTestCase(unittest.TestCase):
    def setUp(self):
        # validate_identity() logs every result. Keep those lines out of the
        # console, including handlers other test modules put on the root logger.
        logger = logging.getLogger("network")
        handler = logging.NullHandler()
        logger.addHandler(handler)
        self.addCleanup(logger.removeHandler, handler)
        patcher = mock.patch.object(logger, "propagate", False)
        patcher.start()
        self.addCleanup(patcher.stop)


class BuildPayloadTests(RegistrationTestCase):
    def test_build_registration_payload_returns_copy(self):
        identity = dict(VALID_PAYLOAD)
        with mock.patch.object(boot_registration, "get_identity", return_value=identity):
            built = build_registration_payload()
        self.assertEqual(built, identity)
        self.assertIsNot(built, identity)

    def test_payload_is_identity_unchanged(self):
        identity = dict(VALID_PAYLOAD, cpu_serial=None, ip=None)
        with mock.patch.object(boot_registration, "get_identity", return_value=identity):
            built = build_registration_payload()
        self.assertEqual(list(built), list(identity))  # no fields added, removed or reordered
        self.assertEqual(built, identity)

    def test_changing_payload_leaves_identity_alone(self):
        identity = dict(VALID_PAYLOAD)
        with mock.patch.object(boot_registration, "get_identity", return_value=identity):
            built = build_registration_payload()
        built["node_id"] = "tampered"
        self.assertEqual(identity["node_id"], "Node-001")

    def test_built_payload_is_valid(self):
        with mock.patch.object(boot_registration, "get_identity", return_value=dict(VALID_PAYLOAD)):
            self.assertEqual(validate_registration_payload(build_registration_payload()), [])


class ValidationTests(RegistrationTestCase):
    def test_valid_payload(self):
        self.assertEqual(validate_registration_payload(payload()), [])

    def test_payload_none(self):
        self.assertEqual(validate_registration_payload(None), ["payload must be a dictionary"])

    def test_payload_not_dict(self):
        for value in ("Node-001", 42, ["node_id"], ("node_id", "Node-001")):
            self.assertEqual(validate_registration_payload(value), ["payload must be a dictionary"], value)

    def test_missing_node_id(self):
        errors = validate_registration_payload(without("node_id"))
        self.assertIn("missing node_id", errors)

    def test_missing_mac(self):
        errors = validate_registration_payload(without("mac"))
        self.assertIn("missing mac", errors)
        self.assertIn("mac address is not available", errors)  # from validate_identity()

    def test_missing_camera_name(self):
        errors = validate_registration_payload(without("camera_name"))
        self.assertIn("missing camera_name", errors)

    def test_invalid_identity_version(self):
        for value in (2, 0, "1", 1.0, True, None):
            errors = validate_registration_payload(payload(identity_version=value))
            self.assertEqual(errors, ["invalid identity_version"], value)

    def test_invalid_ipv4(self):
        for value in ("abc", "999.999.999.999", "192.168.1", "", 3232235887):
            self.assertEqual(validate_registration_payload(payload(ip=value)), ["invalid ip"], value)

    def test_invalid_ipv6(self):
        for value in ("2001:db8::zz", "2001:::1", "2001:db8::1::2"):
            self.assertEqual(validate_registration_payload(payload(ip=value)), ["invalid ip"], value)

    def test_valid_ip_addresses(self):
        for value in ("192.168.1.111", "10.0.0.5", "2001:db8::1", None):
            self.assertEqual(validate_registration_payload(payload(ip=value)), [], value)

    def test_cpu_serial_none(self):
        self.assertEqual(validate_registration_payload(payload(cpu_serial=None)), [])

    def test_invalid_cpu_serial(self):
        for value in (10000000, 1.5, {"serial": "x"}, ["x"]):
            errors = validate_registration_payload(payload(cpu_serial=value))
            self.assertEqual(errors, ["invalid cpu_serial"], value)

    def test_hostname_none(self):
        self.assertEqual(validate_registration_payload(payload(hostname=None)), [])

    def test_invalid_hostname(self):
        for value in (1, 1.5, {"name": "x"}, ["x"]):
            self.assertEqual(
                validate_registration_payload(payload(hostname=value)), ["invalid hostname"], value
            )

    def test_identity_rules_come_from_validate_identity(self):
        errors = validate_registration_payload(payload(node_id="Node 001", mac="88-A2-9E-C1-68-6F"))
        self.assertEqual(len(errors), 2)
        self.assertTrue(any("node_id 'Node 001' is invalid" in e for e in errors), errors)
        self.assertTrue(any("unicast MAC" in e for e in errors), errors)

    def test_multiple_missing_fields(self):
        errors = validate_registration_payload(without("node_id", "camera_name", "identity_version"))
        for key in ("node_id", "camera_name", "identity_version"):
            self.assertIn(f"missing {key}", errors)
        self.assertNotIn("invalid identity_version", errors)  # reported once, as missing

    def test_empty_payload_reports_every_key(self):
        errors = validate_registration_payload({})
        for key in VALID_PAYLOAD:
            self.assertIn(f"missing {key}", errors)

    def test_unexpected_error_is_contained(self):
        with (
            mock.patch.object(boot_registration, "validate_identity", side_effect=RuntimeError("boom")),
            self.assertLogs("network.boot_registration", "ERROR"),
        ):
            self.assertEqual(validate_registration_payload(payload()), ["internal validation error"])


if __name__ == "__main__":
    unittest.main()
