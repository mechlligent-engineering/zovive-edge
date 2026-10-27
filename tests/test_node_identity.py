"""network/node_identity.py: config load, hardware discovery, caching, validation.

Hardware is faked: /sys/class/net, the device-tree serial and /proc/cpuinfo
are redirected into a temp directory, and socket/uuid lookups are mocked, so
these tests give the same result on a Raspberry Pi, a Linux CI runner and a
Windows laptop.
"""

from __future__ import annotations

import os
import socket
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import paths
from network import node_identity
from network.node_identity import (
    IDENTITY_VERSION,
    get_cpu_serial,
    get_hostname,
    get_identity,
    get_ip_address,
    get_mac_address,
    normalize_mac,
    validate_identity,
)
from utils import config_loader

REPO_NODE_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "node_config.yaml"
LOGGER = "network.node_identity"

PI_MAC = "88:a2:9e:c1:68:6f"
WIFI_MAC = "88:a2:9e:c1:68:70"
PI_SERIAL = "10000000abcdef12"
# uuid.getnode() sets the multicast bit when it had to invent an address.
RANDOM_UUID_NODE = 0x010000000001

NODE_CONFIG = """\
node:
  node_id: "Node-001"
  camera_id: "cam01"
  camera_name: "Tower-01"
  forest_name: "Bandipur"
  identity_interface: ""
"""

VALID_IDENTITY = {
    "identity_version": IDENTITY_VERSION,
    "node_id": "Node-001",
    "camera_id": "cam01",
    "forest_name": "Bandipur",
    "camera_name": "Tower-01",
    "mac": PI_MAC,
    "cpu_serial": PI_SERIAL,
    "ip": "192.168.1.111",
    "hostname": "zovive-node-001",
}


class NodeIdentityTestCase(unittest.TestCase):
    """Fake hardware tree + temp config files; identity and config caches reset."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.sys_net = root / "sys" / "class" / "net"
        self.sys_net.mkdir(parents=True)
        self.dt_serial = root / "serial-number"
        self.cpuinfo = root / "cpuinfo"
        self.node_config = root / "node_config.yaml"
        self.transfer_config = root / "transfer_config.yaml"

        self.patch(node_identity, "_SYS_NET", self.sys_net)
        self.patch(node_identity, "_DT_SERIAL", self.dt_serial)
        self.patch(node_identity, "_CPUINFO", self.cpuinfo)
        self.patch(paths, "NODE_CONFIG", self.node_config)
        self.patch(paths, "TRANSFER_CONFIG", self.transfer_config)
        self.patch(uuid, "getnode", return_value=RANDOM_UUID_NODE)  # never this machine's real MAC

        config_loader.clear_cache()
        node_identity.reset_identity_cache()
        self.addCleanup(config_loader.clear_cache)
        self.addCleanup(node_identity.reset_identity_cache)

    def patch(self, target, attribute: str, *args, **kwargs):
        patcher = mock.patch.object(target, attribute, *args, **kwargs)
        mocked = patcher.start()
        self.addCleanup(patcher.stop)
        return mocked

    def add_interface(self, name: str, mac: str, physical: bool = True) -> None:
        iface = self.sys_net / name
        iface.mkdir()
        (iface / "address").write_text(mac + "\n", encoding="ascii")
        if physical:
            (iface / "device").mkdir()

    def write_node_config(self, text: str = NODE_CONFIG) -> None:
        self.node_config.write_text(text, encoding="utf-8")

    def fake_network(self, ip: str | None = "192.168.1.111", hostname: str = "zovive-node-001"):
        """Patch the route lookup and hostname; returns the route mock."""
        self.patch(node_identity, "_hostname_ip", return_value=None)
        self.patch(socket, "gethostname", return_value=hostname)
        return self.patch(node_identity, "_route_source_ip", return_value=ip)

    def full_node(self) -> None:
        self.write_node_config()
        self.add_interface("eth0", PI_MAC)
        self.dt_serial.write_bytes(PI_SERIAL.encode("ascii") + b"\x00")


class ConfigLoadTests(NodeIdentityTestCase):
    def test_identity_combines_config_and_hardware(self):
        self.full_node()
        self.fake_network()
        identity = get_identity()
        self.assertEqual(identity, VALID_IDENTITY)
        self.assertEqual(list(identity), list(VALID_IDENTITY))  # stable field order for the base station

    def test_node_id_comes_from_environment(self):
        self.write_node_config(NODE_CONFIG.replace('"Node-001"', '"${ZOVIVE_NODE_ID:-zovive-node-01}"'))
        self.fake_network()
        with mock.patch.dict(os.environ, {"ZOVIVE_NODE_ID": "Node-042"}):
            self.assertEqual(get_identity()["node_id"], "Node-042")

        config_loader.clear_cache()
        with mock.patch.dict(os.environ):
            os.environ.pop("ZOVIVE_NODE_ID", None)
            self.assertEqual(get_identity(refresh=True)["node_id"], "zovive-node-01")

    def test_repo_node_config_provides_every_identity_field(self):
        self.add_interface("eth0", PI_MAC)
        self.fake_network()
        with (
            mock.patch.object(paths, "NODE_CONFIG", REPO_NODE_CONFIG),
            mock.patch.dict(os.environ, {"ZOVIVE_NODE_ID": "Node-007"}),
        ):
            node = config_loader.load(REPO_NODE_CONFIG)["node"]
            for key in (*node_identity.REQUIRED_CONFIG_FIELDS, "identity_interface"):
                self.assertIn(key, node)
            identity = get_identity()
        self.assertEqual(identity["node_id"], "Node-007")
        self.assertEqual(identity["camera_id"], "cam01")


class MissingConfigTests(NodeIdentityTestCase):
    def setUp(self):
        super().setUp()
        self.add_interface("eth0", PI_MAC)
        self.fake_network()

    def assert_config_fields_missing(self, identity: dict) -> None:
        for key in node_identity.REQUIRED_CONFIG_FIELDS:
            self.assertIsNone(identity[key], key)
        self.assertEqual(identity["mac"], PI_MAC)  # hardware still discovered

    def test_missing_file_does_not_raise(self):
        with self.assertLogs(LOGGER, "ERROR") as logs:
            identity = get_identity()
        self.assert_config_fields_missing(identity)
        self.assertTrue(any("missing configuration" in line for line in logs.output))

    def test_missing_node_section(self):
        self.write_node_config("runtime:\n  backend: onnx\n")
        with self.assertLogs(LOGGER, "ERROR") as logs:
            identity = get_identity()
        self.assert_config_fields_missing(identity)
        self.assertTrue(any("no 'node' section" in line for line in logs.output))

    def test_malformed_yaml_does_not_raise(self):
        self.write_node_config("node: [unclosed\n")
        with self.assertLogs(LOGGER, "ERROR") as logs:
            identity = get_identity()
        self.assert_config_fields_missing(identity)
        self.assertTrue(any("cannot read node configuration" in line for line in logs.output))

    def test_missing_key_is_reported_by_name(self):
        self.write_node_config(NODE_CONFIG.replace('  forest_name: "Bandipur"\n', ""))
        with self.assertLogs(LOGGER, "ERROR") as logs:
            identity = get_identity()
        self.assertIsNone(identity["forest_name"])
        self.assertEqual(logs.records[0].missing_keys, ["node.forest_name"])

    def test_missing_config_fails_validation(self):
        with self.assertLogs(LOGGER, "ERROR"):
            problems = validate_identity()
        for key in node_identity.REQUIRED_CONFIG_FIELDS:
            self.assertTrue(any(p.startswith(key) for p in problems), key)


class MacAddressTests(NodeIdentityTestCase):
    def test_prefers_wired_over_wireless(self):
        self.add_interface("wlan0", WIFI_MAC)
        self.add_interface("eth0", PI_MAC)
        self.assertEqual(get_mac_address(), PI_MAC)

    def test_predictable_interface_names_count_as_wired(self):
        self.add_interface("wlan0", WIFI_MAC)
        self.add_interface("end0", PI_MAC)
        self.assertEqual(get_mac_address(), PI_MAC)

    def test_configured_interface_wins(self):
        self.add_interface("eth0", PI_MAC)
        self.add_interface("wlan0", WIFI_MAC)
        self.assertEqual(get_mac_address("wlan0"), WIFI_MAC)

    def test_identity_uses_configured_interface(self):
        self.write_node_config(NODE_CONFIG.replace('identity_interface: ""', 'identity_interface: "wlan0"'))
        self.add_interface("eth0", PI_MAC)
        self.add_interface("wlan0", WIFI_MAC)
        self.fake_network()
        self.assertEqual(get_identity()["mac"], WIFI_MAC)

    def test_missing_configured_interface_falls_back_with_warning(self):
        self.add_interface("eth0", PI_MAC)
        with self.assertLogs(LOGGER, "WARNING"):
            self.assertEqual(get_mac_address("eth1"), PI_MAC)

    def test_interface_name_cannot_escape_sysfs(self):
        self.add_interface("eth0", PI_MAC)
        with self.assertLogs(LOGGER, "WARNING"):
            self.assertEqual(get_mac_address("../net/wlan0"), PI_MAC)

    def test_skips_loopback_and_virtual_interfaces(self):
        self.add_interface("lo", "00:00:00:00:00:00", physical=False)
        self.add_interface("docker0", "02:42:ac:11:00:02", physical=False)
        with self.assertLogs(LOGGER, "ERROR"):
            self.assertIsNone(get_mac_address())

    def test_falls_back_to_uuid_without_sysfs(self):
        # Windows: no /sys/class/net at all.
        with (
            mock.patch.object(node_identity, "_SYS_NET", self.sys_net / "absent"),
            mock.patch.object(uuid, "getnode", return_value=0x88A29EC1686F),
        ):
            self.assertEqual(get_mac_address(), PI_MAC)

    def test_random_uuid_node_is_not_a_mac(self):
        with mock.patch.object(node_identity, "_SYS_NET", self.sys_net / "absent"):
            with self.assertLogs(LOGGER, "ERROR") as logs:
                self.assertIsNone(get_mac_address())
        self.assertIn("hardware discovery failed", logs.output[0])

    def test_normalize_mac(self):
        self.assertEqual(normalize_mac("88-A2-9E-C1-68-6F"), PI_MAC)
        self.assertEqual(normalize_mac(" 88:A2:9E:C1:68:6F\n"), PI_MAC)
        self.assertIsNone(normalize_mac("00:00:00:00:00:00"))
        self.assertIsNone(normalize_mac("01:00:5e:00:00:01"))  # multicast
        self.assertIsNone(normalize_mac("ff:ff:ff:ff:ff:ff"))  # broadcast
        self.assertIsNone(normalize_mac("88:a2:9e:c1:68"))
        self.assertIsNone(normalize_mac("not a mac"))
        self.assertIsNone(normalize_mac(None))


class CpuSerialTests(NodeIdentityTestCase):
    def test_reads_device_tree_serial(self):
        self.dt_serial.write_bytes(PI_SERIAL.encode("ascii") + b"\x00")
        self.assertEqual(get_cpu_serial(), PI_SERIAL)

    def test_falls_back_to_cpuinfo(self):
        self.cpuinfo.write_text(
            "processor\t: 0\nRevision\t: d04170\nSerial\t\t: 10000000ABCDEF12\nModel\t\t: Raspberry Pi 5\n",
            encoding="ascii",
        )
        self.assertEqual(get_cpu_serial(), PI_SERIAL)

    def test_all_zero_serial_is_not_a_serial(self):
        self.cpuinfo.write_text("Serial\t\t: 0000000000000000\n", encoding="ascii")
        with (
            mock.patch.object(node_identity, "_on_linux", return_value=True),
            self.assertLogs(LOGGER, "WARNING"),
        ):
            self.assertIsNone(get_cpu_serial())

    def test_missing_on_linux_is_a_warning(self):
        with mock.patch.object(node_identity, "_on_linux", return_value=True):
            with self.assertLogs(LOGGER, "WARNING") as logs:
                self.assertIsNone(get_cpu_serial())
        self.assertIn("cpu serial not found", logs.output[0])

    def test_missing_on_windows_is_expected(self):
        with mock.patch.object(node_identity, "_on_linux", return_value=False):
            with self.assertLogs(LOGGER, "INFO") as logs:
                self.assertIsNone(get_cpu_serial())
        self.assertEqual([r.levelname for r in logs.records], ["INFO"])


class IpAddressTests(NodeIdentityTestCase):
    def test_uses_route_to_target_host_first(self):
        route = self.fake_network(ip="192.168.1.111")
        self.assertEqual(get_ip_address("192.168.1.110"), "192.168.1.111")
        route.assert_called_once_with("192.168.1.110")

    def test_falls_back_to_default_route(self):
        route = self.fake_network()
        route.side_effect = lambda host: None if host == "192.168.1.110" else "10.0.0.5"
        self.assertEqual(get_ip_address("192.168.1.110"), "10.0.0.5")
        self.assertEqual(route.call_args_list[-1], mock.call(node_identity._DEFAULT_ROUTE_PROBE))

    def test_falls_back_to_hostname_address(self):
        with (
            mock.patch.object(node_identity, "_route_source_ip", return_value=None),
            mock.patch.object(socket, "gethostname", return_value="zovive-node-001"),
            mock.patch.object(
                socket,
                "gethostbyname_ex",
                return_value=("zovive-node-001", [], ["127.0.1.1", "192.168.1.50"]),
            ),
        ):
            self.assertEqual(get_ip_address(), "192.168.1.50")

    def test_offline_returns_none_and_warns_once_per_outage(self):
        route = self.fake_network(ip=None)
        with self.assertLogs(LOGGER, "INFO") as logs:
            self.assertIsNone(get_ip_address())
            self.assertIsNone(get_ip_address())
            route.return_value = "192.168.1.111"
            self.assertEqual(get_ip_address(), "192.168.1.111")
        self.assertEqual([r.levelname for r in logs.records], ["WARNING", "INFO"])

    def test_identity_routes_toward_base_station_url(self):
        self.full_node()
        self.transfer_config.write_text('base_station:\n  url: "http://10.20.0.5:8000"\n', encoding="utf-8")
        route = self.fake_network()
        self.assertEqual(get_identity()["ip"], "192.168.1.111")
        route.assert_called_once_with("10.20.0.5")

    def test_identity_without_base_station_uses_default_route(self):
        self.full_node()
        self.transfer_config.write_text('base_station:\n  url: ""\n', encoding="utf-8")
        route = self.fake_network()
        get_identity()
        route.assert_called_once_with(node_identity._DEFAULT_ROUTE_PROBE)

    def test_unreadable_transfer_config_uses_default_route(self):
        self.full_node()
        self.transfer_config.write_text("base_station: [unclosed\n", encoding="utf-8")
        route = self.fake_network()
        with self.assertLogs(LOGGER, "WARNING"):
            get_identity()
        route.assert_called_once_with(node_identity._DEFAULT_ROUTE_PROBE)

    def test_route_lookup_on_loopback_is_rejected(self):
        # Real socket call; a UDP connect() sends nothing.
        self.assertIsNone(node_identity._route_source_ip("127.0.0.1"))

    def test_route_lookup_survives_resolution_failure(self):
        with mock.patch.object(socket, "getaddrinfo", side_effect=socket.gaierror("no such host")):
            self.assertIsNone(node_identity._route_source_ip("base-station.invalid"))

    def test_usable_ip(self):
        usable = node_identity._usable_ip
        self.assertEqual(usable("192.168.1.111"), "192.168.1.111")
        self.assertEqual(usable("fd00::5%eth0"), "fd00::5")
        for value in ("127.0.0.1", "127.0.1.1", "::1", "0.0.0.0", "169.254.10.2", "fe80::1%eth0", "bogus"):  # nosec B104 - test data, not a bind address
            self.assertIsNone(usable(value), value)


class HostnameTests(NodeIdentityTestCase):
    def test_reads_socket_hostname(self):
        with mock.patch.object(socket, "gethostname", return_value=" zovive-node-001\n"):
            self.assertEqual(get_hostname(), "zovive-node-001")

    def test_falls_back_to_platform_node(self):
        with (
            mock.patch.object(socket, "gethostname", side_effect=OSError("no hostname")),
            mock.patch.object(node_identity.platform, "node", return_value="dev-laptop"),
        ):
            self.assertEqual(get_hostname(), "dev-laptop")

    def test_unavailable_hostname_is_none(self):
        with (
            mock.patch.object(socket, "gethostname", return_value=""),
            mock.patch.object(node_identity.platform, "node", return_value=""),
            self.assertLogs(LOGGER, "WARNING"),
        ):
            self.assertIsNone(get_hostname())


class CachingTests(NodeIdentityTestCase):
    def setUp(self):
        super().setUp()
        self.full_node()
        self.route = self.fake_network()

    def test_hardware_is_read_once(self):
        with mock.patch.object(node_identity, "get_mac_address", wraps=get_mac_address) as mac:
            get_identity()
            get_identity()
        mac.assert_called_once()

    def test_refresh_rebuilds(self):
        get_identity()
        self.write_node_config(NODE_CONFIG.replace("Tower-01", "Tower-02"))
        config_loader.clear_cache()
        self.assertEqual(get_identity()["camera_name"], "Tower-01")
        self.assertEqual(get_identity(refresh=True)["camera_name"], "Tower-02")

    def test_ip_is_checked_on_every_call(self):
        self.route.side_effect = ["192.168.1.111", "192.168.1.112"]
        self.assertEqual(get_identity()["ip"], "192.168.1.111")
        identity = get_identity()
        self.assertEqual(identity["ip"], "192.168.1.112")
        self.assertEqual(identity["node_id"], "Node-001")  # IP change, same node

    def test_callers_get_independent_copies(self):
        get_identity()["node_id"] = "tampered"
        self.assertEqual(get_identity()["node_id"], "Node-001")


class ValidationTests(NodeIdentityTestCase):
    def assert_invalid(self, expected_fragment: str, **changes) -> None:
        identity = {**VALID_IDENTITY, **changes}
        with self.assertLogs(LOGGER, "ERROR"):
            problems = validate_identity(identity)
        self.assertTrue(any(expected_fragment in p for p in problems), problems)

    def test_valid_identity_passes(self):
        with self.assertLogs(LOGGER, "INFO") as logs:
            self.assertEqual(validate_identity(dict(VALID_IDENTITY)), [])
        self.assertIn("validation passed", logs.output[-1])

    def test_validates_current_identity_by_default(self):
        self.full_node()
        self.fake_network()
        with self.assertLogs(LOGGER, "INFO"):
            self.assertEqual(validate_identity(), [])

    def test_missing_node_id(self):
        self.assert_invalid("node_id is missing", node_id=None)

    def test_blank_node_id(self):
        self.assert_invalid("node_id is missing", node_id="  ")

    def test_invalid_node_id(self):
        self.assert_invalid("node_id 'Node 001' is invalid", node_id="Node 001")

    def test_missing_camera_id(self):
        self.assert_invalid("camera_id is missing", camera_id=None)

    def test_missing_forest(self):
        self.assert_invalid("forest_name is missing", forest_name="")

    def test_missing_camera_name(self):
        self.assert_invalid("camera_name is missing", camera_name=None)

    def test_missing_mac(self):
        self.assert_invalid("mac address is not available", mac=None)

    def test_malformed_mac(self):
        self.assert_invalid("not a lowercase, colon-separated unicast MAC", mac="88-A2-9E-C1-68-6F")

    def test_missing_cpu_serial_passes_with_warning_on_linux(self):
        with (
            mock.patch.object(node_identity, "_on_linux", return_value=True),
            self.assertLogs(LOGGER, "INFO") as logs,
        ):
            self.assertEqual(validate_identity({**VALID_IDENTITY, "cpu_serial": None}), [])
        self.assertEqual([r.levelname for r in logs.records], ["WARNING", "INFO"])

    def test_missing_cpu_serial_is_optional_on_windows(self):
        with (
            mock.patch.object(node_identity, "_on_linux", return_value=False),
            self.assertLogs(LOGGER, "INFO") as logs,
        ):
            self.assertEqual(validate_identity({**VALID_IDENTITY, "cpu_serial": None}), [])
        self.assertEqual([r.levelname for r in logs.records], ["INFO"])


if __name__ == "__main__":
    unittest.main()
