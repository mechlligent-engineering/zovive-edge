"""The camera address: one deployment default, overridable per node.

configs/rtsp_config.yaml (RtspReader, the ONVIF lens camera,
tools/camera_probe.py) and configs/camera.yaml (load_config()) both read
the host as ${ZOVIVE_CAMERA_HOST:-<default>}. These tests pin that default,
check that the environment variable overrides it, that every component
builds its addresses from the same configured host, and that load()
rejects an unusable host (empty, IPv6, URL, host:port, ...) before the
service starts. Everything runs against fakes; no camera is contacted.
"""

from __future__ import annotations

import io
import os
import re
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr
from pathlib import Path
from unittest import mock
from urllib.parse import urlsplit

import edge_main
import paths
from camera_control import camera_api
from capture.rtsp_reader import RtspReader
from logger_setup import JsonFormatter
from queues.frame_queue import LatestFrameSlot
from tests.helpers import TEST_CAMERA_ENV
from tools import camera_probe
from utils import config_loader
from utils.config_loader import (
    ConfigError,
    IssueKind,
    camera_host_problem,
    camera_host_source,
    load,
    load_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_CONFIGS = REPO_ROOT / "configs"
HOST_VAR = "ZOVIVE_CAMERA_HOST"
DEFAULT_CAMERA_HOST = "192.168.1.250"  # the CP Plus camera of the current deployment
OTHER_DEPLOYMENTS = ("10.20.30.40", "cam-node-07.local")
CREDENTIAL_VARS = tuple(TEST_CAMERA_ENV)

_HOST_REF_RE = re.compile(r"\$\{" + HOST_VAR + r"(?::-([^}]*))?\}")
_REQUIRED_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")  # no default = required
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
# Camera-related code must take the address from configuration, never a literal.
CAMERA_SOURCES = ("camera_control", "capture", "tools", "edge_main.py")


@contextmanager
def camera_env(host: str | None = None, drop: tuple[str, ...] = ()) -> Iterator[None]:
    """os.environ with fake credentials minus `drop`, and ZOVIVE_CAMERA_HOST
    set to `host` or removed when None (even if the developer exported it)."""
    config_loader.clear_cache()
    try:
        with mock.patch.dict(os.environ, TEST_CAMERA_ENV):
            for var in (HOST_VAR, *drop):
                os.environ.pop(var, None)
            if host is not None:
                os.environ[HOST_VAR] = host
            yield
    finally:
        config_loader.clear_cache()


def typed_env(host: str | None = None) -> dict[str, str]:
    env = dict(TEST_CAMERA_ENV)
    if host is not None:
        env[HOST_VAR] = host
    return env


def env_example_values(include_commented: bool = False) -> dict[str, str]:
    values = {}
    for line in (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#"):
            if not include_commented:
                continue
            line = line.lstrip("#").strip()
        name, sep, value = line.partition("=")
        if sep and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            values[name] = value
    return values


def runtime_host(host: str | None = None) -> str:
    with camera_env(host):
        return load(paths.RTSP_CONFIG)["camera"]["host"]


def reader_for(host: str | None) -> RtspReader:
    with camera_env(host):
        return RtspReader(LatestFrameSlot(name="test_camera_host_slot"))


class DefaultHostTests(unittest.TestCase):
    def test_runtime_config_defaults_to_deployed_camera(self):
        self.assertEqual(runtime_host(), DEFAULT_CAMERA_HOST)

    def test_typed_config_defaults_to_deployed_camera(self):
        cfg = load_config(REPO_CONFIGS, env=typed_env())
        self.assertEqual(urlsplit(cfg.camera.rtsp_url).hostname, DEFAULT_CAMERA_HOST)
        self.assertEqual(cfg.camera.ptz.host, DEFAULT_CAMERA_HOST)

    def test_every_config_reference_shares_one_default(self):
        defaults: dict[str, set[str | None]] = {}
        for path in sorted(REPO_CONFIGS.glob("*.yaml")):
            for default in _HOST_REF_RE.findall(path.read_text(encoding="utf-8")):
                defaults.setdefault(path.name, set()).add(default)
        self.assertLessEqual({"rtsp_config.yaml", "camera.yaml"}, set(defaults))
        self.assertEqual(set().union(*defaults.values()), {DEFAULT_CAMERA_HOST}, defaults)

    def test_env_example_documents_the_same_default(self):
        self.assertEqual(env_example_values(include_commented=True).get(HOST_VAR), DEFAULT_CAMERA_HOST)
        # Left commented out: a blank ZOVIVE_CAMERA_HOST= does not fall back
        # to the default; load() rejects the empty host instead.
        self.assertNotIn(HOST_VAR, env_example_values())

    def test_camera_code_hardcodes_no_address(self):
        found = []
        for name in CAMERA_SOURCES:
            root = REPO_ROOT / name
            for path in sorted(root.rglob("*.py")) if root.is_dir() else [root]:
                for ip in _IPV4_RE.findall(path.read_text(encoding="utf-8")):
                    found.append(f"{path.relative_to(REPO_ROOT)}: {ip}")
        self.assertEqual(found, [])


class OverrideTests(unittest.TestCase):
    def test_environment_overrides_runtime_default(self):
        for host in OTHER_DEPLOYMENTS:
            with self.subTest(host=host):
                self.assertEqual(runtime_host(host), host)

    def test_environment_overrides_typed_default(self):
        for host in OTHER_DEPLOYMENTS:
            with self.subTest(host=host):
                cfg = load_config(REPO_CONFIGS, env=typed_env(host))
                self.assertEqual(urlsplit(cfg.camera.rtsp_url).hostname, host)
                self.assertEqual(cfg.camera.ptz.host, host)


class StreamUrlTests(unittest.TestCase):
    def test_sub_and_main_stream_use_configured_host(self):
        for host, expected in ((None, DEFAULT_CAMERA_HOST), *((h, h) for h in OTHER_DEPLOYMENTS)):
            reader = reader_for(host)
            for name, url, subtype in (
                ("sub", reader.sub_url, "subtype=1"),
                ("main", reader.main_url, "subtype=0"),
            ):
                with self.subTest(host=expected, stream=name):
                    parts = urlsplit(url)
                    self.assertEqual(parts.hostname, expected)
                    self.assertEqual(parts.port, 554)
                    self.assertEqual(parts.username, TEST_CAMERA_ENV["ZOVIVE_CAMERA_USERNAME"])
                    self.assertIn(subtype, parts.query)


class CameraProbeTests(unittest.TestCase):
    @staticmethod
    def _unreachable_streams():
        seen: list[str] = []

        def measure(url, seconds):
            seen.append(url)
            return {"url": url, "opened": False}

        return measure, seen

    def test_probe_uses_configured_host(self):
        for host, expected in ((None, DEFAULT_CAMERA_HOST), (OTHER_DEPLOYMENTS[0], OTHER_DEPLOYMENTS[0])):
            with self.subTest(host=expected):
                with camera_env(host):
                    rtsp_cfg = load(paths.RTSP_CONFIG)
                measure, seen = self._unreachable_streams()
                onvif_hosts: list[str] = []

                def no_camera(cfg, hosts=onvif_hosts):
                    hosts.append(cfg["camera"]["host"])
                    raise ConnectionRefusedError("fake: no camera here")

                report = camera_probe.run_probe(
                    rtsp_cfg, 0.0, 0, False, camera_factory=no_camera, stream_measurer=measure
                )
                self.assertEqual(report["host"], expected)
                self.assertEqual(onvif_hosts, [expected])
                self.assertEqual(len(seen), 2)  # configured sub + main streams
                self.assertEqual({urlsplit(u).hostname for u in seen}, {expected})

    def test_probe_onvif_connects_to_configured_host(self):
        host = OTHER_DEPLOYMENTS[0]
        with camera_env(host):
            rtsp_cfg = load(paths.RTSP_CONFIG)
        onvif_cls = mock.Mock(side_effect=ConnectionRefusedError("fake: no camera here"))
        measure, _ = self._unreachable_streams()
        with (
            mock.patch.object(camera_api, "_import_onvif", return_value=onvif_cls),
            mock.patch.object(camera_api, "_onvif_transport", return_value=None),
        ):
            report = camera_probe.run_probe(rtsp_cfg, 0.0, 0, False, stream_measurer=measure)
        onvif_cls.assert_called_once()
        self.assertEqual(
            onvif_cls.call_args.args,
            (host, 80, TEST_CAMERA_ENV["ZOVIVE_ONVIF_USERNAME"], TEST_CAMERA_ENV["ZOVIVE_ONVIF_PASSWORD"]),
        )
        self.assertIn("ConnectionRefusedError", report["onvif_error"])


class MissingCredentialTests(unittest.TestCase):
    """No credentials means a ConfigError naming the variables, raised before
    any URL is built or connection attempted, whatever the host."""

    def test_reader_refuses_to_start_without_credentials(self):
        for host in (None, OTHER_DEPLOYMENTS[0]):
            with self.subTest(host=host), camera_env(host, drop=CREDENTIAL_VARS):
                with mock.patch("capture.rtsp_reader.open_rtsp_capture") as opener:
                    with self.assertRaises(ConfigError) as ctx:
                        RtspReader(LatestFrameSlot(name="test_camera_host_slot"))
                opener.assert_not_called()
                self.assertEqual(ctx.exception.kinds(), {IssueKind.MISSING_ENV})
                for var in CREDENTIAL_VARS:
                    self.assertIn(var, str(ctx.exception))

    def test_probe_exits_with_config_error_without_credentials(self):
        drop = ("ZOVIVE_CAMERA_PASSWORD", "ZOVIVE_ONVIF_PASSWORD")
        stderr = io.StringIO()
        with (
            camera_env(OTHER_DEPLOYMENTS[0], drop=drop),
            mock.patch.object(camera_probe, "run_probe") as run_probe,
            mock.patch.object(camera_probe.logging, "basicConfig"),
            redirect_stderr(stderr),
        ):
            code = camera_probe.main([])
        self.assertEqual(code, camera_probe.EXIT_CONFIG)
        run_probe.assert_not_called()
        for var in drop:
            self.assertIn(var, stderr.getvalue())

    def test_env_example_lists_every_required_variable_blank(self):
        values = env_example_values()
        for name in ("rtsp_config.yaml", "camera.yaml"):
            required = set(_REQUIRED_REF_RE.findall((REPO_CONFIGS / name).read_text(encoding="utf-8")))
            self.assertTrue(required, name)
            for var in sorted(required):
                with self.subTest(file=name, var=var):
                    self.assertIn(var, values, f"{var} is required by {name} but missing from .env.example")
                    self.assertEqual(values[var], "", f"{var} must be blank in .env.example")


# value -> the reason camera_host_problem() gives for it
INVALID_HOSTS = {
    "": "is empty",
    "   ": "is empty",
    "\t": "is empty",
    "192.168.1.250 ": "contains whitespace",
    "192.168.1.256": "is not a valid IPv4 address",
    "192.168.1": "is not a valid IPv4 address",
    "192.168.001.250": "is not a valid IPv4 address",
    "cam_01.lan": "is not a valid IPv4 address or hostname",
    "-cam.lan": "is not a valid IPv4 address or hostname",
    "cam..lan": "is not a valid IPv4 address or hostname",
    "rtsp://192.168.1.250": "is a URL",
    "rtsp://admin:hunter2@192.168.1.250:554/video/live": "is a URL",
    "192.168.1.250:554": "includes a port",
    "cam.lan:554": "includes a port",
    "fe80::1": "is an IPv6 address",
    "::1": "is an IPv6 address",
    "[2001:db8::1]": "is an IPv6 address",
}
VALID_HOSTS = (DEFAULT_CAMERA_HOST, "10.20.30.40", "cam-node-07.local", "camera", "CAM-01.lan", "cam.lan.")


@contextmanager
def rtsp_config_file(text: str) -> Iterator[Path]:
    """A throwaway rtsp_config.yaml (the name selects load()'s camera check)."""
    config_loader.clear_cache()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rtsp_config.yaml"
            path.write_text(text, encoding="utf-8")
            yield path
    finally:
        config_loader.clear_cache()


class HostRulesTests(unittest.TestCase):
    def test_rejected_values_and_reasons(self):
        for host, reason in INVALID_HOSTS.items():
            with self.subTest(host=host):
                self.assertIn(reason, camera_host_problem(host) or "")

    def test_ipv4_and_hostnames_accepted(self):
        for host in VALID_HOSTS:
            with self.subTest(host=host):
                self.assertIsNone(camera_host_problem(host))

    def test_source_tells_environment_from_default(self):
        self.assertEqual(camera_host_source(DEFAULT_CAMERA_HOST, {}), "config default")
        self.assertEqual(camera_host_source("10.20.30.40", {HOST_VAR: "10.20.30.40"}), HOST_VAR)
        self.assertEqual(camera_host_source("", {HOST_VAR: ""}), HOST_VAR)
        # A literal host in YAML ignores the variable: not from the environment.
        self.assertEqual(camera_host_source("10.0.0.9", {HOST_VAR: "10.20.30.40"}), "config default")


class HostValidationTests(unittest.TestCase):
    """load(paths.RTSP_CONFIG) refuses an unusable camera.host with one clear
    ConfigError, like a missing credential."""

    def test_invalid_override_refuses_to_load(self):
        expected = [("rtsp_config.yaml", "camera.host", IssueKind.INVALID_VALUE)]
        for host, reason in INVALID_HOSTS.items():
            with self.subTest(host=host):
                with camera_env(host), self.assertRaises(ConfigError) as ctx:
                    load(paths.RTSP_CONFIG)
                self.assertEqual([(i.source, i.key, i.kind) for i in ctx.exception.issues], expected)
                message = str(ctx.exception)
                self.assertIn(reason, message)
                self.assertIn(f"from {HOST_VAR}", message)
                self.assertNotIn("hunter2", message)  # URL credentials are never echoed
                for secret in TEST_CAMERA_ENV.values():
                    self.assertNotIn(secret, message)

    def test_unset_and_valid_overrides_still_load(self):
        self.assertEqual(runtime_host(), DEFAULT_CAMERA_HOST)
        for host in OTHER_DEPLOYMENTS:
            with self.subTest(host=host):
                self.assertEqual(runtime_host(host), host)

    def test_bad_default_names_the_config_file(self):
        text = 'camera:\n  host: "${ZOVIVE_CAMERA_HOST:-192.168.1.256}"\n'
        with camera_env(), rtsp_config_file(text) as path, self.assertRaises(ConfigError) as ctx:
            load(path)
        self.assertIn("the default in rtsp_config.yaml", str(ctx.exception))
        self.assertIn("is not a valid IPv4 address", str(ctx.exception))

    def test_missing_host_and_wrong_types(self):
        cases = {
            "reconnect:\n  stall_timeout_sec: 5.0\n": ("camera", IssueKind.MISSING_KEY),
            'camera: "192.168.1.250"\n': ("camera", IssueKind.INVALID_TYPE),
            "camera:\n  username: u\n": ("camera.host", IssueKind.MISSING_KEY),
            "camera:\n  host:\n": ("camera.host", IssueKind.MISSING_KEY),
            "camera:\n  host: 1234\n": ("camera.host", IssueKind.INVALID_TYPE),
            "camera:\n  host: [192.168.1.250]\n": ("camera.host", IssueKind.INVALID_TYPE),
        }
        for text, (key, kind) in cases.items():
            with self.subTest(yaml=text), rtsp_config_file(text) as path:
                with self.assertRaises(ConfigError) as ctx:
                    load(path)
                self.assertEqual([(i.key, i.kind) for i in ctx.exception.issues], [(key, kind)])

    def test_invalid_host_reported_with_missing_credentials(self):
        drop = ("ZOVIVE_CAMERA_PASSWORD", "ZOVIVE_ONVIF_PASSWORD")
        with camera_env("", drop=drop), self.assertRaises(ConfigError) as ctx:
            load(paths.RTSP_CONFIG)
        found = {(i.key, i.kind) for i in ctx.exception.issues}
        self.assertEqual(
            found,
            {
                ("camera.password", IssueKind.MISSING_ENV),
                ("camera.onvif_password", IssueKind.MISSING_ENV),
                ("camera.host", IssueKind.INVALID_VALUE),
            },
        )

    def test_failed_load_is_not_cached(self):
        config_loader.clear_cache()
        try:
            with mock.patch.dict(os.environ, TEST_CAMERA_ENV):
                os.environ[HOST_VAR] = "192.168.1.250:554"
                with self.assertRaises(ConfigError):
                    load(paths.RTSP_CONFIG)
                os.environ[HOST_VAR] = OTHER_DEPLOYMENTS[0]
                self.assertEqual(load(paths.RTSP_CONFIG)["camera"]["host"], OTHER_DEPLOYMENTS[0])
        finally:
            config_loader.clear_cache()


class TypedHostValidationTests(unittest.TestCase):
    """configs/camera.yaml through load_config() (python -m utils.config_loader)
    applies the same host rule, so the validation command and the service agree."""

    def test_invalid_override_rejected(self):
        for host in INVALID_HOSTS:
            with self.subTest(host=host):
                with self.assertRaises(ConfigError) as ctx:
                    load_config(REPO_CONFIGS, env=typed_env(host))
                keys = {i.key for i in ctx.exception.issues if i.source == "camera.yaml"}
                self.assertTrue(keys & {"rtsp_url", "ptz.host"}, str(ctx.exception))
                message = str(ctx.exception)
                self.assertNotIn("hunter2", message)
                for value in TEST_CAMERA_ENV.values():  # neither usernames nor passwords
                    self.assertNotIn(value, message)

    def test_url_errors_mask_username_and_password(self):
        cases = {
            "http://user:hunter2@cam.local/stream": "http://***@cam.local/stream",
            "rtsp://user:p@ss@:554/live": "rtsp://***@:554/live",
            "rtsp://cam.local/live?tag=a@b": "rtsp://***@b",  # everything before the last '@' is masked
            "rtsp://192.168.1.250:554/live": "rtsp://192.168.1.250:554/live",
        }
        for url, shown in cases.items():
            with self.subTest(url=url):
                self.assertEqual(config_loader._url_for_error(url), shown)

    def test_valid_hosts_accepted(self):
        for host in VALID_HOSTS:
            with self.subTest(host=host):
                cfg = load_config(REPO_CONFIGS, env=typed_env(host))
                self.assertEqual(cfg.camera.ptz.host, host)
                self.assertEqual(urlsplit(cfg.camera.rtsp_url).hostname, host.lower())


class _ModelLoadReached(Exception):
    pass


@contextmanager
def edge_startup(build: mock.Mock) -> Iterator[None]:
    """edge_main._run with the database stubbed and model loading replaced by `build`."""
    with (
        mock.patch.object(edge_main, "init_db"),
        mock.patch.object(edge_main, "migrate"),
        mock.patch.object(edge_main, "build_detector", build),
    ):
        yield


class StartupTests(unittest.TestCase):
    def test_invalid_host_stops_startup_before_models_load(self):
        for host in ("", "fe80::1", "rtsp://192.168.1.250"):
            build = mock.Mock(side_effect=AssertionError("models must not load with an invalid camera host"))
            with self.subTest(host=host):
                with camera_env(host), edge_startup(build), self.assertRaises(ConfigError) as ctx:
                    edge_main._run({}, None, None)
                build.assert_not_called()
                self.assertIn("camera.host", str(ctx.exception))

    def test_startup_logs_host_and_source_without_secrets(self):
        cases = (
            (None, DEFAULT_CAMERA_HOST, "config default"),
            (OTHER_DEPLOYMENTS[0], OTHER_DEPLOYMENTS[0], HOST_VAR),
        )
        formatter = JsonFormatter("detect")
        for host, expected_host, expected_source in cases:
            build = mock.Mock(side_effect=_ModelLoadReached)
            with self.subTest(host=expected_host):
                with (
                    camera_env(host),
                    edge_startup(build),
                    self.assertLogs("edge_main", level="INFO") as logs,
                    self.assertRaises(_ModelLoadReached),
                ):
                    edge_main._run({}, None, None)
                host_logs = [r for r in logs.records if r.getMessage() == "camera host"]
                self.assertEqual(len(host_logs), 1)
                self.assertEqual(host_logs[0].camera_host, expected_host)
                self.assertEqual(host_logs[0].camera_host_source, expected_source)
                rendered = "\n".join(formatter.format(r) for r in logs.records)
                self.assertNotIn("rtsp://", rendered)
                for secret in TEST_CAMERA_ENV.values():
                    self.assertNotIn(secret, rendered)


class ProbeHostTests(unittest.TestCase):
    def test_probe_exits_with_config_error_for_invalid_host(self):
        for host in ("", "192.168.1.250:554"):
            stderr = io.StringIO()
            with self.subTest(host=host):
                with (
                    camera_env(host),
                    mock.patch.object(camera_probe, "run_probe") as run_probe,
                    mock.patch.object(camera_probe.logging, "basicConfig"),
                    redirect_stderr(stderr),
                ):
                    code = camera_probe.main([])
                self.assertEqual(code, camera_probe.EXIT_CONFIG)
                run_probe.assert_not_called()
                self.assertIn("camera.host", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
