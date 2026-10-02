"""Camera/base-station credentials come only from the environment.

Covers both config paths (utils.config_loader.load() for the runtime YAML
files, ConfigLoader/load_config() for the typed core files), startup
failing when ZOVIVE_CAMERA_USERNAME / ZOVIVE_CAMERA_PASSWORD are missing,
and a guard that no tracked config file carries a credential value.
"""

from __future__ import annotations

import os
import re
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import yaml

import edge_main
import paths
from tests.helpers import TEST_CAMERA_ENV, camera_credentials
from utils import config_loader
from utils.config_loader import ConfigError, IssueKind, load, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_CONFIGS = REPO_ROOT / "configs"
CAMERA_VARS = ("ZOVIVE_CAMERA_USERNAME", "ZOVIVE_CAMERA_PASSWORD")


@contextmanager
def without_camera_credentials(**overrides: str) -> Iterator[None]:
    """os.environ with both camera variables removed (even if the developer
    has them exported), plus optional overrides."""
    config_loader.clear_cache()
    try:
        with mock.patch.dict(os.environ):
            for var in CAMERA_VARS:
                os.environ.pop(var, None)
            os.environ.update(overrides)
            yield
    finally:
        config_loader.clear_cache()


class RuntimeLoadTests(unittest.TestCase):
    """load(paths.RTSP_CONFIG), the path edge_main.py / RtspReader / ONVIF use."""

    def test_credentials_come_from_environment(self):
        with camera_credentials():
            cam = load(paths.RTSP_CONFIG)["camera"]
        self.assertEqual(cam["username"], TEST_CAMERA_ENV["ZOVIVE_CAMERA_USERNAME"])
        self.assertEqual(cam["password"], TEST_CAMERA_ENV["ZOVIVE_CAMERA_PASSWORD"])
        # URL templates are left for RtspReader to fill in.
        self.assertIn("{user}:{pass}@{host}", cam["sub_stream"]["url"])

    def test_missing_credentials_fail_naming_both_variables(self):
        with without_camera_credentials(), self.assertRaises(ConfigError) as ctx:
            load(paths.RTSP_CONFIG)
        err = ctx.exception
        self.assertEqual(err.kinds(), {IssueKind.MISSING_ENV})
        self.assertEqual({i.key for i in err.issues}, {"camera.username", "camera.password"})
        for var in CAMERA_VARS:
            self.assertIn(var, str(err))

    def test_empty_password_counts_as_missing(self):
        # nosec B106: the empty string IS the case under test (a blank
        # password must be rejected); it is not a hardcoded credential.
        with without_camera_credentials(ZOVIVE_CAMERA_USERNAME="u", ZOVIVE_CAMERA_PASSWORD=""):  # nosec B106
            with self.assertRaises(ConfigError) as ctx:
                load(paths.RTSP_CONFIG)
        self.assertEqual([i.key for i in ctx.exception.issues], ["camera.password"])

    def test_error_never_contains_a_credential_value(self):
        with without_camera_credentials(ZOVIVE_CAMERA_USERNAME="visible-user-value"):
            with self.assertRaises(ConfigError) as ctx:
                load(paths.RTSP_CONFIG)
        self.assertNotIn("visible-user-value", str(ctx.exception))

    def test_failed_load_is_not_cached(self):
        with without_camera_credentials(), self.assertRaises(ConfigError):
            load(paths.RTSP_CONFIG)
        with camera_credentials():
            self.assertEqual(load(paths.RTSP_CONFIG)["camera"]["password"], "test-pass")

    def test_optional_api_key_defaults_to_empty(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("ZOVIVE_BASE_STATION_API_KEY", None)
            config_loader.clear_cache()
            try:
                self.assertEqual(load(paths.TRANSFER_CONFIG)["base_station"]["api_key"], "")
            finally:
                config_loader.clear_cache()

    def test_missing_optional_file_is_empty_and_required_file_errors(self):
        missing = REPO_ROOT / "configs" / "does_not_exist.yaml"
        self.assertEqual(load(missing, required=False), {})
        config_loader.clear_cache()
        with self.assertRaises(ConfigError) as ctx:
            load(missing)
        self.assertEqual(ctx.exception.kinds(), {IssueKind.MISSING_FILE})


class TypedLoaderTests(unittest.TestCase):
    """configs/camera.yaml through load_config() uses the same rule."""

    def test_missing_credentials_fail(self):
        with self.assertRaises(ConfigError) as ctx:
            load_config(REPO_CONFIGS, env={})
        err = ctx.exception
        self.assertIn(IssueKind.MISSING_ENV, err.kinds())
        for var in CAMERA_VARS:
            self.assertIn(var, str(err))

    def test_empty_username_counts_as_missing(self):
        env = dict(TEST_CAMERA_ENV, ZOVIVE_CAMERA_USERNAME="")
        with self.assertRaises(ConfigError) as ctx:
            load_config(REPO_CONFIGS, env=env)
        self.assertIn("ZOVIVE_CAMERA_USERNAME", str(ctx.exception))

    def test_credentials_from_environment(self):
        cfg = load_config(REPO_CONFIGS, env=TEST_CAMERA_ENV)
        self.assertIn("test-user:test-pass@", cfg.camera.rtsp_url)
        self.assertNotIn("test-pass", cfg.camera.rtsp_url_redacted)


class EdgeMainStartupTests(unittest.TestCase):
    def test_startup_fails_before_loading_models(self):
        build = mock.Mock(side_effect=AssertionError("models must not load without credentials"))
        with (
            without_camera_credentials(),
            mock.patch.object(edge_main, "init_db"),
            mock.patch.object(edge_main, "migrate"),
            mock.patch.object(edge_main, "build_detector_and_classifier", build),
            self.assertRaises(ConfigError) as ctx,
        ):
            edge_main._run({}, None, None)
        build.assert_not_called()
        self.assertIn("ZOVIVE_CAMERA_PASSWORD", str(ctx.exception))

    def test_main_exits_with_config_error_code(self):
        err = ConfigError(
            [
                config_loader.ConfigIssue(
                    "rtsp_config.yaml", "camera.password", "missing", IssueKind.MISSING_ENV
                )
            ]
        )
        with (
            mock.patch.object(edge_main, "run", side_effect=err),
            mock.patch("sys.argv", ["edge_main.py"]),
            mock.patch("logging.shutdown"),
            self.assertLogs("edge_main", level="CRITICAL") as logs,
            self.assertRaises(SystemExit) as ctx,
        ):
            edge_main.main()
        self.assertEqual(ctx.exception.code, edge_main.EXIT_CONFIG)
        self.assertIn("refusing to start", logs.output[0])


# --- Guard: no credential value in any tracked config file ----------------

_SECRET_KEY_RE = re.compile(
    r"^(user(name)?|pass|passwd|.*password|secret|.*_secret|api_?key|.*_api_?key|token|.*_token)$",
    re.IGNORECASE,
)
# Allowed values: empty, or one ${VAR} with no default or an empty default.
_ENV_REF_RE = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*(:-)?\}$")
# user:pass@ inside a URL: each part must be a {template} or a ${VAR} (no default).
_URL_USERINFO_RE = re.compile(r"://([^/@\s]+)@")
_USERINFO_PART_RE = re.compile(r"^(\{[a-z_]+\}|\$\{[A-Za-z_][A-Za-z0-9_]*\})$")


def _walk(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")
    else:
        yield path, node


def credential_problems(text: str) -> list[str]:
    problems = []
    for key_path, value in _walk(yaml.safe_load(text) or {}):
        leaf = key_path.rsplit(".", 1)[-1]
        is_flag = isinstance(value, bool)  # e.g. a feature switch, never a credential
        if (
            _SECRET_KEY_RE.search(leaf)
            and not is_flag
            and value not in (None, "")
            and not (isinstance(value, str) and _ENV_REF_RE.match(value))
        ):
            problems.append(f"{key_path}: credential must be a ${{VAR}} reference, not a literal")
        if isinstance(value, str):
            for userinfo in _URL_USERINFO_RE.findall(value):
                if not all(_USERINFO_PART_RE.match(part) for part in userinfo.split(":")):
                    problems.append(
                        f"{key_path}: URL credentials must be {{user}}/{{pass}} or ${{VAR}} without a default"
                    )
    return problems


class NoCommittedCredentialsTests(unittest.TestCase):
    def test_tracked_configs_hold_no_credentials(self):
        files = sorted(REPO_CONFIGS.glob("*.yaml"))
        self.assertTrue(files)
        problems = [
            f"{f.name}: {p}" for f in files for p in credential_problems(f.read_text(encoding="utf-8"))
        ]
        self.assertEqual(problems, [], "\n".join(problems))

    def test_env_example_leaves_credentials_blank(self):
        lines = (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        values = dict(
            line.split("=", 1) for line in lines if "=" in line and not line.lstrip().startswith("#")
        )
        for var in (*CAMERA_VARS, "ZOVIVE_BASE_STATION_API_KEY"):
            self.assertEqual(values.get(var), "", f"{var} must be blank in .env.example")

    def test_guard_catches_literals(self):
        self.assertTrue(credential_problems('camera:\n  password: "changeme"\n'))
        self.assertTrue(credential_problems('camera:\n  password: "${ZOVIVE_CAMERA_PASSWORD:-changeme}"\n'))
        self.assertTrue(credential_problems('url: "rtsp://admin:hunter2@10.0.0.1/s"\n'))
        self.assertTrue(credential_problems('url: "rtsp://${U}:${P:-changeme}@h/s"\n'))
        self.assertEqual(credential_problems('camera:\n  password: "${ZOVIVE_CAMERA_PASSWORD}"\n'), [])
        self.assertEqual(credential_problems('url: "rtsp://{user}:{pass}@{host}/s"\n'), [])
        self.assertTrue(credential_problems("camera:\n  password: 123456\n"))  # unquoted number
        self.assertEqual(credential_problems("sahi:\n  run_full_frame_pass: true\n"), [])


if __name__ == "__main__":
    unittest.main()
