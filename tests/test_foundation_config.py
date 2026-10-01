"""Tests for utils/config_loader.py (Phase 1 foundation)."""

from __future__ import annotations

import dataclasses
import datetime as dt
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from utils.config_loader import (
    AppConfig,
    ConfigError,
    ConfigLoader,
    ConfigReader,
    DirectoryCreationError,
    InferenceBackend,
    IssueKind,
    ParseContext,
    SectionSpec,
    load_config,
    redact_url,
)

REPO_CONFIGS = Path(__file__).resolve().parent.parent / "configs"
CORE_FILES = ("node.yaml", "camera.yaml", "inference.yaml", "storage.yaml", "network.yaml")


class ConfigTestCase(unittest.TestCase):
    """Each test gets a temp copy of the real configs/ files and a temp base dir."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="zovive-cfg-"))
        self.config_dir = self.tmp / "configs"
        self.config_dir.mkdir()
        for name in CORE_FILES:
            shutil.copy(REPO_CONFIGS / name, self.config_dir / name)
        self.env: dict[str, str] = {}

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    # helpers -----------------------------------------------------------------

    def load(self, **kwargs) -> AppConfig:
        return ConfigLoader(self.config_dir, base_dir=self.tmp, env=self.env, **kwargs).load()

    def load_error(self, **kwargs) -> ConfigError:
        with self.assertRaises(ConfigError) as ctx:
            self.load(**kwargs)
        return ctx.exception

    def edit(self, filename: str, **changes) -> None:
        """Set top-level or dotted keys, e.g. edit('camera.yaml', fps=0, **{'resolution.width': 1})."""
        path = self.config_dir / filename
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for dotted, value in changes.items():
            node = data
            *parents, leaf = dotted.split(".")
            for part in parents:
                node = node[part]
            if value is _DELETE:
                del node[leaf]
            else:
                node[leaf] = value
        path.write_text(yaml.safe_dump(data), encoding="utf-8")

    def write(self, filename: str, text: str) -> None:
        (self.config_dir / filename).write_text(text, encoding="utf-8")

    def assertIssue(self, err: ConfigError, source: str, key: str, kind: IssueKind) -> None:
        matches = [i for i in err.issues if i.source == source and i.key == key and i.kind == kind]
        self.assertTrue(matches, f"expected {kind} for {source}:{key}; got:\n{err}")


_DELETE = object()


class ValidConfigTests(ConfigTestCase):
    def test_repo_configs_load_with_defaults(self):
        cfg = self.load()
        self.assertEqual(cfg.node.node_id, "zovive-node-01")
        self.assertEqual(cfg.node.installation.installed_on, dt.date(2026, 9, 1))
        self.assertEqual((cfg.camera.resolution.width, cfg.camera.resolution.height), (640, 360))
        self.assertEqual(cfg.inference.backend, InferenceBackend.ONNX)
        self.assertEqual(len(cfg.inference.classes), 7)
        self.assertEqual(cfg.network.base_station_port, 8000)  # "8000" string coerced
        self.assertEqual(cfg.warnings, ())

    def test_real_repo_directory_is_valid(self):
        cfg = load_config(REPO_CONFIGS, env={})
        self.assertEqual(cfg.config_dir, REPO_CONFIGS)

    def test_relative_paths_resolve_against_base_and_storage_root(self):
        cfg = self.load()
        self.assertEqual(cfg.inference.model_path, (self.tmp / "models" / "detector.onnx").resolve())
        self.assertEqual(cfg.storage.root, (self.tmp / "var").resolve())
        self.assertEqual(cfg.storage.image_dir, cfg.storage.root / "snapshots")

    def test_env_interpolation_overrides_defaults(self):
        self.env.update(ZOVIVE_BASE_STATION_IP="10.0.0.5", ZOVIVE_BASE_STATION_PORT="9443", ZOVIVE_CAMERA_PASSWORD="s3cret")
        cfg = self.load()
        self.assertEqual(cfg.network.base_url, "http://10.0.0.5:9443")
        self.assertIn("s3cret", cfg.camera.rtsp_url)

    def test_config_dir_from_environment(self):
        self.env["ZOVIVE_CONFIG_DIR"] = str(self.config_dir)
        cfg = ConfigLoader(base_dir=self.tmp, env=self.env).load()
        self.assertEqual(cfg.config_dir, self.config_dir.resolve())

    def test_config_objects_are_immutable(self):
        cfg = self.load()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cfg.camera.fps = 30  # type: ignore[misc]
        with self.assertRaises(TypeError):
            cfg.node.raw["node_id"] = "x"  # type: ignore[index]

    def test_credentials_never_appear_in_repr_or_summary(self):
        self.env["ZOVIVE_CAMERA_PASSWORD"] = "s3cret"
        cfg = self.load()
        self.assertNotIn("s3cret", repr(cfg.camera))
        self.assertNotIn("s3cret", str(cfg.summary()))
        self.assertIn("***", cfg.camera.rtsp_url_redacted)

    def test_optional_sections_may_be_omitted(self):
        self.edit("camera.yaml", reconnect=_DELETE, ptz=_DELETE)
        self.edit("node.yaml", installation=_DELETE)
        cfg = self.load()
        self.assertTrue(cfg.camera.reconnect.enabled)
        self.assertFalse(cfg.camera.ptz.enabled)
        self.assertIsNone(cfg.node.installation.installed_on)

    def test_unknown_keys_are_warnings_not_errors(self):
        self.edit("camera.yaml", fpss=8, **{"reconnect.jitter": 0.1})
        cfg = self.load()
        keys = {w.key for w in cfg.warnings}
        self.assertEqual(keys, {"fpss", "reconnect.jitter"})
        self.assertTrue(all(w.kind is IssueKind.UNKNOWN_KEY for w in cfg.warnings))
        self.assertEqual(cfg.camera.raw["fpss"], 8)  # still reachable for newer modules

    def test_reconnect_backoff(self):
        rc = self.load().camera.reconnect
        self.assertEqual([rc.delay_for_attempt(n) for n in (1, 2, 3)], [1.0, 2.0, 4.0])
        self.assertEqual(rc.delay_for_attempt(50), rc.max_delay_sec)


class FileLevelErrorTests(ConfigTestCase):
    def test_missing_config_directory(self):
        err = ConfigLoader(self.tmp / "nope", env={}).load
        with self.assertRaises(ConfigError) as ctx:
            err()
        self.assertEqual(ctx.exception.kinds(), {IssueKind.MISSING_DIR})

    def test_missing_file_is_reported_with_path(self):
        (self.config_dir / "network.yaml").unlink()
        err = self.load_error()
        self.assertIssue(err, "network.yaml", "", IssueKind.MISSING_FILE)
        self.assertIn("network.yaml", str(err))

    def test_malformed_yaml_reports_line_number(self):
        self.write("node.yaml", "node_id: abc\ngps:\n  latitude: [1\n")
        err = self.load_error()
        self.assertIssue(err, "node.yaml", "", IssueKind.PARSE)
        self.assertRegex(str(err), r"line \d+, column \d+")

    def test_duplicate_keys_are_rejected(self):
        self.write("network.yaml", (REPO_CONFIGS / "network.yaml").read_text() + "\nupload_retry_count: 9\n")
        err = self.load_error()
        self.assertIn("duplicate key 'upload_retry_count'", str(err))

    def test_empty_file(self):
        self.write("storage.yaml", "# nothing here\n")
        self.assertIn("file is empty", str(self.load_error()))

    def test_top_level_must_be_mapping(self):
        self.write("inference.yaml", "- a\n- b\n")
        self.assertIssue(self.load_error(), "inference.yaml", "", IssueKind.PARSE)

    def test_permission_denied(self):
        real_read = Path.read_text

        def fake_read(path: Path, *a, **kw):
            if path.name == "camera.yaml":
                raise PermissionError(13, "Permission denied")
            return real_read(path, *a, **kw)

        with mock.patch.object(Path, "read_text", fake_read):
            err = self.load_error()
        self.assertIssue(err, "camera.yaml", "", IssueKind.PERMISSION)

    def test_all_problems_reported_together(self):
        (self.config_dir / "network.yaml").unlink()
        self.edit("camera.yaml", fps=0)
        self.edit("inference.yaml", confidence_threshold=1.5)
        err = self.load_error()
        self.assertEqual({i.source for i in err.issues}, {"network.yaml", "camera.yaml", "inference.yaml"})


class FieldValidationTests(ConfigTestCase):
    def test_missing_required_key(self):
        self.edit("node.yaml", node_id=_DELETE)
        self.assertIssue(self.load_error(), "node.yaml", "node_id", IssueKind.MISSING_KEY)

    def test_missing_required_section_reported_once(self):
        self.edit("node.yaml", gps=_DELETE)
        err = self.load_error()
        self.assertIssue(err, "node.yaml", "gps", IssueKind.MISSING_KEY)
        self.assertEqual(len(err.issues), 1)  # no extra noise for gps.latitude / gps.longitude

    def test_wrong_type(self):
        self.edit("camera.yaml", fps="fast")
        self.assertIssue(self.load_error(), "camera.yaml", "fps", IssueKind.INVALID_TYPE)

    def test_bool_is_not_an_integer(self):
        self.edit("network.yaml", upload_retry_count=True)
        self.assertIssue(self.load_error(), "network.yaml", "upload_retry_count", IssueKind.INVALID_TYPE)

    def test_out_of_range_values(self):
        self.edit("node.yaml", **{"gps.latitude": 123.0})
        self.edit("network.yaml", **{"base_station.port": 70000})
        err = self.load_error()
        self.assertIssue(err, "node.yaml", "gps.latitude", IssueKind.INVALID_VALUE)
        self.assertIssue(err, "network.yaml", "base_station.port", IssueKind.INVALID_VALUE)

    def test_invalid_choice(self):
        self.edit("inference.yaml", backend="tpu")
        self.assertIn("must be one of onnx, hailo", str(self.load_error()))

    def test_invalid_host(self):
        self.env["ZOVIVE_BASE_STATION_IP"] = "not a host!"
        self.assertIssue(self.load_error(), "network.yaml", "base_station.host", IssueKind.INVALID_VALUE)

    def test_invalid_node_id(self):
        self.edit("node.yaml", node_id="bad id with spaces")
        self.assertIssue(self.load_error(), "node.yaml", "node_id", IssueKind.INVALID_VALUE)

    def test_rtsp_url_scheme_and_error_is_redacted(self):
        self.edit("camera.yaml", rtsp_url="http://user:hunter2@cam.local/stream")
        err = self.load_error()
        self.assertIssue(err, "camera.yaml", "rtsp_url", IssueKind.INVALID_VALUE)
        self.assertNotIn("hunter2", str(err))

    def test_missing_env_var_without_default(self):
        self.edit("network.yaml", **{"base_station.host": "${SITE_BASE_IP}"})
        err = self.load_error()
        self.assertIssue(err, "network.yaml", "base_station.host", IssueKind.MISSING_ENV)
        self.assertIn("SITE_BASE_IP", str(err))

    def test_dollar_escape(self):
        self.edit("node.yaml", forest_name="Price $$5 Range")
        self.assertEqual(self.load().node.forest_name, "Price $5 Range")

    def test_hailo_backend_requires_hef(self):
        self.edit("inference.yaml", backend="hailo", **{"hailo.hef_path": None})
        self.assertIssue(self.load_error(), "inference.yaml", "hailo.hef_path", IssueKind.MISSING_KEY)

    def test_hailo_backend_selects_hef(self):
        self.edit("inference.yaml", backend="hailo")
        cfg = self.load()
        self.assertEqual(cfg.inference.active_model_path.suffix, ".hef")

    def test_ptz_enabled_requires_host(self):
        self.edit("camera.yaml", **{"ptz.enabled": True, "ptz.host": None})
        self.assertIssue(self.load_error(), "camera.yaml", "ptz.host", IssueKind.MISSING_KEY)

    def test_cross_field_rules(self):
        self.edit("storage.yaml", video_dir="snapshots", **{"disk.critical_percent": 50})
        self.edit("camera.yaml", **{"reconnect.max_delay_sec": 0.5, "reconnect.initial_delay_sec": 5})
        err = self.load_error()
        self.assertIssue(err, "storage.yaml", "video_dir", IssueKind.INVALID_VALUE)
        self.assertIssue(err, "storage.yaml", "disk.critical_percent", IssueKind.INVALID_VALUE)
        self.assertIssue(err, "camera.yaml", "reconnect.max_delay_sec", IssueKind.INVALID_VALUE)

    def test_empty_and_duplicate_classes(self):
        self.edit("inference.yaml", classes=["deer", "deer", ""])
        err = self.load_error()
        self.assertIn("duplicate entries: deer", str(err))
        self.assertIssue(err, "inference.yaml", "classes[2]", IssueKind.INVALID_TYPE)

    def test_newer_schema_version_rejected(self):
        self.edit("node.yaml", schema_version=99)
        self.assertIssue(self.load_error(), "node.yaml", "schema_version", IssueKind.INVALID_VALUE)

    def test_storage_path_that_is_a_file(self):
        root = self.tmp / "var"
        root.mkdir()
        (root / "snapshots").write_text("oops")
        self.assertIssue(self.load_error(), "storage.yaml", "image_dir", IssueKind.FILESYSTEM)


class StorageDirectoryTests(ConfigTestCase):
    def test_ensure_directories_creates_everything(self):
        storage = self.load().storage
        storage.ensure_directories()
        for path in storage.directories.values():
            self.assertTrue(path.is_dir(), path)

    def test_ensure_directories_permission_error_names_path(self):
        storage = self.load().storage
        with mock.patch.object(Path, "mkdir", side_effect=PermissionError(13, "denied")):
            with self.assertRaises(DirectoryCreationError) as ctx:
                storage.ensure_directories()
        self.assertIn("storage.image_dir", str(ctx.exception))


class ExtensionTests(ConfigTestCase):
    """Future modules add their own YAML without changing utils/config_loader.py."""

    @staticmethod
    def parse_watchdog(r: ConfigReader, ctx: ParseContext) -> dict:
        return {"interval": r.number("interval_sec", 5.0, gt=0)}

    def test_register_extension_section(self):
        self.write("watchdog.yaml", "interval_sec: 2\n")
        spec = SectionSpec("watchdog", "watchdog.yaml", self.parse_watchdog)
        cfg = self.load(extra_sections=[spec])
        self.assertEqual(cfg.section("watchdog"), {"interval": 2.0})
        self.assertIs(cfg.section("camera"), cfg.camera)

    def test_optional_extension_file_uses_defaults(self):
        spec = SectionSpec("watchdog", "watchdog.yaml", self.parse_watchdog, required=False)
        self.assertEqual(self.load(extra_sections=[spec]).section("watchdog"), {"interval": 5.0})

    def test_extension_validation_errors_are_aggregated(self):
        self.write("watchdog.yaml", "interval_sec: -1\n")
        spec = SectionSpec("watchdog", "watchdog.yaml", self.parse_watchdog)
        self.assertIssue(self.load_error(extra_sections=[spec]), "watchdog.yaml", "interval_sec", IssueKind.INVALID_VALUE)

    def test_crashing_parser_is_reported_not_raised_raw(self):
        def broken(r, ctx):
            raise RuntimeError("boom")

        self.write("broken.yaml", "a: 1\n")
        err = self.load_error(extra_sections=[SectionSpec("broken", "broken.yaml", broken)])
        self.assertIssue(err, "broken.yaml", "", IssueKind.INTERNAL)

    def test_duplicate_registration_rejected(self):
        loader = ConfigLoader(self.config_dir, env={})
        with self.assertRaises(ValueError):
            loader.register(SectionSpec("camera", "other.yaml", self.parse_watchdog))

    def test_unknown_section_lookup(self):
        with self.assertRaises(KeyError):
            self.load().section("ota")


class RedactUrlTests(unittest.TestCase):
    def test_redacts_password_only(self):
        self.assertEqual(redact_url("rtsp://admin:pw@10.0.0.1:554/s"), "rtsp://admin:***@10.0.0.1:554/s")
        self.assertEqual(redact_url("rtsp://10.0.0.1/s"), "rtsp://10.0.0.1/s")


if __name__ == "__main__":
    unittest.main()
