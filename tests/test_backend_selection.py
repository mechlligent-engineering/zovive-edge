"""Tests for edge_main.build_detector: the ONNX / Hailo backend switch
(config.yaml `runtime.backend`), the CPU fallback added for the Hailo HEF
enhancement request (Hailo init failure -> ONNX instead of a boot
crash-loop), and SAHI wrapping. The node runs one model (single-stage:
the 7-species detector), so an inference config with only a `detector`
section is enough. Uses mocked Onnx/Hailo classes, except where real
HailoDetector runs against a faked HailoRT — this is about wiring, not
real inference.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import edge_main
import paths
from inference.base import InferenceBackendError
from inference.sahi_slicer import SahiDetector

DET_CFG = {"onnx_path": "d.onnx", "hef_path": "d.hef", "classes": ["deer", "tiger"]}
INF_CFG = {"detector": DET_CFG}


class TestBackendSelection(unittest.TestCase):
    def test_onnx_backend_builds_onnx_detector(self):
        with patch("edge_main.OnnxDetector") as mock_det:
            mock_det.return_value = MagicMock(class_names=["deer", "tiger"])

            detector = edge_main.build_detector({"runtime": {"backend": "onnx"}}, INF_CFG)

            mock_det.assert_called_once()
            self.assertIs(detector, mock_det.return_value)

    def test_hailo_backend_builds_hailo_detector_when_available(self):
        with patch("inference.hailo_inference.HailoDetector") as mock_hdet:
            mock_hdet.return_value = MagicMock(class_names=["deer", "tiger"])

            detector = edge_main.build_detector({"runtime": {"backend": "hailo"}}, INF_CFG)

            mock_hdet.assert_called_once()
            self.assertIs(detector, mock_hdet.return_value)
            mock_hdet.return_value.close.assert_not_called()

    def test_hailo_backend_falls_back_to_onnx_on_init_failure(self):
        # Simulates the real Pi-without-HailoRT / laptop case: HailoDetector's
        # __init__ raises InferenceBackendError (see hailo_inference.py's
        # _import_hailo) instead of hailo_platform being importable.
        with (
            patch(
                "inference.hailo_inference.HailoDetector",
                side_effect=InferenceBackendError("no hailo_platform"),
            ),
            patch("edge_main.OnnxDetector") as mock_det,
        ):
            mock_det.return_value = MagicMock(class_names=["deer", "tiger"])

            detector = edge_main.build_detector({"runtime": {"backend": "hailo"}}, INF_CFG)

            # Fell back to the ONNX backend instead of raising / crashing.
            mock_det.assert_called_once()
            self.assertIs(detector, mock_det.return_value)

    def test_sahi_wraps_detector_when_enabled_under_detector_block(self):
        with patch("edge_main.OnnxDetector") as mock_det:
            mock_det.return_value = MagicMock(class_names=["deer", "tiger"])

            det_cfg = {**DET_CFG, "sahi": {"enabled": True, "tile_size": 320}}
            detector = edge_main.build_detector({"runtime": {"backend": "onnx"}}, {"detector": det_cfg})

            self.assertIsInstance(detector, SahiDetector)
            self.assertIs(detector.inner, mock_det.return_value)
            self.assertEqual(detector.config.tile_size, 320)

    def test_sahi_not_wrapped_when_disabled(self):
        with patch("edge_main.OnnxDetector") as mock_det:
            mock_det.return_value = MagicMock(class_names=["deer", "tiger"])

            detector = edge_main.build_detector({"runtime": {"backend": "onnx"}}, INF_CFG)

            self.assertNotIsInstance(detector, SahiDetector)
            self.assertIs(detector, mock_det.return_value)


class TestModelPaths(unittest.TestCase):
    def test_onnx_path_resolves_under_models_dir(self):
        with patch("edge_main.OnnxDetector") as mock_det:
            mock_det.return_value = MagicMock(class_names=["deer", "tiger"])
            edge_main.build_detector({"runtime": {"backend": "onnx"}}, INF_CFG)
        self.assertEqual(mock_det.call_args.args[0], str(paths.MODELS_DIR / "d.onnx"))
        # load() caches config dicts: they must not be modified in place.
        self.assertEqual(DET_CFG["onnx_path"], "d.onnx")

    def test_hef_path_resolves_under_models_dir(self):
        with patch("inference.hailo_inference.HailoDetector") as mock_hdet:
            mock_hdet.return_value = MagicMock(class_names=["deer", "tiger"])
            edge_main.build_detector({"runtime": {"backend": "hailo"}}, INF_CFG)
        self.assertEqual(mock_hdet.call_args.args[0], str(paths.MODELS_DIR / "d.hef"))

    def test_absolute_model_path_is_kept(self):
        absolute = paths.PROJECT_ROOT / "elsewhere" / "d.onnx"
        self.assertEqual(paths.model_file(str(absolute)), absolute)


def _fake_hailort(fail_hef: str | None = None, device_error: Exception | None = None):
    """Stand-in for hailo_platform that records every VDevice it opens.
    RuntimeError plays HailoRT's own exception types (Pi-only)."""
    devices: list[MagicMock] = []

    def hef(path):
        if fail_hef is not None and Path(path).name == fail_hef:
            raise RuntimeError(f"HAILO_INVALID_HEF: {path}")
        h = MagicMock()
        h.get_input_vstream_infos.return_value = [SimpleNamespace(name="in", shape=(640, 640, 3))]
        h.get_output_vstream_infos.return_value = [SimpleNamespace(name="out", shape=(1, 6, 8400))]
        return h

    def vdevice():
        device = MagicMock()
        if device_error is not None:
            device.configure.side_effect = device_error
        devices.append(device)
        return device

    hpf = MagicMock()
    hpf.HEF.side_effect = hef
    hpf.VDevice.side_effect = vdevice
    return hpf, devices


class TestHailoInitFailuresFallBack(unittest.TestCase):
    """Real HailoDetector with HailoRT faked: every Hailo init failure ends
    in the ONNX backend with a warning, and a Hailo device that was already
    opened is released."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def _build_hailo(self, hpf, *, create_hef: bool):
        if create_hef:
            (self.dir / "d.hef").write_bytes(b"")
        det_cfg = {**DET_CFG, "hef_path": str(self.dir / "d.hef")}
        with (
            patch("inference.hailo_inference._import_hailo", return_value=hpf),
            patch("edge_main.OnnxDetector") as mock_det,
            self.assertLogs("edge_main", level="WARNING") as logs,
        ):
            detector = edge_main.build_detector({"runtime": {"backend": "hailo"}}, {"detector": det_cfg})
        self.assertIs(detector, mock_det.return_value)
        [record] = logs.records
        self.assertEqual(record.levelname, "WARNING")
        self.assertIn("falling back to onnx", record.getMessage())
        return record

    def test_missing_hef_falls_back_to_onnx(self):
        hpf, devices = _fake_hailort()
        record = self._build_hailo(hpf, create_hef=False)
        self.assertTrue(record.error.startswith("FileNotFoundError"), record.error)
        hpf.HEF.assert_not_called()
        self.assertEqual(devices, [])

    def test_hailo_device_failure_falls_back_and_releases_the_device(self):
        hpf, devices = _fake_hailort(device_error=RuntimeError("HAILO_OUT_OF_PHYSICAL_DEVICES"))
        record = self._build_hailo(hpf, create_hef=True)
        self.assertTrue(record.error.startswith("InferenceBackendError"), record.error)
        self.assertIn("HAILO_OUT_OF_PHYSICAL_DEVICES", record.error)
        self.assertEqual(len(devices), 1)
        devices[0].release.assert_called_once()

    def test_bad_hef_falls_back_to_onnx(self):
        hpf, devices = _fake_hailort(fail_hef="d.hef")
        record = self._build_hailo(hpf, create_hef=True)
        self.assertTrue(record.error.startswith("InferenceBackendError"), record.error)
        self.assertIn("HAILO_INVALID_HEF", record.error)
        self.assertEqual(devices, [])  # failed before any device was opened


if __name__ == "__main__":
    unittest.main()
