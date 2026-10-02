"""Tests for edge_main.build_detector_and_classifier: the ONNX / Hailo
backend switch (config.yaml `runtime.backend`), the CPU fallback added
for the Hailo HEF enhancement request (Hailo init failure -> ONNX
instead of a boot crash-loop), and SAHI wrapping. Uses mocked
Onnx/Hailo classes throughout — this is about wiring, not real
inference (that's onnx_inference.py / hailo_inference.py's own job,
already covered elsewhere).
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import edge_main
from inference.base import InferenceBackendError
from inference.sahi_slicer import SahiDetector

DET_CFG = {"onnx_path": "d.onnx", "hef_path": "d.hef", "classes": ["deer", "tiger"]}
CLS_CFG = {"onnx_path": "c.onnx", "hef_path": "c.hef", "classes": ["deer", "tiger"]}


class TestBackendSelection(unittest.TestCase):
    def test_onnx_backend_builds_onnx_classes(self):
        with patch("edge_main.OnnxDetector") as mock_det, patch("edge_main.OnnxClassifier") as mock_cls:
            mock_det.return_value = MagicMock(class_names=["deer", "tiger"])
            mock_cls.return_value = MagicMock()

            detector, classifier = edge_main.build_detector_and_classifier(
                {"runtime": {"backend": "onnx"}}, {"detector": DET_CFG, "classifier": CLS_CFG}
            )

            mock_det.assert_called_once()
            mock_cls.assert_called_once()
            self.assertIs(detector, mock_det.return_value)

    def test_hailo_backend_builds_hailo_classes_when_available(self):
        with (
            patch("inference.hailo_inference.HailoDetector") as mock_hdet,
            patch("inference.hailo_inference.HailoClassifier") as mock_hcls,
        ):
            mock_hdet.return_value = MagicMock(class_names=["deer", "tiger"])
            mock_hcls.return_value = MagicMock()

            detector, classifier = edge_main.build_detector_and_classifier(
                {"runtime": {"backend": "hailo"}}, {"detector": DET_CFG, "classifier": CLS_CFG}
            )

            mock_hdet.assert_called_once()
            mock_hcls.assert_called_once()
            self.assertIs(detector, mock_hdet.return_value)

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
            patch("edge_main.OnnxClassifier") as mock_cls,
        ):
            mock_det.return_value = MagicMock(class_names=["deer", "tiger"])
            mock_cls.return_value = MagicMock()

            detector, classifier = edge_main.build_detector_and_classifier(
                {"runtime": {"backend": "hailo"}}, {"detector": DET_CFG, "classifier": CLS_CFG}
            )

            # Fell back to the ONNX backend instead of raising / crashing.
            mock_det.assert_called_once()
            mock_cls.assert_called_once()
            self.assertIs(detector, mock_det.return_value)

    def test_sahi_wraps_detector_when_enabled_under_detector_block(self):
        with patch("edge_main.OnnxDetector") as mock_det, patch("edge_main.OnnxClassifier") as mock_cls:
            mock_det.return_value = MagicMock(class_names=["deer", "tiger"])
            mock_cls.return_value = MagicMock()

            det_cfg = {**DET_CFG, "sahi": {"enabled": True, "tile_size": 320}}
            detector, classifier = edge_main.build_detector_and_classifier(
                {"runtime": {"backend": "onnx"}}, {"detector": det_cfg, "classifier": CLS_CFG}
            )

            self.assertIsInstance(detector, SahiDetector)
            self.assertIs(detector.inner, mock_det.return_value)
            self.assertEqual(detector.config.tile_size, 320)

    def test_sahi_not_wrapped_when_disabled(self):
        with patch("edge_main.OnnxDetector") as mock_det, patch("edge_main.OnnxClassifier") as mock_cls:
            mock_det.return_value = MagicMock(class_names=["deer", "tiger"])
            mock_cls.return_value = MagicMock()

            detector, classifier = edge_main.build_detector_and_classifier(
                {"runtime": {"backend": "onnx"}}, {"detector": DET_CFG, "classifier": CLS_CFG}
            )

            self.assertNotIsInstance(detector, SahiDetector)
            self.assertIs(detector, mock_det.return_value)


if __name__ == "__main__":
    unittest.main()
