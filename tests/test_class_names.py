"""Tests for inference/class_names.py and its use by the detector
loaders: species names come out canonical, and a model whose own class
order differs from the configured list is refused at load time (sessions
are faked; no model files needed). Single-stage: the detector is the
node's only model, so the shipped config and manifest list nothing else."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import paths
from inference.base import InferenceBackendError
from inference.class_names import canonical_class_name, checked_class_names, embedded_class_names
from inference.onnx_inference import OnnxDetector
from utils.config_loader import load

# model.names of ZOVIVE_FINAL_150E_BEST.pt as Ultralytics' ONNX export
# writes it: str(dict), with spaces in two species.
DETECTOR_NAMES = (
    "{0: 'deer', 1: 'gaur', 2: 'leopard', 3: 'elephant', 4: 'sloth bear', 5: 'tiger', 6: 'wild boar'}"
)
DETECTOR_ORDER = ["deer", "gaur", "leopard", "elephant", "sloth_bear", "tiger", "wild_boar"]


def _session(names: str | None) -> MagicMock:
    session = MagicMock()
    session.get_modelmeta.return_value = SimpleNamespace(
        custom_metadata_map={} if names is None else {"names": names}
    )
    session.get_inputs.return_value = [SimpleNamespace(name="images")]
    return session


class TestCanonicalClassName(unittest.TestCase):
    def test_spellings_collapse_to_lowercase_underscore(self):
        for raw in ("sloth bear", "Sloth Bear", "sloth_bear", " sloth-bear ", "SLOTH  BEAR"):
            self.assertEqual(canonical_class_name(raw), "sloth_bear")


class TestEmbeddedClassNames(unittest.TestCase):
    def test_reads_ultralytics_names_in_index_order(self):
        self.assertEqual(embedded_class_names({"names": DETECTOR_NAMES}), DETECTOR_ORDER)

    def test_missing_or_unreadable_names_are_none(self):
        for meta in (
            {},
            {"names": ""},
            {"names": "not a dict"},
            {"names": "['deer', 'gaur']"},
            {"names": "{0: 'deer', 2: 'gaur'}"},
        ):
            self.assertIsNone(embedded_class_names(meta), meta)


class TestCheckedClassNames(unittest.TestCase):
    def test_matching_names_pass_and_come_back_canonical(self):
        got = checked_class_names(["deer", "Sloth Bear"], "m.onnx", model_names=["deer", "sloth_bear"])
        self.assertEqual(got, ["deer", "sloth_bear"])

    def test_a_different_order_is_refused(self):
        swapped = ["deer", "elephant", "gaur", "leopard", "sloth_bear", "tiger", "wild_boar"]
        with self.assertRaises(InferenceBackendError):
            checked_class_names(swapped, "detector.onnx", model_names=DETECTOR_ORDER)

    def test_without_model_information_the_configured_list_is_used(self):
        self.assertEqual(checked_class_names(DETECTOR_ORDER, "detector.hef"), DETECTOR_ORDER)


class TestOnnxDetectorLoad(unittest.TestCase):
    def test_detector_accepts_the_150e_spaced_names(self):
        with patch("inference.onnx_inference._make_session", return_value=_session(DETECTOR_NAMES)):
            self.assertEqual(OnnxDetector("detector.onnx", DETECTOR_ORDER).class_names, DETECTOR_ORDER)

    def test_detector_refuses_a_config_in_another_order(self):
        swapped = ["deer", "leopard", "gaur", "elephant", "sloth_bear", "tiger", "wild_boar"]
        with patch("inference.onnx_inference._make_session", return_value=_session(DETECTOR_NAMES)):
            with self.assertRaises(InferenceBackendError):
                OnnxDetector("detector.onnx", swapped)


class TestShippedConfig(unittest.TestCase):
    def test_detector_is_the_only_model_with_its_trained_order(self):
        cfg = load(paths.INFERENCE_CONFIG)
        self.assertEqual(cfg["detector"]["classes"], DETECTOR_ORDER)
        self.assertNotIn("classifier", cfg)
        self.assertIn("stage2", cfg)

    def test_manifest_agrees_with_config(self):
        cfg = load(paths.INFERENCE_CONFIG)
        manifest = json.loads(paths.MODEL_MANIFEST.read_text(encoding="utf-8"))
        self.assertNotIn("classifier", manifest)
        for key in ("classes", "onnx_path", "hef_path"):
            self.assertEqual(manifest["detector"][key], cfg["detector"][key], f"detector.{key}")
        self.assertIn("ZOVIVE_FINAL_150E", manifest["detector"]["trained_from"])


if __name__ == "__main__":
    unittest.main()
