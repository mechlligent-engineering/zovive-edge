"""Per-class confidence thresholds (inference/class_thresholds.py): the
filter itself, edge_main's inference thread applying it before the
DetectionQueue, and load() rejecting thresholds that are misspelt or could
never take effect."""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

import paths
from edge_main import _inference_loop
from inference.base import Detection
from inference.class_thresholds import DEFAULT_CONFIDENCE_THRESHOLD, ClassThresholds
from queues.detection_queue import DetectionQueue
from queues.frame_queue import LatestFrameSlot
from utils import config_loader
from utils.config_loader import ConfigError, IssueKind, load


def _det(class_name: str, score: float, class_id: int = 0) -> Detection:
    return Detection(box=(0, 0, 10, 10), score=score, class_id=class_id, class_name=class_name)


THRESHOLDS = ClassThresholds(default=0.50, per_class={"elephant": 0.60, "leopard": 0.55})


class TestClassThresholds(unittest.TestCase):
    def test_listed_class_uses_its_own_threshold(self):
        self.assertEqual(THRESHOLDS.for_class("elephant"), 0.60)
        self.assertEqual(THRESHOLDS.for_class("leopard"), 0.55)

    def test_unlisted_class_uses_the_default(self):
        self.assertEqual(THRESHOLDS.for_class("deer"), 0.50)

    def test_keep_drops_only_detections_below_their_class_threshold(self):
        dets = [
            _det("elephant", 0.59),  # below 0.60: dropped
            _det("elephant", 0.60),  # exactly the threshold: kept
            _det("leopard", 0.56),
            _det("deer", 0.52),  # deer is unlisted: default 0.50
            _det("deer", 0.49),
        ]
        kept = THRESHOLDS.keep(dets)
        self.assertEqual(
            [(d.class_name, d.score) for d in kept], [("elephant", 0.60), ("leopard", 0.56), ("deer", 0.52)]
        )

    def test_keep_on_nothing_is_nothing(self):
        self.assertEqual(THRESHOLDS.keep([]), [])

    def test_from_config_reads_the_detector_section(self):
        t = ClassThresholds.from_config(
            {"confidence_threshold": 0.5, "class_confidence_thresholds": {"tiger": 0.5, "gaur": 0.6}}
        )
        self.assertEqual(t.default, 0.5)
        self.assertEqual(dict(t.per_class), {"tiger": 0.5, "gaur": 0.6})

    def test_from_config_without_thresholds_keeps_the_detector_default(self):
        t = ClassThresholds.from_config({})
        self.assertEqual(t.default, DEFAULT_CONFIDENCE_THRESHOLD)
        self.assertEqual(dict(t.per_class), {})
        self.assertEqual(t.for_class("elephant"), DEFAULT_CONFIDENCE_THRESHOLD)


class _FixedDetector:
    def __init__(self, detections: list[Detection]):
        self.detections = detections

    def infer(self, frame):
        return list(self.detections)


class TestInferenceLoopAppliesThresholds(unittest.TestCase):
    def _run_one_frame(self, class_thresholds: ClassThresholds | None) -> list[Detection]:
        slot = LatestFrameSlot(name="test_class_thresholds_slot")
        slot.put(np.zeros((10, 10, 3), dtype=np.uint8))
        queue = DetectionQueue(maxsize=2)
        detector = _FixedDetector(
            [_det("elephant", 0.55, 3), _det("elephant", 0.75, 3), _det("deer", 0.51, 0)]
        )
        _inference_loop(
            detector, slot, queue, "cam01", threading.Event(), 1, None, class_thresholds=class_thresholds
        )
        batch = queue.get(timeout=1.0)
        self.assertIsNotNone(batch)
        return batch.detections

    def test_detections_below_their_class_threshold_never_reach_the_queue(self):
        kept = self._run_one_frame(THRESHOLDS)
        self.assertEqual([(d.class_name, d.score) for d in kept], [("elephant", 0.75), ("deer", 0.51)])

    def test_without_thresholds_every_detection_is_queued(self):
        self.assertEqual(len(self._run_one_frame(None)), 3)


_VALID_DETECTOR = """\
detector:
  classes: [deer, gaur, leopard, elephant, sloth_bear, tiger, wild_boar]
  confidence_threshold: 0.50
"""


class TestInferenceConfigThresholdChecks(unittest.TestCase):
    def setUp(self):
        config_loader.clear_cache()
        self.addCleanup(config_loader.clear_cache)
        self.dir = Path(tempfile.mkdtemp(prefix="zovive_infcfg_"))

    def _load(self, extra: str = "", base: str = _VALID_DETECTOR):
        path = self.dir / "inference_config.yaml"  # the file name selects the check
        path.write_text(base + extra, encoding="utf-8")
        return load(path)

    def _issues(self, extra: str = "", base: str = _VALID_DETECTOR) -> list:
        with self.assertRaises(ConfigError) as ctx:
            self._load(extra, base)
        return list(ctx.exception.issues)

    def test_valid_thresholds_load(self):
        cfg = self._load("  class_confidence_thresholds:\n    elephant: 0.60\n    deer: 0.50\n")
        self.assertEqual(cfg["detector"]["class_confidence_thresholds"]["elephant"], 0.60)

    def test_no_per_class_thresholds_is_fine(self):
        self.assertIn("detector", self._load())

    def test_unknown_class_is_rejected(self):
        issues = self._issues("  class_confidence_thresholds:\n    rhino: 0.70\n")
        self.assertEqual(
            [(i.key, i.kind) for i in issues],
            [("detector.class_confidence_thresholds.rhino", IssueKind.UNKNOWN_KEY)],
        )

    def test_threshold_below_the_decode_threshold_is_rejected(self):
        issues = self._issues("  class_confidence_thresholds:\n    tiger: 0.45\n")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].key, "detector.class_confidence_thresholds.tiger")
        self.assertIn("could never take effect", issues[0].message)

    def test_threshold_out_of_range_or_not_a_number_is_rejected(self):
        issues = self._issues(
            "  class_confidence_thresholds:\n    tiger: 1.5\n    gaur: high\n    deer: true\n"
        )
        self.assertEqual(
            sorted((i.key, i.kind) for i in issues),
            [
                ("detector.class_confidence_thresholds.deer", IssueKind.INVALID_TYPE),
                ("detector.class_confidence_thresholds.gaur", IssueKind.INVALID_TYPE),
                ("detector.class_confidence_thresholds.tiger", IssueKind.INVALID_VALUE),
            ],
        )

    def test_thresholds_must_be_a_mapping(self):
        issues = self._issues("  class_confidence_thresholds: [0.6, 0.5]\n")
        self.assertEqual([i.kind for i in issues], [IssueKind.INVALID_TYPE])

    def test_bad_decode_threshold_is_rejected(self):
        base = _VALID_DETECTOR.replace("confidence_threshold: 0.50", "confidence_threshold: 0")
        issues = self._issues(base=base)
        self.assertEqual([i.key for i in issues], ["detector.confidence_threshold"])

    def test_missing_detector_section_is_rejected(self):
        issues = self._issues(base="stage2:\n  min_confidence: 0.55\n")
        self.assertEqual([(i.key, i.kind) for i in issues], [("detector", IssueKind.MISSING_KEY)])

    def test_every_problem_is_reported_at_once(self):
        issues = self._issues("  class_confidence_thresholds:\n    rhino: 0.7\n    tiger: 0.3\n")
        self.assertEqual(len(issues), 2)


class TestDeployedInferenceConfig(unittest.TestCase):
    def setUp(self):
        config_loader.clear_cache()
        self.addCleanup(config_loader.clear_cache)

    def test_repository_config_passes_the_checks_with_the_approved_thresholds(self):
        det = load(paths.INFERENCE_CONFIG)["detector"]
        t = ClassThresholds.from_config(det)
        self.assertEqual(t.default, 0.50)
        self.assertEqual(
            dict(t.per_class),
            {
                "elephant": 0.60,
                "gaur": 0.60,
                "leopard": 0.55,
                "wild_boar": 0.55,
                "tiger": 0.50,
                "sloth_bear": 0.50,
                "deer": 0.50,
            },
        )
        self.assertEqual(set(t.per_class), set(det["classes"]))


if __name__ == "__main__":
    unittest.main()
