"""Tests for pipeline/stage2_verifier.py's two voting strategies.

The "average" tests double as regression coverage for the original
batch-1 behavior (softmax-averaging); this file's main job is proving
the new "majority" strategy added for the species-voting enhancement
request, plus the shared unknown-animal / min-confidence path both
strategies go through identically.
"""

from __future__ import annotations

import unittest

from pipeline.best_snapshot import ScoredCrop
from pipeline.stage2_verifier import UNKNOWN_ANIMAL, verify_track
from tests.helpers import ScriptedClassifier, blank_frame, scored_result

CLASS_NAMES = ["deer", "gaur", "leopard", "elephant", "sloth_bear", "tiger", "wild_boar"]


def _crops(n: int) -> list[ScoredCrop]:
    frame = blank_frame(64, 64)
    return [ScoredCrop(crop=frame, full_frame=frame, score=1.0, sharpness=1.0) for _ in range(n)]


class TestAverageStrategy(unittest.TestCase):
    def test_averages_softmax_across_crops(self):
        # Matches the field-report example: mostly Tiger, one Leopard
        # outlier -> averaging should still land on Tiger.
        script = [
            scored_result(CLASS_NAMES, "tiger", 0.96),
            scored_result(CLASS_NAMES, "tiger", 0.95),
            scored_result(CLASS_NAMES, "tiger", 0.97),
            scored_result(CLASS_NAMES, "leopard", 0.51),
            scored_result(CLASS_NAMES, "tiger", 0.94),
        ]
        classifier = ScriptedClassifier(script, class_names=CLASS_NAMES)

        result = verify_track(classifier, _crops(5), min_confidence=0.55, strategy="average")

        self.assertEqual(result.species, "tiger")
        self.assertFalse(result.is_unknown)
        self.assertEqual(result.strategy, "average")


class TestMajorityStrategy(unittest.TestCase):
    def test_majority_vote_wins_over_one_outlier(self):
        script = [
            scored_result(CLASS_NAMES, "tiger", 0.96),
            scored_result(CLASS_NAMES, "tiger", 0.95),
            scored_result(CLASS_NAMES, "tiger", 0.97),
            scored_result(CLASS_NAMES, "leopard", 0.51),
            scored_result(CLASS_NAMES, "tiger", 0.94),
        ]
        classifier = ScriptedClassifier(script, class_names=CLASS_NAMES)

        result = verify_track(classifier, _crops(5), min_confidence=0.55, strategy="majority")

        self.assertEqual(result.species, "tiger")
        self.assertEqual(result.strategy, "majority")
        # Confidence should be the mean of only the winning (tiger) votes.
        self.assertAlmostEqual(result.confidence, (0.96 + 0.95 + 0.97 + 0.94) / 4, places=5)

    def test_tie_broken_by_summed_confidence(self):
        # 2 votes Tiger (high conf), 2 votes Leopard (low conf) -> tie
        # in vote count, Tiger should win on summed confidence.
        script = [
            scored_result(CLASS_NAMES, "tiger", 0.90),
            scored_result(CLASS_NAMES, "tiger", 0.85),
            scored_result(CLASS_NAMES, "leopard", 0.40),
            scored_result(CLASS_NAMES, "leopard", 0.35),
        ]
        classifier = ScriptedClassifier(script, class_names=CLASS_NAMES)

        result = verify_track(classifier, _crops(4), min_confidence=0.0, strategy="majority")

        self.assertEqual(result.species, "tiger")

    def test_below_min_confidence_is_unknown_animal(self):
        script = [scored_result(CLASS_NAMES, "tiger", 0.40) for _ in range(3)]
        classifier = ScriptedClassifier(script, class_names=CLASS_NAMES)

        result = verify_track(classifier, _crops(3), min_confidence=0.55, strategy="majority")

        self.assertEqual(result.species, UNKNOWN_ANIMAL)
        self.assertTrue(result.is_unknown)


class TestSharedBehavior(unittest.TestCase):
    def test_no_crops_is_unknown_animal(self):
        classifier = ScriptedClassifier([], class_names=CLASS_NAMES)
        result = verify_track(classifier, [], strategy="average")
        self.assertEqual(result.species, UNKNOWN_ANIMAL)
        self.assertEqual(result.crops_used, 0)

    def test_unknown_strategy_raises(self):
        classifier = ScriptedClassifier([scored_result(CLASS_NAMES, "tiger", 0.9)], class_names=CLASS_NAMES)
        with self.assertRaises(ValueError):
            verify_track(classifier, _crops(1), strategy="bogus")


if __name__ == "__main__":
    unittest.main()
