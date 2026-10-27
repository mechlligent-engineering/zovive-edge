"""Tests for pipeline/stage2_verifier.py's two voting strategies, which
combine a track's per-frame detector (class_name, score) votes into one
species (single-stage: there is no classifier model), plus the shared
unknown-animal / min-confidence path both strategies go through, and the
tracker side that records those votes.
"""

from __future__ import annotations

import unittest

from inference.base import Detection
from inference.bytetrack_wrapper import IouTracker
from pipeline.stage2_verifier import UNKNOWN_ANIMAL, verify_track

# Field-report example: mostly Tiger, one ambiguous Leopard frame.
MOSTLY_TIGER = [("tiger", 0.96), ("tiger", 0.95), ("tiger", 0.97), ("leopard", 0.51), ("tiger", 0.94)]


class TestAverageStrategy(unittest.TestCase):
    def test_one_outlier_frame_does_not_change_the_species(self):
        result = verify_track(MOSTLY_TIGER, min_confidence=0.55, strategy="average")

        self.assertEqual(result.species, "tiger")
        self.assertFalse(result.is_unknown)
        self.assertEqual(result.strategy, "average")
        self.assertEqual(result.frames_used, 5)
        # Summed tiger score over ALL frames: the outlier frame counts as 0.
        self.assertAlmostEqual(result.confidence, (0.96 + 0.95 + 0.97 + 0.94) / 5, places=6)

    def test_flip_flopping_track_falls_below_the_floor(self):
        votes = [("tiger", 0.9), ("leopard", 0.9), ("tiger", 0.9), ("leopard", 0.9)]
        result = verify_track(votes, min_confidence=0.55, strategy="average")

        self.assertEqual(result.species, UNKNOWN_ANIMAL)
        self.assertTrue(result.is_unknown)
        self.assertAlmostEqual(result.confidence, 0.45, places=6)


class TestMajorityStrategy(unittest.TestCase):
    def test_majority_vote_wins_over_one_outlier(self):
        result = verify_track(MOSTLY_TIGER, min_confidence=0.55, strategy="majority")

        self.assertEqual(result.species, "tiger")
        self.assertEqual(result.strategy, "majority")
        # Confidence is the mean of only the winning (tiger) frames.
        self.assertAlmostEqual(result.confidence, (0.96 + 0.95 + 0.97 + 0.94) / 4, places=5)

    def test_tie_broken_by_summed_score(self):
        # 2 frames Tiger (high score), 2 frames Leopard (low score): tied
        # vote count, Tiger wins on summed score whatever the order.
        votes = [("leopard", 0.40), ("tiger", 0.90), ("leopard", 0.35), ("tiger", 0.85)]
        result = verify_track(votes, min_confidence=0.0, strategy="majority")

        self.assertEqual(result.species, "tiger")

    def test_below_min_confidence_is_unknown_animal(self):
        result = verify_track([("tiger", 0.50)] * 3, min_confidence=0.55, strategy="majority")

        self.assertEqual(result.species, UNKNOWN_ANIMAL)
        self.assertTrue(result.is_unknown)


class TestSharedBehavior(unittest.TestCase):
    def test_no_votes_is_unknown_animal(self):
        result = verify_track([], strategy="average")
        self.assertEqual(result.species, UNKNOWN_ANIMAL)
        self.assertEqual(result.frames_used, 0)

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            verify_track([("tiger", 0.9)], strategy="bogus")


class TestTrackerRecordsVotes(unittest.TestCase):
    def test_each_matched_frame_adds_one_vote_and_old_votes_drop_off(self):
        tracker = IouTracker(min_hits_to_confirm=1, max_class_votes=3)
        box = (100.0, 80.0, 180.0, 160.0)
        for name, score in [("tiger", 0.9), ("leopard", 0.6), ("tiger", 0.8), ("tiger", 0.7)]:
            [track] = tracker.update([Detection(box=box, score=score, class_id=0, class_name=name)])

        self.assertEqual(track.class_votes, [("leopard", 0.6), ("tiger", 0.8), ("tiger", 0.7)])
        # The track's own class is still the latest frame's.
        self.assertEqual((track.class_name, track.score), ("tiger", 0.7))

    def test_missed_frames_add_no_vote(self):
        tracker = IouTracker(min_hits_to_confirm=1)
        det = Detection(box=(10.0, 10.0, 50.0, 50.0), score=0.9, class_id=0, class_name="deer")
        tracker.update([det])
        [track] = tracker.update([])

        self.assertEqual(track.class_votes, [("deer", 0.9)])


if __name__ == "__main__":
    unittest.main()
