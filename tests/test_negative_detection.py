"""Regression test for the field report's "empty forest -> leopard"
false-positive problem: a single spurious one-frame detection must
NOT be enough to create an event. Only a track seen in at least
min_hits_in_window of the last window_size frames should ever reach
CONFIRMED.
"""

import unittest

from inference.base import Detection
from inference.bytetrack_wrapper import IouTracker
from pipeline.stage1_gate import Stage1Config, Stage1Gate
from pipeline.track_state_machine import TrackState, TrackStateMachine


class TestNegativeDetection(unittest.TestCase):
    def setUp(self):
        self.tracker = IouTracker(
            iou_match_threshold=0.3, max_age_frames=15, min_hits_to_confirm=1, presence_window=5
        )
        self.gate = Stage1Gate(Stage1Config(min_hits_in_window=3, window_size=5))
        self.sm = TrackStateMachine(self.gate)

    def _run_frame(self, detections, t):
        tracks = self.tracker.update(detections)
        for track in tracks:
            self.sm.advance(track, now=t)
        return tracks

    def test_single_spurious_detection_never_confirms(self):
        # 10 empty frames with exactly one spurious "leopard" in the middle.
        spurious = Detection(box=(50, 50, 100, 100), score=0.6, class_id=2, class_name="leopard")
        for i in range(10):
            dets = [spurious] if i == 5 else []
            self._run_frame(dets, t=float(i))

        confirmed = self.sm.in_state(TrackState.CONFIRMED)
        self.assertEqual(confirmed, [], "a single-frame false positive must not reach CONFIRMED")

    def test_two_out_of_five_never_confirms(self):
        # Below the 3-of-5 threshold — still should not confirm.
        det = Detection(box=(50, 50, 100, 100), score=0.7, class_id=5, class_name="tiger")
        pattern = [True, False, True, False, False]
        for i, present in enumerate(pattern):
            self._run_frame([det] if present else [], t=float(i))

        confirmed = self.sm.in_state(TrackState.CONFIRMED)
        self.assertEqual(confirmed, [])

    def test_three_out_of_five_does_confirm(self):
        # At the 3-of-5 threshold — this SHOULD confirm (sanity check that
        # the gate isn't simply always rejecting).
        det = Detection(box=(50, 50, 100, 100), score=0.85, class_id=5, class_name="tiger")
        pattern = [True, True, False, True, False]
        for i, present in enumerate(pattern):
            self._run_frame([det] if present else [], t=float(i))

        confirmed = self.sm.in_state(TrackState.CONFIRMED)
        self.assertEqual(len(confirmed), 1)


if __name__ == "__main__":
    unittest.main()
