import unittest

from pipeline.motion_gate import MotionGate
from tests.helpers import blank_frame, frame_with_box


class TestMotionGate(unittest.TestCase):
    def test_first_frame_scores_zero(self):
        gate = MotionGate()
        self.assertEqual(gate.score(blank_frame()), 0.0)

    def test_identical_frames_no_motion(self):
        gate = MotionGate()
        frame = blank_frame()
        gate.score(frame)
        self.assertFalse(gate.has_motion(frame.copy()))

    def test_changed_frame_detects_motion(self):
        gate = MotionGate(min_changed_ratio=0.01)
        gate.score(blank_frame())
        moved = frame_with_box(value=250)  # big bright block = big diff
        self.assertTrue(gate.has_motion(moved))

    def test_reset_clears_baseline(self):
        gate = MotionGate()
        gate.score(blank_frame())
        gate.reset()
        # after reset, first score() call again returns 0.0 regardless of content
        self.assertEqual(gate.score(frame_with_box()), 0.0)


if __name__ == "__main__":
    unittest.main()
