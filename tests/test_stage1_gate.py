import unittest

from inference.bytetrack_wrapper import Track
from pipeline.stage1_gate import Stage1Config, Stage1Gate


def _track(track_id="t1", presence=(True, True, True)):
    return Track(
        track_id=track_id,
        box=(0, 0, 10, 10),
        class_id=5,
        class_name="tiger",
        score=0.9,
        presence_history=list(presence),
    )


class TestStage1Gate(unittest.TestCase):
    def test_passes_with_enough_hits_in_window(self):
        gate = Stage1Gate(Stage1Config(min_hits_in_window=3, window_size=5))
        track = _track(presence=(True, False, True, False, True))  # 3 of 5
        self.assertTrue(gate.evaluate(track, now=0.0))

    def test_fails_with_too_few_hits(self):
        gate = Stage1Gate(Stage1Config(min_hits_in_window=3, window_size=5))
        track = _track(presence=(True, False, False, False, True))  # 2 of 5
        self.assertFalse(gate.evaluate(track, now=0.0))

    def test_only_considers_last_window_size_entries(self):
        gate = Stage1Gate(Stage1Config(min_hits_in_window=3, window_size=3))
        # 10 hits far in the past, but the recent window only has 1 hit
        track = _track(presence=[True] * 10 + [False, False, True])
        self.assertFalse(gate.evaluate(track, now=0.0))

    def test_cooldown_suppresses_repeat_alert(self):
        gate = Stage1Gate(Stage1Config(min_hits_in_window=1, window_size=1, track_alert_cooldown_sec=100.0))
        track = _track(presence=(True,))
        self.assertTrue(gate.evaluate(track, now=1000.0))
        gate.mark_alerted(track.track_id, now=1000.0)
        # Still within cooldown window
        self.assertTrue(gate.in_cooldown(track, now=1050.0))
        self.assertFalse(gate.evaluate(track, now=1050.0))
        # Cooldown expired
        self.assertFalse(gate.in_cooldown(track, now=1200.0))
        self.assertTrue(gate.evaluate(track, now=1200.0))

    def test_evaluate_all_filters_list(self):
        gate = Stage1Gate(Stage1Config(min_hits_in_window=2, window_size=2))
        passing = _track("t-pass", presence=(True, True))
        failing = _track("t-fail", presence=(False, True))
        result = gate.evaluate_all([passing, failing], now=0.0)
        self.assertEqual([t.track_id for t in result], ["t-pass"])


if __name__ == "__main__":
    unittest.main()
