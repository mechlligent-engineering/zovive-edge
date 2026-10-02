import unittest

from inference.bytetrack_wrapper import Track
from pipeline.stage1_gate import Stage1Config, Stage1Gate
from pipeline.track_state_machine import TrackState, TrackStateMachine


def _track(track_id="t1", presence=(True,)):
    return Track(
        track_id=track_id,
        box=(0, 0, 10, 10),
        class_id=5,
        class_name="tiger",
        score=0.9,
        presence_history=list(presence),
    )


class TestTrackStateMachine(unittest.TestCase):
    def setUp(self):
        self.gate = Stage1Gate(
            Stage1Config(min_hits_in_window=2, window_size=2, track_alert_cooldown_sec=100.0)
        )
        self.sm = TrackStateMachine(self.gate)

    def test_new_track_starts_confirming(self):
        track = _track(presence=(False,))
        rec = self.sm.advance(track, now=0.0)
        self.assertEqual(rec.state, TrackState.CONFIRMING)

    def test_confirms_once_gate_passes(self):
        track = _track(presence=(True, True))
        rec = self.sm.advance(track, now=0.0)
        self.assertEqual(rec.state, TrackState.CONFIRMED)

    def test_full_lifecycle(self):
        track = _track(presence=(True, True))
        self.sm.advance(track, now=0.0)
        rec = self.sm.mark_classified("t1", species="tiger", confidence=0.8)
        self.assertEqual(rec.state, TrackState.CLASSIFIED)
        rec = self.sm.mark_alerted("t1", event_id="evt-1")
        self.assertEqual(rec.state, TrackState.ALERTED)
        self.assertEqual(rec.event_id, "evt-1")
        # gate cooldown should now suppress a re-evaluation
        self.assertTrue(self.gate.in_cooldown(track, now=1.0))

    def test_sweep_lost_removes_missing_tracks(self):
        track = _track("t1")
        self.sm.advance(track, now=0.0)
        self.sm.advance(_track("t2"), now=0.0)
        removed = self.sm.sweep_lost(active_track_ids={"t2"})
        self.assertEqual(removed, ["t1"])
        self.assertIsNone(self.sm.get("t1"))
        self.assertIsNotNone(self.sm.get("t2"))


if __name__ == "__main__":
    unittest.main()
