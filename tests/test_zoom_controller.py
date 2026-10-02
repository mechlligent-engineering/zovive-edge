import unittest

from camera_control.camera_api import FakeLensCamera
from camera_control.mode_manager import ModeManager
from camera_control.zoom_controller import WIDE_VIEW, ZOOMED_VIEW, AnimalZoomController, ZoomState

FRAME = (360, 640)
FAR_BOX = (300, 170, 340, 190)  # small and centred


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.now = start

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now

    def __call__(self) -> float:
        return self.now


class TestAnimalZoomController(unittest.TestCase):
    def setUp(self):
        self.cam = FakeLensCamera()
        self.clock = FakeClock()
        self.zoom = AnimalZoomController(
            self.cam,
            zoom_settle_sec=2.0,
            focus_settle_sec=1.0,
            zoomed_timeout_sec=5.0,
            suppress_realert_sec=30.0,
            clock=self.clock,
        )

    def _engage(self):
        mag = self.zoom.plan(FAR_BOX, FRAME)
        self.assertIsNotNone(mag)
        self.zoom.engage("t-1", "tiger", FAR_BOX, mag)

    def _to_zoomed(self):
        self._engage()
        self.clock.advance(2.0)
        self.zoom.tick()
        self.clock.advance(1.0)
        self.zoom.tick()

    def test_zoom_then_focus_then_zoomed(self):
        self._engage()
        self.assertEqual(self.zoom.state, ZoomState.ZOOMING)
        self.assertFalse(self.zoom.frames_usable())
        self.assertGreater(self.cam.get_zoom(), 0.0)

        self.clock.advance(2.0)
        self.zoom.tick()
        self.assertEqual(self.zoom.state, ZoomState.FOCUSING)
        self.assertEqual(self.cam.focus_count, 1)

        self.clock.advance(1.0)
        self.zoom.tick()
        self.assertEqual(self.zoom.state, ZoomState.ZOOMED)
        self.assertEqual(self.zoom.view_name, ZOOMED_VIEW)
        self.assertTrue(self.zoom.frames_usable())

    def test_complete_zooms_out_and_refocuses(self):
        self._to_zoomed()
        self.zoom.complete("evt-1", "tiger", 0.9)
        self.assertEqual(self.zoom.state, ZoomState.RETURNING)
        self.assertEqual(self.cam.get_zoom(), 0.0)
        self.clock.advance(2.0)
        self.zoom.tick()
        self.assertEqual(self.cam.focus_count, 2)
        self.clock.advance(1.0)
        self.zoom.tick()
        self.assertEqual(self.zoom.state, ZoomState.WIDE)
        self.assertEqual(self.zoom.view_name, WIDE_VIEW)

    def test_zoomed_timeout_returns_session_for_fallback(self):
        self._to_zoomed()
        self.clock.advance(5.0)
        session = self.zoom.tick()
        self.assertIsNotNone(session)
        self.assertEqual(session.track_id, "t-1")

    def test_same_animal_remembered_after_session(self):
        self._to_zoomed()
        self.zoom.complete("evt-1", "tiger", 0.9)
        known = self.zoom.match_known("tiger", (302, 171, 342, 191))
        self.assertIsNotNone(known)
        self.assertEqual(known.event_id, "evt-1")
        self.assertIsNone(self.zoom.match_known("deer", FAR_BOX))
        self.clock.advance(31.0)
        self.assertIsNone(self.zoom.match_known("tiger", FAR_BOX))

    def test_no_new_session_while_busy_or_manual(self):
        self._engage()
        self.assertIsNone(self.zoom.plan(FAR_BOX, FRAME))

        mm = ModeManager()
        mm.set_manual()
        manual = AnimalZoomController(FakeLensCamera(), mode_manager=mm, clock=self.clock)
        self.assertIsNone(manual.plan(FAR_BOX, FRAME))

    def test_disabled_never_zooms(self):
        zoom = AnimalZoomController(FakeLensCamera(), enabled=False, clock=self.clock)
        self.assertIsNone(zoom.plan(FAR_BOX, FRAME))

    def test_camera_only_receives_zoom_and_focus_commands(self):
        self._to_zoomed()
        self.zoom.complete("evt-1", "tiger", 0.9)
        self.assertTrue(
            {name for name, _ in self.cam.call_log} <= {"zoom_to", "trigger_autofocus", "stop_zoom"}
        )

    def test_from_config(self):
        zoom = AnimalZoomController.from_config(
            FakeLensCamera(), {"mode": {"zoom": {"enabled": False, "zoom_settle_sec": 3.5}}}
        )
        self.assertFalse(zoom.enabled)
        self.assertEqual(zoom.zoom_settle_sec, 3.5)


if __name__ == "__main__":
    unittest.main()
