import unittest

from camera_control import camera_api
from camera_control.camera_api import FakeLensCamera


class TestFakeLensCamera(unittest.TestCase):
    def test_zoom_to_sets_level(self):
        cam = FakeLensCamera()
        self.assertTrue(cam.zoom_to(0.5))
        self.assertEqual(cam.get_zoom(), 0.5)
        self.assertEqual(cam.call_log[-1][0], "zoom_to")

    def test_trigger_autofocus_records_call(self):
        cam = FakeLensCamera()
        self.assertTrue(cam.trigger_autofocus())
        self.assertEqual(cam.focus_count, 1)
        self.assertEqual(cam.call_log[-1][0], "trigger_autofocus")


class TestNoPanTilt(unittest.TestCase):
    def test_lens_interface_has_no_pan_tilt_or_presets(self):
        forbidden = ("pan", "tilt", "preset", "relative_move", "continuous_move")
        for cls in (camera_api.LensCamera, camera_api.FakeLensCamera, camera_api.OnvifLensCamera):
            for name in dir(cls):
                if name.startswith("_"):
                    continue
                self.assertFalse(any(f in name.lower() for f in forbidden), f"{cls.__name__}.{name}")


if __name__ == "__main__":
    unittest.main()
