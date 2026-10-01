import unittest

from camera_control.mode_manager import ModeManager


class TestModeManager(unittest.TestCase):
    def test_defaults_to_auto(self):
        mm = ModeManager()
        self.assertTrue(mm.is_auto())

    def test_manual_override(self):
        mm = ModeManager()
        mm.set_manual()
        self.assertFalse(mm.is_auto())
        mm.set_auto()
        self.assertTrue(mm.is_auto())

    def test_from_config(self):
        mm = ModeManager.from_config({"mode": {"default_mode": "manual"}})
        self.assertFalse(mm.is_auto())


if __name__ == "__main__":
    unittest.main()
