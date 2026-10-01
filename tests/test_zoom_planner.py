import unittest

from camera_control.zoom_planner import (
    ZoomPlanConfig,
    box_fill_ratio,
    magnification_to_zoom_level,
    max_in_frame_magnification,
    plan_zoom,
)

FRAME = (360, 640)  # (h, w)


def _centered_box(bw, bh, frame=FRAME):
    h, w = frame
    return (w / 2 - bw / 2, h / 2 - bh / 2, w / 2 + bw / 2, h / 2 + bh / 2)


class TestZoomPlanner(unittest.TestCase):
    def setUp(self):
        self.cfg = ZoomPlanConfig(
            far_fill_ratio=0.25, target_fill_ratio=0.5, max_optical_magnification=4.0,
            min_useful_magnification=1.5, edge_margin_ratio=0.1,
        )

    def test_close_animal_is_not_zoomed(self):
        self.assertIsNone(plan_zoom(_centered_box(320, 180), FRAME, self.cfg))

    def test_far_centered_animal_zooms_to_target_fill(self):
        box = _centered_box(64, 36)  # fills 10% of the frame
        self.assertAlmostEqual(box_fill_ratio(box, FRAME), 0.1)
        self.assertAlmostEqual(plan_zoom(box, FRAME, self.cfg), 4.0)  # 0.5/0.1 = 5, capped at lens max

    def test_off_center_animal_zoom_limited_to_stay_in_frame(self):
        # Small box near the right edge: can't zoom much without a pan.
        box = (560, 170, 600, 190)
        fit = max_in_frame_magnification(box, FRAME, 0.1)
        self.assertLess(fit, 1.5)
        self.assertIsNone(plan_zoom(box, FRAME, self.cfg))

    def test_partly_off_center_animal_zooms_less(self):
        box = (400, 170, 440, 190)  # right edge at 0.375 of half-width from centre
        mag = plan_zoom(box, FRAME, self.cfg)
        self.assertIsNotNone(mag)
        self.assertAlmostEqual(mag, 0.9 / ((440 - 320) / 320))
        self.assertLess(mag, 4.0)

    def test_zoom_level_mapping_is_linear_between_wide_and_max(self):
        self.assertAlmostEqual(magnification_to_zoom_level(1.0, self.cfg), 0.0)
        self.assertAlmostEqual(magnification_to_zoom_level(4.0, self.cfg), 1.0)
        self.assertAlmostEqual(magnification_to_zoom_level(2.5, self.cfg), 0.5)
        self.assertAlmostEqual(magnification_to_zoom_level(9.0, self.cfg), 1.0)

    def test_from_config_defaults(self):
        cfg = ZoomPlanConfig.from_config({"far_fill_ratio": 0.3})
        self.assertEqual(cfg.far_fill_ratio, 0.3)
        self.assertEqual(cfg.max_optical_magnification, 4.0)


if __name__ == "__main__":
    unittest.main()
