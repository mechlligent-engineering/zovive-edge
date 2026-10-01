import unittest

from inference.base import Detection
from pipeline.zone_filter import ZoneFilter


class TestZoneFilter(unittest.TestCase):
    def test_empty_zone_list_means_no_filtering(self):
        zf = ZoneFilter({"zones": {"north": []}})
        det = Detection(box=(0, 0, 10, 10), score=0.9, class_id=0, class_name="deer")
        kept = zf.filter([det], "north", (100, 100))
        self.assertEqual(len(kept), 1)

    def test_unknown_preset_means_no_filtering(self):
        zf = ZoneFilter({"zones": {"north": [[[0, 0], [1, 0], [1, 1], [0, 1]]]}})
        det = Detection(box=(0, 0, 10, 10), score=0.9, class_id=0, class_name="deer")
        kept = zf.filter([det], "unmapped_preset", (100, 100))
        self.assertEqual(len(kept), 1)

    def test_detection_inside_polygon_kept(self):
        # polygon covering the left half of the frame (normalized coords)
        polygon = [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]
        zf = ZoneFilter({"zones": {"north": [polygon]}})
        inside = Detection(box=(0, 0, 20, 20), score=0.9, class_id=0, class_name="deer")  # center ~(10,10)/100
        kept = zf.filter([inside], "north", (100, 100))
        self.assertEqual(len(kept), 1)

    def test_detection_outside_polygon_dropped(self):
        polygon = [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]
        zf = ZoneFilter({"zones": {"north": [polygon]}})
        outside = Detection(box=(80, 80, 95, 95), score=0.9, class_id=0, class_name="deer")
        kept = zf.filter([outside], "north", (100, 100))
        self.assertEqual(len(kept), 0)


if __name__ == "__main__":
    unittest.main()
