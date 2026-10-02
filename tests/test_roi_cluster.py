"""ROI cluster tests."""

import unittest

from pipeline.roi_cluster import cluster_area_ratio, cluster_box

SHAPE = (2160, 3840)


class TestRoiCluster(unittest.TestCase):
    def test_encloses_every_box(self):
        c = cluster_box([(100, 100, 200, 200), (900, 700, 1000, 800)], SHAPE, min_size=0)
        self.assertLessEqual(c[0], 100)
        self.assertLessEqual(c[1], 100)
        self.assertGreaterEqual(c[2], 1000)
        self.assertGreaterEqual(c[3], 800)

    def test_min_size_expands_a_tiny_target(self):
        c = cluster_box([(100, 100, 140, 130)], SHAPE, min_size=640)
        self.assertGreaterEqual(c[2] - c[0], 640)
        self.assertGreaterEqual(c[3] - c[1], 640)

    def test_clamped_to_frame(self):
        c = cluster_box([(3800, 2100, 3839, 2159)], SHAPE, min_size=640)
        self.assertLessEqual(c[2], 3840)
        self.assertLessEqual(c[3], 2160)
        self.assertGreaterEqual(c[0], 0)
        self.assertGreaterEqual(c[1], 0)

    def test_area_ratio(self):
        self.assertEqual(cluster_area_ratio(None, SHAPE), 0.0)


if __name__ == "__main__":
    unittest.main()
