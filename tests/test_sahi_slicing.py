"""SAHI tile geometry and merge tests."""
import unittest

import numpy as np

from inference.sahi_batched import compute_tiles, merge_detections, remap_tile_detections


class TestSahiSlicing(unittest.TestCase):
    def test_small_region_is_one_tile(self):
        self.assertEqual(compute_tiles(500, 400), [(0, 0, 500, 400)])

    def test_tiles_cover_the_far_edge(self):
        tiles = compute_tiles(1500, 700, 640, 640, 0.2, 0.2)
        self.assertEqual(max(t[2] for t in tiles), 1500)
        self.assertEqual(max(t[3] for t in tiles), 700)

    def test_remap_shifts_into_region_coordinates(self):
        boxes = np.array([[10.0, 20.0, 50.0, 60.0]])
        out = remap_tile_detections(boxes, (512, 128))
        self.assertEqual(out.tolist(), [[522.0, 148.0, 562.0, 188.0]])

    def test_merge_deduplicates_across_overlapping_tiles(self):
        a = (np.array([[10.0, 10.0, 110.0, 110.0]]), np.array([0.9]), np.array([0]), (0, 0))
        b = (np.array([[8.0, 8.0, 108.0, 108.0]]),   np.array([0.8]), np.array([0]), (0, 0))
        boxes, scores, _ = merge_detections([a, b], iou_thr=0.5)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(scores[0], 0.9)


if __name__ == "__main__":
    unittest.main()
