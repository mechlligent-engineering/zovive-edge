"""Tests for inference/sahi_slicer.py: tile geometry, transparent
pass-through on small frames, and merging detections found in
different tiles back into original-frame coordinates.
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from inference.base import ClassificationResult, Classifier, Detection, Detector
from inference.sahi_slicer import SahiConfig, SahiDetector, compute_tiles, is_sahi_enabled
from tests.helpers import blank_frame


class BrightBoxDetector(Detector):
    """Fake detector: finds the bounding box of any pixels brighter than
    a threshold in whatever image it's given (full frame or a tile) and
    reports it as one detection. Used so SAHI tests exercise real
    tile-offset math instead of a canned detection list."""

    class_names = ["deer", "gaur", "leopard", "elephant", "sloth_bear", "tiger", "wild_boar"]

    def __init__(self, threshold: int = 150, score: float = 0.9, class_id: int = 5):
        self.threshold = threshold
        self.score = score
        self.class_id = class_id
        self.calls = 0

    def infer(self, frame) -> list[Detection]:
        self.calls += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        mask = gray > self.threshold
        if not mask.any():
            return []
        ys, xs = np.where(mask)
        box = (float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1)
        return [
            Detection(
                box=box, score=self.score, class_id=self.class_id, class_name=self.class_names[self.class_id]
            )
        ]

    def warmup(self) -> None:
        return None

    def close(self) -> None:
        return None


class TestComputeTiles(unittest.TestCase):
    def test_frame_smaller_than_tile_gives_single_tile(self):
        tiles = compute_tiles(width=320, height=240, tile_size=640, overlap_ratio=0.2)
        self.assertEqual(tiles, [(0, 0, 320, 240)])

    def test_tiles_cover_entire_frame(self):
        width, height, tile_size = 1000, 700, 400
        tiles = compute_tiles(width, height, tile_size, overlap_ratio=0.2)
        # Every pixel must fall inside at least one tile.
        covered = np.zeros((height, width), dtype=bool)
        for x1, y1, x2, y2 in tiles:
            covered[y1:y2, x1:x2] = True
        self.assertTrue(covered.all())

    def test_tiles_overlap_neighbors(self):
        tiles = compute_tiles(width=1000, height=400, tile_size=400, overlap_ratio=0.25)
        xs = sorted({t[0] for t in tiles})
        # Consecutive tile starts should be closer together than tile_size
        # (i.e. they overlap), not spaced a full tile_size apart.
        for a, b in zip(xs, xs[1:], strict=False):
            self.assertLess(b - a, 400)


class TestIsSahiEnabled(unittest.TestCase):
    def test_defaults_to_disabled(self):
        self.assertFalse(is_sahi_enabled({}))

    def test_reads_top_level_sahi_block(self):
        self.assertTrue(is_sahi_enabled({"sahi": {"enabled": True}}))

    def test_reads_detector_nested_sahi_block(self):
        self.assertTrue(is_sahi_enabled({"detector": {"sahi": {"enabled": True}}}))


class TestSahiDetector(unittest.TestCase):
    def test_passthrough_when_frame_fits_one_tile(self):
        inner = BrightBoxDetector()
        sahi = SahiDetector(inner, SahiConfig(tile_size=640))
        frame = blank_frame(320, 240)  # smaller than tile_size in both dims

        sahi.infer(frame)

        self.assertEqual(inner.calls, 1, "should not slice a frame that already fits in one tile")

    def test_recovers_detection_only_visible_when_tiled(self):
        # A 900x300 frame with a small bright box near the right edge.
        # tile_size=400 forces slicing; the box should be found in
        # whichever tile contains it and mapped back to global coords.
        frame = blank_frame(900, 300)
        frame[140:160, 860:880] = 220  # small bright patch near x=870

        inner = BrightBoxDetector()
        sahi = SahiDetector(inner, SahiConfig(tile_size=400, overlap_ratio=0.2, run_full_frame_pass=False))

        detections = sahi.infer(frame)

        self.assertEqual(len(detections), 1)
        x1, y1, x2, y2 = detections[0].box
        # Box should land back at its true global location, not a
        # tile-local one.
        self.assertTrue(850 <= x1 <= 865)
        self.assertTrue(875 <= x2 <= 885)

    def test_duplicate_detections_across_full_frame_and_tile_are_merged(self):
        # A box large/bright enough to be found by both the full-frame
        # pass and a tile pass should collapse to one detection, not two.
        frame = blank_frame(900, 300)
        frame[100:200, 100:250] = 220

        inner = BrightBoxDetector()
        sahi = SahiDetector(inner, SahiConfig(tile_size=400, overlap_ratio=0.2, run_full_frame_pass=True))

        detections = sahi.infer(frame)

        self.assertEqual(len(detections), 1)

    def test_class_names_and_lifecycle_delegate_to_inner(self):
        inner = BrightBoxDetector()
        sahi = SahiDetector(inner, SahiConfig())
        self.assertEqual(sahi.class_names, inner.class_names)
        sahi.warmup()  # should not raise
        sahi.close()  # should not raise


if __name__ == "__main__":
    unittest.main()
