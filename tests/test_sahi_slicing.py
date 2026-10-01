"""SAHI tile geometry and merge tests."""
import numpy as np

from inference.sahi_batched import compute_tiles, merge_detections, remap_tile_detections


def test_small_region_is_one_tile():
    assert compute_tiles(500, 400) == [(0, 0, 500, 400)]


def test_tiles_cover_the_far_edge():
    tiles = compute_tiles(1500, 700, 640, 640, 0.2, 0.2)
    assert max(t[2] for t in tiles) == 1500
    assert max(t[3] for t in tiles) == 700


def test_remap_shifts_into_region_coordinates():
    boxes = np.array([[10.0, 20.0, 50.0, 60.0]])
    out = remap_tile_detections(boxes, (512, 128))
    assert out.tolist() == [[522.0, 148.0, 562.0, 188.0]]


def test_merge_deduplicates_across_overlapping_tiles():
    a = (np.array([[10.0, 10.0, 110.0, 110.0]]), np.array([0.9]), np.array([0]), (0, 0))
    b = (np.array([[8.0, 8.0, 108.0, 108.0]]),   np.array([0.8]), np.array([0]), (0, 0))
    boxes, scores, _ = merge_detections([a, b], iou_thr=0.5)
    assert len(boxes) == 1 and scores[0] == 0.9
