"""ROI cluster tests."""
from pipeline.roi_cluster import cluster_area_ratio, cluster_box

SHAPE = (2160, 3840)


def test_encloses_every_box():
    c = cluster_box([(100, 100, 200, 200), (900, 700, 1000, 800)], SHAPE, min_size=0)
    assert c[0] <= 100 and c[1] <= 100 and c[2] >= 1000 and c[3] >= 800


def test_min_size_expands_a_tiny_target():
    c = cluster_box([(100, 100, 140, 130)], SHAPE, min_size=640)
    assert (c[2] - c[0]) >= 640 and (c[3] - c[1]) >= 640


def test_clamped_to_frame():
    c = cluster_box([(3800, 2100, 3839, 2159)], SHAPE, min_size=640)
    assert c[2] <= 3840 and c[3] <= 2160 and c[0] >= 0 and c[1] >= 0


def test_area_ratio():
    assert cluster_area_ratio(None, SHAPE) == 0.0
