import numpy as np

from deepspacedb_xenium_pipeline.geometry import clean_ring, polygon_area_centroid


def test_unit_square_area_and_centroid():
    area, cx, cy = polygon_area_centroid([0, 1, 1, 0], [0, 0, 1, 1])
    assert area == 1.0
    assert cx == 0.5
    assert cy == 0.5


def test_closed_ring_is_cleaned():
    x, y = clean_ring(np.array([0, 1, 1, 0, 0]), np.array([0, 0, 1, 1, 0]))
    assert len(x) == 4 and len(y) == 4


def test_degenerate_ring_returns_zero_area():
    area, cx, cy = polygon_area_centroid([0, 1], [0, 1])
    assert area == 0.0
    assert cx == 0.5 and cy == 0.5


def test_triangle_area():
    area, _, _ = polygon_area_centroid([0, 4, 0], [0, 0, 3])
    assert area == 6.0
