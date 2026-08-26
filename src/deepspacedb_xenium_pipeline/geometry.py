"""Pure-numpy polygon geometry helpers (no I/O, easily unit-testable)."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def clean_ring(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Drop a duplicated closing vertex from a polygon ring, if present."""
    if len(x) >= 2 and x[0] == x[-1] and y[0] == y[-1]:
        return x[:-1], y[:-1]
    return x, y


def polygon_area_centroid(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Return ``(area, centroid_x, centroid_y)`` via the shoelace formula.

    Degenerate rings (< 3 vertices, or zero signed area) fall back to the mean
    of the vertices for the centroid and an area of ``0.0``.
    """
    x, y = clean_ring(np.asarray(x, float), np.asarray(y, float))
    n = len(x)
    if n < 3:
        return (
            0.0,
            float(np.mean(x) if n else np.nan),
            float(np.mean(y) if n else np.nan),
        )
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    cross = x * y_next - x_next * y
    signed_double_area = np.sum(cross)
    if signed_double_area == 0:
        return 0.0, float(np.mean(x)), float(np.mean(y))
    cx = np.sum((x + x_next) * cross) / (3 * signed_double_area)
    cy = np.sum((y + y_next) * cross) / (3 * signed_double_area)
    return abs(0.5 * signed_double_area), cx, cy
