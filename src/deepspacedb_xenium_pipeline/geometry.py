"""Pure-numpy polygon geometry helpers (no I/O, easily unit-testable)."""

from __future__ import annotations

from typing import Optional, Tuple

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


def nearest_match_assignment(
    points: np.ndarray,
    candidates: np.ndarray,
    max_dist: float = 10.0,
) -> Optional[np.ndarray]:
    """Match each row of ``points`` to a unique row of ``candidates`` by distance.

    Used to recover identity when two per-cell tables have different row counts
    (e.g. some cells failed to produce a valid circle geometry) and so cannot
    simply be lined up positionally.

    Uses greedy closest-first assignment -- points are processed in order of
    their single closest candidate distance, each claiming the nearest
    not-yet-taken candidate among its 10 nearest -- rather than plain
    nearest-neighbour, since two points can share the same nearest candidate.
    Returns ``None`` (callers should fail loudly rather than guess) unless every
    point gets a unique match within ``max_dist``.
    """
    from scipy.spatial import cKDTree

    if len(points) == 0 or len(candidates) == 0:
        return None

    tree = cKDTree(candidates)
    k = min(10, len(candidates))
    dist_k, idx_k = tree.query(points, k=k)
    if k == 1:  # query() drops the k-axis when k == 1
        dist_k, idx_k = dist_k[:, None], idx_k[:, None]

    n = len(points)
    assigned = -np.ones(n, dtype=np.int64)
    assigned_dist = np.full(n, np.inf)
    taken = np.zeros(len(candidates), dtype=bool)

    for ci in np.argsort(dist_k[:, 0]):
        for rank in range(k):
            bi = idx_k[ci, rank]
            if not taken[bi]:
                taken[bi] = True
                assigned[ci] = bi
                assigned_dist[ci] = dist_k[ci, rank]
                break

    if (assigned < 0).any() or assigned_dist.max() > max_dist:
        return None
    return assigned
