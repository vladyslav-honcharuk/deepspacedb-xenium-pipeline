"""Tiny SpatialData helpers shared across analysis components."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .geometry import nearest_match_assignment


def get_table(sdata):
    """Return the main expression table, tolerating the legacy ``mydata`` key."""
    if "table" in sdata.tables:
        return sdata.tables["table"]
    return sdata.tables["mydata"]


def set_table(sdata, adata) -> None:
    """Write ``adata`` back to whichever table key the object uses."""
    if "table" in sdata.tables:
        sdata.tables["table"] = adata
    else:
        sdata.tables["mydata"] = adata


def reindex_by_cell_id(df, cell_names: List):
    """Reindex a shapes GeoDataFrame to ``cell_names`` (the table's cell order).

    Retries with a numeric cast when the direct reindex leaves everything
    unmatched: some samples use a plain sequential integer ``cell_id``, stored
    as int in the shapes index but as strings in ``adata.obs['cell_id']`` (e.g.
    ``'116'`` vs ``116``). That is a dtype mismatch, not an identity mismatch,
    and ``reindex()`` only matches on exact value+type equality.
    """
    candidate = df.reindex(cell_names)
    if not candidate.geometry.x.isna().any():
        return candidate
    if pd.api.types.is_numeric_dtype(df.index):
        try:
            numeric_names = [int(c) for c in cell_names]
        except (TypeError, ValueError):
            return candidate
        numeric_candidate = df.reindex(numeric_names)
        if not numeric_candidate.geometry.x.isna().any():
            return numeric_candidate
    return candidate


def relabel_circles_by_nearest_centroid(circles_all, ref, max_dist: float = 10.0):
    """Recover ``circles_all``'s row labels from a polygon GeoDataFrame.

    ``ref`` is ``cell_boundaries`` or ``nucleus_boundaries``; each circle is
    matched to its nearest boundary centroid. See
    :func:`~.geometry.nearest_match_assignment`.
    """
    points = np.column_stack([circles_all.geometry.x.to_numpy(), circles_all.geometry.y.to_numpy()])
    ref_centroid = ref.geometry.centroid
    candidates = np.column_stack([ref_centroid.x.to_numpy(), ref_centroid.y.to_numpy()])
    assigned = nearest_match_assignment(points, candidates, max_dist)
    if assigned is None:
        return None
    return circles_all.set_axis(ref.index[assigned], axis=0)


def relabel_circles_by_nearest_point(circles_all, ref_x, ref_y, ref_index, max_dist: float = 10.0):
    """Recover ``circles_all``'s row labels from a flat table of x/y points.

    ``ref_x``/``ref_y``/``ref_index`` come from e.g. ``processed/cells.parquet``
    (``x_centroid``/``y_centroid``/``cell_id``); each circle is matched to its
    nearest point. See :func:`~.geometry.nearest_match_assignment`.
    """
    points = np.column_stack([circles_all.geometry.x.to_numpy(), circles_all.geometry.y.to_numpy()])
    candidates = np.column_stack([np.asarray(ref_x), np.asarray(ref_y)])
    assigned = nearest_match_assignment(points, candidates, max_dist)
    if assigned is None:
        return None
    return circles_all.set_axis(pd.Index(np.asarray(ref_index)[assigned]), axis=0)


def _counts_comparable(n_a: int, n_b: int) -> bool:
    """True when two per-cell tables are close enough in length to match up.

    Guards the nearest-neighbour recovery against nonsense matches to a wildly
    different reference.
    """
    return bool(n_b) and 0.5 <= n_a / n_b <= 2.0


def aligned_cell_circles(sdata, adata, logger):
    """Return ``cell_circles`` shapes aligned to ``adata``'s cells, or ``None``.

    Aligns strictly by ``cell_id``, in ``adata``'s row order -- never by
    position. ``cell_circles`` holds every originally-segmented cell in raw
    FOV/tile order while ``adata`` has already been QC-filtered down to a
    scattered subset, so "take the first n" silently drops whole spatial tiles
    instead of the actual QC-dropped cells, producing spatially corrupted output.

    Some samples (~1 in 3) carry no real ``cell_id`` in the ``cell_circles``
    index at all (a bare positional RangeIndex, geometry/radius only). For those
    the labels are recovered from ``cell_boundaries``/``nucleus_boundaries`` or
    from ``processed/cells.parquet``, all of which are built from the same
    original, unfiltered per-cell list and do carry ``cell_id``. When no
    ``cell_id``-based mapping can be established, this returns ``None`` rather
    than guessing.
    """
    cell_names = (adata.obs["cell_id"] if "cell_id" in adata.obs else adata.obs.index).tolist()
    circles_all = sdata.shapes["cell_circles"]
    circles = reindex_by_cell_id(circles_all, cell_names)
    if not circles.geometry.x.isna().any():
        return circles

    logger.warning(
        "cell_circles index (%s) doesn't carry cell_id labels (%d/%d cells unmatched); "
        "attempting to recover labels from cell_boundaries/nucleus_boundaries",
        circles_all.index.dtype,
        int(circles.geometry.x.isna().sum()),
        len(cell_names),
    )

    for ref_name in ("cell_boundaries", "nucleus_boundaries"):
        ref = sdata.shapes.get(ref_name)
        if ref is None:
            continue
        if len(ref) == len(circles_all):
            relabeled = circles_all.set_axis(ref.index, axis=0)
        else:
            # Different cell counts (e.g. some cells failed to produce a valid
            # circle geometry): set_axis needs equal lengths, so match by
            # nearest boundary centroid instead.
            if not _counts_comparable(len(circles_all), len(ref)):
                continue
            relabeled = relabel_circles_by_nearest_centroid(circles_all, ref)
            if relabeled is None:
                continue
        candidate = reindex_by_cell_id(relabeled, cell_names)
        if not candidate.geometry.x.isna().any():
            logger.info("Recovered cell_id labels from '%s'", ref_name)
            return candidate

    # Some samples have no boundaries at all in sdata.shapes: fall back to the
    # flat processed/cells.parquet, built from the same original, unfiltered
    # per-cell list in the same row order and always carrying the real cell_id.
    sdata_path = getattr(sdata, "path", None)
    cells_parquet = Path(sdata_path).parent / "processed" / "cells.parquet" if sdata_path else None
    if cells_parquet is not None and cells_parquet.exists():
        cells_df = pd.read_parquet(cells_parquet, columns=["cell_id", "x_centroid", "y_centroid"])
        if len(cells_df) == len(circles_all):
            relabeled = circles_all.set_axis(pd.Index(cells_df["cell_id"]), axis=0)
        elif _counts_comparable(len(circles_all), len(cells_df)):
            # Off-by-a-few row counts happen here too; match by nearest x/y
            # rather than assuming aligned order.
            relabeled = relabel_circles_by_nearest_point(
                circles_all, cells_df["x_centroid"], cells_df["y_centroid"], cells_df["cell_id"]
            )
        else:
            relabeled = None
        if relabeled is not None:
            candidate = reindex_by_cell_id(relabeled, cell_names)
            if not candidate.geometry.x.isna().any():
                logger.info("Recovered cell_id labels from processed/cells.parquet")
                return candidate

    logger.error(
        "Could not recover a cell_id-based mapping between cell_circles and the expression "
        "table - refusing to fall back to a positional guess, since that silently produces "
        "spatially-corrupted output (missing rectangular tiles)"
    )
    return None


def expression_and_coords(sdata, logger) -> Optional[Tuple[object, List[str], np.ndarray, np.ndarray]]:
    """Normalized expression matrix plus cell x/y, or None if coordinates are unusable."""
    adata = get_table(sdata)
    gene_expression = adata.layers["normalized"] if "normalized" in adata.layers else adata.X
    gene_names = list(adata.var_names)
    circles = aligned_cell_circles(sdata, adata, logger)
    if circles is None:
        return None
    x_coords = np.asarray(circles.geometry.x)
    y_coords = np.asarray(circles.geometry.y)
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    if any(np.isnan(v) for v in (x_min, x_max, y_min, y_max)):
        logger.error("Invalid coordinate ranges: x=[%s,%s] y=[%s,%s]", x_min, x_max, y_min, y_max)
        return None
    if x_max <= 0 or y_max <= 0:
        logger.error("Invalid coordinate maxima: x=%s y=%s", x_max, y_max)
        return None
    return gene_expression, gene_names, x_coords, y_coords
