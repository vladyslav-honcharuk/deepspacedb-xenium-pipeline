"""Tiny SpatialData helpers shared across analysis components."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


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


def aligned_cell_circles(sdata, adata, logger):
    """Return ``cell_circles`` shapes aligned to ``adata``'s cells.

    Tries index-based filtering first; if the index types mismatch (e.g. a
    RangeIndex vs string cell IDs) it falls back to positional alignment, taking
    the first ``len(adata)`` circles. Shared by binning and single-cell export.
    """
    cell_names = (adata.obs["cell_id"] if "cell_id" in adata.obs else adata.obs.index).tolist()
    circles_all = sdata.shapes["cell_circles"]
    filtered = circles_all[circles_all.index.isin(cell_names)]
    if filtered.empty or filtered.geometry.x.isna().all():
        logger.warning(
            "cell_circles index (%s) doesn't match cell_names; using positional alignment",
            circles_all.index.dtype,
        )
        filtered = circles_all.iloc[: len(adata)]
        logger.info("Positional alignment: %d circles for %d cells", len(filtered), len(adata))
    return filtered


def expression_and_coords(sdata, logger) -> Optional[Tuple[object, List[str], np.ndarray, np.ndarray]]:
    """Normalized expression matrix plus cell x/y, or None if coordinates are unusable."""
    adata = get_table(sdata)
    gene_expression = adata.layers["normalized"] if "normalized" in adata.layers else adata.X
    gene_names = list(adata.var_names)
    circles = aligned_cell_circles(sdata, adata, logger)
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
