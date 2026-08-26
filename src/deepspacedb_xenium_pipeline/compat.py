"""Compatibility patches for ``spatialdata_io``.

Applies fixes to ``spatialdata_io.readers.xenium._get_polygons`` when
:func:`apply_spatialdata_patches` is called. The patch is idempotent.

Fixes applied:

* Compares polygon index values directly to avoid spurious series name mismatches.
* Handles degenerate cell/nucleus boundary rings (<3 distinct vertices) gracefully
  by dropping invalid cells with a warning instead of raising an unhandled error.
"""

from __future__ import annotations

import warnings

import numpy as np

from .logging_setup import get_logger

_PATCHED = False
_logger = get_logger(__name__)


def apply_spatialdata_patches() -> None:
    """Patch ``spatialdata_io`` polygon reading. Safe to call more than once."""
    global _PATCHED
    if _PATCHED:
        return

    import geopandas as gpd
    import pandas as pd
    import pyarrow.parquet as pq
    import spatialdata_io.readers.xenium as xenium_mod
    from shapely import Polygon

    def patched_get_polygons(path, file, specs, n_jobs, idx=None):
        def _poly(arr):
            return Polygon(arr[:-1])

        df = pq.read_table(path / file).to_pandas()
        group_by = df.groupby(xenium_mod.XeniumKeys.CELL_ID)
        index = pd.Series(group_by.indices.keys())
        index.index = index.index.astype(str)
        index = xenium_mod._decode_cell_id_column(index)
        groups = list(
            group_by[
                [
                    xenium_mod.XeniumKeys.BOUNDARIES_VERTEX_X,
                    xenium_mod.XeniumKeys.BOUNDARIES_VERTEX_Y,
                ]
            ]
        )

        # A valid polygon ring needs >= 3 distinct corners. Drop the handful of
        # degenerate cells some Xenium runs ship instead of crashing the sample.
        valid_mask = np.array([len(np.unique(i.to_numpy()[:-1], axis=0)) >= 3 for _, i in groups])
        if not valid_mask.all():
            dropped = [str(cid) for (cid, _), keep in zip(groups, valid_mask, strict=False) if not keep]
            _logger.warning(
                "%d cell(s) in %s had degenerate (<3 point) boundaries and were dropped: %s",
                len(dropped),
                file,
                dropped,
            )
            index = index[valid_mask].reset_index(drop=True)
            # Only reindex idx when it is 1:1 with this boundary file's cells.
            # After keeping cells that lack nucleus boundaries, idx can be the
            # full table and boolean-indexing it here would IndexError.
            if idx is not None and len(idx) == len(valid_mask):
                idx = idx[valid_mask]

        out = [_poly(i.to_numpy()) for keep, (_, i) in zip(valid_mask, groups, strict=False) if keep]
        geo_df = gpd.GeoDataFrame({"geometry": out})
        version = xenium_mod._parse_version_of_xenium_analyzer(specs)
        if version is not None and version < xenium_mod.packaging.version.parse("2.0.0"):
            assert idx is not None
            assert len(idx) == len(geo_df)
            assert np.unique(geo_df.index).size == len(geo_df)
            # PATCH: compare values only, not Series.equals().
            assert np.array_equal(index.values, idx.values)
            geo_df.index = idx
        else:
            geo_df.index = index
            if not np.unique(geo_df.index).size == len(geo_df):
                warnings.warn(
                    "Found non-unique polygon indices, this will be addressed in a "
                    "future version of the reader. For the time being please consider "
                    "merging polygons with non-unique indices into single multi-polygons.",
                    UserWarning,
                    stacklevel=2,
                )
        scale = xenium_mod.Scale([1.0 / specs["pixel_size"], 1.0 / specs["pixel_size"]], axes=("x", "y"))
        return xenium_mod.ShapesModel.parse(geo_df, transformations={"global": scale})

    xenium_mod._get_polygons = patched_get_polygons
    _PATCHED = True
    _logger.debug("Applied spatialdata_io._get_polygons compatibility patch")


def get_polygons(path, file, specs, n_jobs, idx=None):
    """Call the patched polygon reader, applying the patch if needed."""
    apply_spatialdata_patches()
    import spatialdata_io.readers.xenium as xenium_mod

    return xenium_mod._get_polygons(path, file, specs, n_jobs, idx=idx)
