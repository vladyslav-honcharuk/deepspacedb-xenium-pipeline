"""Opt-in compatibility patches for ``spatialdata_io``.

The original module monkey-patched ``spatialdata_io.readers.xenium._get_polygons``
*at import time*, so merely importing the file mutated a third-party module for
the whole process. That is a side effect a library must not impose.

Here the patch is applied only when :func:`apply_spatialdata_patches` is called
explicitly (the CLI and the high-level :class:`~xenium_pipeline.transformer.XeniumTransformer`
do so). The patch is idempotent.

The underlying bug it fixes: ``spatialdata_io`` compares polygon indices with
``Series.equals`` (which also compares the index/name), but the reader builds
the two series with mismatched names, raising a spurious ``AssertionError`` on
otherwise-valid Xenium data. The patched version compares values only.
"""
from __future__ import annotations

import warnings

from .logging_setup import get_logger

_PATCHED = False
_logger = get_logger(__name__)


def apply_spatialdata_patches() -> None:
    """Patch ``spatialdata_io`` polygon reading. Safe to call more than once."""
    global _PATCHED
    if _PATCHED:
        return

    import geopandas as gpd
    import numpy as np
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
        out = [
            _poly(i.to_numpy())
            for _, i in group_by[
                [
                    xenium_mod.XeniumKeys.BOUNDARIES_VERTEX_X,
                    xenium_mod.XeniumKeys.BOUNDARIES_VERTEX_Y,
                ]
            ]
        ]
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
        scale = xenium_mod.Scale(
            [1.0 / specs["pixel_size"], 1.0 / specs["pixel_size"]], axes=("x", "y")
        )
        return xenium_mod.ShapesModel.parse(geo_df, transformations={"global": scale})

    xenium_mod._get_polygons = patched_get_polygons
    _PATCHED = True
    _logger.debug("Applied spatialdata_io._get_polygons compatibility patch")
