"""Unit tests for sdata_utils in deepspacedb_xenium_pipeline.sdata_utils."""

from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from shapely.geometry import Point
from spatialdata import SpatialData
from spatialdata.models import ShapesModel

from deepspacedb_xenium_pipeline.sdata_utils import (
    aligned_cell_circles,
    expression_and_coords,
    get_table,
    set_table,
)


def test_get_set_table():
    adata1 = ad.AnnData(X=np.zeros((2, 2)))
    sdata = SpatialData(tables={"table": adata1})

    assert get_table(sdata) is adata1

    adata2 = ad.AnnData(X=np.ones((2, 2)))
    set_table(sdata, adata2)
    assert get_table(sdata) is adata2


def test_expression_and_coords():
    logger = logging.getLogger("test")
    X = csr_matrix([[1.0, 2.0], [3.0, 4.0]])
    adata = ad.AnnData(X=X, obs=pd.DataFrame(index=["c1", "c2"]), var=pd.DataFrame(index=["g1", "g2"]))
    adata.layers["normalized"] = X

    import geopandas as gpd

    gdf = ShapesModel.parse(
        gpd.GeoDataFrame(
            {
                "geometry": [Point(10.0, 20.0), Point(30.0, 40.0)],
                "radius": [5.0, 5.0],
            },
            index=["c1", "c2"],
        )
    )

    sdata = SpatialData(tables={"table": adata}, shapes={"cell_circles": gdf})
    res = expression_and_coords(sdata, logger)

    assert res is not None
    expr, names, xs, ys = res
    assert len(names) == 2
    assert np.allclose(xs, [10.0, 30.0])
    assert np.allclose(ys, [20.0, 40.0])


def _circles(xs, ys, index):
    import geopandas as gpd

    return ShapesModel.parse(
        gpd.GeoDataFrame(
            {"geometry": [Point(x, y) for x, y in zip(xs, ys, strict=True)], "radius": [5.0] * len(xs)},
            index=index,
        )
    )


def _adata(cell_ids):
    n = len(cell_ids)
    X = csr_matrix(np.arange(n * 2, dtype=float).reshape(n, 2))
    adata = ad.AnnData(X=X, obs=pd.DataFrame(index=list(cell_ids)), var=pd.DataFrame(index=["g1", "g2"]))
    adata.layers["normalized"] = X
    return adata


def test_expression_and_coords_follows_table_order_not_shape_order():
    """Coordinates must follow adata's row order, not cell_circles' on-disk order."""
    logger = logging.getLogger("test")
    adata = _adata(["c2", "c1"])
    gdf = _circles([10.0, 30.0], [20.0, 40.0], ["c1", "c2"])
    sdata = SpatialData(tables={"table": adata}, shapes={"cell_circles": gdf})

    _, _, xs, ys = expression_and_coords(sdata, logger)
    assert np.allclose(xs, [30.0, 10.0])
    assert np.allclose(ys, [40.0, 20.0])


def test_expression_and_coords_matches_qc_filtered_subset():
    """A QC-filtered table must get its own cells, never the first n circles."""
    logger = logging.getLogger("test")
    adata = _adata(["c1", "c3"])  # c2 was dropped by QC
    gdf = _circles([10.0, 20.0, 30.0], [11.0, 21.0, 31.0], ["c1", "c2", "c3"])
    sdata = SpatialData(tables={"table": adata}, shapes={"cell_circles": gdf})

    _, _, xs, ys = expression_and_coords(sdata, logger)
    assert np.allclose(xs, [10.0, 30.0])
    assert np.allclose(ys, [11.0, 31.0])


def test_expression_and_coords_numeric_index_dtype_mismatch():
    """An int shapes index vs string cell_ids is a dtype mismatch, not a real one."""
    logger = logging.getLogger("test")
    adata = _adata(["116", "117"])
    gdf = _circles([10.0, 30.0], [20.0, 40.0], pd.Index([116, 117]))
    sdata = SpatialData(tables={"table": adata}, shapes={"cell_circles": gdf})

    _, _, xs, ys = expression_and_coords(sdata, logger)
    assert np.allclose(xs, [10.0, 30.0])


def test_aligned_cell_circles_recovers_labels_from_boundaries():
    """A label-less RangeIndex on cell_circles is recovered from cell_boundaries."""
    import geopandas as gpd
    from shapely.geometry import Polygon

    logger = logging.getLogger("test")
    adata = _adata(["c1", "c3"])
    gdf = _circles([10.0, 20.0, 30.0], [11.0, 21.0, 31.0], pd.RangeIndex(3))
    boundaries = ShapesModel.parse(
        gpd.GeoDataFrame(
            {
                "geometry": [
                    Polygon([(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 1), (x - 1, y + 1)])
                    for x, y in [(10.0, 11.0), (20.0, 21.0), (30.0, 31.0)]
                ]
            },
            index=["c1", "c2", "c3"],
        )
    )
    sdata = SpatialData(tables={"table": adata}, shapes={"cell_circles": gdf, "cell_boundaries": boundaries})

    circles = aligned_cell_circles(sdata, adata, logger)
    assert circles is not None
    assert np.allclose(circles.geometry.x.to_numpy(), [10.0, 30.0])


def test_aligned_cell_circles_returns_none_rather_than_guessing():
    """With no recoverable cell_id mapping, refuse to fall back to positions."""
    logger = logging.getLogger("test")
    adata = _adata(["c1", "c3"])
    gdf = _circles([10.0, 20.0, 30.0], [11.0, 21.0, 31.0], pd.RangeIndex(3))
    sdata = SpatialData(tables={"table": adata}, shapes={"cell_circles": gdf})

    assert aligned_cell_circles(sdata, adata, logger) is None
    assert expression_and_coords(sdata, logger) is None
