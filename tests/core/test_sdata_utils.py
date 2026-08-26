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
