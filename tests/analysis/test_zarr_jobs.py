"""Unit tests for run_binning_and_export_parallel in analysis.zarr_jobs."""

from __future__ import annotations

import anndata as ad
import pandas as pd
from scipy.sparse import csr_matrix
from shapely.geometry import Point
from spatialdata import SpatialData
from spatialdata.models import ShapesModel

from deepspacedb_xenium_pipeline.analysis.zarr_jobs import run_binning_and_export_parallel
from deepspacedb_xenium_pipeline.config import PipelineConfig


def test_run_binning_and_export_parallel(tmp_path):
    X = csr_matrix([[1.0, 2.0], [3.0, 4.0], [0.0, 5.0]])
    obs = pd.DataFrame(index=["cell_1", "cell_2", "cell_3"])
    var = pd.DataFrame(index=["GeneA", "GeneB"])
    adata = ad.AnnData(X=X, obs=obs, var=var)

    import geopandas as gpd

    gdf = ShapesModel.parse(
        gpd.GeoDataFrame(
            {
                "geometry": [Point(10.0, 10.0), Point(20.0, 20.0), Point(30.0, 30.0)],
                "radius": [5.0, 5.0, 5.0],
            },
            index=["cell_1", "cell_2", "cell_3"],
        )
    )

    sdata = SpatialData(tables={"table": adata}, shapes={"cell_circles": gdf})
    # Binning is opt-in, so enable it explicitly to exercise the bin export path.
    config = PipelineConfig(bin_sizes=[10, 20], zarr_export_workers=2, skip_binning=False)

    bins, sc_files = run_binning_and_export_parallel(sdata, tmp_path, config)

    assert len(bins) > 0
    assert len(sc_files) > 0
    assert (tmp_path / "zarr" / "sparse_gene_expression_csr.zarr.zip").exists()
