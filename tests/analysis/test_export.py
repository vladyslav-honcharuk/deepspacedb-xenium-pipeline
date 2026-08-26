"""Unit tests for SingleCellExporter and export_single_cell in analysis.export."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from shapely.geometry import Point
from spatialdata import SpatialData
from spatialdata.models import ShapesModel

from deepspacedb_xenium_pipeline.analysis.export import SingleCellExporter, export_single_cell
from deepspacedb_xenium_pipeline.config import PipelineConfig


def test_export_single_cell(tmp_path):
    mat = csr_matrix([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    gene_names = ["GeneA", "GeneB"]
    x_coords = np.array([10.0, 20.0, 30.0])
    y_coords = np.array([15.0, 25.0, 35.0])

    created = export_single_cell(mat, gene_names, x_coords, y_coords, str(tmp_path))

    assert "sparse_gene_expression_csr.zarr.zip" in created
    assert "sparse_gene_expression_csc.zarr.zip" in created
    assert "sparse_gene_expression_chunked_per_gene.zarr.zip" in created
    assert "cell_coordinates.zarr.zip" in created

    zarr_dir = tmp_path / "zarr"
    assert (zarr_dir / "sparse_gene_expression_csr.zarr.zip").exists()
    assert (zarr_dir / "cell_coordinates.zarr.zip").exists()


def test_single_cell_exporter_class(tmp_path):
    X = csr_matrix([[1.0, 2.0], [3.0, 4.0]])
    obs = pd.DataFrame(index=["cell_1", "cell_2"])
    var = pd.DataFrame(index=["GeneA", "GeneB"])
    adata = ad.AnnData(X=X, obs=obs, var=var)

    import geopandas as gpd

    gdf = ShapesModel.parse(
        gpd.GeoDataFrame(
            {
                "geometry": [Point(10.0, 10.0), Point(20.0, 20.0)],
                "radius": [5.0, 5.0],
            },
            index=["cell_1", "cell_2"],
        )
    )

    sdata = SpatialData(tables={"table": adata}, shapes={"cell_circles": gdf})
    exporter = SingleCellExporter(PipelineConfig())
    created = exporter.export(sdata, tmp_path)

    assert len(created) == 4
