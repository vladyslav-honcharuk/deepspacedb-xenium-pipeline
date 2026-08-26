"""Unit tests for spatial binning and single cell export in deepspacedb_xenium_pipeline.analysis."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from deepspacedb_xenium_pipeline.analysis.binning import bin_one_size
from deepspacedb_xenium_pipeline.analysis.export import export_single_cell


def test_bin_one_size(tmp_path):
    # 3 cells, 2 genes
    expr = csr_matrix(np.array([[5.0, 1.0], [2.0, 0.0], [0.0, 3.0]]))
    gene_names = ["GeneA", "GeneB"]
    x_coords = np.array([10.0, 25.0, 50.0])
    y_coords = np.array([10.0, 30.0, 60.0])

    created = bin_one_size(expr, gene_names, x_coords, y_coords, 20, str(tmp_path))
    assert len(created) == 2
    assert "bins_size_20.zarr.zip" in created
    assert "bins_size_20_spatial.zarr.zip" in created
    assert (tmp_path / "zarr" / "bins_size_20.zarr.zip").exists()


def test_export_single_cell(tmp_path):
    expr = csr_matrix(np.array([[5.0, 1.0], [2.0, 0.0], [0.0, 3.0]]))
    gene_names = ["GeneA", "GeneB"]
    x_coords = np.array([10.0, 25.0, 50.0])
    y_coords = np.array([10.0, 30.0, 60.0])

    created = export_single_cell(expr, gene_names, x_coords, y_coords, str(tmp_path))
    assert len(created) == 4
    assert "sparse_gene_expression_csr.zarr.zip" in created
    assert "sparse_gene_expression_csc.zarr.zip" in created
    assert "sparse_gene_expression_chunked_per_gene.zarr.zip" in created
    assert "cell_coordinates.zarr.zip" in created
