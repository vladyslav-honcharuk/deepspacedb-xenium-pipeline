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


def _read_bins(path):
    import zarr

    store = zarr.storage.ZipStore(str(path), mode="r")
    try:
        return np.asarray(zarr.open_array(store, mode="r")[:])
    finally:
        store.close()


def test_bin_one_size_sums_cells_per_bin(tmp_path):
    """Two cells share a bin; their per-gene counts must be summed into it."""
    expr = csr_matrix(np.array([[5.0, 1.0], [2.0, 4.0], [0.0, 3.0]]))
    gene_names = ["GeneA", "GeneB"]
    # Cells 0 and 1 fall in the same 20um bin; cell 2 is elsewhere.
    x_coords = np.array([10.0, 15.0, 50.0])
    y_coords = np.array([10.0, 12.0, 60.0])

    bin_one_size(expr, gene_names, x_coords, y_coords, 20, str(tmp_path))
    binned = _read_bins(tmp_path / "zarr" / "bins_size_20.zarr.zip")

    assert binned.shape[0] == 2
    assert binned[0].sum() == 5.0 + 2.0  # GeneA total
    assert binned[1].sum() == 1.0 + 4.0 + 3.0  # GeneB total
    assert binned[0, 0, 0] == 7.0  # cells 0+1 share the first bin
    assert binned[1, 0, 0] == 5.0


def test_bin_one_size_spatial_copy_matches(tmp_path):
    expr = csr_matrix(np.array([[5.0, 1.0], [2.0, 0.0], [0.0, 3.0]]))
    bin_one_size(
        expr, ["GeneA", "GeneB"], np.array([10.0, 25.0, 50.0]), np.array([10.0, 30.0, 60.0]), 20, str(tmp_path)
    )

    plain = _read_bins(tmp_path / "zarr" / "bins_size_20.zarr.zip")
    spatial = _read_bins(tmp_path / "zarr" / "bins_size_20_spatial.zarr.zip")
    assert np.array_equal(plain, spatial)


def test_export_single_cell_duplicate_gene_names(tmp_path):
    """A repeated gene symbol must not collide on the same zarr dataset path."""
    import zarr

    expr = csr_matrix(np.array([[5.0, 1.0, 2.0], [2.0, 0.0, 0.0]]))
    created = export_single_cell(
        expr, ["GeneA", "GeneB", "GeneA"], np.array([10.0, 25.0]), np.array([10.0, 30.0]), str(tmp_path)
    )
    assert len(created) == 4

    store = zarr.storage.ZipStore(str(tmp_path / "zarr" / "sparse_gene_expression_chunked_per_gene.zarr.zip"), mode="r")
    try:
        keys = set(zarr.group(store=store).array_keys())
    finally:
        store.close()
    assert "data_GeneA" in keys
    assert "data_GeneB" in keys
    assert "data_000002_GeneA" in keys  # the repeat, disambiguated by index
