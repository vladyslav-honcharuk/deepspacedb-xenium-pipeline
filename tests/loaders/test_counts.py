"""Unit tests for MexCountsLoader and CellsParquetBuilder in loaders.counts."""

from __future__ import annotations

import gzip
import io

import pandas as pd
from scipy.io import mmwrite
from scipy.sparse import csr_matrix

from deepspacedb_xenium_pipeline import constants
from deepspacedb_xenium_pipeline.loaders.counts import (
    CellsParquetBuilder,
    MexCountsLoader,
    zero_counts_frame,
)


def test_zero_counts_frame():
    idx = pd.Index(["c1", "c2"], name="cell_id")
    df = zero_counts_frame(idx)
    assert len(df) == 2
    assert "total_counts" in df.columns
    assert df["total_counts"].sum() == 0


def test_mex_counts_loader_empty(tmp_path):
    loader = MexCountsLoader()
    # No barcodes file exists
    df = loader.load(tmp_path)
    assert df.empty


def test_mex_counts_loader_valid(tmp_path):
    # Create valid barcodes.tsv.gz, features.tsv.gz, matrix.mtx.gz
    with gzip.open(tmp_path / constants.BARCODES_TSV_GZ, "wt") as fh:
        fh.write("c1\nc2\n")

    with gzip.open(tmp_path / constants.FEATURES_TSV_GZ, "wt") as fh:
        fh.write("g1\tGeneA\tGene Expression\ng2\tGeneB\tGene Expression\n")

    mat = csr_matrix([[5, 0], [2, 3]])
    buf = io.BytesIO()
    mmwrite(buf, mat)
    with gzip.open(tmp_path / constants.MATRIX_MTX_GZ, "wb") as fh:
        fh.write(buf.getvalue())

    loader = MexCountsLoader()
    df = loader.load(tmp_path)
    assert len(df) == 2
    assert "total_counts" in df.columns
    assert "cell_id" in df.columns or df.index.name == "cell_id"


def test_cells_parquet_builder_already_exists(tmp_path):
    cells_file = tmp_path / constants.CELLS_PARQUET
    pd.DataFrame({"cell_id": [1]}).to_parquet(cells_file)

    builder = CellsParquetBuilder()
    assert builder.ensure(tmp_path) is True


def test_cells_parquet_builder_from_boundaries(tmp_path):
    bnd_file = tmp_path / "cell_boundaries.parquet"
    df_bnd = pd.DataFrame(
        {
            "cell_id": ["c1", "c1", "c1", "c1"],
            "label_id": [1, 1, 1, 1],
            "vertex_x": [0.0, 10.0, 10.0, 0.0],
            "vertex_y": [0.0, 0.0, 10.0, 10.0],
        }
    )
    df_bnd.to_parquet(bnd_file)

    builder = CellsParquetBuilder()
    ok = builder.ensure(tmp_path)
    assert ok is True

    cells_file = tmp_path / constants.CELLS_PARQUET
    assert cells_file.exists()
    df_cells = pd.read_parquet(cells_file)
    assert "cell_id" in df_cells.columns
    assert "cell_area" in df_cells.columns
    assert "x_centroid" in df_cells.columns
