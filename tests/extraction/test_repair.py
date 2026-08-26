"""Unit tests for FileRepair in deepspacedb_xenium_pipeline.extraction.repair."""

from __future__ import annotations

import gzip
import zlib

import pandas as pd

from deepspacedb_xenium_pipeline.extraction.repair import FileRepair


def test_repair_raw_zlib_gz(tmp_path):
    # A file named .gz that is actually raw DEFLATE (no gzip header).
    payload = b"barcode-1\nbarcode-2\n"
    raw_deflate = zlib.compressobj(6, zlib.DEFLATED, -15)
    blob = raw_deflate.compress(payload) + raw_deflate.flush()
    (tmp_path / "barcodes.tsv.gz").write_bytes(blob)
    assert FileRepair().fix_gz_files(tmp_path) is True
    with gzip.open(tmp_path / "barcodes.tsv.gz", "rb") as fh:
        assert fh.read() == payload


def test_repair_valid_gz_untouched(tmp_path):
    payload = b"test data"
    with gzip.open(tmp_path / "matrix.mtx.gz", "wb") as fh:
        fh.write(payload)
    assert FileRepair().fix_gz_files(tmp_path) is True
    with gzip.open(tmp_path / "matrix.mtx.gz", "rb") as fh:
        assert fh.read() == payload


def test_repair_csv_as_parquet(tmp_path):
    # A file named .parquet that is actually CSV.
    (tmp_path / "cells.parquet").write_text("cell_id,x\nA,1\nB,2\n")
    assert FileRepair().fix_parquet_files(tmp_path) is True
    df = pd.read_parquet(tmp_path / "cells.parquet")
    assert list(df.columns) == ["cell_id", "x"]
    assert len(df) == 2


def test_clean_zarr_underscore_files(tmp_path):
    zarr_path = tmp_path / "sample.zarr"
    points_dir = zarr_path / "points" / "transcripts" / "points.parquet"
    points_dir.mkdir(parents=True, exist_ok=True)

    good_file = points_dir / "part-0.parquet"
    good_file.write_bytes(b"valid")
    junk_1 = points_dir / "._part-0.parquet"
    junk_1.write_bytes(b"junk")
    junk_2 = points_dir / "data_.part"
    junk_2.write_bytes(b"junk")

    repair = FileRepair()
    repair.clean_zarr_underscore_files(zarr_path)

    remaining = {p.name for p in points_dir.iterdir()}
    assert remaining == {"part-0.parquet"}
