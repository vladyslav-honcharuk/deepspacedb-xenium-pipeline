"""Unit tests for TableLoader and MexCountsLoader in deepspacedb_xenium_pipeline.loaders."""

from __future__ import annotations

import gzip

import pandas as pd

from deepspacedb_xenium_pipeline.loaders.counts import MexCountsLoader, zero_counts_frame
from deepspacedb_xenium_pipeline.loaders.tables import TableLoader


def test_zero_counts_frame():
    index = pd.Index(["c1", "c2", "c3"], name="cell_id")
    df = zero_counts_frame(index)
    assert len(df) == 3
    assert "total_counts" in df.columns
    assert "transcript_counts" in df.columns
    assert (df["total_counts"] == 0).all()


def test_mex_counts_loader_empty_when_no_barcodes(tmp_path):
    loader = MexCountsLoader()
    df = loader.load(tmp_path)
    assert df.empty


def test_fix_features_file(tmp_path):
    feat_file = tmp_path / "features.tsv.gz"
    with gzip.open(feat_file, "wt") as f:
        f.write("ENSG0001\tGeneA\nENSG0002\tGeneB\n")

    loader = TableLoader()
    loader.fix_features_file(feat_file)

    with gzip.open(feat_file, "rt") as f:
        lines = f.readlines()
    assert len(lines) == 2
    assert lines[0].strip().endswith("Gene Expression")
