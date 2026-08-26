"""Unit tests for SpatialDataBuilder in deepspacedb_xenium_pipeline.loaders.builder."""

from __future__ import annotations

import pandas as pd

from deepspacedb_xenium_pipeline.config import PipelineConfig
from deepspacedb_xenium_pipeline.loaders.builder import SpatialDataBuilder


def test_transcript_counts_reconstruct_assigned_genes(tmp_path):
    transcripts = pd.DataFrame(
        {
            "cell_id": ["cell-a", "cell-a", "cell-b", "UNASSIGNED", "cell-b"],
            "feature_name": ["GeneA", "GeneA", "GeneB", "GeneA", "NegControlProbe_1"],
            "codeword_category": [
                "predesigned_gene",
                "predesigned_gene",
                "predesigned_gene",
                "predesigned_gene",
                "control_probe",
            ],
        }
    )
    transcripts_path = tmp_path / "transcripts.parquet"
    transcripts.to_parquet(transcripts_path)

    builder = SpatialDataBuilder(PipelineConfig())
    adata = builder._build_counts_from_transcripts(tmp_path)

    assert list(adata.obs_names) == ["cell-a", "cell-b"]
    assert list(adata.var_names) == ["GeneA", "GeneB"]
    assert adata.X.toarray().tolist() == [[2, 0], [0, 1]]


def test_builder_init():
    cfg = PipelineConfig()
    builder = SpatialDataBuilder(cfg)
    assert builder.config is cfg
    assert builder.tables is not None
    assert builder.morphology is not None
