"""Unit tests for Salvager in deepspacedb_xenium_pipeline.salvager."""

from __future__ import annotations

from unittest.mock import patch

from deepspacedb_xenium_pipeline.config import PipelineConfig
from deepspacedb_xenium_pipeline.salvager import Salvager


def test_salvager_missing_raw_dir(tmp_path):
    sample_dir = tmp_path / "sample_empty"
    sample_dir.mkdir()

    salvager = Salvager(PipelineConfig(base_dir=tmp_path))
    res = salvager.salvage_sample(sample_dir)
    assert res.success is False
    assert "Raw directory not found" in res.error_message


def test_salvager_already_processed_skip(tmp_path):
    sample_dir = tmp_path / "sample_done"
    raw_dir = sample_dir / "raw"
    raw_dir.mkdir(parents=True)
    proc_dir = sample_dir / "processed"
    proc_dir.mkdir(parents=True)
    (proc_dir / "experiment.xenium").write_text("{}")

    salvager = Salvager(PipelineConfig(base_dir=tmp_path, overwrite_existing=False))
    res = salvager.salvage_sample(sample_dir)
    assert res.success is True
    assert "experiment.xenium" in res.files_processed


def test_salvager_extraction_and_organization_flow(tmp_path):
    sample_dir = tmp_path / "sample_1"
    raw_dir = sample_dir / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "sample.tar.gz").write_bytes(b"content")

    salvager = Salvager(PipelineConfig(base_dir=tmp_path))
    with patch.object(salvager.extractor, "extract_all", return_value=2):
        with patch.object(salvager.organizer, "organize", return_value=2):
            res = salvager.salvage_sample(sample_dir)
            assert res.success is True
            assert res.processed_path == sample_dir / "processed"
