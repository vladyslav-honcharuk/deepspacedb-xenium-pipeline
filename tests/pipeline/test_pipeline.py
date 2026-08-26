"""Unit tests for XeniumPipeline facade in deepspacedb_xenium_pipeline.pipeline."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from deepspacedb_xenium_pipeline.config import PipelineConfig
from deepspacedb_xenium_pipeline.finder import SampleFinder
from deepspacedb_xenium_pipeline.pipeline import XeniumPipeline
from deepspacedb_xenium_pipeline.results import DownloadResult, RawProcessingResult


def test_pipeline_init(tmp_path):
    cfg = PipelineConfig(base_dir=tmp_path)
    pipe = XeniumPipeline(cfg)
    assert pipe.config is cfg
    assert isinstance(pipe.finder, SampleFinder)
    assert pipe.downloader is not None


def test_pipeline_find_writes_output(tmp_path):
    pipe = XeniumPipeline(PipelineConfig(base_dir=tmp_path))
    mock_df = pd.DataFrame([{"platform": "GPL1", "series": "GSE1", "sample": "GSM1"}])
    out_csv = tmp_path / "samples.csv"

    with patch.object(pipe.finder, "find", return_value=mock_df):
        res = pipe.find(out_csv)
        assert len(res) == 1
        assert out_csv.exists()


def test_pipeline_download_delegation(tmp_path):
    pipe = XeniumPipeline(PipelineConfig(base_dir=tmp_path))
    mock_res = DownloadResult(sample_path=tmp_path / "GSM1", success=True, gsm_id="GSM1")

    with patch.object(pipe.downloader, "download_gsm_sample", return_value=mock_res) as mock_dl:
        res = pipe.download(gsm_id="GSM1")
        assert "GSM1" in res
        assert res["GSM1"].success is True
        mock_dl.assert_called_once_with("GSM1")


def test_pipeline_run_chain(tmp_path):
    pipe = XeniumPipeline(PipelineConfig(base_dir=tmp_path))
    out_csv = tmp_path / "samples.csv"
    mock_df = pd.DataFrame([{"platform": "GPL1", "series": "GSE1", "sample": "GSM1"}])

    with (
        patch.object(pipe, "find", return_value=mock_df) as mock_find,
        patch.object(pipe, "download") as mock_down,
        patch.object(pipe, "discover_samples", return_value=[tmp_path / "GSM1"]),
        patch.object(
            pipe,
            "salvage_sample",
            return_value=RawProcessingResult(sample_path=tmp_path / "GSM1", success=True),
        ),
        patch.object(pipe, "process_all", return_value=({}, [])) as mock_proc,
    ):
        results, summaries = pipe.run(out_csv)
        assert mock_find.called
        assert mock_down.called
        assert mock_proc.called
