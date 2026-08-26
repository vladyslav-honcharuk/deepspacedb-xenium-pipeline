"""Unit tests for CLI commands in deepspacedb_xenium_pipeline.cli."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from deepspacedb_xenium_pipeline.cli import _build_parser, main
from deepspacedb_xenium_pipeline.results import DownloadResult


def test_cli_parser_find_arguments():
    parser = _build_parser()
    args = parser.parse_args(["find", "-o", "custom.csv"])
    assert args.command == "find"
    assert args.output == Path("custom.csv")


def test_cli_parser_download_arguments():
    parser = _build_parser()
    args = parser.parse_args(["download", "--gsm-id", "GSM123", "--max-samples", "5"])
    assert args.command == "download"
    assert args.gsm_id == "GSM123"
    assert args.max_samples == 5


def test_cli_parser_salvage_arguments():
    parser = _build_parser()
    args = parser.parse_args(["salvage", "/path/to/data"])
    assert args.command == "salvage"
    assert args.data_dir == Path("/path/to/data")


def test_cli_parser_process_arguments():
    parser = _build_parser()
    args = parser.parse_args(["process", "/path/to/data", "--enable-binning", "--zarr-export-workers", "4"])
    assert args.command == "process"
    assert args.data_dir == Path("/path/to/data")
    assert args.enable_binning is True
    assert args.zarr_export_workers == 4


def test_cli_main_find_execution(tmp_path):
    mock_df = pd.DataFrame([{"platform": "GPL1", "series": "GSE1", "sample": "GSM1"}])
    with patch("deepspacedb_xenium_pipeline.pipeline.XeniumPipeline.find", return_value=mock_df):
        ret = main(["find", "-o", str(tmp_path / "out.csv")])
        assert ret == 0


def test_cli_main_download_execution(tmp_path):
    mock_res = {"GSM1": DownloadResult(sample_path=tmp_path / "GSM1", success=True, gsm_id="GSM1")}
    with patch("deepspacedb_xenium_pipeline.pipeline.XeniumPipeline.download", return_value=mock_res):
        ret = main(["download", "--gsm-id", "GSM1"])
        assert ret == 0
