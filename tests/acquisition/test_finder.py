"""Unit tests for SampleFinder in deepspacedb_xenium_pipeline.finder."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from deepspacedb_xenium_pipeline import FindError, PipelineConfig, SampleFinder, XeniumPipeline
from deepspacedb_xenium_pipeline.cli import _build_parser


def test_finder_imports_and_types():
    assert issubclass(FindError, Exception)
    cfg = PipelineConfig()
    finder = SampleFinder(cfg)
    assert finder.config is cfg


def test_finder_save_empty_and_non_empty(tmp_path):
    empty_df = pd.DataFrame(columns=["platform", "series", "sample"])
    assert SampleFinder.save(empty_df, str(tmp_path / "empty.csv")) is None
    assert not (tmp_path / "empty.csv").exists()

    df = pd.DataFrame([{"platform": "GPL1", "series": "GSE1", "sample": "GSM1"}])
    out_file = tmp_path / "samples.csv"
    res = SampleFinder.save(df, str(out_file))
    assert res == str(out_file)
    assert out_file.exists()


def test_get_xenium_platforms_mock():
    finder = SampleFinder()
    summary_xml = """<DocumentSummarySet>
        <DocumentSummary><Accession>GPL31942</Accession></DocumentSummary>
        <DocumentSummary><Accession>GPL31943</Accession></DocumentSummary>
    </DocumentSummarySet>"""
    root = ET.fromstring(summary_xml)

    with (
        patch.object(finder, "_search_gds", return_value=["1001", "1002"]),
        patch.object(finder, "_get_summaries", return_value=root),
    ):
        platforms = finder.get_xenium_platforms()
        assert platforms == ["GPL31942", "GPL31943"]


def test_get_samples_for_platform_mock():
    finder = SampleFinder()
    summary_xml = """<DocumentSummarySet>
        <DocumentSummary>
            <Accession>GSM8253807</Accession>
            <GSE>263881; 263882</GSE>
        </DocumentSummary>
    </DocumentSummarySet>"""
    root = ET.fromstring(summary_xml)

    with (
        patch.object(finder, "_search_gds", return_value=["3001"]),
        patch.object(finder, "_get_summaries", return_value=root),
    ):
        samples = finder.get_samples_for_platform("GPL31942")
        assert len(samples) == 1
        assert samples[0] == {
            "platform": "GPL31942",
            "series": "GSE263881",
            "sample": "GSM8253807",
        }


def test_find_all_samples():
    finder = SampleFinder()
    mock_samples = [{"platform": "GPL1", "series": "GSE1", "sample": "GSM1"}]

    with (
        patch.object(finder, "get_xenium_platforms", return_value=["GPL1"]),
        patch.object(finder, "get_samples_for_platform", return_value=mock_samples),
    ):
        df = finder.find()
        assert len(df) == 1
        assert list(df.columns) == ["platform", "series", "sample"]


def test_api_call_handling():
    finder = SampleFinder()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b"<eSearchResult><IdList><Id>101</Id></IdList></eSearchResult>"

    with patch.object(finder.session, "get", return_value=mock_resp):
        ids = finder._search_gds("test_query")
        assert ids == ["101"]

    # HTTP error handling
    mock_resp.status_code = 500
    with patch.object(finder.session, "get", return_value=mock_resp):
        ids = finder._search_gds("test_query")
        assert ids == []


def test_pipeline_find_delegation(tmp_path):
    pipeline = XeniumPipeline(PipelineConfig(base_dir=tmp_path))
    assert hasattr(pipeline, "finder")
    assert isinstance(pipeline.finder, SampleFinder)

    mock_df = pd.DataFrame([{"platform": "GPL1", "series": "GSE1", "sample": "GSM1"}])
    with patch.object(pipeline.finder, "find", return_value=mock_df) as mock_find:
        out_csv = tmp_path / "found_samples.csv"
        res_df = pipeline.find(out_csv)
        assert mock_find.called
        assert len(res_df) == 1
        assert out_csv.exists()


def test_cli_parser_find():
    parser = _build_parser()
    args = parser.parse_args(["find", "-o", "custom.csv"])
    assert args.command == "find"
    assert args.output == Path("custom.csv")
