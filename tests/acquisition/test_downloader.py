"""Unit tests for SampleDownloader in deepspacedb_xenium_pipeline.downloader."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from deepspacedb_xenium_pipeline import DownloadResult, PipelineConfig
from deepspacedb_xenium_pipeline.downloader import SampleDownloader


def test_downloader_init():
    cfg = PipelineConfig()
    dl = SampleDownloader(cfg)
    assert dl.config is cfg
    assert dl.session is not None


def test_load_sample_metadata_and_lookup(tmp_path):
    csv_file = tmp_path / "test_samples.csv"
    df = pd.DataFrame(
        [
            {"platform": "GPL31942", "series": "GSE263881", "sample": "GSM8253807"},
            {"platform": "GPL31942", "series": "GSE263881", "sample": "GSM8253808"},
        ]
    )
    df.to_csv(csv_file, index=False)

    dl = SampleDownloader(PipelineConfig(base_dir=tmp_path))
    dl.load_sample_metadata(csv_file)

    gpl, gse = dl._lookup_metadata("GSM8253807")
    assert gpl == "GPL31942"
    assert gse == "GSE263881"

    gpl_none, gse_none = dl._lookup_metadata("NONEXISTENT")
    assert gpl_none is None and gse_none is None


def test_load_samples_from_csv(tmp_path):
    csv_file = tmp_path / "test_samples.csv"
    df = pd.DataFrame(
        [
            {"platform": "GPL1", "series": "GSE1", "sample": "GSM1"},
            {"platform": "GPL2", "series": "GSE2", "sample": "GSM2"},
        ]
    )
    df.to_csv(csv_file, index=False)

    dl = SampleDownloader(PipelineConfig(base_dir=tmp_path))
    samples = dl.load_samples_from_csv(csv_file)
    assert len(samples) == 2
    assert samples[0] == ("GSM1", "GPL1", "GSE1")
    assert samples[1] == ("GSM2", "GPL2", "GSE2")


def test_destination_path_structure(tmp_path):
    cfg = PipelineConfig(base_dir=tmp_path)
    sample_dir = cfg.data_dir / "GPL31942" / "GSE263881" / "GSM8253807"
    assert sample_dir == tmp_path / "data" / "GPL31942" / "GSE263881" / "GSM8253807"


def test_get_download_info(mock_miniml_xml):
    dl = SampleDownloader()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = mock_miniml_xml.encode("utf-8")

    with patch.object(dl.session, "get", return_value=mock_resp):
        info = dl.get_download_info("GSM8253807")
        assert info is not None
        assert info.gsm_id == "GSM8253807"
        assert len(info.supplementary_files) == 2
        assert "GSM8253807_outs.tar.gz" in info.supplementary_files[0]


def test_download_gsm_sample_resume(tmp_path):
    """If .download_complete marker exists, skip downloading."""
    sample_dir = tmp_path / "data" / "GPL1" / "GSE1" / "GSM1"
    raw_dir = sample_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    marker = sample_dir / ".download_complete"
    marker.write_text("done")

    fake_file = raw_dir / "sample.tar.gz"
    fake_file.write_bytes(b"content")

    dl = SampleDownloader(PipelineConfig(base_dir=tmp_path))
    result = dl.download_gsm_sample("GSM1", "GPL1", "GSE1")
    assert result.success is True
    assert result.gsm_id == "GSM1"


def test_download_multiple(tmp_path):
    dl = SampleDownloader(PipelineConfig(base_dir=tmp_path))
    sample_list = [("GSM1", "GPL1", "GSE1"), ("GSM2", "GPL1", "GSE1")]

    mock_result = DownloadResult(sample_path=tmp_path / "GSM1", success=True, gsm_id="GSM1")
    with patch.object(dl, "download_gsm_sample", return_value=mock_result) as mock_dl:
        res = dl.download_multiple(sample_list)
        assert len(res) == 2
        assert mock_dl.call_count == 2
