from pathlib import Path

from deepspacedb_xenium_pipeline import PipelineConfig
from deepspacedb_xenium_pipeline.results import (
    BaseResult,
    DownloadResult,
    RawProcessingResult,
    failure,
)


def test_config_no_side_effects(tmp_path):
    # Constructing a config must not create directories.
    cfg = PipelineConfig(base_dir=tmp_path / "root")
    assert not (tmp_path / "root").exists()
    assert cfg.data_dir == tmp_path / "root" / "data"


def test_config_banksy_from_env(monkeypatch, tmp_path):
    py = tmp_path / "python"
    py.write_text("#!/bin/sh\n")
    monkeypatch.setenv("XENIUM_BANKSY_PYTHON", str(py))
    monkeypatch.setenv("XENIUM_BANKSY_DIR", str(tmp_path))
    monkeypatch.setenv("XENIUM_BANKSY_TIMEOUT", "120")
    cfg = PipelineConfig()
    assert cfg.banksy_python == py
    assert cfg.banksy_dir == tmp_path
    assert cfg.banksy_timeout_seconds == 120
    assert cfg.banksy_available is True


def test_config_banksy_unset(monkeypatch):
    monkeypatch.delenv("XENIUM_BANKSY_PYTHON", raising=False)
    monkeypatch.delenv("XENIUM_BANKSY_DIR", raising=False)
    cfg = PipelineConfig()
    assert cfg.banksy_available is False


def test_result_hierarchy_and_bool():
    ok = RawProcessingResult(sample_path=Path("/x"), success=True)
    assert isinstance(ok, BaseResult)
    assert bool(ok) is True
    bad = failure(RawProcessingResult, Path("/x"), "boom")
    assert bad.success is False and bad.error_message == "boom"
    assert bool(bad) is False


def test_download_result_fields():
    r = DownloadResult(sample_path=Path("/s"), success=True, gsm_id="GSM1", raw_path=Path("/s/raw"))
    assert isinstance(r, BaseResult)
    assert r.gsm_id == "GSM1"
    assert r.files_processed == []
