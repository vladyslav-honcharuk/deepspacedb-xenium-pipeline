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


def test_config_skip_binning_default():
    cfg = PipelineConfig()
    # Binning is opt-in: skipped by default.
    assert cfg.skip_binning is True
    assert cfg.zarr_export_workers is None


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
