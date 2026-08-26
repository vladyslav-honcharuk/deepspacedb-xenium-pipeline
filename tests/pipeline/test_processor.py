"""Unit tests for XeniumProcessor in deepspacedb_xenium_pipeline.processor."""

from __future__ import annotations

from pathlib import Path

from deepspacedb_xenium_pipeline.config import PipelineConfig
from deepspacedb_xenium_pipeline.processor import XeniumProcessor, outputs_look_complete


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_outputs_look_complete_requires_all_outputs(tmp_path):
    final = tmp_path / "processed.zarr"
    final.mkdir()
    _touch(tmp_path / "images" / "overview.png")
    _touch(tmp_path / "zarr" / "sparse_gene_expression_csr.zarr.zip")

    assert not outputs_look_complete(tmp_path, final, skip_binning=False)

    _touch(tmp_path / "zarr" / "bins_size_10.zarr.zip")
    assert outputs_look_complete(tmp_path, final, skip_binning=False)


def test_outputs_look_complete_skip_binning(tmp_path):
    final = tmp_path / "processed.zarr"
    final.mkdir()
    _touch(tmp_path / "images" / "overview.png")
    _touch(tmp_path / "zarr" / "sparse_gene_expression_csr.zarr.zip")

    assert outputs_look_complete(tmp_path, final, skip_binning=True)
    assert not outputs_look_complete(tmp_path, final, skip_binning=False)

    _touch(tmp_path / "zarr" / "bins_size_10.zarr.zip")
    assert outputs_look_complete(tmp_path, final, skip_binning=False)


def test_discover_downloaded_samples(tmp_path):
    s1 = tmp_path / "GPL1" / "GSE1" / "GSM1" / "raw"
    s1.mkdir(parents=True)
    (s1 / "sample1.tar.gz").write_bytes(b"data")

    s2 = tmp_path / "GPL1" / "GSE1" / "GSM2" / "raw"
    s2.mkdir(parents=True)
    (s2 / "sample2.tar.gz").write_bytes(b"data")

    proc = XeniumProcessor(PipelineConfig(base_dir=tmp_path))
    discovered = proc.discover_downloaded_samples(tmp_path)
    sample_names = {d.name for d in discovered}
    assert sample_names == {"GSM1", "GSM2"}
