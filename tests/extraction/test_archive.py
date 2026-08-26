"""Unit tests for ArchiveExtractor in deepspacedb_xenium_pipeline.extraction.archive."""

from __future__ import annotations

import io
import tarfile
import zipfile

from deepspacedb_xenium_pipeline.extraction.archive import ArchiveExtractor


def test_extract_zip_flattens_and_skips_junk(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    archive = raw / "sample.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("outs/cells.parquet", b"data")
        zf.writestr("outs/._cells.parquet", b"junk")  # AppleDouble fork
        zf.writestr("outs/.DS_Store", b"junk")
    temp = tmp_path / "temp"
    temp.mkdir()
    n = ArchiveExtractor().extract_all(raw, temp)
    names = {p.name for p in temp.rglob("*") if p.is_file()}
    assert "cells.parquet" in names  # flattened, no outs/ prefix
    assert "._cells.parquet" not in names  # junk skipped
    assert ".DS_Store" not in names
    assert n == 1


def test_extract_tar_gz(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    archive = raw / "sample.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        for name in ("experiment.xenium", "transcripts.parquet"):
            data = b"x"
            info = tarfile.TarInfo(f"outs/{name}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    temp = tmp_path / "temp"
    temp.mkdir()
    n = ArchiveExtractor().extract_all(raw, temp)
    names = {p.name for p in temp.rglob("*") if p.is_file()}
    assert {"experiment.xenium", "transcripts.parquet"} <= names
    assert n == 2


def test_outs_zip_priority(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    with zipfile.ZipFile(raw / "a_outs.zip", "w") as zf:
        zf.writestr("good.txt", b"1")
    with zipfile.ZipFile(raw / "a_xe_outs.zip", "w") as zf:
        zf.writestr("bad.txt", b"1")
    temp = tmp_path / "temp"
    temp.mkdir()
    ArchiveExtractor().extract_all(raw, temp)
    names = {p.name for p in temp.rglob("*") if p.is_file()}
    assert "good.txt" in names
    assert "bad.txt" not in names  # xe_outs skipped in favor of _outs


def test_zarr_zip_is_not_treated_as_archive(tmp_path):
    p = tmp_path / "cells.zarr.zip"
    p.write_bytes(b"x")
    assert ArchiveExtractor.is_archive(p) is False
    assert ArchiveExtractor.is_archive(tmp_path / "x.tar.gz") is True
