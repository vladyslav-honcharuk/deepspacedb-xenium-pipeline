"""Unit tests for HEProcessor in deepspacedb_xenium_pipeline.imaging.he."""

from __future__ import annotations

from deepspacedb_xenium_pipeline.imaging.he import HEProcessor


def test_find_alignment_csv_not_found(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    assert HEProcessor.find_alignment_csv(raw_dir) is None


def test_find_alignment_csv_found_in_raw(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    align_csv = raw_dir / "sample_he_imagealignment.csv"
    align_csv.write_text("1,0,0\n0,1,0\n0,0,1\n")
    found = HEProcessor.find_alignment_csv(raw_dir)
    assert found == align_csv


def test_find_alignment_csv_found_in_subfolder(tmp_path):
    raw_dir = tmp_path / "raw"
    sub = raw_dir / "alignment_files"
    sub.mkdir(parents=True)
    matrix_csv = sub / "matrix.csv"
    matrix_csv.write_text("1,0,0\n0,1,0\n0,0,1\n")
    found = HEProcessor.find_alignment_csv(raw_dir)
    assert found == matrix_csv
