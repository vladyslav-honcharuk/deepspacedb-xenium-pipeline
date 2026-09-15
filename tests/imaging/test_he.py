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


def test_find_alignment_csv_skips_keypoints_file(tmp_path):
    """A control-points file matching an earlier pattern must not shadow the real matrix."""
    raw_dir = tmp_path / "raw"
    sub = raw_dir / "alignment_files"
    sub.mkdir(parents=True)
    keypoints = raw_dir / "sample_he_imagealignment.csv"
    keypoints.write_text("fixedX,fixedY,alignmentX,alignmentY\n1,2,3,4\n5,6,7,8\n")
    matrix_csv = sub / "matrix.csv"
    matrix_csv.write_text("1,0,0\n0,1,0\n0,0,1\n")

    assert HEProcessor.find_alignment_csv(raw_dir) == matrix_csv


def test_find_alignment_csv_keypoints_only_returns_none(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "sample_he_imagealignment.csv").write_text("fixedX,fixedY,alignmentX,alignmentY\n1,2,3,4\n")
    assert HEProcessor.find_alignment_csv(raw_dir) is None


def test_find_alignment_csv_in_sample_root(tmp_path):
    """matrix_fitted_from_control_points.csv sits beside raw/, not inside it."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    matrix_csv = tmp_path / "matrix_fitted_from_control_points.csv"
    matrix_csv.write_text("1,0,0\n0,1,0\n0,0,1\n")
    assert HEProcessor.find_alignment_csv(raw_dir) == matrix_csv


def test_is_plain_matrix_csv(tmp_path):
    matrix_csv = tmp_path / "matrix.csv"
    matrix_csv.write_text("1,0,0\n0,1,0\n0,0,1\n")
    keypoints = tmp_path / "keypoints.csv"
    keypoints.write_text("fixedX,fixedY\n1,2\n")

    assert HEProcessor.is_plain_matrix_csv(matrix_csv) is True
    assert HEProcessor.is_plain_matrix_csv(keypoints) is False


def test_find_unaligned_he_source_prefers_processed_copy(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    he = processed / "he_image.ome.tif"
    he.write_bytes(b"")
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "sample_HE.ome.tif").write_bytes(b"")

    assert HEProcessor.find_unaligned_he_source(tmp_path) == he


def test_find_unaligned_he_source_from_raw(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    he = raw_dir / "sample_HE.ome.tif"
    he.write_bytes(b"")
    # Already registered by another method, plus non-image files: all excluded.
    (raw_dir / "sample_HE_registered.ome.tif").write_bytes(b"")
    (raw_dir / "morphology_mip.ome.tif").write_bytes(b"")
    (raw_dir / "matrix.csv").write_text("1,0,0\n")

    assert HEProcessor.find_unaligned_he_source(tmp_path) == he


def test_find_unaligned_he_source_ambiguous_returns_none(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "sample_a_HE.ome.tif").write_bytes(b"")
    (raw_dir / "sample_b_HE.ome.tif").write_bytes(b"")
    assert HEProcessor.find_unaligned_he_source(tmp_path) is None
