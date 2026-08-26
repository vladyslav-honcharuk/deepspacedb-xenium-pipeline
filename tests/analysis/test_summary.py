import csv
from pathlib import Path

from deepspacedb_xenium_pipeline import PipelineConfig
from deepspacedb_xenium_pipeline.summary import (
    SampleAnalyzer,
    SampleSummary,
    write_summary_csv,
)


def test_write_summary_csv_writes_every_row(tmp_path):
    summaries = [SampleSummary(sample_name=f"S{i}", sample_path=f"/p/{i}") for i in range(5)]
    out = tmp_path / "summary.csv"
    write_summary_csv(summaries, out)
    with open(out) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 5
    assert [r["sample_name"] for r in rows] == ["S0", "S1", "S2", "S3", "S4"]
    # Booleans render as Yes/No.
    assert rows[0]["has_raw_dir"] == "No"


def test_analyzer_detects_raw_files(tmp_path):
    sample = tmp_path / "GSM1"
    raw = sample / "raw"
    raw.mkdir(parents=True)
    (raw / "cell_feature_matrix.h5").write_bytes(b"\x00")
    (raw / "transcripts.parquet").write_bytes(b"\x00")
    (raw / "morphology_focus.ome.tif").write_bytes(b"\x00")

    analyzer = SampleAnalyzer(PipelineConfig(base_dir=tmp_path))
    summary = analyzer.analyze(sample)
    assert summary.has_raw_dir
    assert summary.has_cell_feature_matrix
    assert summary.has_transcripts
    assert summary.has_morphology_focus
    assert summary.files_in_raw == 3
    assert not summary.has_spatialdata_zarr


def test_update_with_result_sets_error():
    from deepspacedb_xenium_pipeline.results import CompleteProcessingResult, failure

    analyzer = SampleAnalyzer(PipelineConfig())
    summary = SampleSummary(sample_name="S", sample_path="/p")
    analyzer.update_with_result(summary, failure(CompleteProcessingResult, Path("/p"), "boom"), "complete")
    assert summary.complete_processing_success is False
    assert summary.complete_processing_error == "boom"
