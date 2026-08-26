"""Per-sample file inventory and CSV reporting."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

from .config import PipelineConfig
from .logging_setup import get_logger
from .results import BaseResult


@dataclass
class SampleSummary:
    """Availability flags and processing status for one sample."""

    sample_name: str
    sample_path: str

    has_raw_dir: bool = False
    has_experiment_file: bool = False
    has_cell_feature_matrix: bool = False
    has_transcripts: bool = False
    has_cells_parquet: bool = False
    has_cell_boundaries: bool = False
    has_nucleus_boundaries: bool = False
    has_morphology: bool = False
    has_morphology_mip: bool = False
    has_morphology_focus: bool = False
    has_he_image: bool = False
    has_gene_panel: bool = False
    has_matrix_mtx: bool = False
    has_barcodes: bool = False
    has_features: bool = False
    has_cells_zarr: bool = False
    has_transcripts_zarr: bool = False

    raw_processing_success: bool = False
    raw_processing_error: str = ""
    spatialdata_success: bool = False
    spatialdata_error: str = ""
    transcriptomics_success: bool = False
    transcriptomics_error: str = ""
    complete_processing_success: bool = False
    complete_processing_error: str = ""

    has_processed_dir: bool = False
    has_spatialdata_zarr: bool = False
    has_final_processed_zarr: bool = False
    has_images: bool = False
    has_bins: bool = False
    has_single_cell_exports: bool = False

    processing_timestamp: str = ""
    files_in_raw: int = 0
    files_in_processed: int = 0
    processing_duration: str = ""


_CSV_FIELDS = [
    "sample_name",
    "sample_path",
    "processing_timestamp",
    "files_in_raw",
    "files_in_processed",
    "has_raw_dir",
    "has_experiment_file",
    "has_cell_feature_matrix",
    "has_transcripts",
    "has_cells_parquet",
    "has_cell_boundaries",
    "has_nucleus_boundaries",
    "has_morphology",
    "has_morphology_mip",
    "has_morphology_focus",
    "has_he_image",
    "has_gene_panel",
    "has_matrix_mtx",
    "has_barcodes",
    "has_features",
    "has_cells_zarr",
    "has_transcripts_zarr",
    "raw_processing_success",
    "raw_processing_error",
    "spatialdata_success",
    "spatialdata_error",
    "transcriptomics_success",
    "transcriptomics_error",
    "complete_processing_success",
    "complete_processing_error",
    "has_processed_dir",
    "has_spatialdata_zarr",
    "has_final_processed_zarr",
    "has_images",
    "has_bins",
    "has_single_cell_exports",
]

_OUTPUT_FLAG_FIELDS = [
    "has_processed_dir",
    "has_spatialdata_zarr",
    "has_final_processed_zarr",
    "has_images",
    "has_bins",
    "has_single_cell_exports",
    "files_in_processed",
]


class SampleAnalyzer:
    """Inspect a sample directory and produce/refresh :class:`SampleSummary`."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.logger = get_logger(__name__)

    def analyze(self, sample_dir: Path) -> SampleSummary:
        summary = SampleSummary(
            sample_name=sample_dir.name,
            sample_path=str(sample_dir),
            processing_timestamp=datetime.now().isoformat(),
        )
        raw_dir = sample_dir / "raw"
        summary.has_raw_dir = raw_dir.exists()
        if summary.has_raw_dir:
            self._scan_raw(raw_dir, summary)

        processed_dir = sample_dir / "processed"
        summary.has_processed_dir = processed_dir.exists()
        if summary.has_processed_dir:
            summary.files_in_processed = sum(1 for f in processed_dir.rglob("*") if f.is_file())

        self._scan_outputs(sample_dir, summary)
        return summary

    def refresh_outputs(self, sample_dir: Path, summary: SampleSummary) -> None:
        """Update only the output flags after processing has run."""
        post = self.analyze(sample_dir)
        for field_name in _OUTPUT_FLAG_FIELDS:
            setattr(summary, field_name, getattr(post, field_name))

    def update_with_result(self, summary: SampleSummary, result: BaseResult, stage: str) -> None:
        mapping = {
            "raw": ("raw_processing_success", "raw_processing_error"),
            "spatialdata": ("spatialdata_success", "spatialdata_error"),
            "complete": ("complete_processing_success", "complete_processing_error"),
        }
        if stage not in mapping:
            return
        ok_field, err_field = mapping[stage]
        setattr(summary, ok_field, result.success)
        if not result.success:
            setattr(summary, err_field, result.error_message or "Unknown error")

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _scan_raw(self, raw_dir: Path, summary: SampleSummary) -> None:
        files = [f for f in raw_dir.rglob("*") if f.is_file()]
        summary.files_in_raw = len(files)
        names = [f.name.lower() for f in files]

        def any_match(pred) -> bool:
            return any(pred(n) for n in names)

        summary.has_experiment_file = any_match(lambda n: "experiment" in n and (".xenium" in n or ".json" in n))
        summary.has_he_image = any_match(lambda n: ("he" in n and ".tif" in n) or ("h&e" in n and ".tif" in n))
        summary.has_cell_feature_matrix = any_match(
            lambda n: "cell_feature_matrix" in n and (".h5" in n or ".zarr" in n)
        )
        summary.has_transcripts = any_match(lambda n: "transcript" in n and (".parquet" in n or ".csv" in n))
        summary.has_cells_parquet = any_match(
            lambda n: "cells" in n and (".parquet" in n or ".csv" in n) and "boundaries" not in n
        )
        summary.has_cell_boundaries = any_match(lambda n: "cell_boundaries" in n or "cell_boundary" in n)
        summary.has_nucleus_boundaries = any_match(lambda n: "nucleus_boundaries" in n or "nucleus_boundary" in n)
        summary.has_morphology = any_match(
            lambda n: "morphology" in n and ".tif" in n and "mip" not in n and "focus" not in n
        )
        summary.has_morphology_mip = any_match(lambda n: "morphology_mip" in n or ("morphology" in n and "mip" in n))
        summary.has_morphology_focus = any_match(
            lambda n: "morphology_focus" in n or ("morphology" in n and "focus" in n)
        )
        summary.has_gene_panel = any_match(lambda n: "gene_panel" in n or "panel" in n)
        summary.has_matrix_mtx = any_match(lambda n: "matrix.mtx" in n)
        summary.has_barcodes = any_match(lambda n: "barcodes" in n)
        summary.has_features = any_match(lambda n: "features" in n)
        summary.has_cells_zarr = any_match(lambda n: "cells.zarr" in n)
        summary.has_transcripts_zarr = any_match(lambda n: "transcripts.zarr" in n)

    def _scan_outputs(self, sample_dir: Path, summary: SampleSummary) -> None:
        summary.has_spatialdata_zarr = (sample_dir / self.config.output_suffix).exists()
        summary.has_final_processed_zarr = (sample_dir / self.config.processed_suffix).exists()

        images_dir = sample_dir / "images"
        if images_dir.exists():
            image_files = list(images_dir.rglob("*.png")) + list(images_dir.rglob("*.pdf"))
        else:
            image_files = []
        summary.has_images = len(image_files) > 0

        zarr_dir = sample_dir / "zarr"
        if zarr_dir.exists():
            bins = list(zarr_dir.glob("bins_size_*.zarr.zip"))
            sc_files = list(zarr_dir.glob("sparse_gene_expression_*.zarr.zip")) + list(
                zarr_dir.glob("cell_coordinates.zarr.zip")
            )
        else:
            bins, sc_files = [], []
        summary.has_bins = len(bins) > 0
        summary.has_single_cell_exports = len(sc_files) > 0


def write_summary_csv(summaries: List[SampleSummary], output_file: Path) -> None:
    """Write a list of summaries to ``output_file`` (one row per sample)."""
    logger = get_logger(__name__)
    logger.info("Writing summary report to %s", output_file)
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for summary in summaries:
            row = {}
            for field_name in _CSV_FIELDS:
                value = getattr(summary, field_name, "")
                if isinstance(value, bool):
                    row[field_name] = "Yes" if value else "No"
                elif isinstance(value, str):
                    # Collapse embedded newlines so cells never wrap across rows.
                    row[field_name] = " ".join(value.split())
                else:
                    row[field_name] = value
            writer.writerow(row)


def write_successful_samples_csv(summaries: List[SampleSummary], output_file: Path) -> None:
    """Write only successfully processed samples to a CSV file."""
    logger = get_logger(__name__)
    successful = [s for s in summaries if s.complete_processing_success]
    logger.info("Writing %d successfully processed samples to %s", len(successful), output_file)
    fields = [
        "sample_name",
        "sample_path",
        "has_final_processed_zarr",
        "has_images",
        "has_bins",
        "has_single_cell_exports",
        "processing_timestamp",
        "processing_duration",
    ]
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for s in successful:
            writer.writerow(
                {
                    "sample_name": s.sample_name,
                    "sample_path": s.sample_path,
                    "has_final_processed_zarr": "Yes" if s.has_final_processed_zarr else "No",
                    "has_images": "Yes" if s.has_images else "No",
                    "has_bins": "Yes" if s.has_bins else "No",
                    "has_single_cell_exports": "Yes" if s.has_single_cell_exports else "No",
                    "processing_timestamp": s.processing_timestamp,
                    "processing_duration": s.processing_duration,
                }
            )
