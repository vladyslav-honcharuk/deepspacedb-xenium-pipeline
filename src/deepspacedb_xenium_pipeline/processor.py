"""Thin orchestrator wiring the focused pipeline components together.

``XeniumProcessor`` owns no domain logic of its own; it sequences the
single-responsibility components (builder, analysis, and imaging) and translates
their exceptions into typed result objects at
well-defined stage boundaries. This is the only public entrypoint most callers
need.
"""

from __future__ import annotations

import gc
import glob
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import spatialdata as sd

from . import constants
from .analysis import Binner, SingleCellExporter, TranscriptomicsProcessor
from .analysis.zarr_jobs import run_binning_and_export_parallel
from .compat import apply_spatialdata_patches
from .config import PipelineConfig
from .imaging import ImageRenderer, MorphologyProcessor
from .loaders import SpatialDataBuilder, TableLoader
from .logging_setup import get_logger
from .results import (
    CompleteProcessingResult,
    TransformationResult,
    failure,
)
from .salvager import Salvager
from .summary import SampleAnalyzer, SampleSummary, write_successful_samples_csv, write_summary_csv


def outputs_look_complete(
    sample_dir: Path,
    final_path: Path,
    *,
    skip_binning: bool,
) -> bool:
    """True when a sample has the outputs the current config would produce.

    When ``skip_binning`` is True, ``bins_size_*.zarr.zip`` is not required, or
    samples would never be recognized as done.
    """
    if not final_path.exists():
        return False
    images_dir = sample_dir / "images"
    zarr_dir = sample_dir / "zarr"
    has_images = images_dir.exists() and bool(list(images_dir.rglob("*.png")) + list(images_dir.rglob("*.pdf")))
    has_bins = zarr_dir.exists() and bool(list(zarr_dir.glob("bins_size_*.zarr.zip")))
    bins_ok = has_bins or skip_binning
    has_single = zarr_dir.exists() and bool(
        list(zarr_dir.glob("sparse_gene_expression_*.zarr.zip")) + list(zarr_dir.glob("cell_coordinates.zarr.zip"))
    )
    return has_images and bins_ok and has_single


class XeniumProcessor:
    """Orchestrate the full Xenium raw -> processed -> analysis pipeline."""

    def __init__(self, config: Optional[PipelineConfig] = None, *, salvager: Optional[Salvager] = None) -> None:
        self.config = config or PipelineConfig()
        self.logger = get_logger(__name__)
        self.salvager = salvager or Salvager(self.config)

        # Apply the spatialdata_io polygon patch explicitly (no import-time magic).
        apply_spatialdata_patches()

        # Compose the components. Dependencies flow one way; everything is
        # injectable for testing.
        self.he = self.salvager.he
        self.morphology = MorphologyProcessor()
        self.tables = TableLoader()
        self.builder = SpatialDataBuilder(self.config, table_loader=self.tables, morphology=self.morphology)
        self.transcriptomics = TranscriptomicsProcessor(self.config)
        self.renderer = ImageRenderer(self.config)
        self.binner = Binner(self.config)
        self.exporter = SingleCellExporter(self.config)
        self.analyzer = SampleAnalyzer(self.config)

    # ================================================================== #
    # Process salvaged data
    # ================================================================== #

    def process_sample(self, sample_dir: Path) -> TransformationResult:
        self.logger.info("Building processed sample: %s", sample_dir)
        processed_dir = sample_dir / "processed"
        if not processed_dir.exists():
            return failure(TransformationResult, sample_dir, f"Processed directory not found: {processed_dir}")
        self.builder.ensure_experiment_xenium(processed_dir)
        output_path = sample_dir / self.config.output_suffix
        if output_path.exists() and not self.config.overwrite_existing:
            self.logger.info("Skipping %s; already exists", output_path)
            return TransformationResult(sample_path=sample_dir, success=True, output_path=output_path)

        try:
            has_h5 = (processed_dir / constants.CELL_FEATURE_MATRIX_H5).exists()
            if has_h5:
                sdata = self.builder.build_standard_flexible(processed_dir)
            else:
                sdata = self.builder.build_custom(processed_dir)

            he_aligned = processed_dir / constants.HE_IMAGE_ALIGNED_FILENAME
            if he_aligned.exists() and "he_image" not in sdata.images:
                try:
                    sdata.images["he_image"] = self.he.load_as_spatialdata_image(he_aligned)
                    self.logger.info("Added aligned H&E image to SpatialData")
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning("Could not add aligned H&E image: %s", exc)

            if output_path.exists():
                shutil.rmtree(output_path)
            self.logger.info("Writing to %s", output_path)
            sdata.write(output_path)
            self.builder.update_experiment_files(processed_dir)
            files = [f.name for f in processed_dir.iterdir()]
            del sdata
            gc.collect()
            return TransformationResult(
                sample_path=sample_dir,
                success=True,
                output_path=output_path,
                files_processed=files,
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            self.logger.error("Failed to transform %s: %s", sample_dir, msg)
            return failure(TransformationResult, sample_dir, msg)

    def process_salvaged_sample(self, sample_dir: Path) -> CompleteProcessingResult:
        """Run all post-salvage stages using an existing ``processed/`` directory."""
        processed_dir = sample_dir / "processed"
        if not processed_dir.exists():
            msg = f"Processed directory not found: {processed_dir}. Run salvage first."
            (sample_dir / ".processing_failed").write_text(msg + "\n")
            complete_marker = sample_dir / ".processed_complete"
            if complete_marker.exists():
                complete_marker.unlink()
            return failure(
                CompleteProcessingResult,
                sample_dir,
                msg,
            )
        spatialdata_path = sample_dir / self.config.output_suffix
        final_path = sample_dir / self.config.processed_suffix
        try:
            return self._process_existing(sample_dir, processed_dir, spatialdata_path, final_path)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            (sample_dir / ".processing_failed").write_text(msg + "\n")
            complete_marker = sample_dir / ".processed_complete"
            if complete_marker.exists():
                complete_marker.unlink()
            return failure(CompleteProcessingResult, sample_dir, msg)

    def _process_existing(
        self, sample_dir: Path, processed_dir: Path, spatialdata_path: Path, final_path: Path
    ) -> CompleteProcessingResult:
        if not spatialdata_path.exists() or self.config.overwrite_existing:
            self.logger.info("Step 1: build SpatialData")
            transform_result = self.process_sample(sample_dir)
            if not transform_result.success:
                error_message = transform_result.error_message or "Unknown error"
                (sample_dir / ".processing_failed").write_text(error_message + "\n")
                complete_marker = sample_dir / ".processed_complete"
                if complete_marker.exists():
                    complete_marker.unlink()
                return failure(CompleteProcessingResult, sample_dir, error_message)

        if not final_path.exists() or self.config.overwrite_existing:
            self.logger.info("Step 2: transcriptomics")
            sdata = sd.read_zarr(str(spatialdata_path))
            sdata = self.transcriptomics.process(sdata, sample_dir)
            if final_path.exists():
                shutil.rmtree(final_path)
            sdata.write(str(final_path))
            del sdata
            gc.collect()

        res = self._post_analysis(sample_dir, processed_dir, spatialdata_path, final_path)
        if res.success:
            (sample_dir / ".processed_complete").write_text(f"Processed successfully at {datetime.now().isoformat()}\n")
            failed_marker = sample_dir / ".processing_failed"
            if failed_marker.exists():
                failed_marker.unlink()
        return res

    def _post_analysis(
        self,
        sample_dir: Path,
        processed_dir: Optional[Path],
        spatialdata_path: Path,
        final_path: Path,
    ) -> CompleteProcessingResult:
        self.logger.info("Generating images and exports")
        sdata = sd.read_zarr(str(final_path))
        images = self.renderer.generate_images(sdata, sample_dir)
        self.renderer.calculate_correction_factor(sdata, sample_dir)
        self.renderer.generate_cell_shapes(sample_dir)

        self.logger.info("Creating binned data + exporting single-cell data (parallel)")
        bins, single_cell = run_binning_and_export_parallel(sdata, sample_dir, self.config)

        del sdata
        gc.collect()
        return CompleteProcessingResult(
            sample_path=sample_dir,
            success=True,
            raw_processed_path=processed_dir,
            spatialdata_path=spatialdata_path,
            final_processed_path=final_path,
            images_created=images,
            bins_created=bins,
            single_cell_files=single_cell,
        )

    def _fully_processed(self, sample_dir: Path, final_path: Path) -> bool:
        return outputs_look_complete(
            sample_dir,
            final_path,
            skip_binning=self.config.skip_binning,
        )

    def discover_downloaded_samples(self, base_dir: Path) -> List[Path]:
        sample_dirs = []
        for raw_dir in glob.glob(str(base_dir / "**/raw/"), recursive=True):
            sample_dir = Path(raw_dir).parent
            if list(Path(raw_dir).iterdir()):
                sample_dirs.append(sample_dir)
        self.logger.info("Found %d samples with raw/ directories", len(sample_dirs))
        return sample_dirs

    def process_multiple_complete_with_summary(
        self,
        sample_dirs: List[Path],
        summary_file: Optional[Path] = None,
        successful_samples_file: Optional[Path] = None,
    ) -> Tuple[Dict[str, CompleteProcessingResult], List[SampleSummary]]:
        from datetime import datetime

        results: Dict[str, CompleteProcessingResult] = {}
        summaries: List[SampleSummary] = []
        total = len(sample_dirs)
        success = 0
        for idx, sample_dir in enumerate(sample_dirs, 1):
            self.logger.info("Sample %d/%d: %s", idx, total, sample_dir)
            start = datetime.now()
            summary = self.analyzer.analyze(sample_dir)
            result = self.process_salvaged_sample(sample_dir)
            results[sample_dir.name] = result
            self.analyzer.update_with_result(summary, result, "complete")
            summary.processing_duration = str(datetime.now() - start)
            self.analyzer.refresh_outputs(sample_dir, summary)
            summaries.append(summary)
            if result.success:
                success += 1
                self.logger.info("Succeeded: %s", sample_dir.name)
            else:
                self.logger.error("Failed: %s: %s", sample_dir.name, result.error_message)

        if summary_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = Path(f"xenium_processing_summary_{timestamp}.csv")
        write_summary_csv(summaries, summary_file)

        if successful_samples_file is not None:
            write_successful_samples_csv(summaries, successful_samples_file)

        self.logger.info("Finished: %d/%d successful; summary -> %s", success, total, summary_file)
        return results, summaries
