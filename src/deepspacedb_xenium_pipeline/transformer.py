"""Thin orchestrator wiring the focused pipeline components together.

``XeniumTransformer`` owns no domain logic of its own; it sequences the
single-responsibility components (extractor, organizer, builder, analysis,
imaging) and translates their exceptions into typed result objects at
well-defined stage boundaries. This is the only public entrypoint most callers
need.
"""
from __future__ import annotations

import gc
import glob
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import spatialdata as sd

from . import constants
from .analysis import Binner, SingleCellExporter, TranscriptomicsProcessor
from .compat import apply_spatialdata_patches
from .config import PipelineConfig
from .extraction import ArchiveExtractor, FileOrganizer, FileRepair
from .imaging import HEProcessor, ImageRenderer, MorphologyProcessor
from .loaders import SpatialDataBuilder, TableLoader
from .logging_setup import get_logger
from .results import (
    BaseResult,
    CompleteProcessingResult,
    RawProcessingResult,
    TransformationResult,
    failure,
)
from .summary import SampleAnalyzer, SampleSummary, write_summary_csv

_SKIP_DIR_PATTERNS = (".zarr", ".git", ".DS_Store", "__pycache__", ".pytest_cache")


class XeniumTransformer:
    """Orchestrate the full Xenium raw -> processed -> analysis pipeline."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.logger = get_logger(__name__)

        # Apply the spatialdata_io polygon patch explicitly (no import-time magic).
        apply_spatialdata_patches()

        # Compose the components. Dependencies flow one way; everything is
        # injectable for testing.
        self.repair = FileRepair()
        self.he = HEProcessor(self.config)
        self.morphology = MorphologyProcessor()
        self.tables = TableLoader()
        self.extractor = ArchiveExtractor()
        self.organizer = FileOrganizer(self.config, he_processor=self.he, repair=self.repair)
        self.builder = SpatialDataBuilder(
            self.config, table_loader=self.tables, morphology=self.morphology
        )
        self.transcriptomics = TranscriptomicsProcessor(self.config)
        self.renderer = ImageRenderer(self.config)
        self.binner = Binner(self.config)
        self.exporter = SingleCellExporter(self.config)
        self.analyzer = SampleAnalyzer(self.config)

    # ================================================================== #
    # Stage 1: raw extraction
    # ================================================================== #

    def process_raw_data(self, sample_dir: Path) -> RawProcessingResult:
        self.logger.info("Processing raw data for: %s", sample_dir)
        raw_dir = sample_dir / "raw"
        if not raw_dir.exists():
            return failure(RawProcessingResult, sample_dir, f"Raw directory not found: {raw_dir}")

        processed_dir = sample_dir / "processed"
        if processed_dir.exists() and not self.config.overwrite_existing:
            existing = list(processed_dir.iterdir())
            if existing:
                self.logger.info("processed/ exists with %d files; skipping raw", len(existing))
                return RawProcessingResult(
                    sample_path=sample_dir, success=True, processed_path=processed_dir,
                    files_processed=[f.name for f in existing],
                )

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            try:
                if self.extractor.extract_all(raw_dir, temp_dir) == 0:
                    raise RuntimeError("No files could be extracted from raw directory")
                if processed_dir.exists() and self.config.overwrite_existing:
                    shutil.rmtree(processed_dir)
                if self.organizer.organize(temp_dir, processed_dir) == 0:
                    raise RuntimeError("Failed to organize files into processed directory")
                files = [f.name for f in processed_dir.rglob("*") if f.is_file()]
                self.logger.info("Raw processing succeeded for %s", sample_dir)
                return RawProcessingResult(
                    sample_path=sample_dir, success=True, processed_path=processed_dir,
                    files_processed=files,
                )
            except Exception as exc:  # noqa: BLE001 - boundary: convert to result
                msg = f"Error processing raw data for {sample_dir}: {exc}"
                self.logger.error(msg)
                return failure(RawProcessingResult, sample_dir, msg)

    def process_zarr_directly(self, sample_dir: Path) -> RawProcessingResult:
        self.logger.info("Processing zarr directly for: %s", sample_dir)
        raw_dir = sample_dir / "raw"
        if not raw_dir.exists():
            return failure(RawProcessingResult, sample_dir, f"Raw directory not found: {raw_dir}")
        zarr_files = list(raw_dir.glob("*.zarr.tar.gz"))
        if not zarr_files:
            return failure(RawProcessingResult, sample_dir, f"No zarr.tar.gz file in {raw_dir}")
        if len(zarr_files) > 1:
            self.logger.warning("Multiple zarr.tar.gz; using %s", zarr_files[0].name)

        zarr_file = zarr_files[0]
        output_path = sample_dir / self.config.output_suffix
        if output_path.exists() and not self.config.overwrite_existing:
            self.logger.info("Output exists; skipping zarr extraction")
            return RawProcessingResult(
                sample_path=sample_dir, success=True, processed_path=output_path,
                files_processed=[output_path.name],
            )
        try:
            if output_path.exists():
                shutil.rmtree(output_path)
            self.logger.info("Extracting %s as %s", zarr_file.name, self.config.output_suffix)
            with tarfile.open(zarr_file, "r:gz") as tar:
                tar.extractall(sample_dir)
            extracted = [d for d in sample_dir.iterdir() if d.is_dir() and d.name.endswith(".zarr")]
            if extracted and extracted[0].name != self.config.output_suffix:
                extracted[0].rename(output_path)
            if not output_path.exists():
                raise RuntimeError("Zarr extraction succeeded but output directory not found")
            self.repair.clean_zarr_underscore_files(output_path)
            self.logger.info("Extracted zarr to %s", output_path)
            return RawProcessingResult(
                sample_path=sample_dir, success=True, processed_path=output_path,
                files_processed=[self.config.output_suffix],
            )
        except Exception as exc:  # noqa: BLE001
            msg = f"Error extracting zarr for {sample_dir}: {exc}"
            self.logger.error(msg)
            return failure(RawProcessingResult, sample_dir, msg)

    # ================================================================== #
    # Stage 2: SpatialData transform
    # ================================================================== #

    def transform_sample(self, sample_dir: Path) -> TransformationResult:
        self.logger.info("Transforming: %s", sample_dir)
        processed_dir = sample_dir / "processed"
        if not processed_dir.exists():
            return failure(
                TransformationResult, sample_dir, f"Processed directory not found: {processed_dir}"
            )
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
                sample_path=sample_dir, success=True, output_path=output_path,
                files_processed=files,
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            self.logger.error("Failed to transform %s: %s", sample_dir, msg)
            (sample_dir / "error_001.txt").write_text(msg)
            return failure(TransformationResult, sample_dir, msg)

    # ================================================================== #
    # Complete pipeline
    # ================================================================== #

    def process_sample_complete(self, sample_dir: Path) -> CompleteProcessingResult:
        self.logger.info("Starting complete processing for: %s", sample_dir)
        raw_dir = sample_dir / "raw"
        if not raw_dir.exists():
            return failure(CompleteProcessingResult, sample_dir, f"Raw directory not found: {raw_dir}")

        processed_dir = sample_dir / "processed"
        spatialdata_path = sample_dir / self.config.output_suffix
        final_path = sample_dir / self.config.processed_suffix

        if not self.config.overwrite_existing and self._fully_processed(sample_dir, final_path):
            self.logger.info("Sample fully processed; skipping: %s", sample_dir)
            return CompleteProcessingResult(
                sample_path=sample_dir, success=True,
                raw_processed_path=processed_dir if processed_dir.exists() else None,
                spatialdata_path=spatialdata_path if spatialdata_path.exists() else None,
                final_processed_path=final_path,
            )

        try:
            if self.config.process_zarr_directly and list(raw_dir.glob("*.zarr.tar.gz")):
                return self._complete_via_zarr(sample_dir, spatialdata_path, final_path)
            return self._complete_via_raw(sample_dir, processed_dir, spatialdata_path, final_path)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            self.logger.error("Failed to process %s: %s", sample_dir, msg)
            (sample_dir / "error_complete_processing.txt").write_text(msg)
            return failure(CompleteProcessingResult, sample_dir, msg)

    def _complete_via_raw(
        self, sample_dir: Path, processed_dir: Path, spatialdata_path: Path, final_path: Path
    ) -> CompleteProcessingResult:
        if not processed_dir.exists() or self.config.overwrite_existing:
            self.logger.info("Step 1: raw processing")
            raw_result = self.process_raw_data(sample_dir)
            if not raw_result.success:
                return failure(CompleteProcessingResult, sample_dir, raw_result.error_message)

        if not spatialdata_path.exists() or self.config.overwrite_existing:
            self.logger.info("Step 2: SpatialData transform")
            transform_result = self.transform_sample(sample_dir)
            if not transform_result.success:
                return failure(CompleteProcessingResult, sample_dir, transform_result.error_message)

        if not final_path.exists() or self.config.overwrite_existing:
            self.logger.info("Step 3: transcriptomics")
            sdata = sd.read_zarr(str(spatialdata_path))
            sdata = self.transcriptomics.process(sdata, sample_dir)
            if final_path.exists():
                shutil.rmtree(final_path)
            sdata.write(str(final_path))
            del sdata
            gc.collect()

        return self._post_analysis(sample_dir, processed_dir, spatialdata_path, final_path)

    def _complete_via_zarr(
        self, sample_dir: Path, spatialdata_path: Path, final_path: Path
    ) -> CompleteProcessingResult:
        self.logger.info("Direct zarr processing (bypassing raw)")
        if not spatialdata_path.exists() or self.config.overwrite_existing:
            zarr_result = self.process_zarr_directly(sample_dir)
            if not zarr_result.success:
                return failure(CompleteProcessingResult, sample_dir, zarr_result.error_message)

        if not final_path.exists() or self.config.overwrite_existing:
            self.logger.info("Step 2: transcriptomics (skipping transform)")
            sdata = sd.read_zarr(str(spatialdata_path))
            sdata = self.transcriptomics.process(sdata, sample_dir)
            if final_path.exists():
                shutil.rmtree(final_path)
            sdata.write(str(final_path))
            del sdata
            gc.collect()

        return self._post_analysis(sample_dir, None, spatialdata_path, final_path)

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

        if self.config.skip_zarr_exports:
            self.logger.info("Skipping zarr exports (skip_zarr_exports=True)")
            bins, single_cell = [], []
        else:
            self.logger.info("Creating binned data")
            bins = self.binner.bin_to_zarr(sdata, sample_dir)
            self.logger.info("Exporting single-cell data")
            single_cell = self.exporter.export(sdata, sample_dir)

        del sdata
        gc.collect()
        return CompleteProcessingResult(
            sample_path=sample_dir, success=True,
            raw_processed_path=processed_dir,
            spatialdata_path=spatialdata_path,
            final_processed_path=final_path,
            images_created=images, bins_created=bins, single_cell_files=single_cell,
        )

    def _fully_processed(self, sample_dir: Path, final_path: Path) -> bool:
        if not final_path.exists():
            return False
        images_dir = sample_dir / "images"
        zarr_dir = sample_dir / "zarr"
        has_images = images_dir.exists() and bool(
            list(images_dir.rglob("*.png")) + list(images_dir.rglob("*.pdf"))
        )
        has_bins = zarr_dir.exists() and bool(list(zarr_dir.glob("bins_size_*.zarr.zip")))
        has_single = zarr_dir.exists() and bool(
            list(zarr_dir.glob("sparse_gene_expression_*.zarr.zip"))
            + list(zarr_dir.glob("cell_coordinates.zarr.zip"))
        )
        return has_images and has_bins and has_single

    # ================================================================== #
    # Images + haystack only (regeneration helper)
    # ================================================================== #

    def process_images_and_haystack_only(self, sample_dir: Path) -> CompleteProcessingResult:
        self.logger.info("Images + haystack only for: %s", sample_dir)
        final_path = sample_dir / self.config.processed_suffix
        if not final_path.exists():
            return failure(
                CompleteProcessingResult, sample_dir,
                f"Final processed data not found: {final_path}. Run complete processing first.",
            )
        try:
            sdata = sd.read_zarr(str(final_path))
            self.transcriptomics.run_haystack_only(sdata, sample_dir)
            images = self.renderer.generate_images(sdata, sample_dir)
            self.renderer.calculate_correction_factor(sdata, sample_dir)
            self.renderer.generate_cell_shapes(sample_dir)
            del sdata
            gc.collect()
            return CompleteProcessingResult(
                sample_path=sample_dir, success=True, final_processed_path=final_path,
                images_created=images,
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            self.logger.error("Images+haystack failed for %s: %s", sample_dir, msg)
            (sample_dir / "error_images_haystack.txt").write_text(msg)
            return failure(CompleteProcessingResult, sample_dir, msg)

    # ================================================================== #
    # Multi-sample drivers
    # ================================================================== #

    def discover_downloaded_samples(self, base_dir: Path) -> List[Path]:
        sample_dirs = []
        for raw_dir in glob.glob(str(base_dir / "**/raw/"), recursive=True):
            sample_dir = Path(raw_dir).parent
            if list(Path(raw_dir).iterdir()):
                sample_dirs.append(sample_dir)
        self.logger.info("Found %d samples with raw/ directories", len(sample_dirs))
        return sample_dirs

    def _valid_sample_dirs(self, sample_dirs: List[Path]) -> List[Path]:
        return [
            d
            for d in sample_dirs
            if d.is_dir() and not any(p in d.name for p in _SKIP_DIR_PATTERNS)
        ]

    def process_multiple_complete(
        self, sample_dirs: List[Path]
    ) -> Dict[str, CompleteProcessingResult]:
        return self._run_many(sample_dirs, self.process_sample_complete, skip_invalid=False)

    def process_multiple_images_haystack(
        self, sample_dirs: List[Path]
    ) -> Dict[str, CompleteProcessingResult]:
        return self._run_many(
            sample_dirs, self.process_images_and_haystack_only, skip_invalid=True
        )

    def process_multiple_inject_he(
        self, sample_dirs: List[Path]
    ) -> Dict[str, CompleteProcessingResult]:
        return self._run_many(sample_dirs, self.process_sample_complete, skip_invalid=True)

    def _run_many(self, sample_dirs, fn, *, skip_invalid: bool):
        dirs = self._valid_sample_dirs(sample_dirs) if skip_invalid else sample_dirs
        results: Dict[str, BaseResult] = {}
        total = len(dirs)
        success = 0
        self.logger.info("Processing %d samples", total)
        for idx, sample_dir in enumerate(dirs, 1):
            self.logger.info("Sample %d/%d: %s", idx, total, sample_dir)
            result = fn(sample_dir)
            results[sample_dir.name] = result
            if result.success:
                success += 1
                self.logger.info("Succeeded: %s", sample_dir.name)
            else:
                self.logger.error("Failed: %s: %s", sample_dir.name, result.error_message)
        self.logger.info("Finished: %d/%d successful", success, total)
        return results

    def process_multiple_complete_with_summary(
        self, sample_dirs: List[Path], summary_file: Optional[Path] = None
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
            result = self.process_sample_complete(sample_dir)
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
        self.logger.info("Finished: %d/%d successful; summary -> %s", success, total, summary_file)
        return results, summaries
