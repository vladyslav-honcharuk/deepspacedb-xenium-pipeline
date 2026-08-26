"""Salvage malformed or incomplete Xenium uploads into processed/ directories."""

from __future__ import annotations

import shutil
import tarfile
import tempfile
from pathlib import Path

from .config import PipelineConfig
from .extraction import ArchiveExtractor, FileOrganizer, FileRepair
from .imaging import HEProcessor
from .logging_setup import get_logger
from .results import RawProcessingResult, failure


class Salvager:
    """Extract, repair, and organize raw Xenium files."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.logger = get_logger(__name__)
        self.repair = FileRepair()
        self.he = HEProcessor(config)
        self.extractor = ArchiveExtractor()
        self.organizer = FileOrganizer(config, he_processor=self.he, repair=self.repair)

    def salvage_sample(self, sample_dir: Path) -> RawProcessingResult:
        """Create or refresh ``sample_dir/processed`` from ``sample_dir/raw``."""
        raw_dir = sample_dir / "raw"
        if not raw_dir.exists():
            return failure(RawProcessingResult, sample_dir, f"Raw directory not found: {raw_dir}")

        processed_dir = sample_dir / "processed"
        if processed_dir.exists() and not self.config.overwrite_existing:
            existing = list(processed_dir.iterdir())
            if existing:
                return RawProcessingResult(
                    sample_path=sample_dir,
                    success=True,
                    processed_path=processed_dir,
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
                (sample_dir / ".salvage_complete").write_text(f"Salvaged successfully with {len(files)} files\n")
                failed_marker = sample_dir / ".salvage_failed"
                if failed_marker.exists():
                    failed_marker.unlink()
                return RawProcessingResult(
                    sample_path=sample_dir,
                    success=True,
                    processed_path=processed_dir,
                    files_processed=files,
                )
            except Exception as exc:  # noqa: BLE001
                msg = f"Error salvaging raw data for {sample_dir}: {exc}"
                self.logger.error(msg)
                (sample_dir / ".salvage_failed").write_text(msg + "\n")
                complete_marker = sample_dir / ".salvage_complete"
                if complete_marker.exists():
                    complete_marker.unlink()
                return failure(RawProcessingResult, sample_dir, msg)

    def salvage_zarr(self, sample_dir: Path) -> RawProcessingResult:
        """Extract a directly supplied Xenium Zarr archive."""
        raw_dir = sample_dir / "raw"
        if not raw_dir.exists():
            return failure(RawProcessingResult, sample_dir, f"Raw directory not found: {raw_dir}")
        zarr_files = list(raw_dir.glob("*.zarr.tar.gz"))
        if not zarr_files:
            return failure(RawProcessingResult, sample_dir, f"No zarr.tar.gz file in {raw_dir}")

        zarr_file = zarr_files[0]
        output_path = sample_dir / self.config.output_suffix
        if output_path.exists() and not self.config.overwrite_existing:
            return RawProcessingResult(
                sample_path=sample_dir,
                success=True,
                processed_path=output_path,
                files_processed=[output_path.name],
            )
        try:
            if output_path.exists():
                shutil.rmtree(output_path)
            with tarfile.open(zarr_file, "r:gz") as tar:
                tar.extractall(sample_dir)
            extracted = [d for d in sample_dir.iterdir() if d.is_dir() and d.name.endswith(".zarr")]
            if extracted and extracted[0].name != self.config.output_suffix:
                extracted[0].rename(output_path)
            if not output_path.exists():
                raise RuntimeError("Zarr extraction succeeded but output directory was not found")
            self.repair.clean_zarr_underscore_files(output_path)
            return RawProcessingResult(
                sample_path=sample_dir,
                success=True,
                processed_path=output_path,
                files_processed=[self.config.output_suffix],
            )
        except Exception as exc:  # noqa: BLE001
            msg = f"Error salvaging Zarr for {sample_dir}: {exc}"
            self.logger.error(msg)
            return failure(RawProcessingResult, sample_dir, msg)
