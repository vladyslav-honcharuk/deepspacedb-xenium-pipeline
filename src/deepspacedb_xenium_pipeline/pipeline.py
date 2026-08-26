"""The unified pipeline facade.

``XeniumPipeline`` is the single public entrypoint. It exposes the four stages
as methods on one object:

    1. :meth:`find`            - discover Xenium samples from NCBI GEO
    2. :meth:`download`        - download sample files
    3. :meth:`salvage_sample`  - salvage and normalize raw upload layouts
    4. :meth:`process_all`     - processed data -> SpatialData -> analysis for every sample

plus :meth:`run`, which chains all four. The heavy per-sample transform logic
lives in :class:`~deepspacedb_xenium_pipeline.processor.XeniumProcessor`,
which this facade composes and delegates to; the most-used transform methods are
surfaced here so callers never have to juggle separate stage objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import PipelineConfig
from .downloader import SampleDownloader
from .finder import SampleFinder
from .logging_setup import get_logger
from .processor import XeniumProcessor
from .results import CompleteProcessingResult, DownloadResult, RawProcessingResult
from .salvager import Salvager
from .summary import SampleSummary


class XeniumPipeline:
    """One object for discovery, download, salvage, and processing."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.logger = get_logger(__name__)
        self.finder = SampleFinder(self.config)
        self.downloader = SampleDownloader(self.config)
        self.salvager = Salvager(self.config)
        self.processor = XeniumProcessor(self.config)

    # ================================================================== #
    # Stage 1: find
    # ================================================================== #

    def find(self, output_csv: Optional[Path] = None):
        """Discover Xenium samples; optionally write them to ``output_csv``.

        Returns the found DataFrame. When ``output_csv`` is given and the
        result is non-empty, it is written there.
        """
        df = self.finder.find()
        if output_csv is not None and not df.empty:
            self.finder.save(df, str(output_csv))
            self.logger.info("Wrote %d samples to %s", len(df), output_csv)
        return df

    # ================================================================== #
    # Stage 2: download
    # ================================================================== #

    def download(
        self,
        samples_csv: Optional[Path] = None,
        *,
        gsm_id: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> Dict[str, DownloadResult]:
        """Download a single ``gsm_id`` or a batch from ``samples_csv``."""
        if gsm_id:
            if samples_csv:
                self.downloader.load_sample_metadata(samples_csv)
            return {gsm_id: self.downloader.download_gsm_sample(gsm_id)}
        if not samples_csv:
            raise ValueError("download() needs either gsm_id or samples_csv")
        samples = self.downloader.load_samples_from_csv(samples_csv)
        if max_samples:
            samples = samples[:max_samples]
            self.logger.info("Limiting to first %d samples", len(samples))
        return self.downloader.download_multiple(samples)

    # ================================================================== #
    # Stage 3: salvage
    # ================================================================== #

    def discover_samples(self, base_dir: Optional[Path] = None) -> List[Path]:
        return self.processor.discover_downloaded_samples(base_dir or self.config.data_dir)

    def discover_processed_samples(self, base_dir: Optional[Path] = None) -> List[Path]:
        root = base_dir or self.config.data_dir
        return [path.parent for path in root.glob("**/processed") if path.is_dir()]

    def salvage_sample(self, sample_dir: Path) -> RawProcessingResult:
        return self.salvager.salvage_sample(sample_dir)

    # Stage 4: processing after processed/ exists

    def process_all(
        self,
        data_dir: Optional[Path] = None,
        summary_file: Optional[Path] = None,
        successful_samples_file: Optional[Path] = None,
    ) -> Tuple[Dict[str, CompleteProcessingResult], List[SampleSummary]]:
        """Process already-salvaged samples through SpatialData and analysis."""
        sample_dirs = self.discover_processed_samples(data_dir)
        if not sample_dirs:
            self.logger.warning("No samples with processed/ directories under %s", data_dir or self.config.data_dir)
            return {}, []
        return self.processor.process_multiple_complete_with_summary(
            sample_dirs, summary_file=summary_file, successful_samples_file=successful_samples_file
        )

    # ================================================================== #
    # All four stages
    # ================================================================== #

    def run(
        self,
        output_csv: Path,
        *,
        max_samples: Optional[int] = None,
        summary_file: Optional[Path] = None,
        successful_samples_file: Optional[Path] = None,
    ) -> Tuple[Dict[str, CompleteProcessingResult], List[SampleSummary]]:
        """Find -> download -> salvage -> process end to end.

        Discovers samples to ``output_csv``, downloads them under
        ``config.data_dir``, then processes every salvaged sample.
        """
        self.logger.info("Pipeline step 1/4: find")
        df = self.find(output_csv)
        if df.empty:
            self.logger.warning("Search returned no samples; nothing to download/process")
            return {}, []

        self.logger.info("Pipeline step 2/4: download")
        self.download(output_csv, max_samples=max_samples)

        self.logger.info("Pipeline step 3/4: salvage")
        sample_dirs = self.discover_samples(self.config.data_dir)
        for sample_dir in sample_dirs:
            result = self.salvage_sample(sample_dir)
            if not result.success:
                self.logger.error("Salvage failed for %s: %s", sample_dir, result.error_message)
        self.logger.info("Pipeline step 4/4: process")
        return self.process_all(
            self.config.data_dir,
            summary_file=summary_file,
            successful_samples_file=successful_samples_file,
        )
