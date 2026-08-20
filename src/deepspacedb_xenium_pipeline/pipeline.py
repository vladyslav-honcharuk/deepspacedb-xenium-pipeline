"""The unified pipeline facade.

``XeniumPipeline`` is the single public entrypoint. It exposes the three stages
as methods on one object:

    1. :meth:`collect`       - discover Xenium samples from NCBI GEO
    2. :meth:`download`      - download sample files
    3. :meth:`transform_all` - raw -> SpatialData -> analysis for every sample

plus :meth:`run`, which chains all three. The heavy per-sample transform logic
lives in :class:`~deepspacedb_xenium_pipeline.transformer.XeniumTransformer`,
which this facade composes and delegates to; the most-used transform methods are
surfaced here so callers never have to juggle separate stage objects.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .collector import SampleCollector
from .config import PipelineConfig
from .downloader import SampleDownloader
from .logging_setup import get_logger
from .results import CompleteProcessingResult, DownloadResult, RawProcessingResult, TransformationResult
from .summary import SampleSummary
from .transformer import XeniumTransformer


class XeniumPipeline:
    """One object, three stages: collect -> download -> transform."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.logger = get_logger(__name__)
        self.collector = SampleCollector(self.config)
        self.downloader = SampleDownloader(self.config)
        self.transformer = XeniumTransformer(self.config)

    # ================================================================== #
    # Stage 1: collect
    # ================================================================== #

    def collect(self, output_csv: Optional[Path] = None):
        """Discover Xenium samples; optionally write them to ``output_csv``.

        Returns the collected DataFrame. When ``output_csv`` is given and the
        result is non-empty, it is written there.
        """
        df = self.collector.collect()
        if output_csv is not None and not df.empty:
            self.collector.save(df, str(output_csv))
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
    # Stage 3: transform (delegated to the transform engine)
    # ================================================================== #

    def discover_samples(self, base_dir: Optional[Path] = None) -> List[Path]:
        return self.transformer.discover_downloaded_samples(base_dir or self.config.data_dir)

    def process_raw_data(self, sample_dir: Path) -> RawProcessingResult:
        return self.transformer.process_raw_data(sample_dir)

    def transform_sample(self, sample_dir: Path) -> TransformationResult:
        return self.transformer.transform_sample(sample_dir)

    def process_sample_complete(self, sample_dir: Path) -> CompleteProcessingResult:
        return self.transformer.process_sample_complete(sample_dir)

    def process_images_and_haystack_only(self, sample_dir: Path) -> CompleteProcessingResult:
        return self.transformer.process_images_and_haystack_only(sample_dir)

    def transform_all(
        self, data_dir: Optional[Path] = None, summary_file: Optional[Path] = None
    ) -> Tuple[Dict[str, CompleteProcessingResult], List[SampleSummary]]:
        """Run the complete per-sample pipeline for every discovered sample."""
        sample_dirs = self.discover_samples(data_dir)
        if not sample_dirs:
            self.logger.warning("No samples with raw/ directories under %s", data_dir or self.config.data_dir)
            return {}, []
        return self.transformer.process_multiple_complete_with_summary(sample_dirs, summary_file)

    # ================================================================== #
    # All three stages
    # ================================================================== #

    def run(
        self,
        output_csv: Path,
        *,
        max_samples: Optional[int] = None,
        summary_file: Optional[Path] = None,
    ) -> Tuple[Dict[str, CompleteProcessingResult], List[SampleSummary]]:
        """Collect -> download -> transform end to end.

        Discovers samples to ``output_csv``, downloads them under
        ``config.data_dir``, then transforms every downloaded sample.
        """
        self.logger.info("Pipeline step 1/3: collect")
        df = self.collect(output_csv)
        if df.empty:
            self.logger.warning("Collection returned no samples; nothing to download/transform")
            return {}, []

        self.logger.info("Pipeline step 2/3: download")
        self.download(output_csv, max_samples=max_samples)

        self.logger.info("Pipeline step 3/3: transform")
        return self.transform_all(self.config.data_dir, summary_file)
