"""Typed result objects returned by pipeline stages.

The three original dataclasses (``RawProcessingResult``, ``TransformationResult``,
``CompleteProcessingResult``) duplicated the same four fields. They now share
:class:`BaseResult`, giving callers a uniform supertype and a single ``ok``/error
contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class BaseResult:
    """Common shape for every stage result."""

    sample_path: Path
    success: bool
    error_message: Optional[str] = None
    files_processed: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.success


@dataclass
class RawProcessingResult(BaseResult):
    """Result of extracting/organizing raw downloaded data."""

    processed_path: Optional[Path] = None


@dataclass
class TransformationResult(BaseResult):
    """Result of transforming a sample into a SpatialData zarr."""

    output_path: Optional[Path] = None


@dataclass
class CompleteProcessingResult(BaseResult):
    """Result of the full raw -> processed -> analysis pipeline."""

    raw_processed_path: Optional[Path] = None
    spatialdata_path: Optional[Path] = None
    final_processed_path: Optional[Path] = None
    images_created: List[str] = field(default_factory=list)
    bins_created: List[str] = field(default_factory=list)
    single_cell_files: List[str] = field(default_factory=list)


@dataclass
class DownloadResult(BaseResult):
    """Result of downloading one GSM sample from NCBI GEO.

    ``sample_path`` is the sample directory; ``files_processed`` holds the names
    of the downloaded files (inherited from :class:`BaseResult`).
    """

    gsm_id: str = ""
    raw_path: Optional[Path] = None


def failure(result_cls, sample_path: Path, error_message: str, **extra) -> BaseResult:
    """Build a failed result of ``result_cls`` with a message."""
    return result_cls(
        sample_path=sample_path,
        success=False,
        error_message=error_message,
        **extra,
    )
