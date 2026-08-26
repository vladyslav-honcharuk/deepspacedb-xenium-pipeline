"""deepspacedb_xenium_pipeline - find, download, salvage, and process Xenium data.

A single :class:`XeniumPipeline` exposes the pipeline stages as methods on one
object: ``find`` (discover samples from NCBI GEO), ``download`` (fetch raw archives),
``salvage_sample`` (repair malformed uploads), and ``process_all`` (SpatialData construction,
transcriptomics, imaging, and Zarr exports), plus ``run`` for the complete end-to-end workflow.

Importing this package is cheap and side-effect free: it configures no logging,
writes no files, and patches no third-party modules. The heavy classes (which
pull in scanpy/spatialdata/requests) are imported lazily on first access so
lightweight consumers (config, CLI parsing) stay fast.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import PipelineConfig
from .exceptions import (
    ConfigurationError,
    DownloadError,
    ExtractionError,
    FindError,
    ImagingError,
    LoaderError,
    OrganizationError,
    SpatialDataBuildError,
    XeniumPipelineError,
)
from .logging_setup import configure_logging, get_logger
from .results import (
    BaseResult,
    CompleteProcessingResult,
    DownloadResult,
    RawProcessingResult,
    TransformationResult,
)

__version__ = "0.1.0"

if TYPE_CHECKING:  # for type checkers/IDEs without triggering heavy imports
    from .downloader import SampleDownloader
    from .finder import SampleFinder
    from .pipeline import XeniumPipeline
    from .processor import XeniumProcessor

_LAZY = {
    "XeniumPipeline": ("pipeline", "XeniumPipeline"),
    "XeniumProcessor": ("processor", "XeniumProcessor"),
    "SampleFinder": ("finder", "SampleFinder"),
    "SampleDownloader": ("downloader", "SampleDownloader"),
}

__all__ = [
    "__version__",
    "PipelineConfig",
    "configure_logging",
    "get_logger",
    "XeniumPipeline",
    "XeniumProcessor",
    "SampleFinder",
    "SampleDownloader",
    "BaseResult",
    "RawProcessingResult",
    "TransformationResult",
    "CompleteProcessingResult",
    "DownloadResult",
    "XeniumPipelineError",
    "FindError",
    "DownloadError",
    "ExtractionError",
    "OrganizationError",
    "LoaderError",
    "SpatialDataBuildError",
    "ImagingError",
    "ConfigurationError",
]


def __getattr__(name: str):
    if name in _LAZY:
        from importlib import import_module

        module_name, attr = _LAZY[name]
        module = import_module(f".{module_name}", __name__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
