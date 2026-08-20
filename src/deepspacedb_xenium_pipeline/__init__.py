"""deepspacedb_xenium_pipeline - collect, download, and transform 10x Xenium data.

A single :class:`XeniumPipeline` exposes the three stages as methods on one
object: ``collect`` (discover samples from NCBI GEO), ``download`` (fetch them),
and ``transform_all`` (raw -> SpatialData -> analysis), plus ``run`` for all
three.

Importing this package is cheap and side-effect free: it configures no logging,
writes no files, and patches no third-party modules. The heavy classes (which
pull in scanpy/spatialdata/requests) are imported lazily on first access so
lightweight consumers (config, CLI parsing) stay fast.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .config import PipelineConfig
from .exceptions import (
    BanksyError,
    CollectionError,
    ConfigurationError,
    DownloadError,
    ExtractionError,
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
    from .collector import SampleCollector
    from .downloader import SampleDownloader
    from .pipeline import XeniumPipeline
    from .transformer import XeniumTransformer

_LAZY = {
    "XeniumPipeline": ("pipeline", "XeniumPipeline"),
    "XeniumTransformer": ("transformer", "XeniumTransformer"),
    "SampleCollector": ("collector", "SampleCollector"),
    "SampleDownloader": ("downloader", "SampleDownloader"),
}

__all__ = [
    "__version__",
    "PipelineConfig",
    "configure_logging",
    "get_logger",
    "XeniumPipeline",
    "XeniumTransformer",
    "SampleCollector",
    "SampleDownloader",
    "BaseResult",
    "RawProcessingResult",
    "TransformationResult",
    "CompleteProcessingResult",
    "DownloadResult",
    "XeniumPipelineError",
    "CollectionError",
    "DownloadError",
    "ExtractionError",
    "OrganizationError",
    "LoaderError",
    "SpatialDataBuildError",
    "ImagingError",
    "BanksyError",
    "ConfigurationError",
]


def __getattr__(name: str):
    if name in _LAZY:
        from importlib import import_module

        module_name, attr = _LAZY[name]
        module = import_module(f".{module_name}", __name__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
