"""Exception hierarchy for the Xenium pipeline.

The original code mixed four incompatible error-handling styles (``bool`` flags,
bare ``raise``, ``except:`` swallowing, and ``None`` sentinels). The package now
uses a single contract: **stages raise a subclass of :class:`XeniumPipelineError`
on failure**. The orchestrator catches these at well-defined boundaries and
converts them into the typed result objects in :mod:`xenium_pipeline.results`.
"""
from __future__ import annotations


class XeniumPipelineError(Exception):
    """Base class for every error raised intentionally by the pipeline."""


class CollectionError(XeniumPipelineError):
    """Raised when sample discovery against NCBI GEO fails."""


class DownloadError(XeniumPipelineError):
    """Raised when downloading a sample's files fails."""


class ExtractionError(XeniumPipelineError):
    """Raised when raw archives cannot be extracted or organized."""


class OrganizationError(XeniumPipelineError):
    """Raised when extracted files cannot be organized into ``processed/``."""


class LoaderError(XeniumPipelineError):
    """Raised when expression/metadata tables cannot be loaded or aligned."""


class SpatialDataBuildError(XeniumPipelineError):
    """Raised when a :class:`spatialdata.SpatialData` object cannot be built."""


class ImagingError(XeniumPipelineError):
    """Raised for unrecoverable image-processing failures."""


class BanksyError(XeniumPipelineError):
    """Raised when the BANKSY subprocess cannot be run or fails."""


class ConfigurationError(XeniumPipelineError):
    """Raised for invalid pipeline configuration."""
