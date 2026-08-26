"""Exception hierarchy for the Xenium pipeline.

Provides structured, typed exceptions subclassing :class:`XeniumPipelineError`.
"""

from __future__ import annotations


class XeniumPipelineError(Exception):
    """Base class for every error raised intentionally by the pipeline."""


class FindError(XeniumPipelineError):
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


class ConfigurationError(XeniumPipelineError):
    """Raised for invalid pipeline configuration."""
