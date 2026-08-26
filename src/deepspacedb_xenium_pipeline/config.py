"""Pipeline configuration.

Defines all pipeline settings, paths, execution toggles, and tunables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


def _default_bin_sizes() -> List[int]:
    return [5, 10, 20, 40, 80, 160]


def _default_image_dpis() -> List[int]:
    return [100, 200, 300, 500, 1000]


@dataclass
class PipelineConfig:
    """Central configuration for discovery, download, salvage, and processing."""

    # Output naming.
    output_suffix: str = "data.zarr"
    processed_suffix: str = "processed.zarr"

    # Behaviour toggles.
    overwrite_existing: bool = False
    # Spatial binning is opt-in (heavy); single-cell Zarr export always runs.
    skip_binning: bool = True
    # Bin sizes + single-cell export run as separate processes.
    zarr_export_workers: Optional[int] = None

    # Tunables.
    bin_sizes: List[int] = field(default_factory=_default_bin_sizes)
    image_dpis: List[int] = field(default_factory=_default_image_dpis)

    # Acquisition (find + download). ``base_dir`` is the root under which the
    # ``data_subdir`` tree (``{GPL}/{GSE}/{GSM}/raw/``) is created. No directory
    # is created at construction time - the downloader makes them on demand.
    base_dir: Path = field(default_factory=Path.cwd)
    data_subdir: str = "data"
    request_timeout: int = 300
    request_retries: int = 3
    download_delay: float = 2.0
    ncbi_delay: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.base_dir, Path):
            self.base_dir = Path(self.base_dir)

    @property
    def data_dir(self) -> Path:
        """Root of the acquisition data tree (``base_dir/data_subdir``)."""
        return self.base_dir / self.data_subdir
