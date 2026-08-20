"""Pipeline configuration.

Unlike the original ``XeniumTransformerConfig``, constructing this object has
**no global side effects**. It does not touch the root logger, does not open log
files, and does not monkey-patch any third-party module. Logging is configured
explicitly via :func:`xenium_pipeline.logging_setup.configure_logging`, and the
``spatialdata_io`` compatibility patch is applied explicitly via
:func:`xenium_pipeline.compat.apply_spatialdata_patches`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


def _default_bin_sizes() -> List[int]:
    return [5, 10, 20, 40, 80, 160]


def _default_image_dpis() -> List[int]:
    return [100, 200, 300, 500, 1000]


@dataclass
class PipelineConfig:
    """Immutable-ish configuration for the transformer.

    All machine-specific values that used to be hardcoded in the source are now
    fields here, each defaulting from an environment variable so the package can
    run on any machine without source edits.
    """

    # Output naming.
    output_suffix: str = "data.zarr"
    processed_suffix: str = "processed.zarr"

    # Behaviour toggles.
    overwrite_existing: bool = False
    skip_focus_copy: bool = False
    process_zarr_directly: bool = False
    skip_zarr_exports: bool = False

    # Tunables.
    bin_sizes: List[int] = field(default_factory=_default_bin_sizes)
    image_dpis: List[int] = field(default_factory=_default_image_dpis)

    # Acquisition (collect + download). ``base_dir`` is the root under which the
    # ``data_subdir`` tree (``{GPL}/{GSE}/{GSM}/raw/``) is created. No directory
    # is created at construction time - the downloader makes them on demand.
    base_dir: Path = field(default_factory=Path.cwd)
    data_subdir: str = "data"
    request_timeout: int = 300
    request_retries: int = 3
    download_delay: float = 2.0
    ncbi_delay: float = 1.0

    # BANKSY subprocess configuration. Previously hardcoded to
    # ``/home/vlad/xenium_mundus/...``; now environment-driven and optional.
    banksy_python: Optional[Path] = None
    banksy_dir: Optional[Path] = None
    banksy_top_genes: int = 1000
    banksy_timeout_seconds: Optional[int] = None

    def __post_init__(self) -> None:
        # Resolve BANKSY paths from the environment when not provided explicitly.
        if self.banksy_python is None:
            env = os.environ.get("XENIUM_BANKSY_PYTHON")
            self.banksy_python = Path(env) if env else None
        elif not isinstance(self.banksy_python, Path):
            self.banksy_python = Path(self.banksy_python)

        if self.banksy_dir is None:
            env = os.environ.get("XENIUM_BANKSY_DIR")
            self.banksy_dir = Path(env) if env else None
        elif not isinstance(self.banksy_dir, Path):
            self.banksy_dir = Path(self.banksy_dir)

        if self.banksy_timeout_seconds is None:
            env = os.environ.get("XENIUM_BANKSY_TIMEOUT")
            self.banksy_timeout_seconds = int(env) if env else None

        if not isinstance(self.base_dir, Path):
            self.base_dir = Path(self.base_dir)

    @property
    def data_dir(self) -> Path:
        """Root of the acquisition data tree (``base_dir/data_subdir``)."""
        return self.base_dir / self.data_subdir

    @property
    def banksy_available(self) -> bool:
        """True when a usable BANKSY interpreter and source dir are configured."""
        return (
            self.banksy_python is not None
            and self.banksy_dir is not None
            and self.banksy_python.exists()
        )
