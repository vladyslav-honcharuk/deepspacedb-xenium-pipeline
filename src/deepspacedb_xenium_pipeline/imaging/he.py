"""H&E image handling: decompression, alignment, and lazy chunked loading.

Alignment is delegated to the bundled :mod:`.transform_he` module (skimage-based,
unaffected by OpenCV's ``SHRT_MAX`` dimension limit), imported lazily so that the
heavy image stack is only pulled in when alignment actually runs.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

from .. import constants
from ..config import PipelineConfig
from ..io_utils import gzip_decompress, is_gzip_file
from ..logging_setup import get_logger

_ALIGNMENT_CSV_PATTERNS: List[str] = [
    "*he_imagealignment.csv*",
    "*homography*.csv*",
    "*alignment*.csv*",
    "*transform*.csv*",
    "matrix.csv",
]


class HEProcessor:
    """Process and align Xenium H&E images."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config
        self.logger = get_logger(__name__)

    def process_he_file(self, source: Path, target: Path) -> bool:
        """Place the H&E image at ``target`` and trigger alignment when possible."""
        try:
            if is_gzip_file(source):
                self.logger.info("Decompressing H&E file: %s", source.name)
                gzip_decompress(source, target)
            else:
                shutil.copy2(source, target)

            processed_dir = target.parent
            already_aligned = (processed_dir / constants.HE_IMAGE_ALIGNED_FILENAME).exists()
            raw_dir = processed_dir.parent / "raw"
            if not already_aligned and self.find_alignment_csv(raw_dir) is not None:
                self.align_to_xenium(processed_dir)
            return True
        except OSError as exc:
            self.logger.error("Failed to process H&E file %s: %s", source.name, exc)
            return False

    @staticmethod
    def find_alignment_csv(raw_dir: Path) -> Optional[Path]:
        """Search ``raw/`` and ``raw/alignment_files/`` for an alignment CSV."""
        if not raw_dir.exists():
            return None
        for search_dir in (raw_dir, raw_dir / "alignment_files"):
            if not search_dir.exists():
                continue
            for pattern in _ALIGNMENT_CSV_PATTERNS:
                matches = sorted(search_dir.glob(pattern))
                if matches:
                    return matches[0]
        return None

    def align_to_xenium(self, processed_dir: Path) -> bool:
        """Align the H&E image to the Xenium coordinate system."""
        try:
            he_image_path = processed_dir / constants.HE_IMAGE_FILENAME
            if not he_image_path.exists():
                self.logger.warning("H&E image not found; skipping alignment")
                return False

            raw_dir = processed_dir.parent / "raw"
            alignment_file = self.find_alignment_csv(raw_dir)
            if alignment_file is None:
                self.logger.warning("Alignment CSV not found in %s; skipping", raw_dir)
                return False

            pixel_size = constants.read_pixel_size(processed_dir / constants.EXPERIMENT_FILENAME)
            # Bundled in this package (runs in-process, same env as the pipeline).
            from .transform_he import transform_he_auto

            output_path = processed_dir / constants.HE_IMAGE_ALIGNED_FILENAME
            self.logger.info("Aligning H&E via transform_he_auto (skimage)")
            transform_he_auto(
                he_image_path=str(he_image_path),
                transform_csv=str(alignment_file),
                output_path=str(output_path),
                pixel_size=pixel_size,
                bigtiff=True,
                generate_pngs=True,
                pyramid_level=0,
            )
            self.logger.info("Saved aligned H&E image to %s", output_path)
            return True
        except Exception as exc:  # noqa: BLE001 - alignment is best-effort
            self.logger.error("Failed to align H&E image: %s", exc)
            return False

    def load_as_spatialdata_image(self, he_aligned_path: Path, chunk_xy: int = 4096):
        """Read an aligned H&E OME-TIFF as a chunked ``Image2DModel``.

        Reads via tifffile's zarr store to avoid the >2 GB single-codec error,
        wrapping in a dask array with explicit tile chunks.
        """
        import dask.array as da
        import tifffile
        from spatialdata.models import Image2DModel
        from spatialdata.transformations import Identity

        with tifffile.TiffFile(str(he_aligned_path)) as tif:
            try:
                arr = da.from_zarr(tif.aszarr(level=0))
            except Exception:  # noqa: BLE001
                arr = da.from_array(tif.asarray(), chunks="auto")

        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            arr = arr.transpose(2, 0, 1)
        n_c = arr.shape[0]
        arr = arr.rechunk((n_c, chunk_xy, chunk_xy))
        return Image2DModel.parse(
            arr, dims=("c", "y", "x"), transformations={"global": Identity()}
        )
