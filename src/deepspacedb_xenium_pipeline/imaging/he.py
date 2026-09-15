"""H&E image handling: decompression, alignment, and lazy chunked loading.

Alignment is delegated to the bundled :mod:`.transform_he` module (skimage-based,
unaffected by OpenCV's ``SHRT_MAX`` dimension limit), imported lazily so that the
heavy image stack is only pulled in when alignment actually runs.
"""

from __future__ import annotations

import re
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
    "matrix_fitted_from_control_points.csv",
]

#: Image files that are already registered by some other method, or that belong
#: to a different modality, and so are never the raw unaligned H&E source.
_NON_HE_SOURCE_RE = re.compile(r"registered|aligned|_if_|morphology", re.IGNORECASE)
_HE_IMAGE_EXTENSIONS = (".ome.tif", ".ome.tiff", ".tif", ".tiff")


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
    def is_plain_matrix_csv(path: Path) -> bool:
        """True when ``path`` parses as a headerless numeric matrix.

        False for a keypoints/control-points file (a header row such as
        ``fixedX,fixedY,alignmentX,alignmentY`` - point correspondences, not a
        matrix) and for anything else unparseable.
        """
        import pandas as pd

        try:
            pd.read_csv(path, header=None).values.astype(float)
            return True
        except Exception:  # noqa: BLE001 - any parse failure means "not a matrix"
            return False

    @staticmethod
    def find_alignment_csv(raw_dir: Path) -> Optional[Path]:
        """Search ``raw/``, ``raw/alignment_files/`` and the sample root for a matrix CSV.

        Some series ship a keypoints file (control-point correspondences, *not*
        a matrix) whose name matches an earlier pattern than the real pre-fitted
        matrix sitting right next to it (e.g. GSE280314's
        ``*_he_imagealignment.csv`` vs ``alignment_files/matrix.csv``), so each
        candidate is validated as an actual numeric matrix before being
        returned rather than handed to the caller to crash on.
        """
        if not raw_dir.exists():
            return None
        seen = set()
        for search_dir in (raw_dir, raw_dir / "alignment_files", raw_dir.parent):
            if not search_dir.exists():
                continue
            for pattern in _ALIGNMENT_CSV_PATTERNS:
                for match in sorted(search_dir.glob(pattern)):
                    if match in seen or not match.is_file():
                        continue
                    seen.add(match)
                    if HEProcessor.is_plain_matrix_csv(match):
                        return match
        return None

    @staticmethod
    def find_unaligned_he_source(sample_dir: Path) -> Optional[Path]:
        """Find the raw, unaligned H&E source image for a sample.

        Prefers ``processed/he_image.ome.tif`` (the pipeline's own standardized
        copy) and otherwise falls back to a single unambiguous H&E-named image
        in ``raw/``, excluding anything already registered/aligned by another
        method as well as non-image files (keypoints/matrix CSVs, zip archives).
        """
        from ..extraction.organizer import find_he_image_file

        processed_he = sample_dir / "processed" / constants.HE_IMAGE_FILENAME
        if processed_he.exists():
            return processed_he

        raw_dir = sample_dir / "raw"
        if not raw_dir.exists():
            return None
        candidates = [
            f
            for f in raw_dir.iterdir()
            if f.is_file()
            and f.name.lower().endswith(_HE_IMAGE_EXTENSIONS)
            and not _NON_HE_SOURCE_RE.search(f.name)
            and find_he_image_file([f]) is not None
        ]
        return candidates[0] if len(candidates) == 1 else None

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

    def load_as_spatialdata_image(self, he_aligned_path: Path, chunk_xy: int = 4096, transformation=None):
        """Read an H&E OME-TIFF as a chunked ``Image2DModel``.

        Reads via tifffile's zarr store to avoid the >2 GB single-codec error,
        wrapping in a dask array with explicit tile chunks.

        ``transformation`` defaults to ``Identity`` - the image is assumed to be
        already aligned to the global frame, e.g. the ``he_image_aligned.ome.tif``
        produced by :meth:`align_to_xenium`'s pixel warp. Pass an ``Affine``
        (see :meth:`load_with_affine`) when reading a raw, unaligned source.
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
            arr,
            dims=("c", "y", "x"),
            transformations={"global": transformation if transformation is not None else Identity()},
        )

    def load_with_affine(self, he_path: Path, alignment_csv: Path, chunk_xy: int = 4096):
        """Load a raw, unaligned H&E image with the ``Affine`` from ``alignment_csv``.

        Same convention as spatialdata_io's own ``xenium_aligned_image()``
        (``pd.read_csv(header=None).values`` as a homogeneous matrix), paired
        with this package's alignment-CSV discovery instead of requiring a
        same-directory, same-stem filename match.
        """
        import pandas as pd
        from spatialdata.transformations import Affine

        matrix = pd.read_csv(alignment_csv, header=None).values
        affine = Affine(matrix, input_axes=("x", "y"), output_axes=("x", "y"))
        return self.load_as_spatialdata_image(he_path, chunk_xy=chunk_xy, transformation=affine)
