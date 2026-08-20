"""Morphology image derivatives: MIP, focus stacks, pyramidal OME-TIFFs.

These methods return ``bool`` success flags *internally* (they degrade
gracefully - a failed MIP should not abort a whole sample) but the orchestration
layer treats a fully-missing morphology as a logged condition, not an exception.
"""
from __future__ import annotations

import shutil
import tarfile
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import tifffile

from .. import constants
from ..logging_setup import get_logger


class MorphologyProcessor:
    """Generate and reconcile morphology MIP/focus derivatives."""

    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    # ------------------------------------------------------------------ #
    # MIP
    # ------------------------------------------------------------------ #

    def generate_mip(self, input_path: Path, output_path: Path) -> bool:
        try:
            file_size_gb = input_path.stat().st_size / (1024 ** 3)
            self.logger.info("Generating MIP from %s (%.2f GB)", input_path, file_size_gb)
            with tifffile.TiffFile(input_path) as tif:
                num_pages = len(tif.pages)
                self.logger.info("Morphology has %d pages", num_pages)
                if num_pages == 1:
                    mip = tif.pages[0].asarray().astype(np.uint16)
                else:
                    mip = self._mip_multipage(tif, num_pages, file_size_gb)
            tifffile.imwrite(output_path, mip.astype(np.uint16))
            return True
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Failed to generate MIP: %s", exc)
            return False

    def _mip_multipage(self, tif, num_pages: int, file_size_gb: float) -> np.ndarray:
        try:
            if file_size_gb <= 5.0:
                self.logger.info("Full-stack MIP for small file")
                stack = tif.series[0].asarray().astype(np.uint16)
                return stack if stack.ndim == 2 else np.max(stack, axis=0)
            raise ValueError("Large file; use page-by-page")
        except Exception as series_err:  # noqa: BLE001
            if file_size_gb <= 5.0:
                self.logger.warning("Series load failed (%s); page-by-page", series_err)
        mip = None
        expected_shape = None
        pages_ok = 0
        for i in range(num_pages):
            try:
                page = tif.pages[i].asarray().astype(np.uint16)
                if page.ndim != 2 or page.size == 0:
                    continue
                if expected_shape is None:
                    expected_shape = page.shape
                elif page.shape != expected_shape:
                    continue
            except Exception as page_err:  # noqa: BLE001
                self.logger.warning("Skipping corrupted page %d: %s", i, page_err)
                continue
            mip = page.copy() if mip is None else np.maximum(mip, page)
            pages_ok += 1
        if mip is None:
            raise RuntimeError("All pages corrupted; cannot generate MIP")
        self.logger.info("MIP from %d/%d readable pages", pages_ok, num_pages)
        return mip

    # ------------------------------------------------------------------ #
    # Focus
    # ------------------------------------------------------------------ #

    def focus_measure(self, image: np.ndarray) -> float:
        """Tenengrad focus measure; higher means sharper."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        if gray.dtype == np.uint16:
            gray = (gray / 256).astype(np.uint8)
        elif gray.dtype not in (np.uint8, np.float32):
            gray = gray.astype(np.uint8)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.sum(sobelx ** 2 + sobely ** 2))

    def write_pyramidal_ome_tiff(
        self,
        filename: Path,
        image: np.ndarray,
        pixel_size: float,
        subresolutions: int = constants.DEFAULT_PYRAMID_SUBRESOLUTIONS,
    ) -> bool:
        try:
            metadata = {
                "axes": "YX",
                "PhysicalSizeX": pixel_size,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": pixel_size,
                "PhysicalSizeYUnit": "µm",
            }
            options = dict(
                photometric="minisblack",
                tile=(constants.DEFAULT_TILE_SIZE, constants.DEFAULT_TILE_SIZE),
                compression="deflate",
                resolutionunit="CENTIMETER",
            )
            with tifffile.TiffWriter(filename, bigtiff=True) as tif:
                self.logger.info("Writing pyramid level 0")
                tif.write(
                    image,
                    subifds=subresolutions,
                    resolution=(1e4 / pixel_size, 1e4 / pixel_size),
                    metadata=metadata,
                    **options,
                )
                scale = 1.0
                current = image.copy()
                for i in range(subresolutions):
                    scale /= 2
                    width = int(np.floor(current.shape[1] * 0.5))
                    height = int(np.floor(current.shape[0] * 0.5))
                    current = cv2.resize(current, (width, height), interpolation=cv2.INTER_AREA)
                    self.logger.info("Writing pyramid level %d", i + 1)
                    tif.write(
                        current,
                        subfiletype=1,
                        resolution=(1e4 / scale / pixel_size, 1e4 / scale / pixel_size),
                        **options,
                    )
            return True
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Failed to write pyramidal OME-TIFF: %s", exc)
            return False

    def generate_focus_stack(
        self, input_path: Path, output_path: Path, patch_size: int = constants.DEFAULT_FOCUS_PATCH_SIZE
    ) -> bool:
        try:
            file_size_gb = input_path.stat().st_size / (1024 ** 3)
            self.logger.info("Generating focus stack from %s (%.2f GB)", input_path, file_size_gb)
            processed_dir = output_path.parent
            pixel_size = constants.read_pixel_size(processed_dir / constants.EXPERIMENT_FILENAME)

            with tifffile.TiffFile(input_path) as tif:
                num_pages = len(tif.pages)
                self.logger.info("Morphology has %d pages", num_pages)
                if num_pages == 1:
                    focus_image = tif.pages[0].asarray()
                else:
                    focus_image = self._best_focus_plane(tif, num_pages)

            self.logger.info("Creating pyramidal OME-TIFF")
            if not self.write_pyramidal_ome_tiff(output_path, focus_image, pixel_size):
                self.logger.warning("Pyramidal write failed; simple write fallback")
                tifffile.imwrite(
                    output_path,
                    focus_image.astype(np.uint16),
                    tile=(constants.DEFAULT_TILE_SIZE, constants.DEFAULT_TILE_SIZE),
                    compression="deflate",
                )

            target_dir = processed_dir / constants.MORPHOLOGY_FOCUS_DIRNAME
            target_dir.mkdir(exist_ok=True)
            target_file = target_dir / "morphology_focus_0000.ome.tif"
            if not target_file.exists():
                shutil.copy2(output_path, target_file)
                self.logger.info("Created focus directory entry: %s", target_file)
            self.logger.info("Generated focus file: %s", output_path)
            return True
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Failed to generate focus stack: %s", exc)
            return False

    def _best_focus_plane(self, tif, num_pages: int) -> np.ndarray:
        self.logger.info("Finding best-focused plane among %d via Tenengrad", num_pages)
        best_score = -1.0
        best_idx = 0
        for i in range(num_pages):
            try:
                score = self.focus_measure(tif.pages[i].asarray())
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Skipping corrupted plane %d: %s", i, exc)
                continue
            if score > best_score:
                best_score = score
                best_idx = i
        self.logger.info("Best plane %d/%d (score %.2e)", best_idx, num_pages, best_score)
        return tif.pages[best_idx].asarray()

    def generate_focus_stack_as_directory(
        self, input_path: Path, output_dir: Path, patch_size: int = constants.DEFAULT_FOCUS_PATCH_SIZE
    ) -> bool:
        try:
            output_dir.mkdir(exist_ok=True)
            temp_focus = output_dir / "morphology_focus_0000.ome.tif"
            return self.generate_focus_stack(input_path, temp_focus, patch_size)
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Failed to generate focus directory: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    # Derivative reconciliation
    # ------------------------------------------------------------------ #

    def ensure_derivatives(self, processed_dir: Path) -> Tuple[bool, bool, bool]:
        """Reconcile focus file/dir, generate missing MIP/focus. Returns flags."""
        morphology_path = processed_dir / constants.MORPHOLOGY_FILENAME
        mip_path = processed_dir / constants.MORPHOLOGY_MIP_FILENAME
        he_path = processed_dir / constants.HE_IMAGE_FILENAME
        focus_file = processed_dir / constants.MORPHOLOGY_FOCUS_FILENAME
        focus_dir = processed_dir / constants.MORPHOLOGY_FOCUS_DIRNAME

        self._maybe_extract_focus_tar(focus_file, focus_dir)

        has_mip = mip_path.exists()
        has_he = he_path.exists()
        has_focus_file = focus_file.exists()
        has_focus_dir = focus_dir.exists() and focus_dir.is_dir()
        has_files_in_dir = bool(has_focus_dir and any(focus_dir.iterdir()))
        has_focus = has_focus_file or (has_focus_dir and has_files_in_dir)

        # dir -> file
        if not has_focus_file and has_files_in_dir:
            focus_files = list(focus_dir.glob("*.ome.tif"))
            if focus_files:
                shutil.copy2(focus_files[0], focus_file)
                has_focus_file = True
                self.logger.info("Copied %s to single focus file", focus_files[0].name)
        # file -> dir
        if has_focus_file and not has_files_in_dir:
            focus_dir.mkdir(exist_ok=True)
            target = focus_dir / "morphology_focus_0000.ome.tif"
            if not target.exists():
                shutil.copy2(focus_file, target)
                has_files_in_dir = True
                self.logger.info("Created focus dir from single file")
        # normalize file count to 1 or 4
        if has_focus_file and has_files_in_dir:
            files_in_dir = sorted(focus_dir.glob("*.ome.tif"))
            if len(files_in_dir) not in (1, 4):
                self.logger.warning(
                    "morphology_focus/ has %d files (reader wants 1 or 4); resetting to 1",
                    len(files_in_dir),
                )
                for f in files_in_dir:
                    f.unlink()
                shutil.copy2(focus_file, focus_dir / "morphology_focus_0000.ome.tif")

        has_morphology_source = morphology_path.exists()
        if not has_morphology_source and not has_he:
            self.logger.warning("No morphology.ome.tif or H&E image found")
            return has_mip, has_focus, has_he
        if has_he and not has_morphology_source:
            self.logger.info("Using H&E image directly")
            return has_mip, has_focus, has_he

        if not has_mip and has_morphology_source:
            self.logger.info("Generating MIP")
            if self.generate_mip(morphology_path, mip_path):
                has_mip = True
        if not has_focus and has_morphology_source:
            self.logger.info("Generating focus")
            if self.generate_focus_stack(morphology_path, focus_file):
                has_focus = True
        return has_mip, has_focus, has_he

    def _maybe_extract_focus_tar(self, focus_file: Path, focus_dir: Path) -> None:
        """Handle the case where ``morphology_focus.ome.tif`` is really a tar."""
        if not focus_file.exists():
            return
        try:
            with open(focus_file, "rb") as fh:
                magic = fh.read(262)
            if magic[257:262] == b"ustar":
                self.logger.info("%s is a tar archive; extracting", focus_file)
                focus_dir.mkdir(exist_ok=True)
                with tarfile.open(focus_file, "r") as tar:
                    tar.extractall(focus_dir)
                focus_file.unlink()
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Could not check/extract %s: %s", focus_file, exc)
