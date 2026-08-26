"""Stage 2 of raw processing: organize extracted files into ``processed/``.

Takes the flat temp directory produced by :class:`ArchiveExtractor` and copies
only the files the downstream loaders need, normalizing names and formats
(``.csv.gz`` -> ``.parquet``, gzip decompression, focus-stack handling). H&E
alignment and ``cells.parquet`` synthesis are delegated to dedicated components.
"""

from __future__ import annotations

import fnmatch
import gzip
import re
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import List, Optional

import pandas as pd
import tifffile

from .. import constants
from ..config import PipelineConfig
from ..io_utils import clean_appledouble_file, gzip_decompress, is_gzip_file
from ..logging_setup import get_logger
from .repair import FileRepair


class FileOrganizer:
    """Organize extracted Xenium files into a clean ``processed/`` directory."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        he_processor=None,
        cells_builder=None,
        repair: Optional[FileRepair] = None,
    ) -> None:
        self.config = config
        self.logger = get_logger(__name__)
        self.repair = repair or FileRepair()
        # Imported lazily to keep this module importable without the heavy stack.
        if he_processor is None:
            from ..imaging.he import HEProcessor

            he_processor = HEProcessor(config)
        if cells_builder is None:
            from ..loaders.counts import CellsParquetBuilder

            cells_builder = CellsParquetBuilder()
        self.he_processor = he_processor
        self.cells_builder = cells_builder

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def organize(self, temp_dir: Path, processed_dir: Path) -> int:
        """Organize ``temp_dir`` into ``processed_dir``; return files organized."""
        self.logger.info("Organizing files into %s", processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)
        all_temp_files = list(temp_dir.rglob("*"))

        files_organized = self._organize_morphology_focus(temp_dir, processed_dir, all_temp_files)

        for target_name, patterns in constants.ESSENTIAL_FILES.items():
            if target_name == constants.MORPHOLOGY_FOCUS_FILENAME:
                continue  # handled above
            if target_name == constants.HE_IMAGE_FILENAME:
                source = find_he_image_file(all_temp_files)
            else:
                source = self._find_best_match(all_temp_files, patterns)
            if source is None:
                continue
            target = processed_dir / target_name
            if target_name == constants.HE_IMAGE_FILENAME:
                if self.he_processor.process_he_file(source, target):
                    files_organized += 1
                    self.logger.info("Organized H&E: %s -> %s", source.name, target_name)
            elif self._process_and_copy(source, target):
                clean_appledouble_file(target)
                files_organized += 1
                self.logger.debug("Organized: %s -> %s", source.name, target_name)

        focus_dir = processed_dir / constants.MORPHOLOGY_FOCUS_DIRNAME
        if focus_dir.exists():
            for focus_file in focus_dir.glob("*.tif"):
                clean_appledouble_file(focus_file)

        self.logger.info("Repairing gzip files...")
        self.repair.fix_gz_files(processed_dir)
        self.logger.info("Validating parquet files...")
        self.repair.fix_parquet_files(processed_dir)

        self.logger.info("Ensuring cells.parquet...")
        if self.cells_builder.ensure(processed_dir):
            files_organized += 1

        self.logger.info("Organization complete: %d files organized", files_organized)
        return files_organized

    # ------------------------------------------------------------------ #
    # Matching / copying
    # ------------------------------------------------------------------ #

    @staticmethod
    def _find_best_match(files: List[Path], patterns: List[str]) -> Optional[Path]:
        for pattern in patterns:
            for file_path in files:
                if fnmatch.fnmatch(file_path.name.lower(), pattern.lower()):
                    return file_path
        return None

    def _process_and_copy(self, source: Path, target: Path) -> bool:
        """Copy ``source`` to ``target``, decompressing/converting CSV->parquet."""
        needs_decompress = source.name.endswith(".gz") and not target.name.endswith(".gz")
        source_format = source.name[:-3] if needs_decompress else source.name
        needs_csv_to_parquet = source_format.endswith(".csv") and target.name.endswith(".parquet")
        try:
            if needs_decompress and needs_csv_to_parquet:
                self.logger.info("Decompress + CSV->Parquet: %s -> %s", source.name, target.name)
                with tempfile.NamedTemporaryFile(mode="w+b", suffix=".csv", delete=False) as tmp:
                    tmp_csv = Path(tmp.name)
                    with gzip.open(source, "rb") as f_in:
                        shutil.copyfileobj(f_in, tmp)
                try:
                    pd.read_csv(tmp_csv).to_parquet(target, index=False)
                finally:
                    tmp_csv.unlink(missing_ok=True)
            elif needs_decompress:
                self.logger.info("Decompress: %s -> %s", source.name, target.name)
                gzip_decompress(source, target)
            elif needs_csv_to_parquet:
                self.logger.info("CSV->Parquet: %s -> %s", source.name, target.name)
                pd.read_csv(source).to_parquet(target, index=False)
            else:
                shutil.copy2(source, target)
            return True
        except Exception as exc:  # noqa: BLE001 - fall back to a plain copy
            self.logger.warning("Conversion of %s failed (%s); falling back to copy", source.name, exc)
            try:
                if needs_decompress:
                    gzip_decompress(source, target)
                else:
                    shutil.copy2(source, target)
                return True
            except OSError as fallback_exc:
                self.logger.error("Fallback copy of %s failed: %s", source.name, fallback_exc)
                return False

    # ------------------------------------------------------------------ #
    # Morphology focus
    # ------------------------------------------------------------------ #

    def _organize_morphology_focus(self, temp_dir: Path, processed_dir: Path, all_temp_files: List[Path]) -> int:
        """Materialize focus planes and a single tiled ``morphology_focus.ome.tif``."""
        target_focus_dir = processed_dir / constants.MORPHOLOGY_FOCUS_DIRNAME
        single_focus_file = processed_dir / constants.MORPHOLOGY_FOCUS_FILENAME
        focus_tars = [
            f
            for f in all_temp_files
            if f.is_file() and "morphology_focus" in f.name.lower() and f.name.endswith(".tar")
        ]

        first_focus_file: Optional[Path] = None
        files_organized = 0

        if focus_tars:
            target_focus_dir.mkdir(exist_ok=True)
            with tarfile.open(focus_tars[0], "r") as tar:
                count = 0
                for member in tar.getmembers():
                    if member.isfile() and member.name.endswith((".tif", ".tiff")):
                        target = target_focus_dir / f"morphology_focus_{count:04d}.ome.tif"
                        source = tar.extractfile(member)
                        if source is not None:
                            with source, open(target, "wb") as out:
                                shutil.copyfileobj(source, out)
                        if count == 0:
                            first_focus_file = target
                        count += 1
                files_organized = count
                self.logger.info("Extracted %d focus files from tar", count)
        else:
            focus_files = sorted(
                f
                for f in all_temp_files
                if f.is_file() and "focus" in f.name.lower() and "morphology" in f.name.lower()
            )
            if focus_files:
                target_focus_dir.mkdir(exist_ok=True)
                for i, source_file in enumerate(focus_files):
                    target = target_focus_dir / f"morphology_focus_{i:04d}.ome.tif"
                    if is_gzip_file(source_file):
                        gzip_decompress(source_file, target)
                    else:
                        shutil.copy2(source_file, target)
                    if i == 0:
                        first_focus_file = target
                files_organized = len(focus_files)
                self.logger.info("Copied %d focus files", files_organized)

        if first_focus_file and first_focus_file.exists():
            self._write_single_focus(first_focus_file, single_focus_file, processed_dir)
        else:
            self.logger.info("No focus files found; will derive from morphology if available")

        return files_organized

    def _write_single_focus(self, source: Path, target: Path, processed_dir: Path) -> None:
        """Write the first focus plane as a standalone tiled OME-TIFF."""
        try:
            with tifffile.TiffFile(source) as tif:
                page = tif.pages[0]
                assert isinstance(page, tifffile.TiffPage)
                img = page.asarray()
                if page.is_tiled:
                    tile_w, tile_l = page.tilewidth, page.tilelength
                else:
                    tile_w = tile_l = constants.DEFAULT_TILE_SIZE
                compression = getattr(page, "compression", "deflate")

            pixel_size = constants.read_pixel_size(processed_dir / constants.EXPERIMENT_FILENAME)
            ome_metadata = {
                "axes": "YX",
                "PhysicalSizeX": pixel_size,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": pixel_size,
                "PhysicalSizeYUnit": "µm",
            }
            tifffile.imwrite(
                target,
                img,
                tile=(tile_w, tile_l),
                compression=compression,
                photometric="minisblack",
                metadata=ome_metadata,
                bigtiff=True,
            )
            self.logger.info("Wrote tiled %s (tile %dx%d)", target.name, tile_w, tile_l)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Tiled focus write failed (%s); copying instead", exc)
            try:
                shutil.copy2(source, target)
            except OSError as copy_exc:
                self.logger.error("Focus fallback copy failed: %s", copy_exc)


# "he"/"h&e" as a bare substring (the old '*he*.tif*' fallback) false-positives
# on marker names that contain those letters (cadherin, HER2, THelper). Require
# a standalone token: bounded by _/-/./whitespace/start/end, or a camelCase
# boundary (lowercase letter immediately before exact-case "HE").
_HE_TOKEN_RE = re.compile(r"(?:^|[_\-.\s])h&?e(?:[_\-.\s]|$)", re.IGNORECASE)
_HE_CAMELCASE_RE = re.compile(r"(?<=[a-z])HE(?:[_\-.\s]|$|[A-Z0-9])")
_HE_EXTENSIONS = (".ome.tif", ".ome.tiff", ".tif", ".tiff")


def find_he_image_file(files: List[Path]) -> Optional[Path]:
    """Return a genuine H&E image, never a file already claimed by morphology_focus/."""
    for file_path in files:
        if not file_path.is_file() or "morphology_focus" in file_path.parts:
            continue
        stem = file_path.name
        for ext in _HE_EXTENSIONS:
            if stem.lower().endswith(ext):
                stem = stem[: -len(ext)]
                break
        else:
            continue
        if _HE_TOKEN_RE.search(stem) or _HE_CAMELCASE_RE.search(stem):
            return file_path
    return None
