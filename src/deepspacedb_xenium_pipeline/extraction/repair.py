"""On-disk repair of common malformed files in a ``processed/`` directory.

Genomics exports frequently contain files that are *named* like one format but
encoded as another (raw-zlib payloads named ``.gz``, CSV files named
``.parquet``, etc.). These repairs are idempotent and safe to run before loading.
"""
from __future__ import annotations

import gzip
import shutil
import zlib
from pathlib import Path

import pandas as pd

from .. import constants
from ..logging_setup import get_logger


class FileRepair:
    """Detect and fix corrupted gz/parquet/zarr artifacts in place."""

    GZ_FILES = (constants.MATRIX_MTX_GZ, constants.BARCODES_TSV_GZ, constants.FEATURES_TSV_GZ)
    PARQUET_FILES = (
        "transcripts.parquet",
        "cells.parquet",
        "cell_boundaries.parquet",
        "nucleus_boundaries.parquet",
    )

    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    def fix_gz_files(self, processed_dir: Path) -> bool:
        """Re-encode mislabeled ``.gz`` files as proper gzip. Returns True on success."""
        for filename in self.GZ_FILES:
            gz_path = processed_dir / filename
            if not gz_path.exists():
                continue
            try:
                with gzip.open(gz_path, "rb") as fh:
                    fh.read(1)
                self.logger.debug("%s is already valid gzip", filename)
                continue
            except (gzip.BadGzipFile, OSError):
                self.logger.warning("%s is not valid gzip; attempting repair", filename)

            try:
                raw = gz_path.read_bytes()
                try:
                    payload = zlib.decompress(raw, -15)  # raw DEFLATE
                    self.logger.info("%s was raw zlib; converting to gzip", filename)
                except zlib.error:
                    payload = raw
                    self.logger.info("%s appears uncompressed; gzipping", filename)
                with gzip.open(gz_path, "wb", compresslevel=6) as out:
                    out.write(payload)
                self.logger.info("Repaired %s", filename)
            except OSError as exc:
                self.logger.error("Failed to repair %s: %s", filename, exc)
                return False
        return True

    def fix_parquet_files(self, processed_dir: Path) -> bool:
        """Rewrite parquet files that are actually CSV. Returns True if any fixed."""
        fixed_any = False
        for filename in self.PARQUET_FILES:
            file_path = processed_dir / filename
            if not file_path.exists():
                continue
            try:
                pd.read_parquet(file_path, engine="pyarrow").head(1)
                continue
            except Exception:  # noqa: BLE001 - pyarrow raises a wide variety
                self.logger.warning("Corrupted parquet %s; attempting CSV recovery", filename)
            try:
                df = pd.read_csv(file_path)
                backup = file_path.with_suffix(".csv.backup")
                shutil.move(str(file_path), str(backup))
                df.to_parquet(file_path, index=False)
                self.logger.info("Recovered %s from CSV", filename)
                fixed_any = True
            except Exception as exc:  # noqa: BLE001
                self.logger.error("Could not fix %s: %s", filename, exc)
        return fixed_any

    def clean_zarr_underscore_files(self, zarr_path: Path) -> None:
        """Remove AppleDouble/partial files from a transcripts points parquet dir."""
        points_dir = zarr_path / "points" / "transcripts" / "points.parquet"
        if not (points_dir.exists() and points_dir.is_dir()):
            self.logger.debug("No points.parquet directory at %s", points_dir)
            return
        junk = [f for f in points_dir.iterdir() if f.name.startswith("._") or "_.part" in f.name]
        if not junk:
            return
        self.logger.info("Cleaning %d stray files from points.parquet", len(junk))
        for file_path in junk:
            if file_path.is_file():
                file_path.unlink(missing_ok=True)
                self.logger.debug("Removed %s", file_path.name)
