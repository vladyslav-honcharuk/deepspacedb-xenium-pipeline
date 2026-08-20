"""Stage 1 of raw processing: extract every archive into a flat temp directory.

Single responsibility: take whatever shapes up in ``raw/`` (zip, tar, tar.gz,
loose files, gzip-compressed payloads) and materialize the contents into a
working directory, flattening nested structure (except for ``morphology_focus``
which is preserved). It deliberately knows nothing about *which* files matter -
that is the organizer's job.
"""
from __future__ import annotations

import shutil
import tarfile
import zipfile
import zlib
from pathlib import Path
from typing import List, Set

from ..io_utils import is_junk_filename, unique_path
from ..logging_setup import get_logger


class ArchiveExtractor:
    """Extract raw archives into a temp directory."""

    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def extract_all(self, raw_dir: Path, temp_dir: Path) -> int:
        """Extract/copy everything in ``raw_dir`` into ``temp_dir``.

        Returns the number of files materialized. ``_outs.zip`` archives take
        priority over ``xe_outs.zip`` when both are present.
        """
        self.logger.info("Extracting all files from %s", raw_dir)
        all_files = list(raw_dir.rglob("*"))
        files_to_skip = self._resolve_outs_priority(all_files)

        self.logger.info("Found %d items in raw directory", len(all_files))
        files_found = 0
        for file_path in all_files:
            if not file_path.is_file():
                continue
            if file_path in files_to_skip:
                self.logger.info("Skipping superseded archive: %s", file_path.name)
                continue
            try:
                if self.is_archive(file_path):
                    extracted = self._extract_archive(file_path, temp_dir)
                    files_found += extracted
                    self.logger.debug("Extracted %d files from %s", extracted, file_path.name)
                else:
                    self._copy_to_temp(file_path, temp_dir, raw_dir)
                    files_found += 1
            except (OSError, tarfile.TarError, zipfile.BadZipFile, zlib.error) as exc:
                self.logger.warning("Failed to process %s: %s", file_path.name, exc)
                continue

        self.logger.info("Extraction complete: %d files extracted/copied", files_found)
        return files_found

    @staticmethod
    def is_archive(file_path: Path) -> bool:
        """Whether ``file_path`` should be unpacked rather than copied verbatim."""
        name = file_path.name.lower()
        if name.endswith(".zarr.zip"):
            # zarr stores are consumed as-is, never unpacked.
            return False
        return (
            name.endswith(".zip")
            or name.endswith(".tar.gz")
            or name.endswith(".tar")
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _resolve_outs_priority(self, all_files: List[Path]) -> Set[Path]:
        outs = [f for f in all_files if f.is_file() and f.name.endswith("_outs.zip")]
        xe_outs = [f for f in all_files if f.is_file() and f.name.endswith("xe_outs.zip")]
        if outs and xe_outs:
            self.logger.info(
                "Both _outs.zip and xe_outs.zip present; prioritizing _outs.zip "
                "(%s) and skipping %s",
                [f.name for f in outs],
                [f.name for f in xe_outs],
            )
            return set(xe_outs)
        return set()

    def _extract_archive(self, archive_path: Path, temp_dir: Path) -> int:
        name = archive_path.name
        if name.endswith((".tar.gz", ".tar")):
            return self._extract_tar(archive_path, temp_dir)
        if name.endswith(".zip"):
            return self._extract_zip(archive_path, temp_dir)
        self.logger.warning("Unknown archive format: %s", name)
        return 0

    def _extract_tar(self, archive_path: Path, temp_dir: Path) -> int:
        try:
            return self._extract_tar_with_mode(archive_path, temp_dir, "r:*")
        except (tarfile.ReadError, OSError, EOFError, zlib.error) as exc:
            # A ``.tar.gz`` that is actually an uncompressed tar is common in the
            # wild; retry forcing the plain-tar reader before giving up.
            if archive_path.name.endswith(".tar.gz"):
                self.logger.warning(
                    "Failed to read %s as tar.gz (%s); retrying as uncompressed tar",
                    archive_path.name,
                    exc,
                )
                try:
                    return self._extract_tar_with_mode(archive_path, temp_dir, "r:")
                except (tarfile.TarError, OSError) as exc2:
                    self.logger.error("Also failed as plain tar: %s", exc2)
                    raise exc
            raise

    def _extract_tar_with_mode(self, archive_path: Path, temp_dir: Path, mode: str) -> int:
        extracted = 0
        with tarfile.open(archive_path, mode) as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                filename = Path(member.name).name
                if not filename or is_junk_filename(filename):
                    continue
                target = unique_path(temp_dir / filename)
                source = archive.extractfile(member)
                if source is not None:
                    with source, open(target, "wb") as out:
                        shutil.copyfileobj(source, out)
                    extracted += 1
        return extracted

    def _extract_zip(self, archive_path: Path, temp_dir: Path) -> int:
        extracted = 0
        with zipfile.ZipFile(archive_path, "r") as archive:
            for member in archive.namelist():
                if member.endswith("/"):
                    continue
                filename = Path(member).name
                if not filename or is_junk_filename(filename):
                    continue
                target = unique_path(temp_dir / filename)
                with archive.open(member) as source, open(target, "wb") as out:
                    shutil.copyfileobj(source, out)
                extracted += 1
        return extracted

    def _copy_to_temp(self, file_path: Path, temp_dir: Path, raw_dir: Path) -> None:
        relative = file_path.relative_to(raw_dir)
        # Preserve nested structure only for multi-plane focus stacks.
        if "morphology_focus" in str(relative).lower():
            target = temp_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
        else:
            target = unique_path(temp_dir / file_path.name)
        shutil.copy2(file_path, target)
