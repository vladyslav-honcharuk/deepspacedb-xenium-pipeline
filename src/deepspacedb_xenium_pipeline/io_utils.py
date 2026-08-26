"""Low-level filesystem helpers shared across the pipeline.

These functions have no pipeline state and are trivially unit-testable.
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

from . import constants
from .logging_setup import get_logger

_logger = get_logger(__name__)


def is_gzip_file(path: Path) -> bool:
    """Return True if ``path`` starts with the gzip magic number."""
    try:
        with open(path, "rb") as fh:
            return fh.read(2) == constants.GZIP_MAGIC
    except OSError:
        return False


def gzip_decompress(source: Path, target: Path) -> None:
    """Decompress a gzip file to ``target``."""
    with gzip.open(source, "rb") as f_in, open(target, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def unique_path(path: Path) -> Path:
    """Return ``path`` or a ``name_N.ext`` sibling that does not yet exist."""
    if not path.exists():
        return path
    counter = 1
    original = path
    while path.exists():
        path = original.parent / f"{original.stem}_{counter}{original.suffix}"
        counter += 1
    return path


def is_junk_filename(filename: str) -> bool:
    """True for AppleDouble forks and known system junk files."""
    return filename.startswith(constants.APPLEDOUBLE_PREFIX) or filename in constants.JUNK_FILENAMES


def clean_appledouble_file(path: Path) -> bool:
    """Replace an AppleDouble-encoded file in place with its data fork.

    Returns True if the file is usable afterwards (either it was not AppleDouble,
    or the data fork was successfully extracted), False if it was AppleDouble but
    contained no data fork (in which case the corrupt file is removed).
    """
    try:
        with open(path, "rb") as fh:
            if fh.read(4) != constants.APPLEDOUBLE_MAGIC:
                return True  # Not AppleDouble; nothing to do.

            _logger.info("Cleaning AppleDouble file: %s", path.name)
            fh.seek(26)
            num_entries = int.from_bytes(fh.read(2), "big")
            for _ in range(num_entries):
                entry_id = int.from_bytes(fh.read(4), "big")
                offset = int.from_bytes(fh.read(4), "big")
                length = int.from_bytes(fh.read(4), "big")
                if entry_id == 1:  # data fork
                    fh.seek(offset)
                    data = fh.read(length)
                    with open(path, "wb") as out:
                        out.write(data)
                    _logger.info("Recovered data fork for %s", path.name)
                    return True

        _logger.warning("No data fork in AppleDouble file %s; removing", path.name)
        path.unlink(missing_ok=True)
        return False
    except OSError as exc:
        _logger.error("Failed to clean AppleDouble file %s: %s", path.name, exc)
        return False
