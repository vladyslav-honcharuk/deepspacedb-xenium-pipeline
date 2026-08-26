"""Package-level logging configuration.

Provides unified logging via :func:`get_logger` under the package namespace.
Logging handlers can be configured explicitly using :func:`configure_logging`.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Union

PACKAGE_LOGGER_NAME = "deepspacedb_xenium_pipeline"

_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return the package logger or a named child of it.

    ``name`` is usually ``__name__``; the ``deepspacedb_xenium_pipeline.`` prefix is added
    automatically if absent so all package loggers share one parent.
    """
    if not name or name == PACKAGE_LOGGER_NAME:
        return logging.getLogger(PACKAGE_LOGGER_NAME)
    if name.startswith(PACKAGE_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{PACKAGE_LOGGER_NAME}.{name}")


def configure_logging(
    level: Union[int, str] = "INFO",
    *,
    log_file: Optional[Path] = None,
    rotate_max_bytes: int = 10 * 1024 * 1024,
    rotate_backups: int = 3,
    force: bool = False,
) -> logging.Logger:
    """Configure the package logger.

    Parameters
    ----------
    level:
        Logging level name or numeric value.
    log_file:
        Optional path to a rotating log file. When ``None`` (default) only a
        console handler is attached.
    rotate_max_bytes, rotate_backups:
        Rotation policy for the file handler.
    force:
        Replace any handlers already attached to the package logger.
    """
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    numeric_level = level if isinstance(level, int) else getattr(logging, str(level).upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Do not propagate to the root logger; the package owns its handlers. This
    # prevents contaminating (or being contaminated by) other libraries.
    logger.propagate = False

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    if logger.handlers:
        # Already configured; just (re)apply the level and return.
        return logger

    formatter = logging.Formatter(_LOG_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=rotate_max_bytes, backupCount=rotate_backups, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
