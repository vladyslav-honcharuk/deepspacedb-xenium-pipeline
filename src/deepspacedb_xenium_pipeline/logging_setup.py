"""Explicit, opt-in logging configuration.

The original code mutated the *root* logger inside a dataclass ``__post_init__``,
attached an unrotated ``FileHandler('xenium_processing.log')`` to the current
working directory, and silently ignored every later configuration ("first
instance wins"). Library code must never do that.

Here, configuring logging is an explicit call that the *application* (CLI) makes.
Importing the package, or constructing a :class:`~xenium_pipeline.config.PipelineConfig`,
changes nothing about global logging state. Every module obtains its logger via
:func:`get_logger`, which returns a child of the ``xenium_pipeline`` logger so
callers can route or silence the package independently of the root logger.
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

    ``name`` is usually ``__name__``; the ``xenium_pipeline.`` prefix is added
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
    """Configure the ``xenium_pipeline`` logger (not the root logger).

    Parameters
    ----------
    level:
        Logging level name or numeric value.
    log_file:
        Optional path to a rotating log file. When ``None`` (default) only a
        console handler is attached and nothing is written to disk.
    rotate_max_bytes, rotate_backups:
        Rotation policy for the file handler.
    force:
        Replace any handlers already attached to the package logger. Without it,
        repeated calls are idempotent (handlers are not duplicated) but an
        explicit ``force=True`` lets an application reconfigure destinations,
        which the original implementation made impossible.
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
