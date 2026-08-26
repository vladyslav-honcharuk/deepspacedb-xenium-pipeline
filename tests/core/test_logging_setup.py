"""Unit tests for logging_setup in deepspacedb_xenium_pipeline.logging_setup."""

from __future__ import annotations

import logging

from deepspacedb_xenium_pipeline.logging_setup import (
    PACKAGE_LOGGER_NAME,
    configure_logging,
    get_logger,
)


def test_get_logger():
    log1 = get_logger()
    assert log1.name == PACKAGE_LOGGER_NAME

    log2 = get_logger("my_module")
    assert log2.name == f"{PACKAGE_LOGGER_NAME}.my_module"

    log3 = get_logger(f"{PACKAGE_LOGGER_NAME}.sub")
    assert log3.name == f"{PACKAGE_LOGGER_NAME}.sub"


def test_configure_logging_console_and_file(tmp_path):
    log_file = tmp_path / "test.log"
    logger = configure_logging(level="DEBUG", log_file=log_file, force=True)

    assert logger.level == logging.DEBUG
    assert len(logger.handlers) >= 2

    logger.debug("Test log message")
    for handler in logger.handlers:
        handler.flush()

    assert log_file.exists()
    assert "Test log message" in log_file.read_text()
