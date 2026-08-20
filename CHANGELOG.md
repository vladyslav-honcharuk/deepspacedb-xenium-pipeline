# Changelog

All notable changes to this project are documented here. The format loosely
follows [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] - 2026-06-30

### Added
- Initial release. A unified `XeniumPipeline` with three stages — `collect`
  (NCBI GEO discovery), `download` (sample files), and `transform_all`
  (raw → SpatialData → analysis/images/binning/export) — plus `run` for all
  three, and a `xenium-pipeline` CLI.

### Changed (rewrite of the original scripts)
- Decomposed the 5,197-line `xenium_transformer.py` god-class into focused,
  injectable components behind a thin orchestrator.
- Single error-handling contract: stages raise `XeniumPipelineError` subclasses,
  converted to typed result objects at stage boundaries.
- Machine-specific BANKSY paths moved to `XENIUM_BANKSY_PYTHON` /
  `XENIUM_BANKSY_DIR` / `XENIUM_BANKSY_TIMEOUT` environment variables.
- Single source of truth for the pixel size and all well-known filenames.
- Logging is explicit and opt-in (`configure_logging`) on the package logger;
  importing the package has no side effects.
- `spatialdata_io` polygon patch is applied explicitly and idempotently rather
  than at import time.
- BANKSY subprocess paths are passed via `argv` (no source interpolation) and
  success is detected by exit code + results file.
- `transform_he` is bundled in `imaging/` (no more `PYTHONPATH` dependency).

### Fixed
- `write_summary_csv` previously wrote only the last sample's row (the
  `writerow` call sat outside the loop); it now writes one row per sample.
