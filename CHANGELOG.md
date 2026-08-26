# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-26

Initial public release.

### Added
- Unified `XeniumPipeline` entrypoint exposing the four stages as methods on one
  object: `find`, `download`, `salvage_sample`, and `process_all`, plus `run`
  for the complete end-to-end workflow.
- **Find**: NCBI GEO discovery of Xenium samples via E-utilities (`SampleFinder`).
- **Download**: GEO MiniML parsing and streaming download of sample archives and
  supplementary data (`SampleDownloader`).
- **Salvage**: repair and normalization of malformed GEO uploads, including
  multi-format decompression (`.tar.gz`, `.zip`, `.gz`, `.zst`), canonical file
  detection, H&E vs. fluorescent-channel discrimination, `experiment.xenium` metadata
  synthesis, `cells.parquet` geometry regeneration, and matrix repair
  (Cell Ranger H5, AnnData H5, Xenium Zarr, 10x MEX).
- **Process**: `SpatialData` construction, transcriptomics (QC, normalization,
  PCA, UMAP, Leiden clustering), imaging and morphology (multi-scale MIPs, focus
  stacks, aligned H&E overlays, boundary masks), opt-in spatial binning, sparse
  Zarr export (CSR/CSC/per-gene chunked), and per-sample QC summary reporting.
- `xenium-pipeline` command-line interface with `find`, `download`, `salvage`,
  `process`, and `run` subcommands.
- Typed exception hierarchy, typed result dataclasses, and shipped `py.typed`
  marker.
- Test suite (pytest) and CI across Python 3.10 to 3.12 with ruff and mypy.

[0.1.0]: https://github.com/vladyslav-honcharuk/deepspacedb-xenium-pipeline/releases/tag/v0.1.0
