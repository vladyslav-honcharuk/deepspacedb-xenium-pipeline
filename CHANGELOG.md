# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Cell/expression alignment**: `cell_circles` shapes are now matched to the
  expression table strictly by `cell_id`, in the table's row order. The previous
  positional fallback ("take the first n circles") silently dropped whole spatial
  tiles for QC-filtered tables, producing spatially corrupted binned and
  single-cell output. Samples whose `cell_circles` index carries no real
  `cell_id` recover their labels from `cell_boundaries`/`nucleus_boundaries` or
  `processed/cells.parquet` (by nearest centroid when row counts differ), and
  processing of a sample fails loudly rather than guessing when no `cell_id`
  mapping can be established.
- **H&E alignment**: alignment-CSV candidates are validated as an actual numeric
  matrix, so a keypoints/control-points file no longer shadows the real
  pre-fitted matrix next to it; the sample root and
  `matrix_fitted_from_control_points.csv` are searched as well. A raw H&E source
  plus its matrix is now attached as an `Affine`-transformed image, overwriting
  any unaligned `he_image` that spatialdata_io loaded with an Identity transform.
- **Correction factor**: when no image is available, it is taken from the shapes'
  absolute max-x (already in microns) instead of the min-to-max range scaled by
  `pixel_size` a second time, which under-sized the viewer canvas and clipped
  everything past the right edge.
- **Cell-shape overlays**: the canvas Y extent is derived from the image
  dimensions (`scale0_y * pixel_size`) rather than reusing the X extent, so
  non-square tissue is no longer cropped below `y == correction_factor`.
- **Single-cell export**: duplicate gene symbols in a panel no longer collide on
  the same per-gene Zarr dataset path; repeats are disambiguated by gene index.

### Changed
- **Binning performance**: per-bin aggregation is a single sparse matrix multiply
  over all genes instead of a per-gene Python loop over cells, and the
  spatial-optimized copy is written from memory rather than by re-reading and
  decompressing the just-written store. Small bin sizes now complete in minutes
  rather than days.

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
