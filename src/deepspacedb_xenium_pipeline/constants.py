"""Centralized constants for the Xenium pipeline.

Every magic value that was previously duplicated across the monolith lives here
exactly once. In particular the Xenium pixel size (0.2125 µm) appeared as an
inline literal in 11 different places; it now has a single source of truth.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

# --------------------------------------------------------------------------- #
# Physical / instrument constants
# --------------------------------------------------------------------------- #

#: Default Xenium pixel size in micrometres, used as a fallback when an
#: ``experiment.xenium`` file does not specify ``pixel_size``.
DEFAULT_PIXEL_SIZE: float = 0.2125

#: Default tile edge (pixels) used when (re)writing tiled OME-TIFFs.
DEFAULT_TILE_SIZE: int = 1024

#: Default patch size used by focus-stack tiling.
DEFAULT_FOCUS_PATCH_SIZE: int = 64

#: Number of sub-resolution levels written for pyramidal OME-TIFFs.
DEFAULT_PYRAMID_SUBRESOLUTIONS: int = 7


# --------------------------------------------------------------------------- #
# Well-known output / intermediate filenames
# --------------------------------------------------------------------------- #

EXPERIMENT_FILENAME = "experiment.xenium"
HAYSTACK_RESULTS_FILENAME = "haystack_results.csv"
BANKSY_RESULTS_FILENAME = "banksy_results.json"
CORRECTION_FACTOR_FILENAME = "correction_factor.csv"
DIMS_FILENAME = "dims.csv"
GENES_FILENAME = "genes.csv"

HE_IMAGE_FILENAME = "he_image.ome.tif"
HE_IMAGE_ALIGNED_FILENAME = "he_image_aligned.ome.tif"
MORPHOLOGY_FILENAME = "morphology.ome.tif"
MORPHOLOGY_MIP_FILENAME = "morphology_mip.ome.tif"
MORPHOLOGY_FOCUS_FILENAME = "morphology_focus.ome.tif"
MORPHOLOGY_FOCUS_DIRNAME = "morphology_focus"

CELL_FEATURE_MATRIX_H5 = "cell_feature_matrix.h5"
CELL_FEATURE_MATRIX_ZARR = "cell_feature_matrix.zarr.zip"
CELLS_ZARR = "cells.zarr.zip"
CELLS_PARQUET = "cells.parquet"
MATRIX_MTX_GZ = "matrix.mtx.gz"
BARCODES_TSV_GZ = "barcodes.tsv.gz"
FEATURES_TSV_GZ = "features.tsv.gz"

# Files/system entries that should never be treated as data when extracting.
JUNK_FILENAMES = frozenset({".DS_Store", ".end-of-run"})
APPLEDOUBLE_PREFIX = "._"

# Magic numbers used for content sniffing.
GZIP_MAGIC = b"\x1f\x8b"
APPLEDOUBLE_MAGIC = b"\x00\x05\x16\x07"


# --------------------------------------------------------------------------- #
# Essential-file mapping used by the organizer
# --------------------------------------------------------------------------- #

#: Maps the canonical filename we want in ``processed/`` to the glob patterns
#: (case-insensitive) that may match it in the extracted raw data. Order of the
#: patterns is significant: the first match wins.
ESSENTIAL_FILES: Dict[str, List[str]] = {
    "experiment.xenium": ["*experiment.xenium*", "*experiment*.json"],
    "cell_feature_matrix.h5": ["*cell_feature_matrix*.h5*"],
    "transcripts.parquet": ["*transcripts*.parquet*", "*transcripts*.csv*"],
    "cells.parquet": ["*cells*.parquet*", "*cells*.csv*"],
    "cell_boundaries.parquet": [
        "*cell_boundaries*.parquet*",
        "*cell_boundaries*.csv*",
        "*cell*boundaries*.parquet*",
    ],
    "nucleus_boundaries.parquet": [
        "*nucleus_boundaries*.parquet*",
        "*nucleus_boundaries*.csv*",
        "*nucleus*boundaries*.parquet*",
    ],
    "morphology.ome.tif": ["*morphology.ome.tif*", "*morphology*.tif*"],
    "morphology_mip.ome.tif": ["*morphology_mip*.tif*"],
    "morphology_focus.ome.tif": ["*morphology_focus*.tif*"],
    "he_image.ome.tif": [
        "*H&E*.tif*",
        "*_HE_*.tif*",
        "*_he_*.tif*",
        "*HE.tif*",
        "*he.tif*",
        "*HE_stain*.tif*",
        "*he_stain*.tif*",
        "*he*.tif*",
    ],
    "gene_panel.json": ["*gene_panel*.json*"],
    "analysis_summary.html": ["*analysis_summary*.html*"],
    "matrix.mtx.gz": ["*matrix.mtx*", "*.mtx.gz", "*.mtx.*"],
    "barcodes.tsv.gz": ["*barcodes*", "*barcodes*.txt*"],
    "features.tsv.gz": ["*features*", "*features*.txt*"],
    "cells.zarr.zip": ["*cells.zarr*"],
    "transcripts.zarr.zip": ["*transcripts.zarr*"],
    "cell_feature_matrix.zarr.zip": ["*cell_feature_matrix.zarr*"],
}

# Names of the count columns produced for ``cells.parquet``.
COUNT_COLUMNS: List[str] = [
    "transcript_counts",
    "control_probe_counts",
    "genomic_control_counts",
    "control_codeword_counts",
    "unassigned_codeword_counts",
    "deprecated_codeword_counts",
    "total_counts",
]

# Final column order enforced for generated ``cells.parquet`` files.
CELLS_PARQUET_COLUMN_ORDER: List[str] = [
    "cell_id",
    "x_centroid",
    "y_centroid",
    "transcript_counts",
    "control_probe_counts",
    "genomic_control_counts",
    "control_codeword_counts",
    "unassigned_codeword_counts",
    "deprecated_codeword_counts",
    "total_counts",
    "cell_area",
    "nucleus_area",
    "nucleus_count",
    "segmentation_method",
]

# Feature-name substrings that mark control/blank probes to filter out.
CONTROL_FEATURE_PATTERNS: List[str] = [
    "NegControl",
    "BLANK",
    "DeprecatedCodeword",
    "UnassignedCodeword",
]


def read_pixel_size(experiment_file: Path) -> float:
    """Return ``pixel_size`` from an ``experiment.xenium`` file or the default.

    This replaces the eleven inline ``0.2125`` literals scattered through the
    original monolith. It never raises: a missing or malformed file yields
    :data:`DEFAULT_PIXEL_SIZE`.
    """
    try:
        if experiment_file.exists():
            with open(experiment_file, "r") as fh:
                return float(json.load(fh).get("pixel_size", DEFAULT_PIXEL_SIZE))
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return DEFAULT_PIXEL_SIZE
