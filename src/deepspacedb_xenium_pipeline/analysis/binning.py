"""Spatial binning of normalized expression into multi-resolution zarr arrays."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import scipy.sparse as sp
import zarr

from ..config import PipelineConfig
from ..logging_setup import get_logger
from ..sdata_utils import expression_and_coords


def bin_one_size(
    gene_expression,
    gene_names: Sequence[str],
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    bin_size: int,
    out_path_str: str,
) -> List[str]:
    """Bin expression at one resolution and write both zarr stores. Picklable for workers."""
    logger = get_logger(f"{__name__}.bin_size_{bin_size}")
    out_path = Path(out_path_str)
    zarr_filename = f"bins_size_{bin_size}.zarr.zip"
    spatial_filename = f"bins_size_{bin_size}_spatial.zarr.zip"
    bin_path = out_path / "zarr"
    zarr_path = bin_path / zarr_filename
    spatial_path = bin_path / spatial_filename
    if zarr_path.exists() and spatial_path.exists():
        logger.info("Bin size %d exists; skipping", bin_size)
        return [zarr_filename, spatial_filename]

    logger.info("Processing bin size %d", bin_size)
    try:
        x_bins = np.arange(bin_size, x_coords.max() + bin_size, bin_size)
        y_bins = np.arange(bin_size, y_coords.max() + bin_size, bin_size)
    except (ValueError, FloatingPointError) as exc:
        logger.error("Failed to create bins for size %d: %s", bin_size, exc)
        return []

    x_idx = np.digitize(x_coords, x_bins)
    y_idx = np.digitize(y_coords, y_bins)
    n_bins_x = int(x_idx.max()) + 1
    n_bins_y = int(y_idx.max()) + 1

    bin_path.mkdir(parents=True, exist_ok=True)
    store = zarr.storage.ZipStore(str(zarr_path), mode="w")
    arr = zarr.create(
        store=store,
        mode="w",
        shape=(len(gene_names), n_bins_x, n_bins_y),
        chunks=(1, n_bins_x, n_bins_y),
        dtype="f4",
    )

    # Aggregate every cell's expression into its spatial bin, for all genes at
    # once, as a single sparse matrix multiply rather than a per-gene Python
    # loop (which did a full-column ``.toarray()`` of the n_cells-row matrix
    # plus a np.unique/dict-lookup scan per gene - O(genes x cells)
    # Python-level work, the reason small bin sizes took days).
    # ``indicator[bin, cell]`` is 1 iff that cell falls in that bin (each cell
    # belongs to exactly one bin), so ``(indicator @ expression)[bin, gene]`` is
    # the sum over the cells in that bin: exactly what the nested loop computed.
    logger.info("Aggregating %d cells into %dx%d bins via sparse matmul", len(x_coords), n_bins_x, n_bins_y)
    n_cells = len(x_coords)
    bin_id = x_idx.astype(np.int64) * n_bins_y + y_idx.astype(np.int64)
    indicator = sp.csr_matrix(
        (np.ones(n_cells, dtype=np.float32), (bin_id, np.arange(n_cells))),
        shape=(n_bins_x * n_bins_y, n_cells),
    )
    binned = (indicator @ sp.csr_matrix(gene_expression)).tocsc()  # (n_bins, n_genes)

    logger.info("Writing gene planes")
    # Keep every gene's plane in memory (already computed, and no larger than
    # the output files themselves) so the spatial-optimized copy below writes
    # from memory instead of reading and decompressing zarr_path all over again.
    full = np.empty((len(gene_names), n_bins_x, n_bins_y), dtype=np.float32)
    for gene_idx, gene_name in enumerate(gene_names):
        if gene_idx % 100 == 0:
            logger.info("Gene %d/%d: %s", gene_idx, len(gene_names), gene_name)
        plane = np.asarray(binned[:, gene_idx].todense(), dtype=np.float32).reshape(n_bins_x, n_bins_y)
        np.round(plane, decimals=3, out=plane)
        arr[gene_idx, :, :] = plane
        full[gene_idx] = plane
    store.close()

    # Same data, re-chunked as (n_genes, 100, 100) tiles instead of one chunk
    # per gene plane, written straight from ``full``.
    logger.info("Creating spatial-optimized version for bin size %d", bin_size)
    spatial_store = zarr.storage.ZipStore(str(spatial_path), mode="w")
    spatial_arr = zarr.create(
        store=spatial_store,
        mode="w",
        shape=full.shape,
        chunks=(full.shape[0], 100, 100),
        dtype="f4",
    )
    spatial_arr[:, :, :] = full
    spatial_store.close()
    logger.info("Completed bin size %d", bin_size)
    return [zarr_filename, spatial_filename]


class Binner:
    """Create binned expression arrays at the configured resolutions."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.logger = get_logger(__name__)

    def bin_to_zarr(self, sdata, out_path: Path) -> List[str]:
        if self.config.skip_binning:
            self.logger.info("skip_binning=True - not generating binned expression stores")
            return []
        self.logger.info("Starting binning")
        prep = expression_and_coords(sdata, self.logger)
        if prep is None:
            return []
        gene_expression, gene_names, x_coords, y_coords = prep
        created: List[str] = []
        for bin_size in self.config.bin_sizes:
            created += bin_one_size(gene_expression, gene_names, x_coords, y_coords, bin_size, str(out_path))
        self.logger.info("Binning complete: %d files", len(created))
        return created
