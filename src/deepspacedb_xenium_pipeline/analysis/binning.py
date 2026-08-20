"""Spatial binning of normalized expression into multi-resolution zarr arrays."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import zarr

from ..config import PipelineConfig
from ..logging_setup import get_logger
from ..sdata_utils import aligned_cell_circles, get_table


class Binner:
    """Create binned expression arrays at the configured resolutions."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.logger = get_logger(__name__)

    def bin_to_zarr(self, sdata, out_path: Path) -> List[str]:
        self.logger.info("Starting binning")
        adata = get_table(sdata)
        gene_expression = adata.layers["normalized"]
        gene_names = adata.var_names

        circles = aligned_cell_circles(sdata, adata, self.logger)
        x_coords = circles.geometry.x
        y_coords = circles.geometry.y
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        if any(np.isnan(v) for v in (x_min, x_max, y_min, y_max)):
            self.logger.error("Invalid coordinate ranges: x=[%s,%s] y=[%s,%s]", x_min, x_max, y_min, y_max)
            return []
        if x_max <= 0 or y_max <= 0:
            self.logger.error("Invalid coordinate maxima: x=%s y=%s", x_max, y_max)
            return []

        created: List[str] = []
        for bin_size in self.config.bin_sizes:
            created += self._bin_one_size(
                bin_size, out_path, gene_expression, gene_names, x_coords, y_coords, x_max, y_max
            )
        self.logger.info("Binning complete: %d files", len(created))
        return created

    def _bin_one_size(
        self, bin_size, out_path, gene_expression, gene_names, x_coords, y_coords, x_max, y_max
    ) -> List[str]:
        zarr_filename = f"bins_size_{bin_size}.zarr.zip"
        spatial_filename = f"bins_size_{bin_size}_spatial.zarr.zip"
        bin_path = out_path / "zarr"
        zarr_path = bin_path / zarr_filename
        spatial_path = bin_path / spatial_filename
        if zarr_path.exists() and spatial_path.exists():
            self.logger.info("Bin size %d exists; skipping", bin_size)
            return [zarr_filename, spatial_filename]

        self.logger.info("Processing bin size %d", bin_size)
        try:
            x_bins = np.arange(bin_size, x_max + bin_size, bin_size)
            y_bins = np.arange(bin_size, y_max + bin_size, bin_size)
        except (ValueError, FloatingPointError) as exc:
            self.logger.error("Failed to create bins for size %d: %s", bin_size, exc)
            return []

        x_idx = np.digitize(x_coords, x_bins)
        y_idx = np.digitize(y_coords, y_bins)
        x_min_b, x_max_b = x_idx.min(), x_idx.max()
        y_max_b = y_idx.max()

        bin_path.mkdir(parents=True, exist_ok=True)
        store = zarr.storage.ZipStore(str(zarr_path), mode="w")
        arr = zarr.create(
            store=store, mode="w",
            shape=(len(gene_names), x_max_b + 1, y_max_b + 1),
            chunks=(1, x_max_b + 1, y_max_b + 1), dtype="f4",
        )

        self.logger.info("Mapping cells to bins")
        cells_in_bins = {}
        for i in range(x_min_b, x_max_b):
            cell_indices_i = np.flatnonzero(x_idx == i)
            y_subset = y_idx[cell_indices_i]
            for j in np.unique(y_subset):
                idx = cell_indices_i[y_subset == j]
                if len(idx) > 0:
                    cells_in_bins[f"{i}_{j}"] = idx

        self.logger.info("Aggregating genes")
        for gene_idx in range(len(gene_names)):
            if gene_idx % 100 == 0:
                self.logger.info("Gene %d/%d", gene_idx, len(gene_names))
            temp = np.zeros((x_max_b + 1, y_max_b + 1), dtype=np.float32)
            with_gene = np.flatnonzero(gene_expression[:, gene_idx].toarray() > 0)
            if len(with_gene) > 0:
                combos = np.unique(list(zip(x_idx[with_gene], y_idx[with_gene])), axis=0)
                for i, j in combos:
                    idx = cells_in_bins.get(f"{i}_{j}", [])
                    if len(idx) > 0:
                        temp[i, j] = np.sum(gene_expression[idx, gene_idx])
            arr[gene_idx, :, :] = np.round(temp, decimals=3)
        store.close()

        self.logger.info("Creating spatial-optimized version for bin size %d", bin_size)
        input_store = zarr.storage.ZipStore(str(zarr_path), mode="r")
        z = zarr.open_array(input_store, mode="r")
        spatial_store = zarr.storage.ZipStore(str(spatial_path), mode="w")
        spatial_arr = zarr.create(
            store=spatial_store, mode="w", shape=z.shape,
            chunks=(z.shape[0], 100, 100), dtype="f4",
        )
        spatial_arr[:, :, :] = z
        input_store.close()
        spatial_store.close()
        self.logger.info("Completed bin size %d", bin_size)
        return [zarr_filename, spatial_filename]
