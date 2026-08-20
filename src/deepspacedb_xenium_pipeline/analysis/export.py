"""Export single-cell expression in multiple sparse layouts plus coordinates."""
from __future__ import annotations

from pathlib import Path
from typing import List

import zarr

from ..config import PipelineConfig
from ..logging_setup import get_logger
from ..sdata_utils import aligned_cell_circles, get_table


class SingleCellExporter:
    """Write CSR/CSC/per-gene sparse expression and cell coordinates to zarr."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.logger = get_logger(__name__)

    def export(self, sdata, out_path: Path) -> List[str]:
        self.logger.info("Exporting single-cell expression data")
        adata = get_table(sdata)
        gene_expression = adata.layers["normalized"]
        gene_names = adata.var_names

        circles = aligned_cell_circles(sdata, adata, self.logger)
        x_coords = circles.geometry.x
        y_coords = circles.geometry.y

        bin_path = out_path / "zarr"
        bin_path.mkdir(parents=True, exist_ok=True)
        created: List[str] = []

        created.append(self._write_csr(bin_path, gene_expression))
        gene_expression_csc = gene_expression.tocsc()
        created.append(self._write_csc(bin_path, gene_expression_csc))
        created.append(self._write_per_gene(bin_path, gene_expression, gene_names, gene_expression_csc))
        created.append(self._write_coordinates(bin_path, x_coords, y_coords))

        self.logger.info("Single-cell export complete: %d files", len(created))
        return created

    def _write_csr(self, bin_path: Path, expr) -> str:
        self.logger.info("Creating CSR format")
        zarr_file = bin_path / "sparse_gene_expression_csr.zarr.zip"
        store = zarr.storage.ZipStore(str(zarr_file), mode="w")
        root = zarr.group(store=store)
        root.create_dataset("data", data=expr.data)
        root.create_dataset("indices", data=expr.indices)
        root.create_dataset("indptr", data=expr.indptr)
        root.attrs["shape"] = expr.shape
        store.close()
        return zarr_file.name

    def _write_csc(self, bin_path: Path, expr_csc) -> str:
        self.logger.info("Creating CSC format")
        zarr_file = bin_path / "sparse_gene_expression_csc.zarr.zip"
        store = zarr.storage.ZipStore(str(zarr_file), mode="w")
        root = zarr.group(store=store)
        root.create_dataset("data", data=expr_csc.data)
        root.create_dataset("indices", data=expr_csc.indices)
        root.create_dataset("indptr", data=expr_csc.indptr)
        root.attrs["shape"] = expr_csc.shape
        store.close()
        return zarr_file.name

    def _write_per_gene(self, bin_path: Path, expr, gene_names, expr_csc) -> str:
        self.logger.info("Creating per-gene chunked format")
        zarr_file = bin_path / "sparse_gene_expression_chunked_per_gene.zarr.zip"
        store = zarr.storage.ZipStore(str(zarr_file), mode="w")
        root = zarr.group(store=store)
        for gene_idx, gene_name in enumerate(gene_names):
            if gene_idx % 500 == 0:
                self.logger.info("Gene %d/%d", gene_idx + 1, len(gene_names))
            gene_expr_csc = expr.getcol(gene_idx).tocsc()
            root.create_dataset(f"data_{gene_name}", data=gene_expr_csc.data, chunks=True, compression="blosc")
            root.create_dataset(f"indices_{gene_name}", data=gene_expr_csc.indices, chunks=True, compression="blosc")
            root.create_dataset(f"indptr_{gene_name}", data=gene_expr_csc.indptr, chunks=True, compression="blosc")
        root.attrs["shape"] = expr_csc.shape
        store.close()
        return zarr_file.name

    def _write_coordinates(self, bin_path: Path, x_coords, y_coords) -> str:
        self.logger.info("Saving cell coordinates")
        zarr_file = bin_path / "cell_coordinates.zarr.zip"
        store = zarr.storage.ZipStore(str(zarr_file), mode="w")
        root = zarr.group(store=store)
        root.create_dataset("x_coords", data=x_coords.values)
        root.create_dataset("y_coords", data=y_coords.values)
        store.close()
        return zarr_file.name
