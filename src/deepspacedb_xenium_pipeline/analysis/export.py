"""Export single-cell expression in multiple sparse layouts plus coordinates."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import zarr

from ..config import PipelineConfig
from ..logging_setup import get_logger
from ..sdata_utils import expression_and_coords


def export_single_cell(
    gene_expression,
    gene_names: Sequence[str],
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    out_path_str: str,
) -> List[str]:
    """Write CSR/CSC/per-gene sparse expression and coordinates. Picklable for workers."""
    logger = get_logger(f"{__name__}.single_cell_export")
    bin_path = Path(out_path_str) / "zarr"
    bin_path.mkdir(parents=True, exist_ok=True)
    created: List[str] = []

    logger.info("Creating CSR format")
    zarr_file = bin_path / "sparse_gene_expression_csr.zarr.zip"
    store = zarr.storage.ZipStore(str(zarr_file), mode="w")
    root = zarr.group(store=store)
    root.create_dataset("data", data=gene_expression.data)
    root.create_dataset("indices", data=gene_expression.indices)
    root.create_dataset("indptr", data=gene_expression.indptr)
    root.attrs["shape"] = gene_expression.shape
    store.close()
    created.append(zarr_file.name)

    logger.info("Creating CSC format")
    gene_expression_csc = gene_expression.tocsc()
    zarr_file = bin_path / "sparse_gene_expression_csc.zarr.zip"
    store = zarr.storage.ZipStore(str(zarr_file), mode="w")
    root = zarr.group(store=store)
    root.create_dataset("data", data=gene_expression_csc.data)
    root.create_dataset("indices", data=gene_expression_csc.indices)
    root.create_dataset("indptr", data=gene_expression_csc.indptr)
    root.attrs["shape"] = gene_expression_csc.shape
    store.close()
    created.append(zarr_file.name)

    logger.info("Creating per-gene chunked format")
    zarr_file = bin_path / "sparse_gene_expression_chunked_per_gene.zarr.zip"
    store = zarr.storage.ZipStore(str(zarr_file), mode="w")
    root = zarr.group(store=store)
    seen_gene_names: set = set()
    for gene_idx, gene_name in enumerate(gene_names):
        if gene_idx % 500 == 0:
            logger.info("Gene %d/%d", gene_idx + 1, len(gene_names))
        gene_expr_csc = gene_expression.getcol(gene_idx).tocsc()
        # Duplicate gene symbols in the panel would otherwise collide on the
        # same dataset path; disambiguate repeats with an index prefix, which
        # the reader's gene-chunk key lookup already falls back to.
        key = gene_name if gene_name not in seen_gene_names else f"{gene_idx:06d}_{gene_name}"
        seen_gene_names.add(gene_name)
        root.create_dataset(f"data_{key}", data=gene_expr_csc.data, chunks=True, compression="blosc")
        root.create_dataset(f"indices_{key}", data=gene_expr_csc.indices, chunks=True, compression="blosc")
        root.create_dataset(f"indptr_{key}", data=gene_expr_csc.indptr, chunks=True, compression="blosc")
    root.attrs["shape"] = gene_expression_csc.shape
    store.close()
    created.append(zarr_file.name)

    logger.info("Saving cell coordinates")
    zarr_file = bin_path / "cell_coordinates.zarr.zip"
    store = zarr.storage.ZipStore(str(zarr_file), mode="w")
    root = zarr.group(store=store)
    root.create_dataset("x_coords", data=np.asarray(x_coords))
    root.create_dataset("y_coords", data=np.asarray(y_coords))
    store.close()
    created.append(zarr_file.name)

    logger.info("Single-cell export complete: %d files", len(created))
    return created


class SingleCellExporter:
    """Write CSR/CSC/per-gene sparse expression and cell coordinates to zarr."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.logger = get_logger(__name__)

    def export(self, sdata, out_path: Path) -> List[str]:
        self.logger.info("Exporting single-cell expression data")
        prep = expression_and_coords(sdata, self.logger)
        if prep is None:
            return []
        gene_expression, gene_names, x_coords, y_coords = prep
        return export_single_cell(gene_expression, gene_names, x_coords, y_coords, str(out_path))
