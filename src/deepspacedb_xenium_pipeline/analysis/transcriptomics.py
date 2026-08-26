"""Transcriptomics processing: QC, normalization, clustering, UMAP, and Leiden."""

from __future__ import annotations

from pathlib import Path

import scanpy as sc

from .. import constants
from ..config import PipelineConfig
from ..logging_setup import get_logger
from ..sdata_utils import get_table, set_table


class TranscriptomicsProcessor:
    """Run the single-cell analysis steps and write side-car CSVs."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.logger = get_logger(__name__)

    def process(self, sdata, base_dir: Path):
        """Process the table in ``sdata`` in place and return ``sdata``."""
        self.logger.info("Processing transcriptomics")
        adata = get_table(sdata)

        nan_mask = adata.var_names.isna() | (adata.var_names.astype(str) == "nan")
        if nan_mask.any():
            self.logger.warning("Removing %d genes with NaN names", int(nan_mask.sum()))
            adata = adata[:, ~nan_mask].copy()
            set_table(sdata, adata)

        self.logger.info("Filtering cells and genes")
        sc.pp.filter_cells(adata, min_counts=10)
        sc.pp.filter_genes(adata, min_cells=5)

        dims = adata.shape
        (base_dir / constants.DIMS_FILENAME).write_text(f"{dims[0]},{dims[1]}\n")
        self.logger.info("Saved dimensions %s", dims)

        self.logger.info("Storing raw counts and computing QC metrics")
        adata.layers["counts"] = adata.X.copy()
        sc.pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)

        self.logger.info("Normalizing, scaling, PCA")
        sc.pp.normalize_total(adata)
        adata.layers["normalized"] = adata.X.copy()
        sc.pp.log1p(adata)
        adata.layers["log1p"] = adata.X.copy()
        sc.pp.pca(adata)

        self.logger.info("Neighbors, UMAP, Leiden")
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        sc.tl.leiden(adata)

        genes = adata.var.index
        genes.to_frame().to_csv(base_dir / constants.GENES_FILENAME, index=False, header=False)
        self.logger.info("Saved %d gene names", len(genes))
        return sdata
