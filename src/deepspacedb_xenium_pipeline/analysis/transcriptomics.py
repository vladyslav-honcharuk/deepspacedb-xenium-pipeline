"""Transcriptomics processing: QC, normalization, clustering, UMAP, haystack, BANKSY."""
from __future__ import annotations

import traceback
from pathlib import Path

import scanpy as sc
import spatialdata as sd

from .. import constants
from ..config import PipelineConfig
from ..logging_setup import get_logger
from ..sdata_utils import get_table, set_table
from .banksy import BanksyRunner


class TranscriptomicsProcessor:
    """Run the single-cell analysis steps and write side-car CSVs."""

    def __init__(self, config: PipelineConfig, *, banksy: BanksyRunner | None = None) -> None:
        self.config = config
        self.logger = get_logger(__name__)
        self.banksy = banksy or BanksyRunner(config)

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

        self._run_haystack(adata, base_dir)
        self._run_banksy(sdata, base_dir)

        genes = adata.var.index
        genes.to_frame().to_csv(
            base_dir / constants.GENES_FILENAME, index=False, header=False
        )
        self.logger.info("Saved %d gene names", len(genes))
        return sdata

    def run_haystack_only(self, sdata, base_dir: Path):
        """Run only the haystack step against an already-processed table."""
        self._run_haystack(get_table(sdata), base_dir)

    # ------------------------------------------------------------------ #
    # Optional analyses
    # ------------------------------------------------------------------ #

    def _run_haystack(self, adata, base_dir: Path) -> None:
        self.logger.info("Running singleCellHaystack analysis")
        try:
            import singleCellHaystack as hs
        except ImportError:
            self.logger.warning("singleCellHaystack not installed; skipping")
            return
        try:
            res = hs.haystack(adata, coord="spatial")
            top_genes = res.top_features(n=10000)
            top_genes.to_csv(base_dir / constants.HAYSTACK_RESULTS_FILENAME, index=True)
            self.logger.info("Saved top spatially variable genes")
            adata.var["haystack_kld"] = res.result["KLD"]
            adata.var["haystack_pvalue"] = res.result["pval"]
            adata.var["haystack_padj"] = res.result["pval_adj"]
            self.logger.info("Completed haystack analysis")
        except Exception as exc:  # noqa: BLE001 - optional analysis
            self.logger.error("Haystack analysis failed: %s", exc)
            self.logger.debug("Traceback: %s", traceback.format_exc())

    def _run_banksy(self, sdata, base_dir: Path) -> None:
        self.logger.info("Running BANKSY spatial domain detection")
        try:
            zarr_path = base_dir / "processed.zarr"
            if not zarr_path.exists():
                self.logger.info("Writing temporary zarr for BANKSY")
                sdata.write(str(zarr_path))
            if not self.banksy.run(base_dir, zarr_path):
                return
            self.logger.info("Reloading zarr with BANKSY results")
            updated = sd.read_zarr(str(zarr_path))
            if "banksy_domain" in updated.tables["table"].obs.columns:
                sdata.tables["table"].obs["banksy_domain"] = updated.tables["table"].obs[
                    "banksy_domain"
                ]
                self.logger.info("BANKSY domain column added to sdata")
            else:
                self.logger.warning("BANKSY succeeded but domain column not found")
            del updated
        except Exception as exc:  # noqa: BLE001 - optional analysis
            self.logger.error("BANKSY analysis failed: %s", exc)
            self.logger.debug("Traceback: %s", traceback.format_exc())
