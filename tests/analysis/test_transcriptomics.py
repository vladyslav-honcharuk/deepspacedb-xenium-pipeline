"""Unit tests for TranscriptomicsProcessor in deepspacedb_xenium_pipeline.analysis.transcriptomics."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from spatialdata import SpatialData

from deepspacedb_xenium_pipeline import constants
from deepspacedb_xenium_pipeline.analysis.transcriptomics import TranscriptomicsProcessor
from deepspacedb_xenium_pipeline.config import PipelineConfig


def test_transcriptomics_process(tmp_path):
    # 20 cells, 10 genes with non-zero expression so filter_cells and filter_genes pass
    np.random.seed(42)
    X = np.random.poisson(lam=5.0, size=(20, 10)).astype(np.float32)
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(20)])
    var = pd.DataFrame(index=[f"Gene_{j}" for j in range(10)])

    adata = ad.AnnData(X=X, obs=obs, var=var)
    sdata = SpatialData(tables={"table": adata})

    proc = TranscriptomicsProcessor(PipelineConfig())
    result_sdata = proc.process(sdata, tmp_path)

    res_adata = result_sdata.tables["table"]
    assert "counts" in res_adata.layers
    assert "normalized" in res_adata.layers
    assert "log1p" in res_adata.layers
    assert "X_pca" in res_adata.obsm
    assert "X_umap" in res_adata.obsm
    assert "leiden" in res_adata.obs

    dims_file = tmp_path / constants.DIMS_FILENAME
    assert dims_file.exists()

    genes_file = tmp_path / constants.GENES_FILENAME
    assert genes_file.exists()
