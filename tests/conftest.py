"""Shared pytest fixtures and synthetic data generators for Xenium pipeline tests."""

from __future__ import annotations

import gzip
import io
import json
import tarfile
from pathlib import Path

import pandas as pd

try:
    import pytest

    fixture = pytest.fixture
except ImportError:

    def fixture(fn):
        return fn


@fixture
def mock_miniml_xml() -> str:
    """Return a minimal valid MiniML XML response for a GSM sample."""
    return """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<MINiML
   xmlns="http://www.ncbi.nlm.nih.gov/geo/info/MINiML"
   version="0.5.0">
  <Sample iid="GSM8253807">
    <Status database="GEO">
      <Submission-Date>2024-04-10</Submission-Date>
      <Release-Date>2024-04-12</Release-Date>
      <Last-Update-Date>2024-04-12</Last-Update-Date>
    </Status>
    <Title>Human Breast Cancer Sample 1</Title>
    <Accession database="GEO">GSM8253807</Accession>
    <Type>SRA</Type>
    <Channel-Count>1</Channel-Count>
    <Channel position="1">
      <Source>Primary Breast Tissue</Source>
      <Organism taxid="9606">Homo sapiens</Organism>
    </Channel>
    <Supplementary-Data type="tar">
      ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8253nnn/GSM8253807/suppl/GSM8253807_outs.tar.gz
    </Supplementary-Data>
    <Supplementary-Data type="csv">
      ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8253nnn/GSM8253807/suppl/GSM8253807_cells.csv.gz
    </Supplementary-Data>
  </Sample>
</MINiML>
"""


@fixture
def mock_samples_csv(tmp_path: Path) -> Path:
    """Create a mock samples.csv file with platform, series, sample columns."""
    csv_path = tmp_path / "samples.csv"
    df = pd.DataFrame(
        [
            {"platform": "GPL31942", "series": "GSE263881", "sample": "GSM8253807"},
            {"platform": "GPL31942", "series": "GSE263881", "sample": "GSM8253808"},
        ]
    )
    df.to_csv(csv_path, index=False)
    return csv_path


@fixture
def synthetic_raw_sample_dir(tmp_path: Path) -> Path:
    """Create a sample directory with a mock raw/ directory containing a tar.gz."""
    sample_dir = tmp_path / "GPL31942" / "GSE263881" / "GSM8253807"
    raw_dir = sample_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Create mock tar.gz containing canonical Xenium files
    archive_path = raw_dir / "GSM8253807_outs.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        # experiment.xenium
        exp_data = json.dumps({"major_version": 2, "minor_version": 0, "pixel_size": 0.2125}).encode("utf-8")
        ti = tarfile.TarInfo(name="experiment.xenium")
        ti.size = len(exp_data)
        tar.addfile(ti, io.BytesIO(exp_data))

        # dummy transcripts.parquet
        df_transcripts = pd.DataFrame(
            {
                "feature_name": ["GeneA", "GeneB"],
                "x_location": [10.0, 20.0],
                "y_location": [15.0, 25.0],
                "z_location": [0.0, 0.0],
                "qv": [40.0, 40.0],
                "cell_id": [1, 2],
            }
        )
        pq_bytes = io.BytesIO()
        try:
            df_transcripts.to_parquet(pq_bytes)
            pq_data = pq_bytes.getvalue()
        except Exception:
            pq_data = b"MOCK_PARQUET_DATA"
        ti2 = tarfile.TarInfo(name="transcripts.parquet")
        ti2.size = len(pq_data)
        tar.addfile(ti2, io.BytesIO(pq_data))

    return sample_dir


@fixture
def synthetic_processed_dir(tmp_path: Path) -> Path:
    """Create a sample directory with a canonical processed/ directory."""
    sample_dir = tmp_path / "sample_1"
    proc_dir = sample_dir / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    # Write experiment.xenium
    with open(proc_dir / "experiment.xenium", "w") as f:
        json.dump({"major_version": 2, "minor_version": 0, "pixel_size": 0.2125}, f)

    # Write dummy cell_feature_matrix directory (MEX)
    mex_dir = proc_dir / "cell_feature_matrix"
    mex_dir.mkdir(exist_ok=True)
    with gzip.open(mex_dir / "features.tsv.gz", "wt") as f:
        f.write("ENSG0001\tGeneA\tGene Expression\nENSG0002\tGeneB\tGene Expression\n")
    with gzip.open(mex_dir / "barcodes.tsv.gz", "wt") as f:
        f.write("cell_1\ncell_2\n")
    with gzip.open(mex_dir / "matrix.mtx.gz", "wt") as f:
        f.write("%%MatrixMarket matrix coordinate integer general\n2 2 2\n1 1 5\n2 2 10\n")

    return sample_dir
