# DeepSpaceDB Xenium Pipeline

A command-line pipeline that finds, downloads, repairs, and processes 10x Xenium
spatial-transcriptomics datasets from NCBI GEO into standardized
[`SpatialData`](https://spatialdata.scverse.org/) objects. It is the data-ingestion
engine behind the [DeepSpaceDB](https://genomics.virus.kyoto-u.ac.jp/deepspacedb/)
database 2.0 update, and runs the full path from a GEO accession to analysis-ready output in
four stages: `find`, `download`, `salvage`, and `process`.

## Installation

Using Conda (recommended):

```bash
conda create -n xenium_pipeline python=3.10 -y
conda activate xenium_pipeline
pip install "git+https://github.com/vladyslav-honcharuk/deepspacedb-xenium-pipeline.git"
```

For development:

```bash
git clone https://github.com/vladyslav-honcharuk/deepspacedb-xenium-pipeline.git
cd deepspacedb-xenium-pipeline
pip install -e ".[dev,plot]"
pytest
```

Requires Python >= 3.10 on Linux, macOS, or Windows (WSL2 recommended).

## Usage

Run the four stages from the command line:

```bash
# 1. Find every Xenium sample on NCBI GEO
xenium-pipeline find -o xenium_samples.csv

# 2. Download all samples in the CSV (or cap with --max-samples, or a single --gsm-id)
#    Files land in ./data/{GPL}/{GSE}/{GSM}/raw/ (./data is base_dir/data; base_dir defaults to cwd)
xenium-pipeline download --samples-csv xenium_samples.csv
xenium-pipeline download --samples-csv xenium_samples.csv --max-samples 10
xenium-pipeline download --gsm-id GSM8253807 --samples-csv xenium_samples.csv

# 3. Repair every raw upload found under the given path into a canonical processed/ layout
xenium-pipeline salvage ./data                                  # all samples under ./data
xenium-pipeline salvage ./data/GPL33896/GSE280376/GSM9460460    # just one sample

# 4. Process the samples found under the given path into SpatialData, images, and Zarr exports
xenium-pipeline process ./data                                  # all samples under ./data
xenium-pipeline process ./data/GPL33896/GSE280376/GSM9460460    # just one sample

# Or run all four end-to-end (find -> download -> salvage -> process)
xenium-pipeline run -o xenium_samples.csv --max-samples 10
```

`./data` is created by the download stage. `salvage` and `process` recursively
discover every sample beneath the path you give, so point them at `base_dir/data`
to run the whole batch, or at a single `{GSM}` directory to run just that one.
Set a different root with the global `--base-dir` option.

Or from Python:

```python
from pathlib import Path
from deepspacedb_xenium_pipeline import PipelineConfig, XeniumPipeline

pipeline = XeniumPipeline(PipelineConfig(base_dir=Path("./data")))

pipeline.find(output_csv=Path("samples.csv"))            # discover
pipeline.download(samples_csv=Path("samples.csv"))       # download
pipeline.process_all(summary_file=Path("summary.csv"))   # salvage + process
```

Outputs are written under `base_dir/data/{GPL}/{GSE}/{GSM}/`.

## Pipeline stages

- **find** discovers Xenium samples on NCBI GEO via E-utilities and writes them to a CSV.
- **download** fetches a sample's archives and supplementary files, driven by its GEO MiniML record.
- **salvage** repairs broken or incomplete uploads: decompresses `.tar.gz`, `.zip`, `.gz`, and `.zst`
  archives, standardizes file names, detects true H&E images, rebuilds missing `experiment.xenium`
  metadata, regenerates `cells.parquet` geometry, and reads Cell Ranger H5, AnnData H5, Xenium Zarr,
  and 10x MEX matrices.
- **process** builds a `SpatialData` object, then runs transcriptomics (QC, normalization, PCA, UMAP,
  Leiden clustering), imaging (multi-scale MIPs, focus stacks, H&E overlays, boundary masks),
  optional spatial binning, sparse Zarr export, and a per-sample QC summary.

## Reference

If you find the DeepSpaceDB database or this package useful, please consider
citing our papers:

- Honcharuk V., Zainab A., Horimoto Y., Takemoto K., Diez D., Kawaoka S., Vandenbon A.
  [DeepSpaceDB: a spatial transcriptomics atlas for interactive in-depth analysis of tissues and tissue microenvironments](https://doi.org/10.1093/nar/gkaf1117).
  *Nucleic Acids Research*, 2026.
- Honcharuk V., Takemoto K., Diez D., Kawaoka S., Vandenbon A.
  [DeepSpaceDB 2.0: an interactive spatial transcriptomics database for large-scale Xenium data exploration](https://doi.org/10.64898/2026.01.15.699623).
  bioRxiv, 2026.

```bibtex
@article{deepspacedb,
  title   = {DeepSpaceDB: a spatial transcriptomics atlas for interactive in-depth analysis of tissues and tissue microenvironments},
  author  = {Honcharuk, Vladyslav and Zainab, Afeefa and Horimoto, Yoshiya and Takemoto, Keiko and Diez, Diego and Kawaoka, Shinpei and Vandenbon, Alexis},
  journal = {Nucleic Acids Research},
  year    = {2026},
  doi     = {10.1093/nar/gkaf1117}
}

@article{deepspacedb2,
  title   = {DeepSpaceDB 2.0: an interactive spatial transcriptomics database for large-scale Xenium data exploration},
  author  = {Honcharuk, Vladyslav and Takemoto, Keiko and Diez, Diego and Kawaoka, Shinpei and Vandenbon, Alexis},
  year    = {2026},
  doi     = {10.64898/2026.01.15.699623}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
