# deepspacedb-xenium-pipeline

A production rewrite of the original three Xenium scripts (`xenium_collector.py`,
`xenium_downloader.py`, and the 5,197-line `xenium_transformer.py` god-class)
into **one** focused, testable, installable package with a single pipeline:

```
collect  →  download  →  transform
(NCBI GEO)  (raw files)   (raw → SpatialData → analysis/images/binning/export)
```

## The unified pipeline

One object, three stages:

```python
from pathlib import Path
from deepspacedb_xenium_pipeline import XeniumPipeline, PipelineConfig, configure_logging

configure_logging("INFO", log_file="run.log")          # explicit, opt-in
p = XeniumPipeline(PipelineConfig(base_dir=Path("/data")))

p.collect(Path("samples.csv"))                          # step 1: discover from GEO
p.download(Path("samples.csv"))                         # step 2: download files
p.transform_all(summary_file=Path("summary.csv"))       # step 3: transform all

# or all three end to end:
p.run(Path("samples.csv"))
```

CLI (installed as `xenium-pipeline`):

```bash
xenium-pipeline collect  -o samples.csv
xenium-pipeline download --samples-csv samples.csv --max-samples 5
xenium-pipeline download --gsm-id GSM8253807
xenium-pipeline transform /data --mode complete
xenium-pipeline transform --sample-dir /data/.../GSM123 --mode transform
xenium-pipeline run -o samples.csv
# global: --base-dir DIR --log-level DEBUG --log-file run.log --overwrite
```

## What changed vs. the originals

| Problem | Fix |
| --- | --- |
| Three disconnected scripts | One `XeniumPipeline` facade + one `xenium-pipeline` CLI with `collect`/`download`/`transform`/`run` |
| `xenium_transformer.py` god-class (6 domains, ~78 methods) | Decomposed into focused components behind a thin orchestrator |
| 4 incompatible error styles (`bool`/`raise`/bare `except:`/`None`) | One contract: stages raise `XeniumPipelineError` subclasses; converted to typed results at boundaries |
| Hardcoded `/home/vlad/...` BANKSY paths | `XENIUM_BANKSY_PYTHON` / `XENIUM_BANKSY_DIR` env vars |
| `0.2125` pixel size duplicated 11× | Single `constants.DEFAULT_PIXEL_SIZE` + `read_pixel_size()` |
| Three separate `logging.basicConfig` / root-logger mutations + hardcoded log files | One explicit `configure_logging()` on the package logger, with rotation |
| `spatialdata_io` monkey-patched at import time | `apply_spatialdata_patches()`, applied explicitly and idempotently |
| Unescaped path interpolation into a `python -c` BANKSY script | All paths via `argv`; success via exit code + results file |
| Downloader config `mkdir`'d `data3/` on construction | No side effects; `data_dir` is a derived property, dirs created on demand |
| `print()` + emoji throughout (incl. logs) | Library never prints; emoji confined to CLI; logs are ASCII |
| Near-duplicate result dataclasses | Shared `BaseResult` (incl. `DownloadResult`) |
| `transform_he` imported from `PYTHONPATH` | Bundled at `imaging/transform_he.py`, imported relatively |
| Debug methods shipped in production | Removed |
| `write_summary_csv` wrote only the last row (indentation bug) | Fixed: one row per sample |

## Architecture

```
deepspacedb_xenium_pipeline/
├── constants.py          # magic values (pixel size, filenames, file map)
├── config.py             # PipelineConfig — no side effects on construction
├── logging_setup.py      # explicit configure_logging() + get_logger()
├── exceptions.py         # XeniumPipelineError hierarchy
├── results.py            # BaseResult + typed stage results
├── compat.py             # opt-in spatialdata_io patch
├── geometry.py, io_utils.py, sdata_utils.py
├── collector.py          # STAGE 1: SampleCollector (NCBI GEO discovery)
├── downloader.py         # STAGE 2: SampleDownloader (MiniML → raw files)
├── extraction/           # STAGE 3a: ArchiveExtractor, FileOrganizer, FileRepair
├── loaders/              # STAGE 3b: TableLoader (h5/zarr/mtx), CellsParquetBuilder, SpatialDataBuilder
├── imaging/              # STAGE 3c: MorphologyProcessor, HEProcessor, ImageRenderer, transform_he
├── analysis/             # STAGE 3d: TranscriptomicsProcessor, BanksyRunner, Binner, SingleCellExporter
├── summary.py            # SampleAnalyzer + CSV reporting
├── transformer.py        # XeniumTransformer — stage-3 engine (raw → processed → analysis)
├── pipeline.py           # XeniumPipeline — the unified facade (the public API)
└── cli.py                # argparse entrypoint (the only place that prints / configures logging)
```

Dependencies flow one way; every component takes its collaborators by
constructor injection, so each is unit-testable in isolation.

## Project layout

This uses the standard **src layout**: the distribution name is hyphenated
(`deepspacedb-xenium-pipeline`), the import package is underscored and lives
under `src/`, and tests live outside the package. Putting the code under `src/`
means it must be *installed* to be importable, so tests always run against the
installed package and packaging mistakes surface immediately.

```
deepspacedb-xenium-pipeline/      # repo / distribution name
├── pyproject.toml                # build config, deps, entry point, tooling
├── LICENSE  README.md  CHANGELOG.md
├── .github/workflows/ci.yml      # lint + test on push/PR
├── src/
│   └── deepspacedb_xenium_pipeline/   # the import package (+ py.typed)
└── tests/
```

## Install

Install directly from GitHub:

```bash
pip install "git+https://github.com/DeepSpaceDB/deepspacedb-xenium-pipeline.git"
# with optional extras:
pip install "deepspacedb-xenium-pipeline[plot,haystack] @ git+https://github.com/DeepSpaceDB/deepspacedb-xenium-pipeline.git"
```

For local development (editable install + dev tools):

```bash
git clone https://github.com/DeepSpaceDB/deepspacedb-xenium-pipeline.git
cd deepspacedb-xenium-pipeline
pip install -e ".[plot,haystack,dev]"
pytest
```

Optional extras: `plot` (spatialdata-plot), `haystack` (singleCellHaystack),
`dev` (pytest/pytest-cov/mypy/ruff).

## BANKSY

BANKSY runs in its **own interpreter / Python version**, so it stays behind a
subprocess boundary and is *not* bundled. Point the pipeline at it via env vars:

```bash
export XENIUM_BANKSY_PYTHON=/opt/banksy-env/bin/python
export XENIUM_BANKSY_DIR=/opt/xenium_mundus       # contains run_banksy_all.py
export XENIUM_BANKSY_TIMEOUT=3600                  # optional, seconds
```

The child is called as `run_banksy_on_sample(Path(sample_dir), top_genes=N)`; if
the env vars are unset, BANKSY is skipped with a warning and the rest of the
pipeline runs.

## Behavioral fidelity

The scientific processing logic — GEO traversal, MiniML parsing, cell-ID
alignment, h5/zarr/mtx fallbacks, focus-plane detection, version-mismatch
handling, SpatialData build strategies, H&E warping — is ported verbatim. The
changes are structural, not numerical.
