"""Run independent bin-size and single-cell zarr writes concurrently."""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import PipelineConfig
from ..logging_setup import get_logger
from ..sdata_utils import expression_and_coords
from .binning import bin_one_size
from .export import export_single_cell

_logger = get_logger(__name__)


def run_binning_and_export_parallel(sdata, out_path: Path, config: PipelineConfig) -> Tuple[List[str], List[str]]:
    """Dispatch each bin size and the single-cell export as separate processes."""
    _logger.info("Starting binning + single-cell export (parallel)")
    prep = expression_and_coords(sdata, _logger)
    if prep is None:
        return [], []
    gene_expression, gene_names, x_coords, y_coords = prep
    out_path_str = str(out_path)

    bin_sizes: List[int] = [] if config.skip_binning else list(config.bin_sizes)
    if config.skip_binning:
        _logger.info("skip_binning=True - not generating binned expression stores")
    n_jobs = len(bin_sizes) + 1
    max_workers = config.zarr_export_workers or n_jobs

    bins_created: List[str] = []
    single_cell_files: List[str] = []

    ctx: mp.context.BaseContext
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
        future_to_job: Dict[Any, Tuple[str, Optional[int]]] = {}
        for bin_size in bin_sizes:
            fut = pool.submit(bin_one_size, gene_expression, gene_names, x_coords, y_coords, bin_size, out_path_str)
            future_to_job[fut] = ("bin", bin_size)
        fut = pool.submit(export_single_cell, gene_expression, gene_names, x_coords, y_coords, out_path_str)
        future_to_job[fut] = ("single_cell", None)

        for fut in as_completed(future_to_job):
            job_type, job_arg = future_to_job[fut]
            try:
                files = fut.result()
            except Exception as exc:  # noqa: BLE001 - one job must not abort the rest
                label = f"bin_size={job_arg}" if job_type == "bin" else "single_cell_export"
                _logger.exception("Parallel job failed (%s): %s", label, exc)
                continue
            if job_type == "bin":
                bins_created.extend(files)
                _logger.info("Finished bin size %s", job_arg)
            else:
                single_cell_files.extend(files)
                _logger.info("Finished single-cell export")

    _logger.info(
        "Binning + single-cell export complete: %d bin files, %d single-cell files",
        len(bins_created),
        len(single_cell_files),
    )
    return bins_created, single_cell_files
