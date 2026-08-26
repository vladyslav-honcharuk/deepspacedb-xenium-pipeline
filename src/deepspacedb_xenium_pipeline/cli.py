"""Command-line interface for the unified Xenium pipeline.

This is the *application* layer: the only place that configures logging (via
:func:`configure_logging`) and owns user-facing stdout. Library code never
prints, and emoji/unicode are confined to this module - never the logs.

Subcommands map to the pipeline stages::

    xenium-pipeline find      [-o samples.csv]
    xenium-pipeline download  (--samples-csv CSV | --gsm-id GSM) [--max-samples N]
    xenium-pipeline salvage DATA_DIR
    xenium-pipeline process DATA_DIR
    xenium-pipeline run       -o samples.csv [--max-samples N]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from .config import PipelineConfig
from .logging_setup import configure_logging

# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xenium-pipeline",
        description="Find, download, salvage, and process 10x Xenium data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-dir", type=Path, default=Path.cwd(), help="Root for the data/ tree (default: cwd)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-file", type=Path, help="Optional rotating log file path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    sub = parser.add_subparsers(dest="command", required=True)

    p_find = sub.add_parser("find", help="Discover Xenium samples from NCBI GEO")
    p_find.add_argument("-o", "--output", type=Path, default=Path("xenium_samples.csv"))

    p_download = sub.add_parser("download", help="Download sample files from NCBI GEO")
    p_download.add_argument("--samples-csv", type=Path, help="CSV of samples to download")
    p_download.add_argument("--gsm-id", type=str, help="Single GSM id to download")
    p_download.add_argument("--max-samples", type=int, help="Limit number of samples (testing)")

    p_salvage = sub.add_parser("salvage", help="Repair and organize raw uploads into processed/")
    p_salvage.add_argument("data_dir", type=Path, help="Base dir of sample subdirs")

    p_process = sub.add_parser("process", help="Process samples after salvage")
    p_process.add_argument("data_dir", type=Path, help="Base dir containing salvaged samples")
    p_process.add_argument("--summary-file", type=Path)
    p_process.add_argument(
        "--enable-binning",
        action="store_true",
        help="Write bins_size_*.zarr.zip (skipped by default)",
    )
    p_process.add_argument(
        "--zarr-export-workers",
        type=int,
        default=None,
        help="Process-pool size for bin + single-cell export (default: one worker per job)",
    )

    p_run = sub.add_parser("run", help="Find -> download -> salvage -> process end to end")
    p_run.add_argument("-o", "--output", type=Path, default=Path("xenium_samples.csv"))
    p_run.add_argument("--max-samples", type=int)
    p_run.add_argument("--summary-file", type=Path)
    p_run.add_argument(
        "--enable-binning",
        action="store_true",
        help="Write bins_size_*.zarr.zip (skipped by default)",
    )
    p_run.add_argument("--zarr-export-workers", type=int, default=None)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level, log_file=args.log_file)

    # Heavy imports happen here so `--help`/arg errors stay fast.
    from .pipeline import XeniumPipeline

    config = PipelineConfig(
        base_dir=args.base_dir,
        overwrite_existing=args.overwrite,
        skip_binning=not getattr(args, "enable_binning", False),
        zarr_export_workers=getattr(args, "zarr_export_workers", None),
    )
    pipeline = XeniumPipeline(config)

    if args.command == "find":
        return _cmd_find(pipeline, args)
    if args.command == "download":
        return _cmd_download(pipeline, args)
    if args.command == "salvage":
        return _cmd_salvage(pipeline, args)
    if args.command == "process":
        return _cmd_process(pipeline, args)
    if args.command == "run":
        return _cmd_run(pipeline, args)
    parser.error(f"Unknown command: {args.command}")
    return 2


# --------------------------------------------------------------------------- #
# find / download / run
# --------------------------------------------------------------------------- #


def _cmd_find(pipeline, args) -> int:
    print("Finding Xenium samples from NCBI GEO...")
    df = pipeline.find(args.output)
    if df.empty:
        print("[FAIL] No Xenium samples found")
        return 1
    print(f"[OK] Found {len(df)} samples -> {args.output}")
    print(f"     platforms={df['platform'].nunique()} series={df['series'].nunique()}")
    return 0


def _cmd_download(pipeline, args) -> int:
    if not args.gsm_id and not args.samples_csv:
        print("[FAIL] download needs --gsm-id or --samples-csv")
        return 1
    if args.gsm_id:
        results = pipeline.download(args.samples_csv, gsm_id=args.gsm_id)
    else:
        if not args.samples_csv.exists():
            print(f"[FAIL] CSV not found: {args.samples_csv}")
            return 1
        results = pipeline.download(args.samples_csv, max_samples=args.max_samples)
    successful = sum(1 for r in results.values() if r.success)
    print(f"[{'OK' if successful == len(results) else 'PARTIAL'}] Downloaded {successful}/{len(results)} samples")
    failed = [g for g, r in results.items() if not r.success]
    if failed:
        print(f"Failed: {', '.join(failed)}")
    return 0 if successful == len(results) else 1


def _cmd_salvage(pipeline, args) -> int:
    sample_dirs = pipeline.discover_samples(args.data_dir)
    if not sample_dirs:
        print("[FAIL] No samples found")
        return 1
    results = {sample_dir.name: pipeline.salvage_sample(sample_dir) for sample_dir in sample_dirs}
    return _report(results, "SALVAGE")


def _cmd_run(pipeline, args) -> int:
    print("Running full pipeline: find -> download -> salvage -> process...")
    results, _ = pipeline.run(args.output, max_samples=args.max_samples, summary_file=args.summary_file)
    return _report(results, "RUN")


# --------------------------------------------------------------------------- #
# process (operates after salvage)
# --------------------------------------------------------------------------- #


def _cmd_process(pipeline, args) -> int:
    results, _ = pipeline.process_all(args.data_dir, args.summary_file)
    return _report(results, "PROCESS")


def _report(results, label: str) -> int:
    successful = sum(1 for r in results.values() if r.success)
    print("=" * 50)
    print(f"FINAL SUMMARY ({label})")
    print("=" * 50)
    print(f"Success: {successful}/{len(results)} samples")
    if successful < len(results):
        failed = [name for name, r in results.items() if not r.success]
        print(f"Failed: {', '.join(failed)}")
    return 0 if successful == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
