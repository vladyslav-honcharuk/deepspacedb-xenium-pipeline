"""Command-line interface for the unified Xenium pipeline.

This is the *application* layer: the only place that configures logging (via
:func:`configure_logging`) and owns user-facing stdout. Library code never
prints, and emoji/unicode are confined to this module - never the logs.

Subcommands map to the pipeline stages::

    xenium-pipeline collect   [-o samples.csv]
    xenium-pipeline download  (--samples-csv CSV | --gsm-id GSM) [--max-samples N]
    xenium-pipeline transform (DATA_DIR | --sample-dir DIR) [--sample NAME] [--mode ...]
    xenium-pipeline run       -o samples.csv [--max-samples N]
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .config import PipelineConfig
from .logging_setup import configure_logging
from .summary import SampleSummary, write_summary_csv


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xenium-pipeline",
        description="Collect, download, and transform 10x Xenium data into SpatialData.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-dir", type=Path, default=Path.cwd(),
                        help="Root for the data/ tree (default: cwd)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-file", type=Path, help="Optional rotating log file path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    sub = parser.add_subparsers(dest="command", required=True)

    p_collect = sub.add_parser("collect", help="Discover Xenium samples from NCBI GEO")
    p_collect.add_argument("-o", "--output", type=Path, default=Path("xenium_samples.csv"))

    p_download = sub.add_parser("download", help="Download sample files from NCBI GEO")
    p_download.add_argument("--samples-csv", type=Path, help="CSV of samples to download")
    p_download.add_argument("--gsm-id", type=str, help="Single GSM id to download")
    p_download.add_argument("--max-samples", type=int, help="Limit number of samples (testing)")

    p_transform = sub.add_parser("transform", help="Transform downloaded samples")
    p_transform.add_argument("data_dir", nargs="?", type=Path, help="Base dir of sample subdirs")
    p_transform.add_argument("--sample", type=str, help="Process only this sample by name")
    p_transform.add_argument("--sample-dir", type=Path, help="Process a single sample directory")
    p_transform.add_argument(
        "--mode", default="complete",
        choices=["raw", "transform", "complete", "images-haystack", "inject-he"],
    )
    p_transform.add_argument("--summary-only", action="store_true")
    p_transform.add_argument("--summary-file", type=Path)
    p_transform.add_argument("--skip-focus-copy", action="store_true")
    p_transform.add_argument("--process-zarr-directly", action="store_true")

    p_run = sub.add_parser("run", help="Collect -> download -> transform end to end")
    p_run.add_argument("-o", "--output", type=Path, default=Path("xenium_samples.csv"))
    p_run.add_argument("--max-samples", type=int)
    p_run.add_argument("--summary-file", type=Path)

    return parser


def _timestamped(prefix: str) -> Path:
    return Path(f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level, log_file=args.log_file)

    # Heavy imports happen here so `--help`/arg errors stay fast.
    from .pipeline import XeniumPipeline

    config = PipelineConfig(
        base_dir=args.base_dir,
        overwrite_existing=args.overwrite or (getattr(args, "mode", None) == "inject-he"),
        skip_focus_copy=getattr(args, "skip_focus_copy", False),
        process_zarr_directly=getattr(args, "process_zarr_directly", False),
        skip_zarr_exports=(getattr(args, "mode", None) == "inject-he"),
    )
    pipeline = XeniumPipeline(config)

    if args.command == "collect":
        return _cmd_collect(pipeline, args)
    if args.command == "download":
        return _cmd_download(pipeline, args)
    if args.command == "transform":
        return _cmd_transform(pipeline, args)
    if args.command == "run":
        return _cmd_run(pipeline, args)
    parser.error(f"Unknown command: {args.command}")
    return 2


# --------------------------------------------------------------------------- #
# collect / download / run
# --------------------------------------------------------------------------- #


def _cmd_collect(pipeline, args) -> int:
    print("Collecting Xenium samples from NCBI GEO...")
    df = pipeline.collect(args.output)
    if df.empty:
        print("[FAIL] No Xenium samples found")
        return 1
    print(f"[OK] Collected {len(df)} samples -> {args.output}")
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
    print(f"[{'OK' if successful == len(results) else 'PARTIAL'}] "
          f"Downloaded {successful}/{len(results)} samples")
    failed = [g for g, r in results.items() if not r.success]
    if failed:
        print(f"Failed: {', '.join(failed)}")
    return 0 if successful == len(results) else 1


def _cmd_run(pipeline, args) -> int:
    print("Running full pipeline: collect -> download -> transform...")
    results, _ = pipeline.run(args.output, max_samples=args.max_samples, summary_file=args.summary_file)
    return _report(results, "RUN")


# --------------------------------------------------------------------------- #
# transform (preserves the original modes)
# --------------------------------------------------------------------------- #


def _validate_transform(args) -> Optional[str]:
    if not args.data_dir and not args.sample_dir:
        return "Must specify either data_dir or --sample-dir"
    if args.sample_dir and args.data_dir:
        return "Cannot specify both data_dir and --sample-dir"
    if args.sample and not args.data_dir:
        return "--sample requires data_dir"
    if args.summary_only and args.sample_dir:
        return "--summary-only only works with data_dir"
    return None


def _dispatch_single(pipeline, sample_dir: Path, mode: str):
    if mode == "raw":
        return pipeline.process_raw_data(sample_dir), "raw"
    if mode == "transform":
        return pipeline.transform_sample(sample_dir), "spatialdata"
    if mode == "images-haystack":
        return pipeline.process_images_and_haystack_only(sample_dir), "complete"
    return pipeline.process_sample_complete(sample_dir), "complete"


def _cmd_transform(pipeline, args) -> int:
    err = _validate_transform(args)
    if err:
        print(f"[FAIL] {err}")
        return 2
    analyzer = pipeline.transformer.analyzer

    if args.sample_dir:
        print(f"Processing single sample: {args.sample_dir}")
        result, _ = _dispatch_single(pipeline, args.sample_dir, args.mode)
        print(f"[OK] {args.sample_dir}" if result.success else f"[FAIL] {result.error_message}")
        return 0 if result.success else 1

    if args.summary_only:
        sample_dirs = pipeline.discover_samples(args.data_dir)
        if not sample_dirs:
            print("[FAIL] No samples found with raw/ directories")
            return 1
        print(f"Generating summary for {len(sample_dirs)} samples...")
        summaries = [analyzer.analyze(d) for d in sample_dirs]
        out = args.summary_file or _timestamped("xenium_summary")
        write_summary_csv(summaries, out)
        print(f"[OK] Summary written to: {out}")
        return 0

    if args.sample:
        sample_dir = args.data_dir / args.sample
        if not sample_dir.exists():
            print(f"[FAIL] Sample not found: {sample_dir}")
            return 1
        print(f"Processing sample '{args.sample}' in {args.mode} mode...")
        summary = analyzer.analyze(sample_dir)
        result, stage = _dispatch_single(pipeline, sample_dir, args.mode)
        analyzer.update_with_result(summary, result, stage)
        analyzer.refresh_outputs(sample_dir, summary)
        out = args.summary_file or sample_dir / f"{args.sample}_summary.csv"
        write_summary_csv([summary], out)
        print(f"[OK] {args.sample}" if result.success else f"[FAIL] {result.error_message}")
        print(f"Summary: {out}")
        return 0 if result.success else 1

    return _transform_multiple(pipeline, args, analyzer)


def _transform_multiple(pipeline, args, analyzer) -> int:
    sample_dirs = pipeline.discover_samples(args.data_dir)
    if not sample_dirs:
        print("[FAIL] No samples found with raw/ directories")
        return 1
    print(f"Processing {len(sample_dirs)} samples in {args.mode} mode...")

    if args.mode == "complete":
        results, _ = pipeline.transformer.process_multiple_complete_with_summary(
            sample_dirs, args.summary_file
        )
    elif args.mode in ("images-haystack", "inject-he"):
        if args.mode == "images-haystack":
            results = pipeline.transformer.process_multiple_images_haystack(sample_dirs)
            prefix = "xenium_summary_images_haystack"
        else:
            results = pipeline.transformer.process_multiple_inject_he(sample_dirs)
            prefix = "xenium_summary_inject_he"
        summaries = []
        for sample_dir in sample_dirs:
            summary = analyzer.analyze(sample_dir)
            if sample_dir.name in results:
                analyzer.update_with_result(summary, results[sample_dir.name], "complete")
            summaries.append(summary)
        write_summary_csv(summaries, args.summary_file or _timestamped(prefix))
    else:  # raw / transform
        results = {}
        summaries: List[SampleSummary] = []
        stage = "raw" if args.mode == "raw" else "spatialdata"
        for i, sample_dir in enumerate(sample_dirs, 1):
            print(f"[{i}/{len(sample_dirs)}] Processing {sample_dir.name}...")
            summary = analyzer.analyze(sample_dir)
            result, _ = _dispatch_single(pipeline, sample_dir, args.mode)
            analyzer.update_with_result(summary, result, stage)
            analyzer.refresh_outputs(sample_dir, summary)
            results[sample_dir.name] = result
            summaries.append(summary)
            print("   [OK]" if result.success else f"   [FAIL] {result.error_message}")
        write_summary_csv(summaries, args.summary_file or _timestamped(f"xenium_summary_{args.mode}"))

    return _report(results, args.mode.upper())


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
