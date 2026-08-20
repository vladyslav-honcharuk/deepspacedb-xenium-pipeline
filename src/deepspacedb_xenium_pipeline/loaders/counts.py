"""Loading MEX count matrices and synthesizing ``cells.parquet``.

``MexCountsLoader`` aggregates a 10x MEX matrix into per-cell count columns.
``CellsParquetBuilder`` reconstructs a Xenium-style ``cells.parquet`` (counts +
geometry) when one is missing, using boundary polygons or, failing that,
transcript-based centroids.
"""
from __future__ import annotations

import gzip
import traceback
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .. import constants
from ..geometry import polygon_area_centroid
from ..logging_setup import get_logger


def zero_counts_frame(index: pd.Index) -> pd.DataFrame:
    """Build a zero-filled per-cell count DataFrame indexed by ``cell_id``."""
    return pd.DataFrame(
        {col: 0 for col in constants.COUNT_COLUMNS},
        index=index,
    )


class MexCountsLoader:
    """Aggregate a 10x MEX matrix into per-feature-type count columns."""

    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    def load(self, processed_dir: Path) -> pd.DataFrame:
        """Return a per-cell counts DataFrame indexed by ``cell_id``.

        Always returns a frame covering *every* barcode (zero-filled if the
        matrix/features are missing); returns an empty frame only if barcodes
        themselves are absent.
        """
        from scipy.io import mmread
        from scipy.sparse import csc_matrix

        mtx_gz = processed_dir / constants.MATRIX_MTX_GZ
        features_gz = processed_dir / constants.FEATURES_TSV_GZ
        barcodes_gz = processed_dir / constants.BARCODES_TSV_GZ

        if not barcodes_gz.exists():
            self.logger.warning("No barcodes file; returning empty counts")
            return pd.DataFrame({"cell_id": []}).set_index("cell_id")

        try:
            bars = pd.read_csv(barcodes_gz, header=None, names=["cell_id"])
            n_cells = len(bars)
            self.logger.info("Loaded %d cells from barcodes", n_cells)
            index = pd.Index(bars["cell_id"].values, name="cell_id")

            if not mtx_gz.exists() or not features_gz.exists():
                self.logger.warning("Matrix/features missing; zero-filling all barcodes")
                return zero_counts_frame(index)

            feats = self._read_features(features_gz)
            with gzip.open(mtx_gz, "rb") as fh:
                mtx = csc_matrix(mmread(fh).tocsr())
            feats["feature_type"] = feats["feature_type"].astype(str)

            def sum_rows(ftype: str) -> np.ndarray:
                idx = np.where(feats["feature_type"].values == ftype)[0]
                if len(idx) == 0:
                    return np.zeros(n_cells, dtype=np.int64)
                return np.asarray(mtx[idx, :].sum(axis=0)).ravel()

            counts = pd.DataFrame(
                {
                    "transcript_counts": sum_rows("Gene Expression"),
                    "control_probe_counts": sum_rows("Negative Control Probe"),
                    "control_codeword_counts": sum_rows("Negative Control Codeword"),
                    "genomic_control_counts": sum_rows("Genomic Control"),
                    "unassigned_codeword_counts": sum_rows("Unassigned Codeword"),
                    "deprecated_codeword_counts": sum_rows("Deprecated Codeword"),
                }
            )
            counts["cell_id"] = bars["cell_id"].values
            counts = counts.set_index("cell_id")
            counts["total_counts"] = counts[
                [
                    "transcript_counts",
                    "control_probe_counts",
                    "control_codeword_counts",
                    "genomic_control_counts",
                    "unassigned_codeword_counts",
                ]
            ].sum(axis=1)
            self.logger.debug("Loaded MEX counts for %d cells", len(counts))
            return counts
        except Exception as exc:  # noqa: BLE001 - degrade to zero counts
            self.logger.error("Failed to load MEX counts: %s", exc)
            self.logger.debug("Traceback: %s", traceback.format_exc())
            try:
                bars = pd.read_csv(barcodes_gz, header=None, names=["cell_id"])
                self.logger.warning("Returning zero counts for %d cells", len(bars))
                return zero_counts_frame(pd.Index(bars["cell_id"].values, name="cell_id"))
            except OSError:
                return pd.DataFrame({"cell_id": []}).set_index("cell_id")

    def _read_features(self, features_gz: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(
                features_gz,
                sep="\t",
                header=None,
                names=["feature_id", "feature_name", "feature_type"],
                usecols=[0, 1, 2],
            )
        except pd.errors.ParserError:
            pass
        try:
            feats = pd.read_csv(
                features_gz, sep="\t", header=None, names=["feature_id", "feature_name"], usecols=[0, 1]
            )
            feats["feature_type"] = "Gene Expression"
            self.logger.info("Expanded 2-column features file")
            return feats
        except pd.errors.ParserError:
            feats = pd.read_csv(features_gz, sep="\t", header=None, names=["feature_id"])
            feats["feature_name"] = feats["feature_id"]
            feats["feature_type"] = "Gene Expression"
            self.logger.info("Expanded 1-column features file")
            return feats


class CellsParquetBuilder:
    """Reconstruct ``cells.parquet`` from counts + geometry when it is missing."""

    def __init__(self, counts_loader: MexCountsLoader | None = None) -> None:
        self.logger = get_logger(__name__)
        self.counts_loader = counts_loader or MexCountsLoader()

    def ensure(self, processed_dir: Path) -> bool:
        """Create ``cells.parquet`` if absent. Returns True if it now exists."""
        cells_path = processed_dir / constants.CELLS_PARQUET
        if cells_path.exists():
            self.logger.debug("cells.parquet already exists")
            return True

        self.logger.info("cells.parquet missing; generating from available data")
        try:
            counts_df = self.counts_loader.load(processed_dir)
            cell_geo_df, nuc_geo_df, nuc_count_df = self._load_geometry(processed_dir)

            if counts_df.empty and not cell_geo_df.empty:
                self.logger.warning("No counts; building geometry-only cells.parquet")
                counts_df = zero_counts_frame(cell_geo_df.index)
            elif counts_df.empty and cell_geo_df.empty:
                self.logger.error("No data available to create cells.parquet")
                return False

            out = counts_df.join(cell_geo_df, how="outer")
            out = out.join(nuc_geo_df, how="outer")
            out = out.join(nuc_count_df, how="outer")

            out = self._drop_unassigned(out)
            out["segmentation_method"] = "unknown"
            out = self._finalize(out)
            out.to_parquet(cells_path, index=False)
            self.logger.info("Generated cells.parquet with %d cells", len(out))
            return True
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Failed to generate cells.parquet: %s", exc)
            self.logger.debug("Traceback: %s", traceback.format_exc())
            return False

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #

    def _load_geometry(
        self, processed_dir: Path
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        empty = pd.DataFrame({"cell_id": []}).set_index("cell_id")
        cell_bnds = processed_dir / "cell_boundaries.parquet"
        nuc_bnds = processed_dir / "nucleus_boundaries.parquet"

        if cell_bnds.exists() or nuc_bnds.exists():
            self.logger.info("Using boundary files for geometry")
            cell_geo_df = self._agg_boundaries(cell_bnds, "cell")[0] if cell_bnds.exists() else empty
            if nuc_bnds.exists():
                nuc_geo_df, nuc_poly_counts = self._agg_boundaries(nuc_bnds, "nucleus")
            else:
                nuc_geo_df, nuc_poly_counts = empty, {}
            if nuc_poly_counts:
                nuc_count_df = pd.Series(
                    nuc_poly_counts, name="nucleus_count", dtype="Int64"
                ).to_frame()
            else:
                nuc_count_df = empty
            return cell_geo_df, nuc_geo_df, nuc_count_df

        self.logger.info("Boundaries unavailable; using transcript-based centroids")
        cell_geo_df, nuc_count_df = self._transcript_centroids(processed_dir)
        return cell_geo_df, empty, nuc_count_df

    def _agg_boundaries(self, boundary_file: Path, kind: str) -> Tuple[pd.DataFrame, Dict]:
        empty = pd.DataFrame({"cell_id": []}).set_index("cell_id")
        try:
            df = pd.read_parquet(
                boundary_file, columns=["cell_id", "vertex_x", "vertex_y", "label_id"]
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Could not read %s: %s", boundary_file, exc)
            return empty, {}
        if df.empty:
            return empty, {}

        pieces = []
        for (cell_id, _label), grp in df.groupby(["cell_id", "label_id"], sort=False):
            area, cx, cy = polygon_area_centroid(grp["vertex_x"].values, grp["vertex_y"].values)
            row = {"cell_id": cell_id, "area": area}
            if kind == "cell":
                row.update({"cx": cx, "cy": cy})
            pieces.append(row)

        per_poly = pd.DataFrame(pieces)
        if per_poly.empty:
            return empty, {}

        if kind == "cell":
            agg = (
                per_poly.groupby("cell_id")
                .apply(
                    lambda t: pd.Series(
                        {
                            "cell_area": t["area"].sum(),
                            "x_centroid": np.average(t["cx"], weights=t["area"])
                            if t["area"].sum() > 0
                            else np.nan,
                            "y_centroid": np.average(t["cy"], weights=t["area"])
                            if t["area"].sum() > 0
                            else np.nan,
                        }
                    )
                )
                .reset_index()
                .set_index("cell_id")
            )
            return agg, {}
        agg = per_poly.groupby("cell_id")["area"].sum().rename("nucleus_area").to_frame()
        poly_counts = per_poly.groupby("cell_id").size().to_dict()
        return agg, poly_counts

    def _transcript_centroids(self, processed_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        empty = pd.DataFrame({"cell_id": []}).set_index("cell_id")
        transcripts_file = processed_dir / "transcripts.parquet"
        if not transcripts_file.exists():
            self.logger.warning("No transcripts.parquet for centroid fallback")
            return empty, empty
        try:
            df = pd.read_parquet(transcripts_file, columns=["cell_id", "x_location", "y_location"])
            df = df.dropna(subset=["cell_id"])
            if df.empty:
                return empty, empty
            centroids = df.groupby("cell_id").agg(
                x_centroid=("x_location", "mean"), y_centroid=("y_location", "mean")
            )
            centroids.index.name = "cell_id"
            self.logger.info("Computed %d transcript-based centroids", len(centroids))
            return centroids, pd.DataFrame(index=pd.Index([], name="cell_id"))
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Transcript centroid fallback failed: %s", exc)
            return empty, empty

    # ------------------------------------------------------------------ #
    # Finalization
    # ------------------------------------------------------------------ #

    def _drop_unassigned(self, out: pd.DataFrame) -> pd.DataFrame:
        if out.index.name in ("cell_id", None):
            for variant in ("UNASSIGNED", "Unassigned", "unassigned"):
                if variant in out.index:
                    self.logger.info("Removing '%s' row", variant)
                    out = out.drop(variant, errors="ignore")
        return out

    def _finalize(self, out: pd.DataFrame) -> pd.DataFrame:
        if "cell_id" in out.columns:
            out = out.drop(columns=["cell_id"])
        out = out.reset_index()
        if "index" in out.columns and "cell_id" not in out.columns:
            out = out.rename(columns={"index": "cell_id"})

        for col in constants.CELLS_PARQUET_COLUMN_ORDER:
            if col not in out.columns:
                if "counts" in col:
                    out[col] = 0
                elif "area" in col:
                    out[col] = 1
                else:
                    out[col] = pd.NA

        for col in constants.COUNT_COLUMNS:
            if col in out.columns:
                out[col] = out[col].fillna(0).astype("int64")
        if "nucleus_count" in out.columns:
            out["nucleus_count"] = out["nucleus_count"].fillna(0).astype("Int64")
        for col in ("cell_area", "nucleus_area"):
            if col in out.columns:
                out[col] = out[col].fillna(pd.NA)
        for col in ("x_centroid", "y_centroid"):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype("float64")

        available = [c for c in constants.CELLS_PARQUET_COLUMN_ORDER if c in out.columns]
        out = out[available]
        if "cell_id" in out.columns:
            out["cell_id"] = out["cell_id"].astype(str)
        return out
