"""Build SpatialData ``TableModel`` and cell-circle ``ShapesModel`` from disk.

Three matrix loaders are supported:

* :meth:`TableLoader.from_h5`   - ``cell_feature_matrix.h5``
* :meth:`TableLoader.from_zarr` - ``cell_feature_matrix.zarr.zip``
* :meth:`TableLoader.from_mtx`  - 10x MEX (``matrix.mtx.gz`` + TSV feature/barcode files)

Handles integer and string cell ID conversions, missing ``cells.parquet`` metadata
(with fallbacks to Zarr, cell boundaries, and transcript centroids), control-probe
filtering, and coordinate alignments.
"""

from __future__ import annotations

import gzip
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scanpy as sc
import zarr

from .. import constants
from ..geometry import polygon_area_centroid
from ..logging_setup import get_logger


class TableLoader:
    """Load expression + metadata into ``(TableModel, ShapesModel)`` tuples."""

    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    # ------------------------------------------------------------------ #
    # 10x MEX helpers
    # ------------------------------------------------------------------ #

    def load_10x_fixed(self, path: Path):
        """Load 10x MEX data, repairing 1/2-column features and integer barcodes."""
        path = Path(path)

        features_file = next((path / p for p in ("features.tsv.gz", "features.tsv") if (path / p).exists()), None)
        if features_file is not None:
            self.fix_features_file(features_file)

        barcodes_file = next((path / p for p in ("barcodes.tsv.gz", "barcodes.tsv") if (path / p).exists()), None)
        if barcodes_file is None:
            return sc.read_10x_mtx(path, gex_only=True)

        barcodes_df = pd.read_csv(barcodes_file, sep="\t", header=None)
        first_val = str(barcodes_df.iloc[0, 0])
        if "CELL_" in first_val or "-" in first_val:
            return sc.read_10x_mtx(path, gex_only=True)
        try:
            int(barcodes_df.iloc[0, 0])
        except (ValueError, TypeError):
            return sc.read_10x_mtx(path, gex_only=True)

        backup = str(barcodes_file) + ".tmp_backup"
        shutil.copy2(barcodes_file, backup)
        try:
            barcodes_df.iloc[:, 0] = [f"CELL_{int(i):08d}-1" for i in barcodes_df.iloc[:, 0]]
            if str(barcodes_file).endswith(".gz"):
                temp_txt = path / "temp_barcodes.tsv"
                barcodes_df.to_csv(temp_txt, sep="\t", header=False, index=False)
                with open(temp_txt, "rb") as f_in, gzip.open(barcodes_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                temp_txt.unlink()
            else:
                barcodes_df.to_csv(barcodes_file, sep="\t", header=False, index=False)
            return sc.read_10x_mtx(path, gex_only=True)
        finally:
            shutil.move(backup, barcodes_file)

    def fix_features_file(self, features_file: Path) -> None:
        """Expand a 1/2-column features file to the 3-column 10x format in place."""
        try:
            features_df = pd.read_csv(features_file, sep="\t", header=None)
            n_cols = features_df.shape[1]
            if n_cols == 1:
                fixed = pd.DataFrame({0: features_df.iloc[:, 0], 1: features_df.iloc[:, 0], 2: "Gene Expression"})
            elif n_cols == 2:
                fixed = pd.DataFrame({0: features_df.iloc[:, 0], 1: features_df.iloc[:, 1], 2: "Gene Expression"})
            else:
                return
            shutil.copy2(features_file, str(features_file) + ".original_backup")
            if str(features_file).endswith(".gz"):
                temp_file = features_file.parent / "temp_features.tsv"
                fixed.to_csv(temp_file, sep="\t", header=False, index=False)
                with open(temp_file, "rb") as f_in, gzip.open(features_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                temp_file.unlink()
            else:
                fixed.to_csv(features_file, sep="\t", header=False, index=False)
            self.logger.info("Expanded features file to 3 columns")
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Error fixing features file: %s", exc)
            raise

    # ------------------------------------------------------------------ #
    # Shared utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def _keys():
        from spatialdata_io._constants._constants import XeniumKeys

        return XeniumKeys

    @staticmethod
    def _decode(series):
        from spatialdata_io.readers.xenium import _decode_cell_id_column

        return _decode_cell_id_column(series)

    @staticmethod
    def _decode_zarr_cell_ids(raw) -> np.ndarray:
        """Decode Xenium zarr ``cell_id`` arrays into alphanumeric strings.

        The on-disk layout is a ``(n, 2)`` uint32 pair ``(prefix, dataset_suffix)``.
        Casting only the prefix with ``.astype(str)`` yields IDs like ``"25866"``
        that never match ``cells.parquet`` (``"aaaagfak-1"``) and silently align
        to 0 cells.
        """
        from spatialdata_io.readers.xenium import cell_id_str_from_prefix_suffix_uint32

        raw = np.asarray(raw)
        if raw.ndim == 2 and raw.shape[1] >= 2:
            return np.asarray(cell_id_str_from_prefix_suffix_uint32(raw[:, 0], raw[:, 1]))
        return raw.astype(str)

    @staticmethod
    def is_anndata_h5(path: Path) -> bool:
        """True when ``path`` is a plain AnnData h5ad mislabeled as a 10x h5.

        GEO uploads sometimes ship ``cell_feature_matrix.h5`` that is actually
        an AnnData export (``X``/``obs``/``var`` groups). ``sc.read_10x_h5``
        then treats those groups as genomes and raises "contains more than one
        genome".
        """
        import h5py

        with h5py.File(path, "r") as fh:
            return "matrix" not in fh and "X" in fh and "obs" in fh and "var" in fh

    def read_cell_feature_matrix_h5(self, path: Path, gex_only: bool = True):
        """Read ``cell_feature_matrix.h5``, tolerating a mislabeled AnnData file."""
        if self.is_anndata_h5(path):
            import anndata as ad

            adata = ad.read_h5ad(path)
            if gex_only and "feature_types" in adata.var.columns:
                adata = adata[:, adata.var["feature_types"] == "Gene Expression"].copy()
            return adata
        return sc.read_10x_h5(path, gex_only=gex_only)

    def _filter_control_features(self, adata):
        names = adata.var_names.fillna("").astype(str)
        mask = ~names.str.contains("|".join(constants.CONTROL_FEATURE_PATTERNS), case=False, regex=True)
        if not mask.all():
            self.logger.info("Filtering %d control/blank features", (~mask).sum())
            adata = adata[:, mask].copy()
        return adata

    def _boundary_cell_ids(self, processed_dir: Path):
        keys = self._keys()
        for fname in (keys.NUCLEUS_BOUNDARIES_FILE, keys.CELL_BOUNDARIES_FILE):
            path = processed_dir / fname
            if path.exists():
                df = pq.read_table(path).to_pandas()
                if len(df) > 0:
                    ids = pd.Series(df.groupby(keys.CELL_ID).indices.keys())
                    return self._decode(ids)
        return None

    # ------------------------------------------------------------------ #
    # H5 loader
    # ------------------------------------------------------------------ #

    def from_h5(self, processed_dir: Path, specs: Dict[str, Any]) -> Tuple[Any, Any]:
        from spatialdata.models import ShapesModel, TableModel
        from spatialdata.transformations import Scale

        keys = self._keys()
        self.logger.info("Loading expression from cell_feature_matrix.h5")
        adata = self.read_cell_feature_matrix_h5(processed_dir / constants.CELL_FEATURE_MATRIX_H5, gex_only=False)
        adata = self._filter_control_features(adata)

        cells_parquet = processed_dir / keys.CELL_METADATA_FILE
        cells_zarr = processed_dir / constants.CELLS_ZARR
        nucleus_bnd = processed_dir / keys.NUCLEUS_BOUNDARIES_FILE
        cell_bnd = processed_dir / keys.CELL_BOUNDARIES_FILE

        metadata = self._read_metadata(processed_dir, cells_parquet, cells_zarr, nucleus_bnd, cell_bnd)
        boundary_cell_ids = self._boundary_cell_ids(processed_dir)

        metadata_cell_ids = self._decode(metadata[keys.CELL_ID]).astype(str)
        metadata[keys.CELL_ID] = metadata_cell_ids
        adata.obs_names = adata.obs_names.astype(str)

        if boundary_cell_ids is not None:
            boundary_set = set(boundary_cell_ids.astype(str))
            missing = sum(1 for c in metadata_cell_ids if c not in boundary_set)
            if missing:
                self.logger.warning("%d cells lack boundary data but are kept", missing)

        common = sorted(set(metadata_cell_ids) & set(adata.obs_names))
        if not common:
            raise ValueError(
                f"0 cells aligned between expression ({len(adata)}) and metadata "
                f"({len(metadata)}) - cell ID format/type mismatch. "
                f"Expression e.g. {list(adata.obs_names[:3])}, "
                f"metadata e.g. {metadata_cell_ids.head(3).tolist()}"
            )
        metadata = metadata[metadata[keys.CELL_ID].isin(common)].reset_index(drop=True)
        adata = adata[adata.obs_names.isin(common)].copy()
        self.logger.info("Aligned %d cells (h5)", len(common))

        coords = metadata[[keys.CELL_X, keys.CELL_Y]].to_numpy()
        adata.obsm["spatial"] = coords
        obs = metadata.drop([keys.CELL_X, keys.CELL_Y], axis=1).reset_index(drop=True)
        adata.obs = obs
        adata.obs["region"] = "cell_circles"
        adata.obs[keys.CELL_ID] = self._decode(adata.obs[keys.CELL_ID])
        adata.obs.index = pd.RangeIndex(len(adata.obs))

        table = TableModel.parse(adata, region="cell_circles", region_key="region", instance_key=str(keys.CELL_ID))

        pixel_size = constants.DEFAULT_PIXEL_SIZE
        transform = Scale([1.0 / pixel_size, 1.0 / pixel_size], axes=("x", "y"))
        radii = self._radii_for(adata, keys, nucleus_bnd, cell_bnd)
        circles = ShapesModel.parse(
            coords,
            geometry=0,
            radius=radii,
            transformations={"global": transform},
            index=adata.obs.index.copy(),
        )
        return table, circles

    def _radii_for(self, adata, keys, nucleus_bnd: Path, cell_bnd: Path) -> np.ndarray:
        if keys.CELL_AREA in adata.obs.columns:
            return np.sqrt(adata.obs[keys.CELL_AREA].to_numpy() / np.pi)
        boundary_file = nucleus_bnd if nucleus_bnd.exists() else (cell_bnd if cell_bnd.exists() else None)
        if boundary_file is not None:
            self.logger.info("Computing cell_area from boundary polygons")
            bnd_df = pq.read_table(boundary_file).to_pandas()
            areas: Dict[str, float] = {}
            for cid, grp in bnd_df.groupby(keys.CELL_ID):
                area, _, _ = polygon_area_centroid(grp["vertex_x"].values, grp["vertex_y"].values)
                areas[str(cid)] = abs(area)
            return np.array([np.sqrt(areas.get(str(cid), 100.0) / np.pi) for cid in adata.obs[keys.CELL_ID]])
        self.logger.warning("No cell_area/boundaries; using default radius")
        return np.full(len(adata), np.sqrt(100.0 / np.pi))

    def _read_metadata(self, processed_dir, cells_parquet, cells_zarr, nucleus_bnd, cell_bnd):
        keys = self._keys()
        if cells_parquet.exists():
            self.logger.info("Loading cell metadata from cells.parquet")
            return pd.read_parquet(cells_parquet)
        if cells_zarr.exists():
            self.logger.info("Loading cell metadata from cells.zarr.zip")
            return self._metadata_from_zarr(cells_zarr)
        self.logger.info("cells.parquet/zarr absent; computing metadata from boundaries")
        boundary_file = nucleus_bnd if nucleus_bnd.exists() else cell_bnd
        if boundary_file.exists():
            df = pq.read_table(boundary_file).to_pandas()
            centroids = df.groupby(keys.CELL_ID).agg({"vertex_x": "mean", "vertex_y": "mean"}).reset_index()
            return pd.DataFrame(
                {
                    "cell_id": centroids[keys.CELL_ID].astype(str),
                    keys.CELL_X: centroids["vertex_x"],
                    keys.CELL_Y: centroids["vertex_y"],
                }
            )
        raise FileNotFoundError(
            f"No cell metadata source in {processed_dir}: tried cells.parquet, cells.zarr.zip, and boundary files"
        )

    def _metadata_from_zarr(self, cells_zarr: Path) -> pd.DataFrame:
        store = zarr.ZipStore(str(cells_zarr), mode="r")
        try:
            try:
                root = zarr.hierarchy.Group(store, read_only=True)
            except zarr.errors.GroupNotFoundError:
                root = store
            cell_summary = root["cell_summary"][:].copy()
            cell_ids = self._decode_zarr_cell_ids(root["cell_id"][:].copy())
            col_names = root["cell_summary"].attrs["column_names"]
            metadata = pd.DataFrame(cell_summary, columns=col_names)
            metadata["cell_id"] = cell_ids
            return metadata.rename(columns={"cell_centroid_x": "x_centroid", "cell_centroid_y": "y_centroid"})
        finally:
            store.close()

    # ------------------------------------------------------------------ #
    # Zarr loader
    # ------------------------------------------------------------------ #

    def from_zarr(self, processed_dir: Path, specs: Dict[str, Any]) -> Tuple[Any, Any]:
        from anndata import AnnData
        from scipy.sparse import csc_matrix
        from spatialdata.models import ShapesModel, TableModel
        from spatialdata.transformations import Scale

        keys = self._keys()
        self.logger.info("Loading expression from cell_feature_matrix.zarr.zip")
        zarr_path = processed_dir / constants.CELL_FEATURE_MATRIX_ZARR
        store = zarr.ZipStore(str(zarr_path), mode="r")
        all_keys = list(store.keys())
        self.logger.info("Found %d items in %s", len(all_keys), zarr_path.name)

        if (
            any("matrix.mtx" in k for k in all_keys)
            and any("barcodes" in k for k in all_keys)
            and any("features" in k for k in all_keys)
        ):
            self.logger.info("zarr.zip actually contains MEX; extracting and delegating")
            store.close()
            with zipfile.ZipFile(zarr_path, "r") as zf:
                for item in all_keys:
                    target = processed_dir / item.split("/")[-1]
                    with zf.open(item) as source, open(target, "wb") as dest:
                        dest.write(source.read())
            return self.from_mtx(processed_dir, cells_as_circles=True, specs=specs)

        try:
            cf = zarr.hierarchy.Group(store, path="cell_features", read_only=True)
        except zarr.errors.GroupNotFoundError:
            self.logger.error("No cell_features group in zarr")
            store.close()
            raise

        data = cf["data"][:].copy()
        indices = cf["indices"][:].copy()
        indptr = cf["indptr"][:].copy()
        feature_types = cf.attrs["feature_types"]
        feature_keys = cf.attrs["feature_keys"]
        n_cells = cf.attrs["number_cells"]
        matrix = csc_matrix((data, indices, indptr), shape=(n_cells, len(feature_types)))

        gene_indices = np.where(np.array([ft == "gene" for ft in feature_types]))[0]
        matrix_genes = matrix[:, gene_indices].copy()
        gene_names = [feature_keys[i] for i in gene_indices]

        valid = np.array([isinstance(n, str) and n == n for n in gene_names])
        if not valid.all():
            self.logger.warning("Removing %d features with NaN/invalid names", (~valid).sum())
            keep = np.where(valid)[0]
            matrix_genes = matrix_genes[:, keep]
            gene_names = [gene_names[i] for i in keep]
            gene_indices = gene_indices[keep]

        adata = AnnData(X=matrix_genes.tocsr())
        adata.var_names = gene_names
        adata.var["gene_ids"] = gene_names
        adata.var["feature_types"] = [feature_types[i] for i in gene_indices]
        adata.obs_names = self._decode_zarr_cell_ids(cf["cell_id"][:].copy())
        store.close()

        metadata, adata = self._read_zarr_metadata(processed_dir, adata)
        boundary_cell_ids = self._boundary_cell_ids(processed_dir)

        metadata_cell_ids = self._decode(metadata[keys.CELL_ID])
        metadata[keys.CELL_ID] = metadata_cell_ids
        if len(set(metadata_cell_ids) & set(adata.obs_names.astype(str))) == 0 and len(metadata) == len(adata):
            self.logger.warning("Cell IDs disjoint but counts match; aligning by position")
            metadata[keys.CELL_ID] = adata.obs_names.astype(str).values
            metadata_cell_ids = metadata[keys.CELL_ID]

        if boundary_cell_ids is not None:
            boundary_set = set(boundary_cell_ids)
            missing = sum(1 for c in metadata_cell_ids if c not in boundary_set)
            if missing:
                self.logger.warning("%d cells lack boundary data but are kept", missing)

        common = sorted(set(metadata_cell_ids) & set(adata.obs_names.astype(str)))
        self.logger.info("Aligning %d cells (zarr)", len(common))
        metadata = metadata[metadata[keys.CELL_ID].isin(common)].reset_index(drop=True)
        adata = adata[adata.obs_names.isin(common)].copy()

        coords = metadata[[keys.CELL_X, keys.CELL_Y]].to_numpy()
        adata.obsm["spatial"] = coords
        obs = metadata.drop([keys.CELL_X, keys.CELL_Y], axis=1).reset_index(drop=True)
        adata.obs = obs
        adata.obs["region"] = "cell_circles"
        adata.obs[keys.CELL_ID] = self._decode(adata.obs[keys.CELL_ID])
        adata.obs.index = adata.obs[keys.CELL_ID].astype(str)
        adata.obs["transcript_counts"] = np.asarray(adata.X.sum(axis=1)).flatten().astype(int)

        table = TableModel.parse(adata, region="cell_circles", region_key="region", instance_key=str(keys.CELL_ID))
        transform = Scale([1.0 / constants.DEFAULT_PIXEL_SIZE, 1.0 / constants.DEFAULT_PIXEL_SIZE], axes=("x", "y"))
        if str(keys.CELL_AREA) in adata.obs.columns:
            radii = np.sqrt(adata.obs[str(keys.CELL_AREA)].to_numpy() / np.pi)
        else:
            self.logger.warning("cell_area absent; using default radius 10")
            radii = np.full(len(adata), 10.0)
        circles = ShapesModel.parse(
            coords,
            geometry=0,
            radius=radii,
            transformations={"global": transform},
            index=adata.obs.index.copy(),
        )
        return table, circles

    def _read_zarr_metadata(self, processed_dir: Path, adata) -> Tuple[pd.DataFrame, Any]:
        """Return ``(metadata, adata)``; the transcript fallback may subset adata."""
        keys = self._keys()
        cells_parquet = processed_dir / keys.CELL_METADATA_FILE
        cells_zarr = processed_dir / constants.CELLS_ZARR
        if cells_parquet.exists():
            self.logger.info("Loading cell metadata from cells.parquet")
            return pd.read_parquet(cells_parquet), adata
        if cells_zarr.exists():
            self.logger.info("Loading cell metadata from cells.zarr.zip")
            return self._metadata_from_zarr(cells_zarr), adata

        transcripts_file = processed_dir / keys.TRANSCRIPTS_FILE
        if not transcripts_file.exists():
            raise FileNotFoundError(f"Neither {cells_parquet}, {cells_zarr}, nor {transcripts_file} found")
        self.logger.info("Computing cell metadata from transcripts.parquet")
        df = pd.read_parquet(transcripts_file)
        df = df[df["cell_id"] != "UNASSIGNED"].copy()
        centroids = (
            df.groupby("cell_id")
            .agg({"x_location": "mean", "y_location": "mean"})
            .reset_index()
            .rename(columns={"x_location": "x_centroid", "y_location": "y_centroid"})
        )
        metadata = pd.DataFrame(
            {
                "cell_id": centroids["cell_id"].astype(str),
                keys.CELL_X: centroids["x_centroid"],
                keys.CELL_Y: centroids["y_centroid"],
            }
        )
        if len(metadata) != len(adata):
            self.logger.warning("Transcripts has %d cells but expression has %d", len(metadata), len(adata))
            if len(metadata) < len(adata):
                adata = adata[: len(metadata)].copy()
            else:
                metadata = metadata.iloc[: len(adata)].copy()
        metadata["cell_id"] = adata.obs_names.astype(str).values
        return metadata, adata

    # ------------------------------------------------------------------ #
    # MTX loader
    # ------------------------------------------------------------------ #

    def from_mtx(self, path: Path, cells_as_circles: bool, specs: Dict[str, Any]) -> Union[Any, Tuple[Any, Any]]:
        from spatialdata.models import ShapesModel, TableModel
        from spatialdata.transformations import Scale

        keys = self._keys()
        adata = self.load_10x_fixed(path)
        adata = self._filter_control_features(adata)
        metadata = pd.read_parquet(path / keys.CELL_METADATA_FILE)
        boundary_cell_ids = self._boundary_cell_ids(path)

        if boundary_cell_ids is not None:
            boundary_set = set(boundary_cell_ids)
            missing = (~metadata["cell_id"].isin(boundary_set)).sum()
            if missing:
                self.logger.warning("%d cells lack boundary data but are kept", missing)

        metadata_cell_ids = metadata["cell_id"].astype(str).values
        adata_ids = set(adata.obs_names.astype(str))
        valid = [c for c in metadata_cell_ids if c in adata_ids]
        if len(valid) < len(metadata_cell_ids):
            self.logger.warning("Filtering metadata to %d cells present in expression", len(valid))
            metadata = metadata[metadata["cell_id"].astype(str).isin(valid)].copy()
        adata = adata[metadata["cell_id"].astype(str).values, :].copy()

        np.testing.assert_array_equal(metadata.cell_id.astype(str), adata.obs_names.astype(str))
        self.logger.info("Aligned %d cells (mtx)", len(metadata))

        coords = metadata[[keys.CELL_X, keys.CELL_Y]].to_numpy()
        adata.obsm["spatial"] = coords
        obs = metadata.drop([keys.CELL_X, keys.CELL_Y], axis=1).reset_index(drop=True)
        adata.obs = obs
        adata.obs["region"] = specs["region"]
        adata.obs["region"] = adata.obs["region"].astype("category")
        adata.obs[keys.CELL_ID] = self._decode(adata.obs[keys.CELL_ID])
        if adata.obs.index.dtype in ("int64", "int32"):
            adata.obs.index = adata.obs.index.astype(str)
        elif not isinstance(adata.obs.index, pd.RangeIndex):
            adata.obs = adata.obs.reset_index(drop=True)

        table = TableModel.parse(adata, region=specs["region"], region_key="region", instance_key=str(keys.CELL_ID))
        if not cells_as_circles:
            return table

        transform = Scale([1.0 / specs["pixel_size"], 1.0 / specs["pixel_size"]], axes=("x", "y"))
        radii = np.sqrt(adata.obs[keys.CELL_AREA].to_numpy() / np.pi)
        circle_index = adata.obs[keys.CELL_ID].copy()
        if circle_index.dtype != "object":
            circle_index = circle_index.astype(str)
        circles = ShapesModel.parse(
            coords,
            geometry=0,
            radius=radii,
            transformations={"global": transform},
            index=circle_index,
        )
        return table, circles
