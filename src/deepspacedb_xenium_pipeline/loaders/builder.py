"""Construct :class:`spatialdata.SpatialData` objects from a processed directory.

Two loading strategies are supported:

* :meth:`build_standard_flexible` - uses ``spatialdata_io.xenium`` when
  ``cell_feature_matrix.h5`` is present, with fallbacks for missing ``cells.zarr.zip``
  and legacy v2 HDF5 files.
* :meth:`build_custom` - evaluates available image configurations and assembles
  tables, cell centroids, circles, and polygon boundaries directly.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import packaging.version
import pandas as pd
import pyarrow.parquet as pq

from .. import constants
from ..config import PipelineConfig
from ..logging_setup import get_logger
from .tables import TableLoader


class SpatialDataBuilder:
    """Build SpatialData objects, isolating the upstream reader's quirks."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        table_loader: Optional[TableLoader] = None,
        morphology=None,
    ) -> None:
        self.config = config
        self.logger = get_logger(__name__)
        self.tables = table_loader or TableLoader()
        if morphology is None:
            from ..imaging.morphology import MorphologyProcessor

            morphology = MorphologyProcessor()
        self.morphology = morphology

    # ------------------------------------------------------------------ #
    # experiment.xenium maintenance
    # ------------------------------------------------------------------ #

    def ensure_experiment_xenium(self, processed_dir: Path) -> None:
        experiment_file = processed_dir / constants.EXPERIMENT_FILENAME
        if experiment_file.exists():
            return
        self.logger.info("Creating experiment.xenium template")
        template: Dict[str, Any] = {
            k: ""
            for k in (
                "major_version",
                "minor_version",
                "run_name",
                "run_start_time",
                "region_name",
                "preservation_method",
                "num_cells",
                "transcripts_per_cell",
                "transcripts_per_100um",
                "cassette_name",
                "slide_id",
                "panel_design_id",
                "panel_name",
                "panel_organism",
                "panel_tissue_type",
                "panel_num_targets_predesigned",
                "panel_num_targets_custom",
                "instrument_sn",
                "instrument_sw_version",
                "analysis_sw_version",
                "experiment_uuid",
                "cassette_uuid",
                "roi_uuid",
                "z_step_size",
                "well_uuid",
                "calibration_uuid",
            )
        }
        template["pixel_size"] = constants.DEFAULT_PIXEL_SIZE
        with open(experiment_file, "w") as fh:
            json.dump(template, fh, indent=4)

    def update_experiment_files(self, processed_dir: Path) -> None:
        experiment_file = processed_dir / constants.EXPERIMENT_FILENAME
        if not experiment_file.exists():
            self.logger.warning("experiment.xenium absent; cannot update file sections")
            return
        try:
            with open(experiment_file, "r") as fh:
                specs = json.load(fh)
            images = {
                key: fname
                for key, fname in {
                    "morphology_filepath": constants.MORPHOLOGY_FILENAME,
                    "morphology_mip_filepath": constants.MORPHOLOGY_MIP_FILENAME,
                    "morphology_focus_filepath": constants.MORPHOLOGY_FOCUS_FILENAME,
                }.items()
                if (processed_dir / fname).exists()
            }
            explorer = {
                key: fname
                for key, fname in {
                    "transcripts_zarr_filepath": "transcripts.zarr.zip",
                    "cells_zarr_filepath": constants.CELLS_ZARR,
                    "cell_features_zarr_filepath": constants.CELL_FEATURE_MATRIX_ZARR,
                    "analysis_zarr_filepath": "analysis.zarr.zip",
                    "analysis_summary_filepath": "analysis_summary.html",
                }.items()
                if (processed_dir / fname).exists()
            }
            if images:
                specs["images"] = images
            else:
                specs.pop("images", None)
            if explorer:
                specs["xenium_explorer_files"] = explorer
            else:
                specs.pop("xenium_explorer_files", None)
            with open(experiment_file, "w") as fh:
                json.dump(specs, fh, indent=4)
            self.logger.info("Updated experiment.xenium (%d images, %d explorer files)", len(images), len(explorer))
        except (OSError, json.JSONDecodeError) as exc:
            self.logger.warning("Failed to update experiment.xenium: %s", exc)

    def can_load_cells_labels(self, processed_dir: Path) -> bool:
        """Detect re-segmented data whose labels would crash the upstream reader."""
        cells_zarr = processed_dir / constants.CELLS_ZARR
        if not cells_zarr.exists():
            return False
        try:
            with zipfile.ZipFile(cells_zarr, "r") as zf:
                names = zf.namelist()
            has_seg_mask = any("seg_mask_value" in n for n in names)
            has_polygon_sets = any("polygon_sets" in n for n in names)
            has_masks_1 = any(re.search(r"masks/1[/.]", n) or n == "masks/1" for n in names)
            if not has_masks_1:
                self.logger.warning("cells.zarr.zip lacks masks/1; skipping labels")
                return False
            exp_file = processed_dir / constants.EXPERIMENT_FILENAME
            if exp_file.exists():
                with open(exp_file) as fh:
                    sw = json.load(fh).get("analysis_sw_version", "")
                match = re.search(r"^(?:x|X)enium-(\d+\.\d+\.\d+)", sw)
                if match:
                    version = packaging.version.parse(match.group(1))
                    if version < packaging.version.parse("2.0.0") and not has_seg_mask and has_polygon_sets:
                        self.logger.warning(
                            "cells.zarr.zip is v2.0+ format but experiment reports v%s (re-segmented); skipping labels",
                            version,
                        )
                        return False
            return True
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Could not check cells.zarr.zip compatibility: %s", exc)
            return True

    # ------------------------------------------------------------------ #
    # Builders
    # ------------------------------------------------------------------ #

    def build_standard_flexible(self, processed_dir: Path):
        from spatialdata_io import xenium

        self.logger.info("Using flexible standard processing for %s", processed_dir)
        keys = self.tables._keys()
        has_transcripts = (processed_dir / keys.TRANSCRIPTS_FILE).exists()
        has_cells_zarr = (processed_dir / constants.CELLS_ZARR).exists()
        can_labels = self.can_load_cells_labels(processed_dir) if has_cells_zarr else False
        has_nuc = (processed_dir / keys.NUCLEUS_BOUNDARIES_FILE).exists()
        has_cell = (processed_dir / keys.CELL_BOUNDARIES_FILE).exists()
        has_mip, has_focus, has_he = self.morphology.ensure_derivatives(processed_dir)
        has_morphology = has_mip or has_focus

        can_table = has_cells_zarr
        can_boundaries = has_cells_zarr and (has_cell or has_nuc)

        try:
            sdata = xenium(
                processed_dir,
                cells_table=can_table,
                cells_as_circles=can_table,
                cells_boundaries=can_boundaries and has_cell,
                nucleus_boundaries=can_boundaries and has_nuc,
                cells_labels=can_labels,
                nucleus_labels=can_labels,
                transcripts=has_transcripts,
                morphology_mip=has_mip,
                morphology_focus=has_focus,
                aligned_images=bool(has_he and not has_morphology),
            )
            if not can_table:
                self.logger.info("cells.zarr.zip absent; loading table/circles separately")
                specs = self._load_specs(processed_dir)
                table, circles = self._table_and_circles(processed_dir, specs)
                sdata.shapes = self._polygons(processed_dir, specs, circles, has_nuc, has_cell)
                sdata.tables["table"] = table
            return sdata
        except ValueError as ve:
            if "older than V3" not in str(ve):
                raise
            self.logger.warning("Legacy v2 HDF5 detected; using scanpy + metadata path")
            return self._build_legacy_v2(processed_dir, has_nuc, has_cell)
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Standard flexible processing failed: %s", exc)
            self.logger.info("Falling back to custom processing")
            return self.build_custom(processed_dir)

    def build_custom(self, processed_dir: Path):
        from spatialdata_io import xenium

        self.logger.info("Using custom processing for %s", processed_dir)
        keys = self.tables._keys()
        has_transcripts = (processed_dir / keys.TRANSCRIPTS_FILE).exists()
        has_cells_zarr = (processed_dir / constants.CELLS_ZARR).exists()
        can_labels = self.can_load_cells_labels(processed_dir) if has_cells_zarr else False
        has_nuc = (processed_dir / keys.NUCLEUS_BOUNDARIES_FILE).exists()
        has_cell = (processed_dir / keys.CELL_BOUNDARIES_FILE).exists()
        has_mip, has_focus, has_he = self.morphology.ensure_derivatives(processed_dir)

        configs = [
            {"name": "with_focus", "morphology_focus": has_focus, "morphology_mip": has_mip, "aligned_images": False},
            {"name": "mip_only", "morphology_focus": False, "morphology_mip": has_mip, "aligned_images": False},
            {"name": "he_only", "morphology_focus": False, "morphology_mip": False, "aligned_images": has_he},
            {"name": "no_images", "morphology_focus": False, "morphology_mip": False, "aligned_images": False},
        ]
        sdata = None
        last_error: Optional[Exception] = None
        for cfg in configs:
            try:
                self.logger.info("Trying configuration: %s", cfg["name"])
                sdata = xenium(
                    processed_dir,
                    cells_table=False,
                    cells_as_circles=False,
                    cells_boundaries=False,
                    nucleus_boundaries=False,
                    cells_labels=can_labels,
                    nucleus_labels=can_labels,
                    transcripts=has_transcripts,
                    morphology_mip=cfg["morphology_mip"],
                    morphology_focus=cfg["morphology_focus"],
                    aligned_images=cfg["aligned_images"],
                )
                self.logger.info("Succeeded with config: %s", cfg["name"])
                break
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Config '%s' failed: %s", cfg["name"], exc)
                last_error = exc
        if sdata is None:
            raise RuntimeError(f"All image configurations failed. Last error: {last_error}")

        specs = self._load_specs(processed_dir)
        table, circles = self._table_and_circles(processed_dir, specs)
        sdata.shapes = self._polygons(processed_dir, specs, circles, has_nuc, has_cell)
        sdata.tables["table"] = table
        return sdata

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _load_specs(self, processed_dir: Path) -> Dict[str, Any]:
        with open(processed_dir / constants.EXPERIMENT_FILENAME, "r") as fh:
            specs = json.load(fh)
        specs["region"] = "cell_circles"
        return specs

    def _table_and_circles(self, processed_dir: Path, specs: Dict[str, Any]):
        keys = self.tables._keys()
        has_h5 = (processed_dir / constants.CELL_FEATURE_MATRIX_H5).exists()
        has_zarr = (processed_dir / constants.CELL_FEATURE_MATRIX_ZARR).exists()
        has_mtx = (processed_dir / constants.MATRIX_MTX_GZ).exists()
        has_transcripts = (processed_dir / keys.TRANSCRIPTS_FILE).exists()
        has_cells = (processed_dir / constants.CELLS_PARQUET).exists()
        if has_mtx:
            self.logger.info("Using MTX expression matrix")
            return self.tables.from_mtx(processed_dir, True, specs)
        if has_h5:
            self.logger.info("Loading H5 matrix directly")
            return self.tables.from_h5(processed_dir, specs)
        if has_zarr:
            self.logger.info("Loading ZARR matrix directly")
            return self.tables.from_zarr(processed_dir, specs)
        if has_transcripts and has_cells:
            self.logger.info("Reconstructing expression from transcripts.parquet")
            return self._load_transcripts_derived_table_and_circles(processed_dir, specs)
        raise RuntimeError(
            "No expression matrix found (need cell_feature_matrix.h5/.zarr.zip, matrix.mtx.gz, "
            "or transcripts.parquet with cells.parquet)"
        )

    def _build_counts_from_transcripts(self, processed_dir: Path):
        """Build a cell-by-gene count matrix from assigned transcripts."""
        import numpy as np
        from anndata import AnnData
        from scipy.sparse import coo_matrix

        keys = self.tables._keys()
        transcripts_file = processed_dir / keys.TRANSCRIPTS_FILE
        schema_columns = pq.ParquetFile(transcripts_file).schema_arrow.names
        cell_id_column = str(keys.CELL_ID)
        feature_name_column = str(keys.FEATURE_NAME)
        columns = [cell_id_column, feature_name_column]
        has_codeword_category = "codeword_category" in schema_columns
        if has_codeword_category:
            columns.append("codeword_category")

        dataframe = pq.read_table(transcripts_file, columns=columns).to_pandas()
        total_transcripts = len(dataframe)
        if has_codeword_category:
            dataframe = dataframe[
                (dataframe["codeword_category"] == "predesigned_gene") & (dataframe[cell_id_column] != "UNASSIGNED")
            ]
        else:
            control_patterns = (
                "NegControlProbe_",
                "NegControlCodeword_",
                "BLANK_",
                "UnassignedCodeword_",
                "DeprecatedCodeword_",
                "antisense",
            )
            feature_names = dataframe[feature_name_column].astype(str)
            is_control = feature_names.str.contains("|".join(control_patterns), case=False, regex=True)
            dataframe = dataframe[~is_control & (dataframe[cell_id_column] != "UNASSIGNED")]

        self.logger.info("Using %d/%d transcripts to build counts", len(dataframe), total_transcripts)
        cell_categories = dataframe[cell_id_column].astype(str).astype("category")
        gene_categories = dataframe[feature_name_column].astype(str).astype("category")
        matrix = coo_matrix(
            (
                np.ones(len(dataframe), dtype=np.int32),
                (cell_categories.cat.codes.to_numpy(), gene_categories.cat.codes.to_numpy()),
            ),
            shape=(len(cell_categories.cat.categories), len(gene_categories.cat.categories)),
        ).tocsr()
        matrix.sum_duplicates()
        return AnnData(
            X=matrix,
            obs=pd.DataFrame(index=cell_categories.cat.categories.astype(str)),
            var=pd.DataFrame(index=gene_categories.cat.categories.astype(str)),
        )

    def _load_transcripts_derived_table_and_circles(
        self, processed_dir: Path, specs: Dict[str, Any]
    ) -> Tuple[Any, Any]:
        """Create the table and cell circles when only transcripts are available."""
        import numpy as np
        from spatialdata.models import ShapesModel, TableModel
        from spatialdata.transformations import Scale

        adata = self._build_counts_from_transcripts(processed_dir)
        keys = self.tables._keys()
        cells_file = processed_dir / constants.CELLS_PARQUET
        metadata = pd.read_parquet(cells_file)
        cell_id_column = str(keys.CELL_ID)
        cell_x_column = str(keys.CELL_X)
        cell_y_column = str(keys.CELL_Y)
        cell_area_column = str(keys.CELL_AREA)
        metadata[cell_id_column] = self.tables._decode(metadata[cell_id_column]).astype(str)
        adata.obs_names = adata.obs_names.astype(str)

        common = sorted(set(metadata[cell_id_column]) & set(adata.obs_names))
        if not common:
            raise ValueError("0 cells aligned between transcript-derived counts and cells.parquet metadata")
        metadata = metadata[metadata[cell_id_column].isin(common)].reset_index(drop=True)
        adata = adata[adata.obs_names.isin(common)].copy()

        coords = metadata[[cell_x_column, cell_y_column]].to_numpy()
        adata.obsm["spatial"] = coords
        metadata_for_obs = metadata.drop([cell_x_column, cell_y_column], axis=1).reset_index(drop=True)
        adata.obs = metadata_for_obs
        adata.obs["region"] = "cell_circles"
        adata.obs[cell_id_column] = self.tables._decode(adata.obs[cell_id_column])
        adata.obs.index = pd.RangeIndex(len(adata.obs))

        table = TableModel.parse(
            adata,
            region="cell_circles",
            region_key="region",
            instance_key=cell_id_column,
        )
        pixel_size = specs.get("pixel_size", constants.DEFAULT_PIXEL_SIZE)
        transform = Scale([1.0 / pixel_size, 1.0 / pixel_size], axes=("x", "y"))
        if cell_area_column in adata.obs.columns:
            radii = np.sqrt(adata.obs[cell_area_column].to_numpy() / np.pi)
        else:
            self.logger.warning("No cell_area available, using default radius")
            radii = np.full(len(adata), np.sqrt(100.0 / np.pi))
        circles = ShapesModel.parse(
            coords,
            geometry=0,
            radius=radii,
            transformations={"global": transform},
            index=adata.obs.index.copy(),
        )
        return table, circles

    def _polygons(self, processed_dir, specs, circles, has_nuc, has_cell) -> Dict[str, Any]:
        from ..compat import get_polygons

        keys = self.tables._keys()
        polygons: Dict[str, Any] = {}
        debug_index = None
        if has_nuc:
            df = pq.read_table(processed_dir / keys.NUCLEUS_BOUNDARIES_FILE).to_pandas()
            if len(df) > 0:
                debug_index = self.tables._decode(pd.Series(df.groupby(keys.CELL_ID).indices.keys()))
                polygons["nucleus_boundaries"] = get_polygons(
                    processed_dir, keys.NUCLEUS_BOUNDARIES_FILE, specs, 1, idx=debug_index
                )
        if has_cell:
            if debug_index is None:
                df = pq.read_table(processed_dir / keys.CELL_BOUNDARIES_FILE).to_pandas()
                if len(df) > 0:
                    debug_index = self.tables._decode(pd.Series(df.groupby(keys.CELL_ID).indices.keys()))
            polygons["cell_boundaries"] = get_polygons(
                processed_dir, keys.CELL_BOUNDARIES_FILE, specs, 1, idx=debug_index
            )
        polygons["cell_circles"] = circles
        return polygons

    def _build_legacy_v2(self, processed_dir: Path, has_nuc: bool, has_cell: bool):
        import numpy as np
        from spatialdata import SpatialData
        from spatialdata.models import ShapesModel, TableModel
        from spatialdata.transformations import Scale

        keys = self.tables._keys()
        adata = self.tables.read_cell_feature_matrix_h5(processed_dir / constants.CELL_FEATURE_MATRIX_H5, gex_only=True)
        metadata = pd.read_parquet(processed_dir / keys.CELL_METADATA_FILE)
        np.testing.assert_array_equal(metadata.cell_id.astype(str), adata.obs_names.values)

        coords = metadata[[keys.CELL_X, keys.CELL_Y]].to_numpy()
        adata.obsm["spatial"] = coords
        adata.obs = metadata.drop([keys.CELL_X, keys.CELL_Y], axis=1)
        adata.obs["region"] = "cell_circles"
        adata.obs[keys.CELL_ID] = self.tables._decode(adata.obs[keys.CELL_ID])

        table = TableModel.parse(adata, region="cell_circles", region_key="region", instance_key=str(keys.CELL_ID))
        transform = Scale([1.0 / constants.DEFAULT_PIXEL_SIZE, 1.0 / constants.DEFAULT_PIXEL_SIZE], axes=("x", "y"))
        radii = np.sqrt(adata.obs[keys.CELL_AREA].to_numpy() / np.pi)
        circles = ShapesModel.parse(
            coords,
            geometry=0,
            radius=radii,
            transformations={"global": transform},
            index=adata.obs[keys.CELL_ID].copy(),
        )

        sdata = SpatialData()
        sdata.tables["table"] = table
        specs = self._load_specs(processed_dir)
        sdata.shapes = self._polygons(processed_dir, specs, circles, has_nuc, has_cell)

        images: Dict[str, Any] = {}
        mip_path = processed_dir / constants.MORPHOLOGY_MIP_FILENAME
        if mip_path.exists():
            images["morphology_mip"] = {"filepath": str(mip_path)}
        focus_path = processed_dir / constants.MORPHOLOGY_FOCUS_FILENAME
        focus_dir = processed_dir / constants.MORPHOLOGY_FOCUS_DIRNAME
        if focus_path.exists():
            images["morphology_focus"] = {"filepath": str(focus_path)}
        elif focus_dir.exists() and any(focus_dir.iterdir()):
            focus_files = sorted(focus_dir.glob("*.ome.tif"))
            if focus_files:
                images["morphology_focus"] = {"filepath": str(focus_files[len(focus_files) // 2])}
        he_path = processed_dir / constants.HE_IMAGE_FILENAME
        if he_path.exists():
            images["he_image"] = {"filepath": str(he_path)}
        sdata.images = images
        return sdata
