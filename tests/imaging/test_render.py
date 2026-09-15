"""Unit tests for ImageRenderer in deepspacedb_xenium_pipeline.imaging.render."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import geopandas as gpd
from shapely.geometry import Polygon
from spatialdata import SpatialData

from deepspacedb_xenium_pipeline import constants
from deepspacedb_xenium_pipeline.config import PipelineConfig
from deepspacedb_xenium_pipeline.imaging.render import ImageRenderer


def test_renderer_init():
    cfg = PipelineConfig()
    renderer = ImageRenderer(cfg)
    assert renderer.config is cfg


def test_calculate_correction_factor_no_image_no_shapes(tmp_path):
    proc_dir = tmp_path / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    exp_file = proc_dir / constants.EXPERIMENT_FILENAME
    exp_file.write_text(json.dumps({"pixel_size": 0.2125}))

    sdata = MagicMock(spec=SpatialData)
    sdata.images = {}
    sdata.shapes = {}

    renderer = ImageRenderer(PipelineConfig())
    factor = renderer.calculate_correction_factor(sdata, tmp_path)

    # The fallback is already in microns; pixel_size must not be applied again.
    assert factor == 1000.0
    cf_file = tmp_path / constants.CORRECTION_FACTOR_FILENAME
    assert cf_file.exists()


def test_calculate_correction_factor_from_shapes(tmp_path):
    """Without an image, the factor is the absolute max-x of the boundaries (microns)."""
    proc_dir = tmp_path / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    (proc_dir / constants.EXPERIMENT_FILENAME).write_text(json.dumps({"pixel_size": 0.2125}))

    poly = Polygon([(100, 50), (400, 50), (400, 300), (100, 300), (100, 50)])
    sdata = MagicMock(spec=SpatialData)
    sdata.images = {}
    sdata.shapes = {"cell_boundaries": gpd.GeoDataFrame({"geometry": [poly]}, index=["cell_1"])}

    factor = ImageRenderer(PipelineConfig()).calculate_correction_factor(sdata, tmp_path)

    # maxx, not the min-to-max range (300) and not scaled by pixel_size.
    assert factor == 400.0


def test_generate_cell_shapes(tmp_path):
    proc_zarr = tmp_path / "processed.zarr" / "shapes"
    proc_zarr.mkdir(parents=True, exist_ok=True)

    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
    gdf_cells = gpd.GeoDataFrame({"geometry": [poly]}, index=["cell_1"])
    gdf_cells.to_parquet(proc_zarr / "cell_boundaries")

    gdf_nucleus = gpd.GeoDataFrame({"geometry": [poly]}, index=["cell_1"])
    gdf_nucleus.to_parquet(proc_zarr / "nucleus_boundaries")

    renderer = ImageRenderer(PipelineConfig())
    renderer.generate_cell_shapes(tmp_path)

    out_dir = tmp_path / "images" / "cell_shapes"
    assert (out_dir / "cells.png").exists()
    assert (out_dir / "nucleus.png").exists()
    assert (out_dir / "cells_nucleus.png").exists()


def test_tissue_height_um_from_image_dims(tmp_path):
    """The Y extent comes from scale0_y * pixel_size, mirroring the correction factor."""
    proc_dir = tmp_path / "processed"
    proc_dir.mkdir(parents=True)
    (proc_dir / constants.EXPERIMENT_FILENAME).write_text(json.dumps({"pixel_size": 0.5}))
    zarray_dir = tmp_path / "processed.zarr" / "images" / "morphology_mip" / "0"
    zarray_dir.mkdir(parents=True)
    (zarray_dir / ".zarray").write_text(json.dumps({"shape": [1, 400, 800]}))

    height = ImageRenderer(PipelineConfig()).tissue_height_um(tmp_path, fallback_maxy=1.0)
    assert height == 200.0  # 400 rows * 0.5 um


def test_tissue_height_um_falls_back_to_shapes(tmp_path):
    height = ImageRenderer(PipelineConfig()).tissue_height_um(tmp_path, fallback_maxy=1234.0)
    assert height == 1234.0


def test_generate_cell_shapes_uses_non_square_canvas(tmp_path):
    """A wide tissue must not be rendered on a square canvas that crops its bottom."""
    from PIL import Image

    proc_zarr = tmp_path / "processed.zarr" / "shapes"
    proc_zarr.mkdir(parents=True)
    poly = Polygon([(0, 0), (800, 0), (800, 200), (0, 200), (0, 0)])
    gpd.GeoDataFrame({"geometry": [poly]}, index=["cell_1"]).to_parquet(proc_zarr / "cell_boundaries")
    (tmp_path / constants.CORRECTION_FACTOR_FILENAME).write_text("800.0\n")

    ImageRenderer(PipelineConfig()).generate_cell_shapes(tmp_path)

    with Image.open(tmp_path / "images" / "cell_shapes" / "cells.png") as img:
        width, height = img.size
    assert width > height
    assert abs(height / width - 200.0 / 800.0) < 0.02
