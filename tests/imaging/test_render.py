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


def test_calculate_correction_factor(tmp_path):
    proc_dir = tmp_path / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    exp_file = proc_dir / constants.EXPERIMENT_FILENAME
    exp_file.write_text(json.dumps({"pixel_size": 0.2125}))

    sdata = MagicMock(spec=SpatialData)
    sdata.images = {}
    sdata.shapes = {}

    renderer = ImageRenderer(PipelineConfig())
    factor = renderer.calculate_correction_factor(sdata, tmp_path)

    assert factor == 1000 * 0.2125
    cf_file = tmp_path / constants.CORRECTION_FACTOR_FILENAME
    assert cf_file.exists()


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
