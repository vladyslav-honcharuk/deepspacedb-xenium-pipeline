"""Unit tests for transform_he in deepspacedb_xenium_pipeline.imaging.transform_he."""

from __future__ import annotations

import numpy as np
import pandas as pd
import tifffile

from deepspacedb_xenium_pipeline.imaging.transform_he import (
    _interpret_axes,
    _write_pyramid_tiff,
    generate_he_pngs,
    transform_he_skimage,
)


def test_interpret_axes():
    # 2D grayscale
    img2d = np.zeros((10, 20), dtype=np.uint8)
    res, h, w, c, is_rgb = _interpret_axes(img2d)
    assert (h, w, c) == (10, 20, 1)
    assert not is_rgb

    # 3D RGB (H, W, C)
    img3d = np.zeros((10, 20, 3), dtype=np.uint8)
    res, h, w, c, is_rgb = _interpret_axes(img3d)
    assert (h, w, c) == (10, 20, 3)
    assert is_rgb


def test_write_pyramid_tiff(tmp_path):
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    out_path = tmp_path / "test_pyramid.ome.tif"
    _write_pyramid_tiff(str(out_path), img, bigtiff=False)

    assert out_path.exists()
    with tifffile.TiffFile(out_path) as tif:
        assert len(tif.pages) > 0


def test_generate_he_pngs(tmp_path):
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    generate_he_pngs(
        img,
        tmp_path / "morphology_he.ome.tif",
        extent_microns=[0, 100, 0, 100],
        dpis=(100, 200),
        output_suffix="he",
    )

    out_dir = tmp_path / "images" / "he"
    assert (out_dir / "morphology_he_dpi100.png").exists()
    assert (out_dir / "morphology_he_dpi200.png").exists()


def test_transform_he_skimage(tmp_path):
    raw_img = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    in_tif = tmp_path / "in.tif"
    tifffile.imwrite(in_tif, raw_img)

    mat = np.eye(3)
    csv_path = tmp_path / "transform.csv"
    pd.DataFrame(mat).to_csv(csv_path, header=False, index=False)

    out_tif = tmp_path / "out.tif"
    transform_he_skimage(
        str(in_tif),
        str(csv_path),
        str(out_tif),
        bigtiff=False,
        generate_pngs=False,
    )

    assert out_tif.exists()
