"""H&E image transformer (scikit-image based).

Bundled into the package (it runs in-process, same env as the pipeline) so H&E
alignment works without juggling ``PYTHONPATH``. Ported verbatim from the
original ``transform_he.py`` except that it no longer calls
``logging.basicConfig`` (which mutated the root logger) and no longer ships a CLI
``main`` - it routes through the package logger and is driven by
:class:`~deepspacedb_xenium_pipeline.imaging.he.HEProcessor`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import AffineTransform, warp
from tifffile import TiffFile, TiffWriter, imread

from ..logging_setup import get_logger

logger = get_logger(__name__)

# Pyramid level used for extent calculation (full resolution by default).
EXTENT_LEVEL = 0


def _write_pyramid_tiff(path: str, image: np.ndarray, bigtiff: bool) -> None:
    """Write a pyramidal TIFF with sub-IFDs so viewers pick zoom-appropriate levels."""
    tile = 1024
    n_levels = 7
    is_rgb = image.ndim == 3
    photometric = "rgb" if is_rgb else "minisblack"
    write_opts = dict(compression="deflate", tile=(tile, tile), photometric=photometric)
    logger.info("Writing pyramid with %d levels for %dx%d image", n_levels, image.shape[1], image.shape[0])
    with TiffWriter(path, bigtiff=bigtiff) as tif:
        current = image
        for i in range(n_levels):
            kwargs = {"subifds": n_levels - 1} if i == 0 else {"subfiletype": 1}
            tif.write(current, **kwargs, **write_opts)
            if i < n_levels - 1:
                new_h = max(1, current.shape[0] // 2)
                new_w = max(1, current.shape[1] // 2)
                pil = Image.fromarray(current)
                current = np.array(pil.resize((new_w, new_h), Image.Resampling.LANCZOS))


def generate_he_pngs(
    transformed_image: np.ndarray,
    output_base_path: Path,
    extent_microns: list,
    dpis: tuple[int, ...] = (100, 200, 300, 500, 1000),
    max_dimension: int = 5000,
    output_suffix: str = "he",
) -> None:
    """Generate H&E PNGs at multiple DPIs and write the extent CSV."""
    logger.info("Generating PNG images at multiple DPIs...")
    sample_dir = output_base_path.parent
    he_dir = sample_dir / "images" / output_suffix
    he_dir.mkdir(parents=True, exist_ok=True)

    extent_file = sample_dir / f"{output_suffix}_position.csv"
    pd.DataFrame([extent_microns], columns=["min_x", "max_x", "min_y", "max_y"]).to_csv(
        extent_file, index=False
    )
    logger.info("Saved extent to %s", extent_file.name)

    if transformed_image.ndim == 3:
        pil_image = Image.fromarray(transformed_image, mode="RGB")
    else:
        pil_image = Image.fromarray(transformed_image, mode="L")

    orig_width, orig_height = pil_image.size
    if max(orig_height, orig_width) > max_dimension:
        downsample_scale = max_dimension / max(orig_height, orig_width)
        base_width = int(orig_width * downsample_scale)
        base_height = int(orig_height * downsample_scale)
        logger.info("Downsampling from %dx%d to %dx%d", orig_width, orig_height, base_width, base_height)
        base_image = pil_image.resize((base_width, base_height), Image.Resampling.LANCZOS)
    else:
        base_image = pil_image
        base_width, base_height = orig_width, orig_height

    for dpi in dpis:
        scale = dpi / 1000.0
        new_width = int(base_width * scale)
        new_height = int(base_height * scale)
        resized = base_image.resize((new_width, new_height), Image.Resampling.LANCZOS) if scale < 1.0 else base_image
        png_file = he_dir / f"morphology_{output_suffix}_dpi{dpi}.png"
        resized.save(png_file, optimize=True)
        logger.info("  DPI %d: %dx%d (%.1f MB)", dpi, new_width, new_height, png_file.stat().st_size / (1024 ** 2))

    logger.info("Generated %d PNG files in images/%s/", len(dpis), output_suffix)


def _interpret_axes(he_image: np.ndarray):
    """Normalize a loaded image to ``(H, W, C)`` (or 2D) and report dimensions."""
    if he_image.ndim != 3:
        height, width = he_image.shape
        return he_image, height, width, 1, False
    dim0, dim1, dim2 = he_image.shape
    if dim2 in (3, 4):
        height, width, channels = dim0, dim1, dim2
    elif dim1 in (3, 4):
        logger.warning("Unusual axis order %s; transposing (H, C, W) -> (H, W, C)", he_image.shape)
        he_image = np.transpose(he_image, (0, 2, 1))
        height, width, channels = he_image.shape
    elif dim0 in (3, 4):
        logger.warning("Unusual axis order %s; transposing (C, H, W) -> (H, W, C)", he_image.shape)
        he_image = np.transpose(he_image, (1, 2, 0))
        height, width, channels = he_image.shape
    else:
        height, width, channels = dim0, dim1, dim2
    return he_image, height, width, channels, True


def _resolve_levels(he_image_path: str, pyramid_level: int):
    """Return ``(actual_level, extent_lvl, level0_width, level0_height, n_levels)``."""
    with TiffFile(he_image_path) as tif:
        if tif.series and hasattr(tif.series[0], "levels"):
            n_levels = len(tif.series[0].levels)
            logger.info("Detected OME-TIFF pyramid with %d levels", n_levels)
            extent_lvl = min(EXTENT_LEVEL, n_levels - 1)
            level0_page = tif.series[0].levels[extent_lvl].pages[0]
        else:
            n_levels = len(tif.pages)
            logger.info("Detected %d pages", n_levels)
            extent_lvl = min(EXTENT_LEVEL, n_levels - 1)
            level0_page = tif.pages[extent_lvl]

        shape = level0_page.shape
        if len(shape) == 3 and shape[2] in (3, 4):
            level0_height, level0_width = shape[0], shape[1]
        elif len(shape) == 3 and shape[1] in (3, 4):
            level0_height, level0_width = shape[0], shape[2]
        elif len(shape) == 3 and shape[0] in (3, 4):
            level0_height, level0_width = shape[1], shape[2]
        else:
            level0_height, level0_width = shape[0], shape[1]

        actual_level = n_levels + pyramid_level if pyramid_level < 0 else pyramid_level
        actual_level = max(0, min(actual_level, n_levels - 1))
        logger.info("Loading pyramid level %d of %d (requested %d)", actual_level, n_levels, pyramid_level)
    return actual_level, extent_lvl, level0_width, level0_height, n_levels


def transform_he_skimage(
    he_image_path: str,
    transform_csv: str,
    output_path: str,
    pixel_size: float = 0.2125,
    bigtiff: bool = True,
    generate_pngs: bool = True,
    png_dpis: tuple[int, ...] = (100, 200, 300, 500, 1000),
    png_max_dimension: int = 5000,
    pyramid_level: int = 0,
    output_suffix: str = "he",
) -> tuple:
    """Transform an H&E image to Xenium coordinates with scikit-image warp."""
    logger.info("Starting H&E transformation")
    transform_mat = pd.read_csv(transform_csv, header=None).values
    actual_level, _, level0_width, level0_height, _ = _resolve_levels(he_image_path, pyramid_level)

    he_image = imread(he_image_path, series=0, level=actual_level)
    logger.info("Raw imread shape: %s, dtype: %s", he_image.shape, he_image.dtype)
    he_image, height, width, channels, is_rgb = _interpret_axes(he_image)
    logger.info("Interpreted as %dx%d, channels %d", width, height, channels)

    corners_he = np.array([[0, 0, 1], [width, 0, 1], [0, height, 1], [width, height, 1]]).T
    corners_transformed = transform_mat @ corners_he
    min_x, max_x = np.min(corners_transformed[0, :]), np.max(corners_transformed[0, :])
    min_y, max_y = np.min(corners_transformed[1, :]), np.max(corners_transformed[1, :])
    output_width = int(np.ceil(max_x - min_x))
    output_height = int(np.ceil(max_y - min_y))
    logger.info("Output dimensions: %dx%d", output_width, output_height)

    shift_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
    inverse_transform = np.linalg.inv(shift_matrix @ transform_mat)
    tform = AffineTransform(matrix=inverse_transform)

    if he_image.dtype == np.uint8:
        he_normalized = he_image.astype(np.float32) / 255.0
        output_dtype, scale_back = np.uint8, 255.0
    elif he_image.dtype == np.uint16:
        he_normalized = he_image.astype(np.float32) / 65535.0
        output_dtype, scale_back = np.uint16, 65535.0
    else:
        he_normalized = he_image.astype(np.float32)
        output_dtype, scale_back = he_image.dtype, 1.0

    if is_rgb:
        channels_out = []
        for c in range(channels):
            logger.info("  Processing channel %d/%d", c + 1, channels)
            channels_out.append(
                warp(
                    he_normalized[:, :, c], tform,
                    output_shape=(output_height, output_width),
                    order=1, mode="constant", cval=1.0, preserve_range=True, clip=False,
                )
            )
        transformed_image = np.stack(channels_out, axis=2)
    else:
        logger.info("Processing grayscale image")
        transformed_image = warp(
            he_normalized, tform, output_shape=(output_height, output_width),
            order=1, mode="constant", cval=1.0, preserve_range=True, clip=False,
        )

    transformed_image = np.clip(transformed_image * scale_back, 0, scale_back).astype(output_dtype)
    logger.info("Saving transformed TIFF")
    _write_pyramid_tiff(output_path, transformed_image, bigtiff)

    corners_level0 = np.array(
        [[0, 0, 1], [level0_width, 0, 1], [0, level0_height, 1], [level0_width, level0_height, 1]]
    ).T
    ct0 = transform_mat @ corners_level0
    extent_microns = [
        np.min(ct0[0, :]) * pixel_size,
        np.max(ct0[0, :]) * pixel_size,
        np.min(ct0[1, :]) * pixel_size,
        np.max(ct0[1, :]) * pixel_size,
    ]

    if generate_pngs:
        generate_he_pngs(
            transformed_image=transformed_image,
            output_base_path=Path(output_path).parent,
            extent_microns=extent_microns,
            dpis=png_dpis,
            max_dimension=png_max_dimension,
            output_suffix=output_suffix,
        )
    logger.info("Transformation complete: %dx%d", output_width, output_height)
    return transformed_image, extent_microns


def transform_he_auto(
    he_image_path: str,
    transform_csv: str,
    output_path: str,
    pixel_size: float = 0.2125,
    bigtiff: bool = True,
    generate_pngs: bool = True,
    png_dpis: tuple[int, ...] = (100, 200, 300, 500, 1000),
    png_max_dimension: int = 5000,
    memory_limit_gb: float = 100,
    pyramid_level: int = 0,
    output_suffix: str = "he",
) -> tuple:
    """Entry point used by the pipeline. Delegates to the scikit-image path."""
    transform_mat = pd.read_csv(transform_csv, header=None).values
    actual_level, _, _, _, _ = _resolve_levels(he_image_path, pyramid_level)
    with TiffFile(he_image_path) as tif:
        if tif.series and hasattr(tif.series[0], "levels"):
            page = tif.series[0].levels[actual_level].pages[0]
        else:
            page = tif.pages[actual_level]
        in_height, in_width = page.shape[:2]
        channels = page.shape[2] if len(page.shape) > 2 else 1

    corners = np.array([[0, 0, 1], [in_width, 0, 1], [0, in_height, 1], [in_width, in_height, 1]]).T
    transformed = transform_mat @ corners
    out_w = int(np.ceil(np.max(transformed[0, :]) - np.min(transformed[0, :])))
    out_h = int(np.ceil(np.max(transformed[1, :]) - np.min(transformed[1, :])))
    logger.info("Estimated output size: %dx%d, ~%.1f GB", out_w, out_h, (out_w * out_h * channels * 4) / (1024 ** 3))

    return transform_he_skimage(
        he_image_path, transform_csv, output_path, pixel_size,
        bigtiff=bigtiff, generate_pngs=generate_pngs,
        png_dpis=png_dpis, png_max_dimension=png_max_dimension,
        pyramid_level=pyramid_level, output_suffix=output_suffix,
    )
