"""Render morphology/H&E figures and cell-shape overlays from a SpatialData.

Imports ``matplotlib`` with the non-interactive ``Agg`` backend so rendering
works headless. ``spatialdata_plot`` is optional; if it is missing, figure
generation is skipped with a warning rather than crashing the pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")  # headless, must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .. import constants  # noqa: E402
from ..config import PipelineConfig  # noqa: E402
from ..logging_setup import get_logger  # noqa: E402

try:  # optional dependency, presence check only
    import spatialdata_plot  # noqa: F401,E402
    _HAS_SDATA_PLOT = True
except ImportError:
    _HAS_SDATA_PLOT = False

_CELL_COLOR = (0.565, 0.565, 0.565, 0.2)
_TARGET_WIDTH = 6000


class ImageRenderer:
    """Produce PNG/PDF renders and correction factors from a SpatialData object."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.logger = get_logger(__name__)

    # ------------------------------------------------------------------ #
    # Figure rendering
    # ------------------------------------------------------------------ #

    def generate_images(self, sdata, base_dir: Path) -> List[str]:
        if not _HAS_SDATA_PLOT:
            self.logger.warning("spatialdata_plot not installed; skipping image generation")
            return []

        available = list(sdata.images.keys())
        self.logger.info("Available images: %s", available)
        he_keys = [k for k in available if "he" in k.lower()]
        groups = {
            "focus": [k for k in available if "morphology_focus" in k],
            "mip": [k for k in available if "morphology_mip" in k],
            "other": [
                k for k in available if "morphology" in k and "focus" not in k and "mip" not in k
            ],
        }
        if not any(groups.values()) and not he_keys:
            self.logger.warning("No images found; skipping image generation")
            return []

        created: List[str] = []
        try:
            for image_type, keys in groups.items():
                for key in keys:
                    created += self._render_morphology(sdata, base_dir, image_type, key)
            for key in he_keys:
                created += self._render_he(sdata, base_dir, key)
            self.logger.info("Generated %d image files", len(created))
        except Exception as exc:  # noqa: BLE001 - rendering is best-effort
            self.logger.error("Error generating images: %s", exc)
            return []
        return created

    def _render_morphology(self, sdata, base_dir: Path, image_type: str, key: str) -> List[str]:
        type_dir = base_dir / "images" / image_type
        type_dir.mkdir(parents=True, exist_ok=True)
        expected = [type_dir / f"{key}_dpi100.pdf"]
        expected += [type_dir / f"{key}_dpi{dpi}.png" for dpi in self.config.image_dpis]
        expected += [
            type_dir / f"{key}_dpi500_gray.png",
            type_dir / f"{key}_dpi500_gray_inverse.png",
        ]
        if all(f.exists() for f in expected):
            self.logger.info("Images for %s exist; skipping", key)
            return [f.name for f in expected]

        created: List[str] = []
        fig, ax = plt.subplots(1, 1)
        sdata.pl.render_images(key).pl.show(ax=ax, colorbar=False)
        ax.set_title("")
        ax.axis("off")
        pdf_file = type_dir / f"{key}_dpi100.pdf"
        fig.savefig(pdf_file, format="pdf", bbox_inches="tight", pad_inches=0)
        created.append(pdf_file.name)
        for dpi in self.config.image_dpis:
            png_file = type_dir / f"{key}_dpi{dpi}.png"
            fig.savefig(png_file, format="png", bbox_inches="tight", pad_inches=0, dpi=dpi,
                        facecolor="none", edgecolor="none")
            created.append(png_file.name)
        plt.close(fig)

        for inverse in (False, True):
            cmap = "gray_r" if inverse else "gray"
            suffix = "_inverse" if inverse else ""
            fig, ax = plt.subplots(1, 1)
            sdata.pl.render_images(key, cmap=cmap).pl.show(ax=ax, colorbar=False)
            ax.set_title("")
            ax.axis("off")
            img_file = type_dir / f"{key}_dpi500_gray{suffix}.png"
            fig.savefig(img_file, format="png", bbox_inches="tight", pad_inches=0, dpi=500,
                        facecolor="none", edgecolor="none")
            created.append(img_file.name)
            plt.close(fig)
        return created

    def _render_he(self, sdata, base_dir: Path, key: str) -> List[str]:
        he_dir = base_dir / "images" / "he"
        he_dir.mkdir(parents=True, exist_ok=True)
        expected = [he_dir / f"{key}_dpi100.pdf"]
        expected += [he_dir / f"{key}_dpi{dpi}.png" for dpi in self.config.image_dpis]
        if all(f.exists() for f in expected):
            self.logger.info("H&E images for %s exist; skipping", key)
            return [f.name for f in expected]

        created: List[str] = []
        fig, ax = plt.subplots(1, 1)
        sdata.pl.render_images(key).pl.show(ax=ax, colorbar=False)
        ax.set_title("")
        ax.axis("off")
        pdf_file = he_dir / f"{key}_dpi100.pdf"
        fig.savefig(pdf_file, format="pdf", bbox_inches="tight", pad_inches=0)
        created.append(pdf_file.name)
        for dpi in self.config.image_dpis:
            png_file = he_dir / f"{key}_dpi{dpi}.png"
            fig.savefig(png_file, format="png", bbox_inches="tight", pad_inches=0, dpi=dpi,
                        facecolor="none", edgecolor="none")
            created.append(png_file.name)
        plt.close(fig)
        return created

    # ------------------------------------------------------------------ #
    # Correction factor
    # ------------------------------------------------------------------ #

    def calculate_correction_factor(self, sdata, sample_dir: Path) -> float:
        self.logger.info("Calculating correction factor")
        correction_factor = 1.0
        try:
            available = list(sdata.images.keys())
            key = None
            if "morphology_mip" in available:
                key = "morphology_mip"
            elif "morphology_focus" in available:
                key = "morphology_focus"
            elif any("he" in k.lower() for k in available):
                key = next(k for k in available if "he" in k.lower())
                self.logger.info("Using H&E image for correction factor: %s", key)
            elif available:
                key = available[0]

            if key is not None:
                scale0_x = sdata.images[key]["scale0"].ds.sizes["x"]
            elif "cell_circles" in sdata.shapes:
                x = sdata.shapes["cell_circles"].geometry.x
                scale0_x = x.max() - x.min()
            else:
                scale0_x = 1000

            pixel_size = constants.read_pixel_size(
                sample_dir / "processed" / constants.EXPERIMENT_FILENAME
            )
            correction_factor = scale0_x * pixel_size
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Error calculating correction factor: %s", exc)

        correction_file = sample_dir / constants.CORRECTION_FACTOR_FILENAME
        correction_file.write_text(f"{correction_factor}\n")
        self.logger.info("Correction factor: %s", correction_factor)
        return correction_factor

    # ------------------------------------------------------------------ #
    # Cell-shape overlays
    # ------------------------------------------------------------------ #

    def generate_cell_shapes(self, sample_dir: Path) -> None:
        import geopandas as gpd

        processed_zarr = sample_dir / "processed.zarr"
        shapes_dir = processed_zarr / "shapes"
        if not shapes_dir.exists():
            self.logger.warning("No shapes in processed.zarr; skipping cell-shape generation")
            return

        output_dir = sample_dir / "images" / "cell_shapes"
        output_dir.mkdir(parents=True, exist_ok=True)

        cb_path = shapes_dir / "cell_boundaries"
        nb_path = shapes_dir / "nucleus_boundaries"
        cell_boundaries = gpd.read_parquet(cb_path) if cb_path.exists() else None
        nucleus_boundaries = gpd.read_parquet(nb_path) if nb_path.exists() else None
        if cell_boundaries is None:
            self.logger.warning("No cell_boundaries; skipping cell-shape generation")
            return
        self.logger.info("Loaded %d cell boundaries", len(cell_boundaries))

        expected = [output_dir / "cells.png"]
        if nucleus_boundaries is not None:
            expected += [output_dir / "nucleus.png", output_dir / "cells_nucleus.png"]
        if all(f.exists() for f in expected):
            self.logger.info("Cell shapes exist; skipping")
            return

        cf_path = sample_dir / constants.CORRECTION_FACTOR_FILENAME
        if cf_path.exists():
            correction_factor = float(pd.read_csv(cf_path, header=None).iloc[0, 0])
        else:
            correction_factor = float(cell_boundaries.total_bounds[2])
            self.logger.warning("No correction_factor.csv; using cell maxx=%.1f", correction_factor)

        def make_fig():
            fig_width = 10
            dpi = _TARGET_WIDTH / fig_width
            fig, ax = plt.subplots(figsize=(fig_width, fig_width), dpi=dpi)
            return fig, ax, dpi

        def save(fig, ax, path, dpi):
            ax.set_xlim(0, correction_factor)
            ax.set_ylim(correction_factor, 0)
            ax.set_aspect("equal")
            ax.axis("off")
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            fig.savefig(path, format="png", dpi=dpi, pad_inches=0, transparent=True)
            plt.close(fig)
            self.logger.info("Saved %s (%.1f MB)", path.name, path.stat().st_size / 1024 / 1024)

        fig, ax, dpi = make_fig()
        cell_boundaries.plot(ax=ax, facecolor=_CELL_COLOR, edgecolor="black", linewidth=0.1)
        save(fig, ax, output_dir / "cells.png", dpi)

        if nucleus_boundaries is not None:
            fig, ax, dpi = make_fig()
            nucleus_boundaries.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=0.1)
            save(fig, ax, output_dir / "nucleus.png", dpi)

            fig, ax, dpi = make_fig()
            cell_boundaries.plot(ax=ax, facecolor=_CELL_COLOR, edgecolor="black", linewidth=0.1)
            nucleus_boundaries.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=0.1)
            save(fig, ax, output_dir / "cells_nucleus.png", dpi)
        self.logger.info("Cell shapes done -> %s", output_dir)
