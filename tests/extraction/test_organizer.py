"""Unit tests for FileOrganizer in deepspacedb_xenium_pipeline.extraction.organizer."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pandas as pd

from deepspacedb_xenium_pipeline.config import PipelineConfig
from deepspacedb_xenium_pipeline.extraction.organizer import FileOrganizer, find_he_image_file


def test_find_he_image_file_token_not_marker_substring(tmp_path):
    cadherin = tmp_path / "Ecadherin.tif"
    cadherin.write_bytes(b"x")
    her2 = tmp_path / "HER2_channel.tif"
    her2.write_bytes(b"x")
    assert find_he_image_file([cadherin, her2]) is None

    genuine = tmp_path / "27670_HE.ome.tif"
    genuine.write_bytes(b"x")
    camel = tmp_path / "XeniumHE_LUAD_No14.tif"
    camel.write_bytes(b"x")
    assert find_he_image_file([cadherin, genuine]).name == "27670_HE.ome.tif"
    assert find_he_image_file([camel]).name == "XeniumHE_LUAD_No14.tif"


def test_find_he_image_file_skips_morphology_focus(tmp_path):
    focus_dir = tmp_path / "morphology_focus"
    focus_dir.mkdir()
    nested = focus_dir / "plane_HE.ome.tif"
    nested.write_bytes(b"x")
    assert find_he_image_file([nested]) is None


def test_organize_basic_sample(tmp_path):
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    proc_dir = tmp_path / "processed"

    # Create dummy experiment.xenium and transcripts.parquet in temp_dir
    exp_file = temp_dir / "sample_experiment.xenium"
    exp_file.write_text(json.dumps({"major_version": 2, "minor_version": 0, "pixel_size": 0.2125}))
    trans_file = temp_dir / "sample_transcripts.parquet"
    df_trans = pd.DataFrame(
        {
            "feature_name": ["GeneA"],
            "x_location": [1.0],
            "y_location": [2.0],
            "qv": [40.0],
            "cell_id": [1],
        }
    )
    df_trans.to_parquet(trans_file)

    mock_he = MagicMock()
    mock_cells = MagicMock()
    mock_cells.ensure.return_value = False

    organizer = FileOrganizer(
        PipelineConfig(base_dir=tmp_path),
        he_processor=mock_he,
        cells_builder=mock_cells,
    )

    n = organizer.organize(temp_dir, proc_dir)
    assert n >= 2
    assert (proc_dir / "experiment.xenium").exists()
    assert (proc_dir / "transcripts.parquet").exists()
