"""Unit tests for MorphologyProcessor in deepspacedb_xenium_pipeline.imaging.morphology."""

from __future__ import annotations

import numpy as np

from deepspacedb_xenium_pipeline.imaging.morphology import MorphologyProcessor


def test_morphology_processor_init():
    proc = MorphologyProcessor()
    assert proc.logger is not None


def test_mip_calculation_math():
    # Test 3D stack MIP math along axis 0
    stack = np.array(
        [
            [[1, 2], [3, 4]],
            [[5, 1], [0, 8]],
            [[2, 6], [7, 3]],
        ],
        dtype=np.uint16,
    )

    mip = np.max(stack, axis=0)
    expected = np.array([[5, 6], [7, 8]], dtype=np.uint16)
    np.testing.assert_array_equal(mip, expected)
