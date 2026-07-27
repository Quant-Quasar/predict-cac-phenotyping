"""Tests for predict.preprocess.hu_handling."""
from __future__ import annotations

import numpy as np
import pytest

from predict.preprocess.hu_handling import clip_hu, flag_metal_artifact


def test_clip_hu_clips_below_min():
    arr = np.array([-500, -200, 0, 100, 3000, 5000], dtype=np.float32)
    out = clip_hu(arr, -200, 3000)
    assert out.min() == -200
    assert out.max() == 3000


def test_clip_hu_preserves_in_range():
    arr = np.array([-200, 0, 1500, 3000], dtype=np.float32)
    out = clip_hu(arr, -200, 3000)
    np.testing.assert_array_equal(out, arr)


def test_clip_hu_does_not_modify_input():
    arr = np.array([5000.0])
    _ = clip_hu(arr, -200, 3000)
    assert arr[0] == 5000.0


def test_clip_hu_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="clip_min"):
        clip_hu(np.zeros(3), 3000, -200)


def test_flag_metal_artifact_detects_high_hu_inside_mask():
    ct = np.full((4, 4, 4), 200.0, dtype=np.float32)
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 1
    ct[2, 2, 2] = 2500
    assert flag_metal_artifact(ct, mask, threshold=2000) is True


def test_flag_metal_artifact_ignores_high_hu_outside_mask():
    ct = np.full((4, 4, 4), 200.0, dtype=np.float32)
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[0, 0, 0] = 1
    ct[2, 2, 2] = 5000  # outside mask
    assert flag_metal_artifact(ct, mask, threshold=2000) is False


def test_flag_metal_artifact_empty_mask_returns_false():
    ct = np.full((4, 4, 4), 5000.0, dtype=np.float32)
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    assert flag_metal_artifact(ct, mask, threshold=2000) is False


def test_flag_metal_artifact_shape_mismatch_raises():
    ct = np.zeros((4, 4, 4), dtype=np.float32)
    mask = np.zeros((4, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="Shape mismatch"):
        flag_metal_artifact(ct, mask, threshold=2000)
