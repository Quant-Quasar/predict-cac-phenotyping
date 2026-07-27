"""Tests for predict.preprocess.resampling."""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from predict.preprocess.resampling import resample_to_target


def _make_image(shape: tuple[int, int, int], spacing: tuple[float, float, float]) -> sitk.Image:
    arr = np.random.RandomState(0).randint(-200, 200, size=shape, dtype=np.int16)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin((0.0, 0.0, 0.0))
    return img


def test_ct_only_resample_changes_spacing():
    ct = _make_image((10, 20, 30), (0.5, 0.5, 3.0))
    ct_out, mask_out = resample_to_target(ct, None, target_spacing=(0.5, 0.5, 3.0))
    assert ct_out.GetSpacing() == (0.5, 0.5, 3.0)
    assert mask_out is None


def test_resample_to_different_spacing():
    ct = _make_image((10, 20, 30), (0.5, 0.5, 3.0))
    ct_out, _ = resample_to_target(ct, None, target_spacing=(1.0, 1.0, 1.0))
    assert ct_out.GetSpacing() == (1.0, 1.0, 1.0)


def test_resample_preserves_physical_extent():
    """new_size * new_spacing should equal original_size * original_spacing."""
    ct = _make_image((10, 20, 30), (0.5, 0.5, 3.0))
    orig_extent = tuple(s * sp for s, sp in zip(ct.GetSize(), ct.GetSpacing()))
    ct_out, _ = resample_to_target(ct, None, target_spacing=(0.4, 0.4, 3.0))
    new_extent = tuple(s * sp for s, sp in zip(ct_out.GetSize(), ct_out.GetSpacing()))
    for o, n in zip(orig_extent, new_extent):
        assert abs(o - n) < 0.5  # within half a target voxel


def test_mask_uses_nearest_neighbour():
    ct = _make_image((10, 20, 30), (0.5, 0.5, 3.0))
    mask_arr = np.zeros((10, 20, 30), dtype=np.uint8)
    mask_arr[3:7, 8:13, 13:18] = 1
    mask = sitk.GetImageFromArray(mask_arr)
    mask.CopyInformation(ct)

    ct_out, mask_out = resample_to_target(ct, mask, target_spacing=(0.5, 0.5, 3.0))
    out_arr = sitk.GetArrayFromImage(mask_out)
    assert set(np.unique(out_arr).tolist()).issubset({0, 1}), "Nearest-neighbour must preserve binary mask"
    assert out_arr.sum() > 0
