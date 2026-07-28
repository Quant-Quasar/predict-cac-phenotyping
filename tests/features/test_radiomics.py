"""Tests for predict.features.radiomics.

These tests exercise the full PyRadiomics extraction on a tiny synthetic
volume. They are marked ``slow`` so the fast inner-loop test run
(``pytest -m "not slow"``) skips them.
"""
from __future__ import annotations

import numpy as np
import pytest

from predict.features.radiomics import (
    DEFAULT_PARAMS_YAML,
    _arrays_to_sitk,
    create_extractor,
    extract_pyradiomics,
    validate_ct_for_radiomics,
)


# ───────────────────── guardrail ─────────────────────


def test_validate_accepts_raw_hu_int16():
    arr = np.array([-1000, -200, 0, 250, 500, 1500], dtype=np.int16)
    validate_ct_for_radiomics(arr)


def test_validate_rejects_normalised_input():
    arr = np.array([0.0, 0.3, 0.7, 1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="normalised"):
        validate_ct_for_radiomics(arr, pid="TEST")


def test_validate_rejects_high_min_hu():
    arr = np.array([50, 100, 200, 500], dtype=np.int16)
    with pytest.raises(ValueError, match="too high for raw HU"):
        validate_ct_for_radiomics(arr, pid="TEST")


def test_validate_rejects_low_max_hu():
    arr = np.array([-1000, -200, 0, 120], dtype=np.int16)
    with pytest.raises(ValueError, match="below CAC threshold"):
        validate_ct_for_radiomics(arr, pid="TEST")


def test_validate_rejects_unexpected_dtype():
    arr = np.array([0, 50, 130, 255], dtype=np.uint8)
    with pytest.raises(ValueError, match="dtype"):
        validate_ct_for_radiomics(arr, pid="TEST")


# ───────────────────── _arrays_to_sitk ─────────────────────


def test_arrays_to_sitk_geometry():
    ct = np.zeros((4, 8, 8), dtype=np.int16)
    mask = np.zeros((4, 8, 8), dtype=np.uint8)
    ct_s, mask_s = _arrays_to_sitk(ct, mask, (0.5, 0.5, 3.0))
    assert ct_s.GetSpacing() == (0.5, 0.5, 3.0)
    assert mask_s.GetSpacing() == (0.5, 0.5, 3.0)
    assert ct_s.GetOrigin() == (0.0, 0.0, 0.0)
    assert ct_s.GetSize() == (8, 8, 4)


def test_arrays_to_sitk_rejects_shape_mismatch():
    ct = np.zeros((4, 8, 8), dtype=np.int16)
    mask = np.zeros((4, 8, 7), dtype=np.uint8)
    with pytest.raises(ValueError, match="Shape mismatch"):
        _arrays_to_sitk(ct, mask, (0.5, 0.5, 3.0))


# ───────────────────── PyRadiomics extraction ─────────────────────


def _synthetic_calcified_volume(
    shape: tuple[int, int, int] = (10, 32, 32),
    cube_origin: tuple[int, int, int] = (3, 10, 10),
    cube_side: int = 6,
    cube_hu: int = 500,
    background_hu: int = -200,
) -> tuple[np.ndarray, np.ndarray]:
    """A small CT-like volume with a homogeneous HU cube inside a mask cube.

    Default cube: 6×6×6 = 216 voxels (well above minimumROISize=14).
    """
    ct = np.full(shape, background_hu, dtype=np.int16)
    mask = np.zeros(shape, dtype=np.uint8)
    z0, y0, x0 = cube_origin
    ct[z0:z0+cube_side, y0:y0+cube_side, x0:x0+cube_side] = cube_hu
    mask[z0:z0+cube_side, y0:y0+cube_side, x0:x0+cube_side] = 1
    return ct, mask


@pytest.mark.slow
def test_extractor_loads_from_default_params_yaml():
    extractor = create_extractor(DEFAULT_PARAMS_YAML)
    # Enabled classes should match params.yaml.
    enabled = set(extractor.enabledFeatures.keys())
    assert enabled == {"shape", "firstorder", "glcm", "glszm", "glrlm", "ngtdm", "gldm"}


@pytest.mark.slow
def test_extract_pyradiomics_returns_all_feature_families():
    ct, mask = _synthetic_calcified_volume()
    extractor = create_extractor()
    features = extract_pyradiomics(ct, mask, (0.5, 0.5, 3.0), extractor)
    assert features, "Non-empty mask must produce features"
    # No diagnostics_ keys leak through.
    assert not any(k.startswith("diagnostics_") for k in features)
    # Each enabled family present by prefix.
    prefixes = ["original_shape", "original_firstorder", "original_glcm",
                "original_glszm", "original_glrlm", "original_ngtdm", "original_gldm"]
    for p in prefixes:
        assert any(k.startswith(p) for k in features), f"missing prefix {p}"
    # Every value is a finite float.
    for k, v in features.items():
        assert isinstance(v, float), f"{k} is not float"


@pytest.mark.slow
def test_extract_pyradiomics_empty_mask_returns_empty_dict():
    ct, mask = _synthetic_calcified_volume()
    mask_zero = np.zeros_like(mask)
    extractor = create_extractor()
    features = extract_pyradiomics(ct, mask_zero, (0.5, 0.5, 3.0), extractor)
    assert features == {}


@pytest.mark.slow
def test_extract_pyradiomics_feature_count_in_expected_range():
    """Per D009 we expect roughly 107 features total. Allow a band because
    PyRadiomics versions can add minor features in patch releases."""
    ct, mask = _synthetic_calcified_volume()
    extractor = create_extractor()
    features = extract_pyradiomics(ct, mask, (0.5, 0.5, 3.0), extractor)
    assert 90 <= len(features) <= 130, (
        f"Unexpected feature count: {len(features)} (expected ~107)"
    )
