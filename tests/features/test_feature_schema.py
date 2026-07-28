"""Tests for predict.features.feature_schema."""
from __future__ import annotations

from predict.features.feature_schema import (
    DENSITY_TIERS,
    GLOBAL_FEATURES,
    PER_VESSEL_STEMS,
    VESSEL_SUFFIXES,
    feature_names,
    n_features,
    zero_features,
)


# Expected: 4 vessels × (10 per-vessel stems + 4 density tiers) + globals
EXPECTED_PER_VESSEL = 4 * (10 + 4)            # 56
EXPECTED_GLOBALS = 12
EXPECTED_TOTAL = EXPECTED_PER_VESSEL + EXPECTED_GLOBALS  # 68


def test_vessel_suffixes_are_canonical_lowercase():
    assert VESSEL_SUFFIXES == ("lad", "rca", "lcx", "lm")


def test_density_tiers_canonical():
    assert DENSITY_TIERS == ("d1", "d2", "d3", "d4")


def test_per_vessel_stems_canonical():
    assert PER_VESSEL_STEMS == (
        "lesion_count",
        "max_hu",
        "mean_hu",
        "volume_mm3",
        "mass",
        "agatston",
        "inter_lesion_dist_mean",
        "inter_lesion_dist_max",
        "first_to_last_dist",
        "diffusivity",
    )


def test_n_features_matches_expected():
    assert n_features() == EXPECTED_TOTAL


def test_feature_names_no_duplicates():
    names = feature_names()
    assert len(names) == len(set(names)), "feature names must be unique"


def test_zero_features_returns_full_schema():
    d = zero_features()
    assert len(d) == n_features()
    assert set(d.keys()) == set(feature_names())


def test_zero_features_all_zero():
    d = zero_features()
    assert all(v == 0.0 for v in d.values())


def test_zero_features_independent_copies():
    a = zero_features()
    a["lesion_count_lad"] = 5.0
    b = zero_features()
    assert b["lesion_count_lad"] == 0.0


def test_per_vessel_keys_use_correct_template():
    names = set(feature_names())
    # volume_mm3 uses the {vessel}_mm3 suffix form, not _volume_mm3
    assert "volume_lad_mm3" in names
    assert "volume_rca_mm3" in names
    assert "volume_lcx_mm3" in names
    assert "volume_lm_mm3" in names
    # other stems use {stem}_{vessel}
    assert "lesion_count_lad" in names
    assert "mass_rca" in names
    assert "agatston_lcx" in names
    assert "diffusivity_lm" in names


def test_density_tier_keys_present_per_vessel():
    names = set(feature_names())
    for suf in VESSEL_SUFFIXES:
        for tier in DENSITY_TIERS:
            assert f"n_rois_{tier}_{suf}" in names


def test_all_globals_present():
    names = set(feature_names())
    for g in GLOBAL_FEATURES:
        assert g in names


def test_feature_names_ordering_per_vessel_then_globals():
    """Per-vessel keys appear before globals; per-vessel grouped by vessel."""
    names = feature_names()
    # First key should be the first stem for LAD.
    assert names[0] == "lesion_count_lad"
    # Last key should be the last global.
    assert names[-1] == GLOBAL_FEATURES[-1]
    # Globals appear contiguously at the end.
    n_global = len(GLOBAL_FEATURES)
    assert names[-n_global:] == GLOBAL_FEATURES
