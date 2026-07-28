"""Tests for predict.features.spatial."""
from __future__ import annotations

from predict.features.lesion_ccl import Lesion
from predict.features.spatial import (
    compute_spatial_features,
    diffusivity,
    gini_coefficient,
)


SLICE_POSITIONS = tuple(float(k) * 3.0 for k in range(30))   # 0..87 mm


def _lesion(
    vessel: str,
    centroid: tuple[float, float, float],
    *,
    n_rois: int = 1,
    total_area_mm2: float = 10.0,
    volume_mm3: float | None = None,
    mean_hu_weighted: float = 200.0,
    max_hu: float = 300.0,
) -> Lesion:
    return Lesion(
        vessel=vessel,
        roi_keys=tuple((0, i) for i in range(n_rois)),
        slice_indices=tuple(range(n_rois)),
        centroid_mm=centroid,
        total_area_mm2=total_area_mm2,
        mean_hu_weighted=mean_hu_weighted,
        max_hu=max_hu,
        volume_mm3=volume_mm3 if volume_mm3 is not None else total_area_mm2 * 3.0,
    )


def _by_vessel(*lesions: Lesion) -> dict[str, list[Lesion]]:
    out: dict[str, list[Lesion]] = {"LAD": [], "RCA": [], "LCx": [], "LM": []}
    for l in lesions:
        out[l.vessel].append(l)
    return out


# ───────────────────────── gini ─────────────────────────


def test_gini_empty_zero():
    assert gini_coefficient([]) == 0.0
    assert gini_coefficient([5.0]) == 0.0


def test_gini_all_equal_zero():
    assert gini_coefficient([3.0, 3.0, 3.0, 3.0]) == 0.0


def test_gini_perfect_inequality_approaches_one_minus_one_over_n():
    # One non-zero among n=4 → Gini = (n-1)/n = 0.75
    assert abs(gini_coefficient([0, 0, 0, 100]) - 0.75) < 1e-9


# ───────────────────────── diffusivity (D016) ─────────────────────────


def test_diffusivity_edge_cases():
    assert diffusivity(0, 0.0) == 0.0
    assert diffusivity(0, 50.0) == 0.0
    assert diffusivity(1, 0.0) == 1.0
    assert diffusivity(1, 50.0) == 1.0
    # N=2 with d below epsilon → 1.0 (D016 fallback)
    assert diffusivity(2, 1e-9) == 1.0
    # Normal case
    assert abs(diffusivity(4, 20.0) - 0.2) < 1e-9


# ───────────────────────── per-vessel counts/distances ─────────────────────────


def test_single_vessel_two_lesions_distances():
    a = _lesion("LAD", (0.0, 0.0, 0.0))
    b = _lesion("LAD", (3.0, 0.0, 4.0))   # 3D distance = 5
    out = compute_spatial_features(_by_vessel(a, b), slice_positions=SLICE_POSITIONS)
    assert out["lesion_count_lad"] == 2.0
    assert out["lesion_count_total"] == 2.0
    assert abs(out["inter_lesion_dist_mean_lad"] - 5.0) < 1e-9
    assert abs(out["inter_lesion_dist_max_lad"] - 5.0) < 1e-9
    assert abs(out["first_to_last_dist_lad"] - 5.0) < 1e-9
    assert abs(out["diffusivity_lad"] - 2 / 5.0) < 1e-9


def test_single_lesion_diffusivity_is_one():
    a = _lesion("RCA", (1.0, 2.0, 3.0))
    out = compute_spatial_features(_by_vessel(a), slice_positions=SLICE_POSITIONS)
    assert out["lesion_count_rca"] == 1.0
    assert out["diffusivity_rca"] == 1.0
    assert out["inter_lesion_dist_mean_rca"] == 0.0
    assert out["first_to_last_dist_rca"] == 0.0


def test_empty_vessel_zero_diffusivity():
    out = compute_spatial_features(_by_vessel(), slice_positions=SLICE_POSITIONS)
    for suf in ("lad", "rca", "lcx", "lm"):
        assert out[f"diffusivity_{suf}"] == 0.0
        assert out[f"lesion_count_{suf}"] == 0.0


def test_first_to_last_uses_z_sorted_extremes():
    """First-to-last is the distance between the lowest-z and highest-z lesions,
    not between the first and last items as ordered in the input list."""
    a = _lesion("LAD", (0.0, 0.0, 30.0))   # mid-z
    b = _lesion("LAD", (0.0, 0.0, 0.0))    # lowest z
    c = _lesion("LAD", (0.0, 0.0, 60.0))   # highest z
    # Pass them in scrambled order:
    by_vessel = {"LAD": [a, b, c], "RCA": [], "LCx": [], "LM": []}
    out = compute_spatial_features(by_vessel, slice_positions=SLICE_POSITIONS)
    assert abs(out["first_to_last_dist_lad"] - 60.0) < 1e-9
    # Consecutive distances after z-sort: (30 - 0)=30, (60 - 30)=30. Mean 30, max 30.
    assert abs(out["inter_lesion_dist_mean_lad"] - 30.0) < 1e-9
    assert abs(out["inter_lesion_dist_max_lad"] - 30.0) < 1e-9


# ───────────────────────── globals ─────────────────────────


def test_n_calcified_arteries():
    by = _by_vessel(
        _lesion("LAD", (0, 0, 0)),
        _lesion("RCA", (0, 0, 0)),
    )
    out = compute_spatial_features(by, slice_positions=SLICE_POSITIONS)
    assert out["n_calcified_arteries"] == 2.0


def test_gini_across_all_lesion_volumes():
    by = _by_vessel(
        _lesion("LAD", (0, 0, 0), volume_mm3=10.0),
        _lesion("LAD", (0, 0, 3), volume_mm3=10.0),
        _lesion("RCA", (0, 0, 0), volume_mm3=10.0),
    )
    out = compute_spatial_features(by, slice_positions=SLICE_POSITIONS)
    # All equal volumes → Gini = 0.
    assert out["gini_lesion_volume"] == 0.0

    by2 = _by_vessel(
        _lesion("LAD", (0, 0, 0), volume_mm3=100.0),
        _lesion("RCA", (0, 0, 0), volume_mm3=0.0),
        _lesion("LCx", (0, 0, 0), volume_mm3=0.0),
        _lesion("LM",  (0, 0, 0), volume_mm3=0.0),
    )
    out2 = compute_spatial_features(by2, slice_positions=SLICE_POSITIONS)
    # Three zeros + one 100 across 4 lesions → Gini = (n-1)/n = 0.75.
    assert abs(out2["gini_lesion_volume"] - 0.75) < 1e-9


def test_dist_from_top_uses_max_slice_position():
    # slice_positions top is 87.0; lesion at z=87 → dist 0; at z=0 → dist 87.
    by = _by_vessel(
        _lesion("LAD", (0, 0, 87.0)),
        _lesion("LAD", (0, 0, 0.0)),
    )
    out = compute_spatial_features(by, slice_positions=SLICE_POSITIONS)
    assert out["dist_from_top_max"] == 87.0
    assert abs(out["dist_from_top_mean"] - 43.5) < 1e-9


def test_dist_from_top_clamps_negative_roundoff_to_zero():
    """Lesion slightly above max(slice_positions) due to float rounding → dist 0."""
    by = _by_vessel(
        _lesion("LAD", (0, 0, 87.0 + 1e-9)),
    )
    out = compute_spatial_features(by, slice_positions=SLICE_POSITIONS)
    assert out["dist_from_top_max"] == 0.0


def test_center_of_mass_z_area_weighted_and_patient_relative():
    """`center_of_mass_z` is the distance from the most-superior slice to the
    weighted centre of mass, NOT the raw weighted Z (which would be scanner-
    table-position-dependent and not comparable across patients)."""
    by = _by_vessel(
        _lesion("LAD", (0, 0, 10.0), total_area_mm2=10.0),
        _lesion("LAD", (0, 0, 20.0), total_area_mm2=30.0),
    )
    # Weighted absolute z: (10*10 + 30*20) / 40 = 17.5.
    # z_top = max(SLICE_POSITIONS) = 87.0. center_of_mass_z = 87.0 - 17.5 = 69.5
    out = compute_spatial_features(by, slice_positions=SLICE_POSITIONS)
    assert abs(out["center_of_mass_z"] - 69.5) < 1e-9


def test_center_of_mass_z_is_invariant_to_slice_position_offset():
    """Shifting slice_positions by a constant offset must not change
    center_of_mass_z (or any dist_from_top_* feature)."""
    base = _by_vessel(_lesion("LAD", (0, 0, 10.0), total_area_mm2=10.0))
    shifted = _by_vessel(_lesion("LAD", (0, 0, 110.0), total_area_mm2=10.0))
    out_base = compute_spatial_features(base, slice_positions=SLICE_POSITIONS)
    shifted_positions = tuple(p + 100.0 for p in SLICE_POSITIONS)
    out_shifted = compute_spatial_features(shifted, slice_positions=shifted_positions)
    assert abs(out_base["center_of_mass_z"] - out_shifted["center_of_mass_z"]) < 1e-9
    assert abs(out_base["dist_from_top_max"] - out_shifted["dist_from_top_max"]) < 1e-9


# ───────────────────────── whole-output shape ─────────────────────────


def test_output_has_26_keys():
    by = _by_vessel(_lesion("LAD", (0, 0, 0)))
    out = compute_spatial_features(by, slice_positions=SLICE_POSITIONS)
    assert len(out) == 26


def test_empty_input_returns_all_zeros():
    by = _by_vessel()
    out = compute_spatial_features(by, slice_positions=SLICE_POSITIONS)
    assert all(v == 0.0 for v in out.values())
