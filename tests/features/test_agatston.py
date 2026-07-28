"""Tests for predict.features.agatston."""
from __future__ import annotations

from predict.features.agatston import (
    AGATSTON_HU_THRESHOLD,
    AgatstonResult,
    agatston_roi_score,
    classify_risk,
    compute_agatston,
    density_factor,
)
from predict.io.xml_parser import ROI, ParseResult, SliceAnnotation


# -- density_factor ----------------------------------------------------------


def test_density_factor_tier_boundaries():
    assert density_factor(130) == 1
    assert density_factor(199) == 1
    assert density_factor(200) == 2
    assert density_factor(299) == 2
    assert density_factor(300) == 3
    assert density_factor(399) == 3
    assert density_factor(400) == 4
    assert density_factor(1500) == 4


# -- agatston_roi_score ------------------------------------------------------


def test_agatston_score_below_hu_threshold_is_zero():
    # max_hu < 130 → 0
    assert agatston_roi_score(area_cm2=0.05, max_hu=129, slice_thickness_mm=3.0) == 0.0


def test_agatston_score_at_threshold():
    # max_hu = 130 → tier 1, factor 1; area 0.05 cm² × 100 × 1 × (3/3) = 5.0
    assert agatston_roi_score(area_cm2=0.05, max_hu=130, slice_thickness_mm=3.0) == 5.0


def test_agatston_score_higher_tiers():
    # max_hu = 250 → tier 2; area 0.10 cm² × 100 × 2 × 1 = 20.0
    assert agatston_roi_score(area_cm2=0.10, max_hu=250, slice_thickness_mm=3.0) == 20.0
    # max_hu = 350 → tier 3
    assert agatston_roi_score(area_cm2=0.10, max_hu=350, slice_thickness_mm=3.0) == 30.0
    # max_hu = 1200 → tier 4
    assert agatston_roi_score(area_cm2=0.10, max_hu=1200, slice_thickness_mm=3.0) == 40.0


def test_agatston_score_thickness_correction_direction():
    base = agatston_roi_score(area_cm2=0.10, max_hu=250, slice_thickness_mm=3.0)
    # 1.5 mm slices → factor 0.5
    half = agatston_roi_score(area_cm2=0.10, max_hu=250, slice_thickness_mm=1.5)
    assert abs(half - 0.5 * base) < 1e-9
    # 6.0 mm slices → factor 2.0
    double = agatston_roi_score(area_cm2=0.10, max_hu=250, slice_thickness_mm=6.0)
    assert abs(double - 2.0 * base) < 1e-9


def test_agatston_score_defensive_against_bad_inputs():
    assert agatston_roi_score(area_cm2=0, max_hu=500, slice_thickness_mm=3.0) == 0.0
    assert agatston_roi_score(area_cm2=-1, max_hu=500, slice_thickness_mm=3.0) == 0.0
    assert agatston_roi_score(area_cm2=0.05, max_hu=500, slice_thickness_mm=0) == 0.0


# -- classify_risk -----------------------------------------------------------


def test_classify_risk_boundaries():
    assert classify_risk(0) == "0"
    assert classify_risk(0.001) == "1-99"
    assert classify_risk(99.9) == "1-99"
    assert classify_risk(100) == "100-399"
    assert classify_risk(399.9) == "100-399"
    assert classify_risk(400) == "400+"
    assert classify_risk(5000) == "400+"


# -- compute_agatston (cohort-style) -----------------------------------------


def _roi(
    vessel: str | None,
    area_cm2: float = 0.05,
    mean_hu: float = 200.0,
    max_hu: float = 250.0,
    vessel_raw: str = "Right Coronary Artery",
) -> ROI:
    return ROI(
        vessel_raw=vessel_raw,
        vessel=vessel,
        area_cm2=area_cm2,
        mean_hu=mean_hu,
        max_hu=max_hu,
        min_hu=130.0,
        total_hu=1000.0,
        n_points=4,
        points_px=((0.0, 0.0),) * 4,
        points_mm=((0.0, 0.0, 0.0),) * 4,
        center_xyz=(0.0, 0.0, 0.0),
    )


def _parse(slice_to_rois: dict[int, list[ROI]]) -> ParseResult:
    slices = tuple(
        SliceAnnotation(image_index=k, rois=tuple(v))
        for k, v in sorted(slice_to_rois.items())
    )
    n_active = sum(len(v) for v in slice_to_rois.values())
    return ParseResult(
        pid="TEST", slices=slices, dirty_vessel_names=(), n_active_rois=n_active,
    )


def test_compute_agatston_breakdown_per_vessel():
    parse = _parse({
        10: [_roi("LAD", area_cm2=0.10, max_hu=250)],   # tier 2, score 20
        11: [_roi("RCA", area_cm2=0.05, max_hu=450)],   # tier 4, score 20
        12: [_roi("LCx", area_cm2=0.03, max_hu=150)],   # tier 1, score 3
        13: [_roi("LM",  area_cm2=0.04, max_hu=350)],   # tier 3, score 12
    })
    result = compute_agatston(parse, slice_thickness_mm=3.0)
    assert result.per_vessel["LAD"] == 20.0
    assert result.per_vessel["RCA"] == 20.0
    assert result.per_vessel["LCx"] == 3.0
    assert result.per_vessel["LM"] == 12.0
    assert result.total == 55.0
    assert result.category == "1-99"


def test_compute_agatston_skips_dirty_rois():
    parse = _parse({
        10: [_roi("LAD", area_cm2=0.10, max_hu=250),
             _roi(None, area_cm2=0.10, max_hu=250, vessel_raw="555614876")],
    })
    result = compute_agatston(parse, slice_thickness_mm=3.0)
    assert result.total == 20.0  # only the LAD ROI counts
    assert result.per_vessel["LAD"] == 20.0


def test_compute_agatston_excluded_ids_skipped():
    parse = _parse({
        10: [_roi("LAD", area_cm2=0.10, max_hu=250),
             _roi("RCA", area_cm2=0.05, max_hu=450)],
    })
    # Exclude the RCA ROI (image_index=10, roi_idx=1).
    result = compute_agatston(
        parse, slice_thickness_mm=3.0, excluded_roi_ids={(10, 1)},
    )
    assert result.per_vessel["LAD"] == 20.0
    assert result.per_vessel["RCA"] == 0.0
    assert result.total == 20.0


def test_compute_agatston_below_threshold_rois_contribute_zero():
    parse = _parse({
        10: [_roi("LAD", area_cm2=0.10, max_hu=129)],   # below threshold
    })
    result = compute_agatston(parse, slice_thickness_mm=3.0)
    assert result.total == 0.0
    assert result.category == "0"


def test_compute_agatston_empty_parse():
    parse = ParseResult(pid="TEST", slices=(), dirty_vessel_names=(), n_active_rois=0)
    result = compute_agatston(parse, slice_thickness_mm=3.0)
    assert result.total == 0.0
    assert result.category == "0"
    assert all(v == 0.0 for v in result.per_vessel.values())


def test_compute_agatston_thickness_correction_unified():
    """D011: the same patient at 3 mm and 1.5 mm should differ by exactly the factor."""
    parse = _parse({
        10: [_roi("LAD", area_cm2=0.10, max_hu=350)],
    })
    base = compute_agatston(parse, slice_thickness_mm=3.0)
    half = compute_agatston(parse, slice_thickness_mm=1.5)
    assert abs(half.total - 0.5 * base.total) < 1e-9
    assert abs(half.per_vessel["LAD"] - 0.5 * base.per_vessel["LAD"]) < 1e-9


def test_to_feature_dict_schema():
    parse = _parse({10: [_roi("LAD", area_cm2=0.10, max_hu=250)]})
    fd = compute_agatston(parse, slice_thickness_mm=3.0).to_feature_dict()
    assert set(fd.keys()) == {
        "agatston_lad", "agatston_rca", "agatston_lcx", "agatston_lm",
        "agatston_total",
    }
    assert fd["agatston_lad"] == 20.0
    assert fd["agatston_total"] == 20.0
