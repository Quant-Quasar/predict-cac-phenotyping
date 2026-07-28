"""Tests for predict.features.density_tiers."""
from __future__ import annotations

from predict.features.density_tiers import (
    DENSE_CALCIUM_HU_THRESHOLD,
    TIER_NAMES,
    compute_density_tiers,
    density_tier,
)
from predict.io.xml_parser import ROI, ParseResult, SliceAnnotation


def _roi(vessel: str | None, max_hu: float,
         vessel_raw: str = "Right Coronary Artery") -> ROI:
    return ROI(
        vessel_raw=vessel_raw, vessel=vessel,
        area_cm2=0.05, mean_hu=200.0, max_hu=max_hu, min_hu=130.0, total_hu=800.0,
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
    return ParseResult(
        pid="TEST", slices=slices, dirty_vessel_names=(),
        n_active_rois=sum(len(v) for v in slice_to_rois.values()),
    )


# ───────────────────── density_tier ─────────────────────


def test_density_tier_below_threshold_is_none():
    assert density_tier(50) is None
    assert density_tier(129) is None


def test_density_tier_boundaries():
    assert density_tier(130) == "d1"
    assert density_tier(199) == "d1"
    assert density_tier(200) == "d2"
    assert density_tier(299) == "d2"
    assert density_tier(300) == "d3"
    assert density_tier(399) == "d3"
    assert density_tier(400) == "d4"
    assert density_tier(2500) == "d4"


# ───────────────────── compute_density_tiers ─────────────────────


def test_output_has_17_keys():
    parse = _parse({0: [_roi("LAD", 250)]})
    out = compute_density_tiers(parse)
    assert len(out) == 17
    for suf in ("lad", "rca", "lcx", "lm"):
        for t in TIER_NAMES:
            assert f"n_rois_{t}_{suf}" in out
    assert "dense_calcium_count" in out


def test_counts_per_tier_per_vessel():
    parse = _parse({
        0: [
            _roi("LAD", 150),    # d1
            _roi("LAD", 250),    # d2
            _roi("LAD", 350),    # d3
            _roi("LAD", 800),    # d4
            _roi("RCA", 250),    # d2
        ],
    })
    out = compute_density_tiers(parse)
    assert out["n_rois_d1_lad"] == 1.0
    assert out["n_rois_d2_lad"] == 1.0
    assert out["n_rois_d3_lad"] == 1.0
    assert out["n_rois_d4_lad"] == 1.0
    assert out["n_rois_d2_rca"] == 1.0
    assert out["n_rois_d1_rca"] == 0.0


def test_below_threshold_rois_not_counted():
    parse = _parse({0: [_roi("LAD", 100), _roi("LAD", 129), _roi("LAD", 200)]})
    out = compute_density_tiers(parse)
    assert out["n_rois_d1_lad"] == 0.0
    assert out["n_rois_d2_lad"] == 1.0


def test_dense_calcium_count_uses_strict_greater_than():
    parse = _parse({
        0: [
            _roi("LAD", 1000),   # not counted (strict >)
            _roi("LAD", 1001),   # counted
            _roi("RCA", 5000),   # counted
        ],
    })
    out = compute_density_tiers(parse)
    assert out["dense_calcium_count"] == 2.0


def test_dirty_roi_skipped():
    parse = _parse({
        0: [_roi("LAD", 250), _roi(None, 250, vessel_raw="555614876")],
    })
    out = compute_density_tiers(parse)
    assert out["n_rois_d2_lad"] == 1.0
    # The dirty ROI's HU still wouldn't contribute to dense_calcium_count either
    # (max_hu=250 < 1000 anyway), but assert we skip it.
    assert sum(out[f"n_rois_d2_{s}"] for s in ("lad", "rca", "lcx", "lm")) == 1.0


def test_excluded_roi_skipped():
    parse = _parse({0: [_roi("LAD", 250), _roi("LAD", 800)]})
    out = compute_density_tiers(parse, excluded_roi_ids={(0, 1)})
    assert out["n_rois_d2_lad"] == 1.0
    assert out["n_rois_d4_lad"] == 0.0


def test_empty_parse_all_zero():
    parse = ParseResult(pid="TEST", slices=(), dirty_vessel_names=(), n_active_rois=0)
    out = compute_density_tiers(parse)
    assert all(v == 0.0 for v in out.values())


def test_dense_count_independent_of_excluded():
    """An excluded high-HU ROI does not inflate dense_calcium_count."""
    parse = _parse({0: [_roi("LAD", 1500)]})
    base = compute_density_tiers(parse)
    excl = compute_density_tiers(parse, excluded_roi_ids={(0, 0)})
    assert base["dense_calcium_count"] == 1.0
    assert excl["dense_calcium_count"] == 0.0
