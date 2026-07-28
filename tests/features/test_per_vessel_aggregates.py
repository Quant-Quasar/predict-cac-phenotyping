"""Tests for predict.features.per_vessel_aggregates."""
from __future__ import annotations

from predict.features.per_vessel_aggregates import compute_per_vessel_aggregates
from predict.io.xml_parser import ROI, ParseResult, SliceAnnotation


def _roi(vessel: str | None, area_cm2: float = 0.10, mean_hu: float = 200.0,
         max_hu: float = 300.0, vessel_raw: str = "Right Coronary Artery") -> ROI:
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
    return ParseResult(
        pid="TEST", slices=slices, dirty_vessel_names=(),
        n_active_rois=sum(len(v) for v in slice_to_rois.values()),
    )


# ──────────────────────────────────────────────────────────────────────────


def test_output_has_canonical_keys():
    parse = _parse({0: [_roi("LAD")]})
    out = compute_per_vessel_aggregates(parse, slice_thickness_mm=3.0)
    expected = (
        {f"{stem}_{suf}" for stem in ("mass", "mean_hu", "max_hu")
         for suf in ("lad", "rca", "lcx", "lm")}
        | {f"volume_{suf}_mm3" for suf in ("lad", "rca", "lcx", "lm")}
        | {"volume_total_mm3", "mass_total", "mean_hu_weighted_global", "max_hu_global"}
    )
    assert set(out.keys()) == expected


def test_volume_formula():
    # 0.10 cm² × 100 = 10 mm²; × 3 mm thickness = 30 mm³.
    parse = _parse({0: [_roi("LAD", area_cm2=0.10)]})
    out = compute_per_vessel_aggregates(parse, slice_thickness_mm=3.0)
    assert out["volume_lad_mm3"] == 30.0
    assert out["volume_rca_mm3"] == 0.0
    assert out["volume_total_mm3"] == 30.0


def test_mass_formula_hu_volume_product():
    # area_mm² × thickness × mean_hu = 10 × 3 × 200 = 6000
    parse = _parse({0: [_roi("LAD", area_cm2=0.10, mean_hu=200)]})
    out = compute_per_vessel_aggregates(parse, slice_thickness_mm=3.0)
    assert out["mass_lad"] == 6000.0
    assert out["mass_total"] == 6000.0


def test_max_hu_is_max_across_rois_in_vessel():
    parse = _parse({
        0: [_roi("LAD", max_hu=200)],
        1: [_roi("LAD", max_hu=450)],
        2: [_roi("LAD", max_hu=380)],
    })
    out = compute_per_vessel_aggregates(parse, slice_thickness_mm=3.0)
    assert out["max_hu_lad"] == 450.0


def test_mean_hu_is_area_weighted():
    # Equal areas → simple average. (10, 30 mean HU) → 20.
    parse = _parse({
        0: [_roi("LAD", area_cm2=0.05, mean_hu=10)],
        1: [_roi("LAD", area_cm2=0.05, mean_hu=30)],
    })
    out = compute_per_vessel_aggregates(parse, slice_thickness_mm=3.0)
    assert abs(out["mean_hu_lad"] - 20.0) < 1e-9


def test_mean_hu_area_weighted_with_unequal_areas():
    # Areas 1.0 cm² and 0.1 cm² with means 100 and 1000.
    # Weighted: (1.0*100 + 0.1*1000) / (1.0+0.1) = (100+100)/1.1 = ~181.82
    parse = _parse({
        0: [_roi("LAD", area_cm2=1.0, mean_hu=100)],
        1: [_roi("LAD", area_cm2=0.1, mean_hu=1000)],
    })
    out = compute_per_vessel_aggregates(parse, slice_thickness_mm=3.0)
    assert abs(out["mean_hu_lad"] - (200.0 / 1.1)) < 1e-9


def test_empty_vessel_returns_zero_sentinel():
    parse = _parse({0: [_roi("LAD")]})  # LAD only
    out = compute_per_vessel_aggregates(parse, slice_thickness_mm=3.0)
    for suf in ("rca", "lcx", "lm"):
        assert out[f"volume_{suf}_mm3"] == 0.0
        assert out[f"mass_{suf}"] == 0.0
        assert out[f"mean_hu_{suf}"] == 0.0
        assert out[f"max_hu_{suf}"] == 0.0


def test_dirty_roi_skipped():
    parse = _parse({
        0: [
            _roi("LAD", area_cm2=0.10, mean_hu=200),
            _roi(None, area_cm2=0.10, mean_hu=999, vessel_raw="555614876"),
        ],
    })
    out = compute_per_vessel_aggregates(parse, slice_thickness_mm=3.0)
    assert out["mass_lad"] == 6000.0           # only LAD ROI counts
    assert out["mean_hu_lad"] == 200.0


def test_excluded_roi_skipped():
    parse = _parse({
        0: [_roi("LAD", area_cm2=0.10), _roi("RCA", area_cm2=0.20)],
    })
    out = compute_per_vessel_aggregates(
        parse, slice_thickness_mm=3.0, excluded_roi_ids={(0, 1)},
    )
    assert out["volume_lad_mm3"] == 30.0       # 0.10 cm² × 100 × 3 mm
    assert out["volume_rca_mm3"] == 0.0
    assert out["volume_total_mm3"] == 30.0


def test_global_max_hu_across_vessels():
    parse = _parse({
        0: [_roi("LAD", max_hu=300)],
        1: [_roi("RCA", max_hu=1500)],
    })
    out = compute_per_vessel_aggregates(parse, slice_thickness_mm=3.0)
    assert out["max_hu_global"] == 1500.0


def test_global_mean_hu_weighted():
    parse = _parse({
        0: [_roi("LAD", area_cm2=0.10, mean_hu=200)],
        1: [_roi("RCA", area_cm2=0.10, mean_hu=400)],
    })
    out = compute_per_vessel_aggregates(parse, slice_thickness_mm=3.0)
    assert abs(out["mean_hu_weighted_global"] - 300.0) < 1e-9


def test_thickness_scales_volume_and_mass():
    parse = _parse({0: [_roi("LAD", area_cm2=0.10, mean_hu=200)]})
    base = compute_per_vessel_aggregates(parse, slice_thickness_mm=3.0)
    half = compute_per_vessel_aggregates(parse, slice_thickness_mm=1.5)
    assert abs(half["volume_lad_mm3"] - 0.5 * base["volume_lad_mm3"]) < 1e-9
    assert abs(half["mass_lad"] - 0.5 * base["mass_lad"]) < 1e-9
    # Mean HU is independent of thickness.
    assert half["mean_hu_lad"] == base["mean_hu_lad"]


def test_empty_parse_result_all_zeros():
    parse = ParseResult(pid="TEST", slices=(), dirty_vessel_names=(), n_active_rois=0)
    out = compute_per_vessel_aggregates(parse, slice_thickness_mm=3.0)
    assert all(v == 0.0 for v in out.values())
