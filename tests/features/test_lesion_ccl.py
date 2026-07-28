"""Tests for predict.features.lesion_ccl.

The test helpers fabricate ROIs by specifying their pixel-coordinate centroid
and ``image_index``; we then call ``group_rois_into_lesions`` with a synthetic
``slice_positions`` list so the matched slice index equals ``image_index``.
"""
from __future__ import annotations

from predict.features.lesion_ccl import group_rois_into_lesions
from predict.io.xml_parser import ROI, ParseResult, SliceAnnotation


# Synthetic slice grid: 30 slices at 3 mm spacing starting at z=0.
SLICE_POSITIONS = tuple(float(k) * 3.0 for k in range(30))
PIXEL_SPACING_XY = (0.5, 0.5)
SLICE_THICKNESS = 3.0


def _square_polygon_px(cx: float, cy: float, half: float = 5.0):
    """Axis-aligned square polygon centred at (cx, cy) with half-side ``half``."""
    return (
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
    )


def _roi(
    *,
    vessel: str | None,
    cx_px: float,
    cy_px: float,
    z_mm: float,
    area_cm2: float = 0.04,
    max_hu: float = 300.0,
    mean_hu: float = 200.0,
    vessel_raw: str = "Left Anterior Descending Artery",
) -> ROI:
    pts_px = _square_polygon_px(cx_px, cy_px)
    return ROI(
        vessel_raw=vessel_raw,
        vessel=vessel,
        area_cm2=area_cm2,
        mean_hu=mean_hu,
        max_hu=max_hu,
        min_hu=130.0,
        total_hu=1000.0,
        n_points=4,
        points_px=pts_px,
        points_mm=tuple((p[0] * 0.5, p[1] * 0.5, z_mm) for p in pts_px),
        center_xyz=(cx_px * 0.5, cy_px * 0.5, z_mm),
    )


def _parse(slice_to_rois: dict[int, list[ROI]]) -> ParseResult:
    slices = tuple(
        SliceAnnotation(image_index=k, rois=tuple(v))
        for k, v in sorted(slice_to_rois.items())
    )
    return ParseResult(
        pid="TEST",
        slices=slices,
        dirty_vessel_names=(),
        n_active_rois=sum(len(v) for v in slice_to_rois.values()),
    )


def _group(parse, **kw):
    return group_rois_into_lesions(
        parse,
        slice_positions=SLICE_POSITIONS,
        pixel_spacing_xy=PIXEL_SPACING_XY,
        slice_thickness_mm=SLICE_THICKNESS,
        **kw,
    )


# ────────────────────────────────────────────────────────────────────────
# Edge-by-edge BFS behaviour
# ────────────────────────────────────────────────────────────────────────


def test_two_adjacent_close_rois_merge():
    """Same vessel, adjacent slices, centroids 2 mm apart in plane → 1 lesion."""
    parse = _parse({
        5: [_roi(vessel="LAD", cx_px=100, cy_px=100, z_mm=SLICE_POSITIONS[5])],
        6: [_roi(vessel="LAD", cx_px=104, cy_px=100, z_mm=SLICE_POSITIONS[6])],
        # XY centroid distance: 4 px × 0.5 mm/px = 2 mm  →  < 5 mm threshold
    })
    out = _group(parse)
    assert len(out["LAD"]) == 1
    assert out["LAD"][0].n_rois == 2


def test_two_adjacent_far_rois_split():
    """Same vessel, adjacent slices, centroids > 5 mm apart → 2 lesions."""
    parse = _parse({
        5: [_roi(vessel="LAD", cx_px=100, cy_px=100, z_mm=SLICE_POSITIONS[5])],
        6: [_roi(vessel="LAD", cx_px=200, cy_px=100, z_mm=SLICE_POSITIONS[6])],
        # XY distance: 100 px × 0.5 = 50 mm  →  > 5 mm
    })
    out = _group(parse)
    assert len(out["LAD"]) == 2


def test_slice_gap_two_splits():
    """Same vessel, |slice_idx diff| = 2 with default gap=1 → 2 lesions."""
    parse = _parse({
        5: [_roi(vessel="LAD", cx_px=100, cy_px=100, z_mm=SLICE_POSITIONS[5])],
        7: [_roi(vessel="LAD", cx_px=100, cy_px=100, z_mm=SLICE_POSITIONS[7])],
    })
    out = _group(parse)
    assert len(out["LAD"]) == 2


def test_slice_gap_two_with_relaxed_gap_merges():
    """Bumping max_slice_gap to 2 merges the previous case."""
    parse = _parse({
        5: [_roi(vessel="LAD", cx_px=100, cy_px=100, z_mm=SLICE_POSITIONS[5])],
        7: [_roi(vessel="LAD", cx_px=100, cy_px=100, z_mm=SLICE_POSITIONS[7])],
    })
    out = _group(parse, max_slice_gap=2)
    assert len(out["LAD"]) == 1
    assert out["LAD"][0].n_rois == 2


def test_same_slice_far_rois_are_separate_lesions():
    """Two ROIs on the SAME slice in the same vessel, > 5 mm apart → 2 lesions."""
    parse = _parse({
        5: [
            _roi(vessel="LAD", cx_px=100, cy_px=100, z_mm=SLICE_POSITIONS[5]),
            _roi(vessel="LAD", cx_px=200, cy_px=100, z_mm=SLICE_POSITIONS[5]),
        ],
    })
    out = _group(parse)
    assert len(out["LAD"]) == 2


def test_same_slice_close_rois_merge():
    """Two ROIs on the SAME slice in the same vessel, close → 1 lesion."""
    parse = _parse({
        5: [
            _roi(vessel="LAD", cx_px=100, cy_px=100, z_mm=SLICE_POSITIONS[5]),
            _roi(vessel="LAD", cx_px=103, cy_px=100, z_mm=SLICE_POSITIONS[5]),
            # 3 px × 0.5 = 1.5 mm < 5 mm
        ],
    })
    out = _group(parse)
    assert len(out["LAD"]) == 1
    assert out["LAD"][0].n_rois == 2


def test_cross_vessel_never_merges():
    """Identical XY/Z but different vessels → always separate."""
    parse = _parse({
        5: [
            _roi(vessel="LAD", cx_px=100, cy_px=100, z_mm=SLICE_POSITIONS[5]),
            _roi(vessel="RCA", cx_px=100, cy_px=100, z_mm=SLICE_POSITIONS[5],
                 vessel_raw="Right Coronary Artery"),
        ],
    })
    out = _group(parse)
    assert len(out["LAD"]) == 1
    assert len(out["RCA"]) == 1


# ────────────────────────────────────────────────────────────────────────
# Eligibility filters
# ────────────────────────────────────────────────────────────────────────


def test_dirty_vessel_ignored():
    parse = _parse({
        5: [
            _roi(vessel="LAD", cx_px=100, cy_px=100, z_mm=SLICE_POSITIONS[5]),
            _roi(vessel=None, cx_px=101, cy_px=100, z_mm=SLICE_POSITIONS[5],
                 vessel_raw="555614876"),
        ],
    })
    out = _group(parse)
    assert len(out["LAD"]) == 1
    assert out["LAD"][0].n_rois == 1


def test_excluded_roi_ids_skipped():
    parse = _parse({
        5: [
            _roi(vessel="LAD", cx_px=100, cy_px=100, z_mm=SLICE_POSITIONS[5]),
            _roi(vessel="LAD", cx_px=101, cy_px=100, z_mm=SLICE_POSITIONS[5]),
        ],
    })
    out = _group(parse, excluded_roi_ids={(5, 1)})
    assert len(out["LAD"]) == 1
    assert out["LAD"][0].n_rois == 1
    assert out["LAD"][0].roi_keys == ((5, 0),)


def test_polygon_too_few_points_ignored():
    parse = ParseResult(
        pid="TEST",
        slices=(
            SliceAnnotation(image_index=5, rois=(
                ROI(
                    vessel_raw="Left Anterior Descending Artery", vessel="LAD",
                    area_cm2=0.04, mean_hu=200, max_hu=300, min_hu=130, total_hu=800,
                    n_points=2,                  # < 3 → dropped
                    points_px=((100.0, 100.0), (101.0, 100.0)),
                    points_mm=((50.0, 50.0, 15.0), (50.5, 50.0, 15.0)),
                    center_xyz=(50.25, 50.0, 15.0),
                ),
            )),
        ),
        dirty_vessel_names=(),
        n_active_rois=1,
    )
    out = _group(parse)
    assert out["LAD"] == []


# ────────────────────────────────────────────────────────────────────────
# Lesion attribute correctness
# ────────────────────────────────────────────────────────────────────────


def test_lesion_volume_is_area_times_thickness():
    parse = _parse({
        5: [_roi(vessel="LAD", cx_px=100, cy_px=100, z_mm=SLICE_POSITIONS[5],
                 area_cm2=0.10)],
        6: [_roi(vessel="LAD", cx_px=100, cy_px=100, z_mm=SLICE_POSITIONS[6],
                 area_cm2=0.05)],
    })
    out = _group(parse)
    lesion = out["LAD"][0]
    assert abs(lesion.total_area_mm2 - (0.10 + 0.05) * 100.0) < 1e-9
    assert abs(lesion.volume_mm3 - lesion.total_area_mm2 * SLICE_THICKNESS) < 1e-9


def test_lesion_centroid_area_weighted():
    parse = _parse({
        5: [_roi(vessel="LAD", cx_px=100, cy_px=100, z_mm=SLICE_POSITIONS[5],
                 area_cm2=0.10)],
        6: [_roi(vessel="LAD", cx_px=104, cy_px=100, z_mm=SLICE_POSITIONS[6],
                 area_cm2=0.10)],
    })
    out = _group(parse)
    lesion = out["LAD"][0]
    # Both areas equal → mean x = 51 mm, y = 50 mm, z = (15+18)/2 = 16.5 mm.
    assert abs(lesion.centroid_mm[0] - 51.0) < 1e-6
    assert abs(lesion.centroid_mm[1] - 50.0) < 1e-6
    assert abs(lesion.centroid_mm[2] - 16.5) < 1e-6


def test_lesion_max_and_weighted_mean_hu():
    parse = _parse({
        5: [_roi(vessel="LAD", cx_px=100, cy_px=100, z_mm=SLICE_POSITIONS[5],
                 area_cm2=0.10, mean_hu=200, max_hu=300)],
        6: [_roi(vessel="LAD", cx_px=100, cy_px=100, z_mm=SLICE_POSITIONS[6],
                 area_cm2=0.10, mean_hu=400, max_hu=500)],
    })
    lesion = _group(parse)["LAD"][0]
    assert lesion.max_hu == 500
    # Equal areas → unweighted average.
    assert abs(lesion.mean_hu_weighted - 300.0) < 1e-6


def test_roi_keys_sorted_and_slice_indices_unique():
    parse = _parse({
        5: [
            _roi(vessel="LAD", cx_px=100, cy_px=100, z_mm=SLICE_POSITIONS[5]),
            _roi(vessel="LAD", cx_px=101, cy_px=100, z_mm=SLICE_POSITIONS[5]),
        ],
        6: [
            _roi(vessel="LAD", cx_px=101, cy_px=100, z_mm=SLICE_POSITIONS[6]),
        ],
    })
    lesion = _group(parse)["LAD"][0]
    # 3 ROIs total in one lesion; roi_keys sorted by (image_index, roi_idx).
    assert lesion.roi_keys == ((5, 0), (5, 1), (6, 0))
    # Two unique slice indices used.
    assert lesion.slice_indices == (5, 6)


# ────────────────────────────────────────────────────────────────────────
# Whole-output shape and determinism
# ────────────────────────────────────────────────────────────────────────


def test_output_always_has_four_vessel_keys():
    parse = _parse({
        5: [_roi(vessel="LAD", cx_px=100, cy_px=100, z_mm=SLICE_POSITIONS[5])],
    })
    out = _group(parse)
    assert set(out.keys()) == {"LAD", "RCA", "LCx", "LM"}
    assert out["RCA"] == [] and out["LCx"] == [] and out["LM"] == []


def test_empty_parse_yields_empty_lists():
    parse = ParseResult(pid="TEST", slices=(), dirty_vessel_names=(), n_active_rois=0)
    out = _group(parse)
    assert all(out[v] == [] for v in ("LAD", "RCA", "LCx", "LM"))


def test_unmatched_roi_skipped():
    """An ROI with center_xyz far outside slice_positions is skipped (no fallback
    needed because we also remove image_index)."""
    # Build a parse with a single ROI whose center is far away and image_index
    # is also outside the n_slices range so the fallback fails too.
    far_roi = ROI(
        vessel_raw="Left Anterior Descending Artery", vessel="LAD",
        area_cm2=0.05, mean_hu=200, max_hu=300, min_hu=130, total_hu=800,
        n_points=4,
        points_px=((100.0, 100.0), (105.0, 100.0), (105.0, 105.0), (100.0, 105.0)),
        points_mm=((50.0, 50.0, 9999.0),) * 4,
        center_xyz=(50.0, 50.0, 9999.0),
    )
    parse = ParseResult(
        pid="TEST",
        slices=(SliceAnnotation(image_index=9999, rois=(far_roi,)),),
        dirty_vessel_names=(),
        n_active_rois=1,
    )
    out = _group(parse)
    assert out["LAD"] == []
