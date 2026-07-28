"""Regression guard for the D016 bypass.

The 68 canonical features bypass the empirical ICC gate (D016) because the
modules that produce them do not read the CT array. This file enforces that
invariant via two complementary checks:

1. **Static check**: grep every module in ``src/predict/features/`` for any
   reference to a CT array or its load helpers. Zero hits is required.
2. **Functional check**: build two synthetic ``ParseResult`` objects that are
   identical apart from arbitrary external state, call every bypass-producing
   function on both, and assert byte-identical output.

If either check fails, the bypass list in
:func:`predict.stability.icc.invariant_by_construction_features` must be
re-examined before D016 can stay active.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from predict.features.agatston import compute_agatston
from predict.features.density_tiers import compute_density_tiers
from predict.features.lesion_ccl import group_rois_into_lesions
from predict.features.per_vessel_aggregates import compute_per_vessel_aggregates
from predict.features.spatial import compute_spatial_features
from predict.io.xml_parser import ROI, ParseResult, SliceAnnotation


# ─────────────────────── module list ───────────────────────


# Every module that contributes to the 68 canonical features. If a new feature
# module is added that produces canonical features, append it here.
CANONICAL_MODULES = (
    "src/predict/features/feature_schema.py",
    "src/predict/features/agatston.py",
    "src/predict/features/per_vessel_aggregates.py",
    "src/predict/features/density_tiers.py",
    "src/predict/features/lesion_ccl.py",
    "src/predict/features/spatial.py",
    "src/predict/preprocess/slice_matcher.py",   # imported by lesion_ccl
)

# Any of these substrings appearing in a CANONICAL_MODULES source file
# constitutes a CT-array dependency and means the bypass list is unsafe.
CT_DEPENDENCY_PATTERNS = (
    r"\bct_array\b",
    r"\bct_np\b",
    r"\bct_sitk\b",
    r"sitk\.GetArrayFromImage",
    r"_ct\.npy",
    r"np\.load.*ct",
    r"load_patient_dicom",       # the CT loader
    r"LoadedPatient",            # the CT-bearing dataclass
)


# ─────────────────────── static check ───────────────────────


@pytest.mark.parametrize("module_rel", CANONICAL_MODULES)
def test_canonical_modules_do_not_reference_ct_array(module_rel: str) -> None:
    """Every canonical-feature module must contain zero CT-array references.

    Strips comments and docstrings before searching so that a documentation
    mention (e.g. 'as load_patient_dicom does') in slice_matcher.py does not
    trip the guard. Only executable code is inspected.
    """
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / module_rel).read_text(encoding="utf-8")

    # Strip triple-quoted strings (docstrings) and single-line comments.
    src_no_doc = re.sub(r'(?s)""".*?"""', "", src)
    src_no_doc = re.sub(r"(?s)'''.*?'''", "", src_no_doc)
    src_no_doc = re.sub(r"#[^\n]*", "", src_no_doc)

    hits: list[tuple[str, str]] = []
    for pattern in CT_DEPENDENCY_PATTERNS:
        for m in re.finditer(pattern, src_no_doc):
            hits.append((pattern, m.group(0)))

    assert not hits, (
        f"{module_rel} contains CT-array references in executable code: {hits}\n"
        f"D016 bypass is unsafe; update the bypass registry or remove the dependency."
    )


# ─────────────────────── functional check ───────────────────────


def _roi(
    image_index: int,
    vessel_label: str,
    max_hu: float = 250.0,
    mean_hu: float = 200.0,
    area_cm2: float = 0.05,
) -> ROI:
    """Build a synthetic ROI matching the real dataclass signature."""
    # 4-vertex square ~7 x 7 px (mapped to mm via spacing in downstream code).
    pts_px = ((100.0, 100.0), (107.0, 100.0), (107.0, 107.0), (100.0, 107.0))
    z = image_index * 3.0
    pts_mm = tuple((float(x), float(y), z) for (x, y) in pts_px)
    return ROI(
        vessel_raw=vessel_label,
        vessel=vessel_label,
        area_cm2=area_cm2,
        mean_hu=mean_hu,
        max_hu=max_hu,
        min_hu=130.0,
        total_hu=mean_hu * 49.0,
        n_points=len(pts_px),
        points_px=pts_px,
        points_mm=pts_mm,
        center_xyz=(103.5, 103.5, z),
    )


def _parse_result() -> ParseResult:
    """Two slices, three ROIs, all calcium-positive."""
    s1 = SliceAnnotation(image_index=10, rois=(
        _roi(10, "LAD", max_hu=350.0),
        _roi(10, "RCA", max_hu=180.0),
    ))
    s2 = SliceAnnotation(image_index=11, rois=(
        _roi(11, "LAD", max_hu=420.0),
    ))
    return ParseResult(pid="test", slices=(s1, s2), n_active_rois=3)


def test_compute_agatston_byte_identical_across_calls():
    pr = _parse_result()
    a = compute_agatston(pr, slice_thickness_mm=3.0, excluded_roi_ids=set())
    b = compute_agatston(pr, slice_thickness_mm=3.0, excluded_roi_ids=set())
    assert a.to_feature_dict() == b.to_feature_dict()
    assert a.category == b.category


def test_compute_per_vessel_aggregates_byte_identical_across_calls():
    pr = _parse_result()
    a = compute_per_vessel_aggregates(pr, slice_thickness_mm=3.0, excluded_roi_ids=set())
    b = compute_per_vessel_aggregates(pr, slice_thickness_mm=3.0, excluded_roi_ids=set())
    assert a == b


def test_compute_density_tiers_byte_identical_across_calls():
    pr = _parse_result()
    a = compute_density_tiers(pr, excluded_roi_ids=set())
    b = compute_density_tiers(pr, excluded_roi_ids=set())
    assert a == b


def test_spatial_features_byte_identical_across_calls():
    pr = _parse_result()
    slice_positions = tuple(float(k) * 3.0 for k in range(20))  # 0..57 mm
    pixel_spacing_xy = (0.5, 0.5)
    lpv_a = group_rois_into_lesions(
        pr, slice_positions=slice_positions,
        pixel_spacing_xy=pixel_spacing_xy, slice_thickness_mm=3.0,
        excluded_roi_ids=set(),
    )
    lpv_b = group_rois_into_lesions(
        pr, slice_positions=slice_positions,
        pixel_spacing_xy=pixel_spacing_xy, slice_thickness_mm=3.0,
        excluded_roi_ids=set(),
    )
    a = compute_spatial_features(lpv_a, slice_positions=slice_positions,
                                 slice_thickness_mm=3.0)
    b = compute_spatial_features(lpv_b, slice_positions=slice_positions,
                                 slice_thickness_mm=3.0)
    assert a == b
