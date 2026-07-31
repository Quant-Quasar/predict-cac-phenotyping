"""Tests for predict.analyse.orthogonality (D024).

Coverage:
* 3-level interpretation classification on the four corner cases
* test_burden_orthogonality raises on empty samples
* burden_tertile_assignment produces equal-sized tertiles on smooth data
* burden_stratified_spatial_replication builds the (tertile x hypothesis) table
* burden_stratified_pass_verdict on synthetic
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predict.analyse.orthogonality import (
    BurdenOrthogonalityResult,
    _classify_interpretation,
    assess_burden_orthogonality,
    burden_stratified_pass_verdict,
    burden_stratified_spatial_replication,
    burden_tertile_assignment,
)


# ─────────────────────── 3-level interpretation ───────────────────────


def test_classify_orthogonal():
    # p > 0.05 AND |delta| < 0.20
    assert _classify_interpretation(pval=0.30, delta=0.05) == "orthogonal"
    assert _classify_interpretation(pval=0.30, delta=-0.05) == "orthogonal"


def test_classify_confounded():
    # p < 0.05 AND |delta| >= 0.20
    assert _classify_interpretation(pval=0.001, delta=0.40) == "confounded"
    assert _classify_interpretation(pval=0.04, delta=-0.30) == "confounded"


def test_classify_marginal_significant_but_small():
    # p < 0.05 AND |delta| < 0.20  (significant but trivial)
    assert _classify_interpretation(pval=0.001, delta=0.08) == "marginal"


def test_classify_marginal_underpowered_visible():
    # p >= 0.05 AND |delta| >= 0.20  (underpowered but visible)
    assert _classify_interpretation(pval=0.12, delta=0.30) == "marginal"


def test_classify_boundary_orthogonal_at_p_exact():
    # p = 0.05 exactly is NOT < 0.05; so it's the "not significant" arm
    assert _classify_interpretation(pval=0.05, delta=0.05) == "orthogonal"


# ─────────────────────── test_burden_orthogonality ───────────────────────


def test_burden_orthogonality_returns_dataclass():
    rng = np.random.default_rng(0)
    focal = rng.normal(100, 10, 50)
    diffuse = rng.normal(100, 10, 50)
    result = assess_burden_orthogonality(focal, diffuse, cohort="t")
    assert isinstance(result, BurdenOrthogonalityResult)
    assert result.cohort == "t"
    assert result.n_focal == 50
    assert result.n_diffuse == 50


def test_burden_orthogonality_orthogonal_on_identical_distributions():
    rng = np.random.default_rng(0)
    focal = rng.normal(100, 10, 200)
    diffuse = rng.normal(100, 10, 200)
    result = assess_burden_orthogonality(focal, diffuse, cohort="t")
    # Same distribution -> high p, small delta
    assert result.interpretation == "orthogonal"
    assert result.passes
    assert abs(result.cliffs_delta_agatston) < 0.20


def test_burden_orthogonality_confounded_on_shifted_distributions():
    rng = np.random.default_rng(0)
    focal = rng.normal(50, 10, 200)  # low burden
    diffuse = rng.normal(200, 10, 200)  # high burden
    result = assess_burden_orthogonality(focal, diffuse, cohort="t")
    assert result.interpretation == "confounded"
    assert not result.passes
    assert abs(result.cliffs_delta_agatston) >= 0.20
    assert result.mannwhitney_pval < 0.05


def test_burden_orthogonality_levene_detects_variance_difference():
    rng = np.random.default_rng(0)
    focal = rng.normal(100, 5, 200)  # same mean, small variance
    diffuse = rng.normal(100, 30, 200)  # same mean, large variance
    result = assess_burden_orthogonality(focal, diffuse, cohort="t")
    # Mann-Whitney medians should NOT differ; Levene SHOULD detect variance
    assert result.levene_pval < 0.05


def test_burden_orthogonality_raises_on_empty_after_nan_drop():
    focal = np.array([np.nan, np.nan])
    diffuse = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="empty focal"):
        assess_burden_orthogonality(focal, diffuse, cohort="t")


# ─────────────────────── burden tertile assignment ───────────────────────


def test_burden_tertile_assignment_smooth_data():
    # 30 patients with uniform burden 0..29
    agatston = pd.Series(np.arange(30, dtype=float), index=[f"p{i}" for i in range(30)])
    tertiles = burden_tertile_assignment(agatston, n_tertiles=3)
    counts = tertiles.value_counts().sort_index().to_dict()
    # Roughly equal tertile sizes
    assert all(9 <= count <= 11 for count in counts.values())
    assert sorted(counts.keys()) == [0, 1, 2]


def test_burden_tertile_assignment_propagates_nan():
    agatston = pd.Series(
        [np.nan, 10.0, 20.0, 30.0, 40.0, 50.0, np.nan],
        index=[f"p{i}" for i in range(7)],
    )
    tertiles = burden_tertile_assignment(agatston, n_tertiles=3)
    assert pd.isna(tertiles.iloc[0])
    assert pd.isna(tertiles.iloc[-1])


# ─────────────────────── burden-stratified replication ───────────────────────


def _focal_diffuse_with_spatial_signal(seed: int = 0):
    """Build a 120-patient cohort with planted focal/diffuse signal that
    holds within each burden tertile."""
    rng = np.random.default_rng(seed)
    n_per = 60
    pids = [f"p{i:03d}" for i in range(2 * n_per)]
    # Burden span 0-300 in each cluster (so tertiles split each cluster ~evenly)
    agatston = np.concatenate([rng.uniform(0, 300, n_per), rng.uniform(0, 300, n_per)])
    # Focal has higher lesion_count_lad, lower dist_from_top_max
    lesion_count_lad = np.concatenate([
        rng.poisson(5, n_per),  # focal: mean 5
        rng.poisson(1, n_per),  # diffuse: mean 1
    ]).astype(float)
    dist_from_top_max = np.concatenate([
        rng.uniform(10, 30, n_per),   # focal: 10-30
        rng.uniform(50, 100, n_per),  # diffuse: 50-100
    ])
    df = pd.DataFrame({
        "agatston_total": agatston,
        "lesion_count_lad": lesion_count_lad,
        "dist_from_top_max": dist_from_top_max,
    }, index=pids)
    labels = pd.Series(["focal"] * n_per + ["diffuse"] * n_per, index=pids)
    return df, labels


def test_burden_stratified_spatial_replication_table_shape():
    df, labels = _focal_diffuse_with_spatial_signal()
    directional = [
        ("lesion_count_lad", "focal>diffuse"),
        ("dist_from_top_max", "focal<diffuse"),
    ]
    result = burden_stratified_spatial_replication(
        df[["lesion_count_lad", "dist_from_top_max"]],
        labels, df["agatston_total"],
        directional_features=directional, cohort="t",
    )
    # 3 tertiles x 2 features = 6 rows
    assert len(result) == 6
    assert set(result["tertile"]) == {0, 1, 2}
    assert set(result["feature"]) == {"lesion_count_lad", "dist_from_top_max"}


def test_burden_stratified_signal_holds_within_tertiles():
    df, labels = _focal_diffuse_with_spatial_signal()
    directional = [
        ("lesion_count_lad", "focal>diffuse"),
        ("dist_from_top_max", "focal<diffuse"),
    ]
    result = burden_stratified_spatial_replication(
        df[["lesion_count_lad", "dist_from_top_max"]],
        labels, df["agatston_total"],
        directional_features=directional, cohort="t",
    )
    # Both features should match predicted direction in all 3 tertiles
    assert result["direction_match"].all()


def test_burden_stratified_pass_verdict_pass():
    # 6 hypotheses x 3 tertiles; assume 5 match in tertile 0, 5 in tertile 1, 2 in tertile 2
    rows = []
    for tertile, n_match in zip([0, 1, 2], [5, 5, 2]):
        for i in range(6):
            rows.append({
                "tertile": tertile,
                "direction_match": i < n_match,
            })
    df = pd.DataFrame(rows)
    verdict = burden_stratified_pass_verdict(df)
    assert verdict["per_tertile_match_count"] == {0: 5, 1: 5, 2: 2}
    assert verdict["tertiles_passing"] == 2
    assert verdict["passes"]


def test_burden_stratified_pass_verdict_fail():
    # Only 1 of 3 tertiles passes
    rows = []
    for tertile, n_match in zip([0, 1, 2], [5, 1, 0]):
        for i in range(6):
            rows.append({
                "tertile": tertile,
                "direction_match": i < n_match,
            })
    df = pd.DataFrame(rows)
    verdict = burden_stratified_pass_verdict(df)
    assert verdict["tertiles_passing"] == 1
    assert not verdict["passes"]
