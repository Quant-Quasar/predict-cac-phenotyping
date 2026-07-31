"""Tests for predict.analyse.profiles (D023).

Coverage:
* cliffs_delta on identical / disjoint / half-overlap / mixed cases
* mannwhitney_u_pval matches scipy reference
* compute_cluster_profile produces expected rows on synthetic 2-cluster data
* apply_fdr_bh adjusts within (cohort, partition) bundles only
* is_robust_discriminator gate combines both criteria
* assert_biological_sanity raises on simulated soft-tissue focal cluster
* assert_label_balance raises on 10/90 split
* determine_focal_diffuse_mapping picks the lower-n_calcified-arteries cluster
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from predict.analyse.profiles import (
    apply_fdr_bh,
    assert_biological_sanity,
    assert_label_balance,
    cliffs_delta,
    compute_cluster_profile,
    determine_focal_diffuse_mapping,
    mannwhitney_u_pval,
)


# ─────────────────────── helpers ───────────────────────


def _two_clusters(seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """20-patient synthetic with one strong discriminator and one null."""
    rng = np.random.default_rng(seed)
    pids = [f"p{i:02d}" for i in range(20)]
    df = pd.DataFrame({
        "pid": pids,
        "discriminator": np.concatenate([
            rng.normal(loc=10.0, scale=1.0, size=10),
            rng.normal(loc=0.0, scale=1.0, size=10),
        ]),
        "null": rng.normal(loc=5.0, scale=1.0, size=20),
        "n_calcified_arteries": np.concatenate([
            np.full(10, 1, dtype=float),
            np.full(10, 4, dtype=float),
        ]),
        "max_hu_global": np.concatenate([
            rng.normal(loc=400, scale=50, size=10),
            rng.normal(loc=420, scale=50, size=10),
        ]),
    }).set_index("pid")
    labels = pd.Series(
        np.concatenate([np.zeros(10, dtype=int), np.ones(10, dtype=int)]),
        index=pids, name="cluster",
    )
    return df, labels


# ─────────────────────── cliffs_delta ───────────────────────


def test_cliffs_delta_identical_samples_zero():
    a = np.array([1.0, 2.0, 3.0])
    assert cliffs_delta(a, a) == 0.0


def test_cliffs_delta_disjoint_positive():
    a = np.array([10.0, 11.0, 12.0])
    b = np.array([0.0, 1.0, 2.0])
    # Every a > every b, so delta = +1.0
    assert cliffs_delta(a, b) == 1.0


def test_cliffs_delta_disjoint_negative():
    a = np.array([0.0, 1.0, 2.0])
    b = np.array([10.0, 11.0, 12.0])
    assert cliffs_delta(a, b) == -1.0


def test_cliffs_delta_symmetric():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 30)
    b = rng.normal(2, 1, 25)
    delta_ab = cliffs_delta(a, b)
    delta_ba = cliffs_delta(b, a)
    assert abs(delta_ab + delta_ba) < 1e-9


def test_cliffs_delta_half_overlap_known_value():
    # Two samples where exactly half of a > half of b.
    # a = [0, 0, 1, 1], b = [0, 0, 1, 1]. Pairs:
    #   (0,0)=tie x4, (0,1)=less x4, (1,0)=greater x4, (1,1)=tie x4.
    # n_greater - n_less = 4 - 4 = 0; total pairs = 16; delta = 0.
    a = np.array([0.0, 0.0, 1.0, 1.0])
    b = np.array([0.0, 0.0, 1.0, 1.0])
    assert cliffs_delta(a, b) == 0.0


def test_cliffs_delta_drops_nans():
    a = np.array([1.0, 2.0, np.nan, 3.0])
    b = np.array([0.0, np.nan, 0.0])
    # After NaN drop: a=[1,2,3], b=[0,0]; all a > all b; delta = +1.0
    assert cliffs_delta(a, b) == 1.0


def test_cliffs_delta_raises_on_empty_after_nan_drop():
    a = np.array([np.nan, np.nan])
    b = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="empty sample"):
        cliffs_delta(a, b)


# ─────────────────────── mannwhitney_u_pval ───────────────────────


def test_mannwhitney_matches_scipy():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 30)
    b = rng.normal(1, 1, 30)
    our_p = mannwhitney_u_pval(a, b, alternative="two-sided")
    ref = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    assert abs(our_p - ref) < 1e-12


def test_mannwhitney_one_sided_matches_scipy():
    rng = np.random.default_rng(0)
    a = rng.normal(2, 1, 30)
    b = rng.normal(0, 1, 30)
    p_greater = mannwhitney_u_pval(a, b, alternative="greater")
    ref_greater = stats.mannwhitneyu(a, b, alternative="greater").pvalue
    assert abs(p_greater - ref_greater) < 1e-12


def test_mannwhitney_returns_1_on_identical_constant():
    a = np.array([5.0, 5.0, 5.0])
    b = np.array([5.0, 5.0, 5.0])
    assert mannwhitney_u_pval(a, b) == 1.0


def test_mannwhitney_returns_1_on_empty_after_nan_drop():
    a = np.array([np.nan, np.nan])
    b = np.array([1.0, 2.0])
    assert mannwhitney_u_pval(a, b) == 1.0


# ─────────────────────── compute_cluster_profile ───────────────────────


def test_compute_cluster_profile_row_count():
    df, labels = _two_clusters()
    result = compute_cluster_profile(
        df, labels,
        feature_names=["discriminator", "null"],
        cohort="test", partition="test_partition",
    )
    # 2 clusters x 2 features = 4 rows
    assert len(result) == 4


def test_compute_cluster_profile_strong_discriminator():
    df, labels = _two_clusters()
    result = compute_cluster_profile(
        df, labels, feature_names=["discriminator"],
        cohort="test", partition="p",
    )
    # The discriminator should have abs(cliffs_delta) close to 1 in cluster 0
    # (cluster 0 has loc=10, cluster 1 has loc=0; one-vs-other is exact opposite)
    cluster_0 = result[result["cluster"] == "0"].iloc[0]
    cluster_1 = result[result["cluster"] == "1"].iloc[0]
    assert cluster_0["cliffs_delta"] > 0.9
    assert cluster_1["cliffs_delta"] < -0.9
    # Mann-Whitney should be highly significant
    assert cluster_0["mannwhitney_u_pval"] < 0.001
    assert cluster_1["mannwhitney_u_pval"] < 0.001


def test_compute_cluster_profile_null_feature_close_to_zero_delta():
    df, labels = _two_clusters()
    result = compute_cluster_profile(
        df, labels, feature_names=["null"],
        cohort="test", partition="p",
    )
    cluster_0 = result[result["cluster"] == "0"].iloc[0]
    assert abs(cluster_0["cliffs_delta"]) < 0.5


def test_compute_cluster_profile_includes_n_and_n_used():
    df, labels = _two_clusters()
    result = compute_cluster_profile(
        df, labels, feature_names=["discriminator"],
        cohort="test", partition="p",
    )
    for _, row in result.iterrows():
        assert row["n"] == 10
        assert row["n_used"] == 10
        assert not row["insufficient_data"]


def test_compute_cluster_profile_skips_missing_feature():
    df, labels = _two_clusters()
    result = compute_cluster_profile(
        df, labels, feature_names=["discriminator", "nonexistent_feature"],
        cohort="test", partition="p",
    )
    # Only discriminator -> 2 clusters x 1 = 2 rows
    assert len(result) == 2
    assert set(result["feature"]) == {"discriminator"}


def test_compute_cluster_profile_flags_insufficient_data():
    # 3 vs 17 split; cluster 0 has only 3 patients (< MIN_SAMPLES_PER_CLUSTER = 5)
    pids = [f"p{i:02d}" for i in range(20)]
    df = pd.DataFrame({
        "f": np.arange(20, dtype=float),
    }, index=pids)
    labels = pd.Series([0, 0, 0] + [1] * 17, index=pids)
    result = compute_cluster_profile(
        df, labels, feature_names=["f"],
        cohort="t", partition="t",
    )
    cluster_0_row = result[result["cluster"] == "0"].iloc[0]
    assert cluster_0_row["insufficient_data"]
    assert np.isnan(cluster_0_row["cliffs_delta"])
    assert np.isnan(cluster_0_row["mannwhitney_u_pval"])


def test_compute_cluster_profile_raises_on_no_shared_pids():
    df = pd.DataFrame({"f": [1.0, 2.0]}, index=["a", "b"])
    labels = pd.Series([0, 1], index=["x", "y"])
    with pytest.raises(ValueError, match="no shared pids"):
        compute_cluster_profile(
            df, labels, feature_names=["f"], cohort="t", partition="t",
        )


# ─────────────────────── apply_fdr_bh ───────────────────────


def test_apply_fdr_bh_adjusts_within_bundle():
    rng = np.random.default_rng(0)
    n_per_bundle = 10
    rows = []
    for cohort in ("A", "B"):
        for partition in ("p1",):
            for i in range(n_per_bundle):
                rows.append({
                    "cohort": cohort,
                    "partition": partition,
                    "cluster": "0",
                    "feature": f"f{i}",
                    "n": 100,
                    "n_used": 100,
                    "n_nonzero": 50,
                    "median": 0.0,
                    "iqr_lower": -1.0,
                    "iqr_upper": 1.0,
                    "cliffs_delta": rng.uniform(-0.5, 0.5),
                    "mannwhitney_u_pval": rng.uniform(0, 0.1),
                    "insufficient_data": False,
                })
    df = pd.DataFrame(rows)
    out = apply_fdr_bh(df)
    assert "fdr_bh_pval" in out.columns
    assert "is_robust_discriminator" in out.columns
    # FDR-adjusted p should be >= raw p
    eligible = out[~out["insufficient_data"]]
    assert (eligible["fdr_bh_pval"] >= eligible["mannwhitney_u_pval"]).all()


def test_apply_fdr_bh_strict_discriminator_gate():
    # One feature: raw p=0.001, delta=0.3 -> should be robust
    # Another: raw p=0.001, delta=0.05 -> NOT robust (delta below 0.20)
    # Another: raw p=0.10, delta=0.5 -> NOT robust (fdr above 0.05)
    df = pd.DataFrame({
        "cohort": ["A", "A", "A"],
        "partition": ["p", "p", "p"],
        "cluster": ["0", "0", "0"],
        "feature": ["strong", "tiny_effect", "weak_evidence"],
        "n": [100, 100, 100],
        "n_used": [100, 100, 100],
        "n_nonzero": [50, 50, 50],
        "median": [0.0, 0.0, 0.0],
        "iqr_lower": [-1.0, -1.0, -1.0],
        "iqr_upper": [1.0, 1.0, 1.0],
        "cliffs_delta": [0.3, 0.05, 0.5],
        "mannwhitney_u_pval": [0.001, 0.001, 0.10],
        "insufficient_data": [False, False, False],
    })
    out = apply_fdr_bh(df)
    out_idx = {row["feature"]: row for _, row in out.iterrows()}
    assert out_idx["strong"]["is_robust_discriminator"]
    assert not out_idx["tiny_effect"]["is_robust_discriminator"]
    assert not out_idx["weak_evidence"]["is_robust_discriminator"]


def test_apply_fdr_bh_excludes_insufficient_data_from_bh_denominator():
    # 10 features; 8 with p=0.04, 2 with insufficient_data.
    # Naive (10-test) BH on the 8 would give 0.05; correct (8-test) BH gives 0.04.
    rows = []
    for i in range(8):
        rows.append({
            "cohort": "A", "partition": "p", "cluster": "0",
            "feature": f"f{i}", "n": 100, "n_used": 100, "n_nonzero": 50,
            "median": 0.0, "iqr_lower": -1.0, "iqr_upper": 1.0,
            "cliffs_delta": 0.3, "mannwhitney_u_pval": 0.04,
            "insufficient_data": False,
        })
    for i in range(2):
        rows.append({
            "cohort": "A", "partition": "p", "cluster": "0",
            "feature": f"insuff{i}", "n": 100, "n_used": 3, "n_nonzero": 0,
            "median": 0.0, "iqr_lower": 0.0, "iqr_upper": 0.0,
            "cliffs_delta": np.nan, "mannwhitney_u_pval": np.nan,
            "insufficient_data": True,
        })
    df = pd.DataFrame(rows)
    out = apply_fdr_bh(df)
    eligible = out[~out["insufficient_data"]]
    # Adjusted p should be exactly 0.04 (since all 8 have raw p=0.04,
    # the BH ranks are uniform and the largest adjusted p is k*p/k = p)
    assert all(abs(p - 0.04) < 1e-12 for p in eligible["fdr_bh_pval"])


def test_apply_fdr_bh_separate_bundles_dont_interact():
    df = pd.DataFrame({
        "cohort": ["A", "A", "B", "B"],
        "partition": ["p", "p", "p", "p"],
        "cluster": ["0", "0", "0", "0"],
        "feature": ["x", "y", "x", "y"],
        "n": [100] * 4, "n_used": [100] * 4, "n_nonzero": [50] * 4,
        "median": [0.0] * 4,
        "iqr_lower": [-1.0] * 4, "iqr_upper": [1.0] * 4,
        "cliffs_delta": [0.3, 0.3, 0.3, 0.3],
        "mannwhitney_u_pval": [0.01, 0.04, 0.01, 0.04],
        "insufficient_data": [False] * 4,
    })
    out = apply_fdr_bh(df)
    # Each bundle has only 2 tests; BH on [0.01, 0.04] gives [0.02, 0.04]
    out_a = out[out["cohort"] == "A"].sort_values("feature")
    out_b = out[out["cohort"] == "B"].sort_values("feature")
    np.testing.assert_allclose(
        out_a["fdr_bh_pval"].to_numpy(),
        out_b["fdr_bh_pval"].to_numpy(),
    )


# ─────────────────────── biological sanity ───────────────────────


def test_assert_biological_sanity_passes_on_similar_calcium():
    """Both clusters at typical calcium levels (~400 HU)."""
    rng = np.random.default_rng(0)
    pids = [f"p{i:02d}" for i in range(20)]
    df = pd.DataFrame({
        "max_hu_global": np.concatenate([
            rng.normal(400, 50, 10),
            rng.normal(420, 50, 10),
        ]),
    }, index=pids)
    labels = pd.Series(["focal"] * 10 + ["diffuse"] * 10, index=pids)
    info = assert_biological_sanity(
        df, labels, focal_label="focal", diffuse_label="diffuse",
        cohort="test",
    )
    assert info["passes"]
    assert not info["warning_low_ratio"]
    assert info["focal_median"] >= 130


def test_assert_biological_sanity_passes_on_production_realistic_ratio():
    """Production COCA cohort pattern: focal ~387 HU, diffuse ~744 HU
    (ratio 0.52). Both unambiguously calcium-positive; the gate should
    PASS but log a warning for the low ratio."""
    rng = np.random.default_rng(0)
    pids = [f"p{i:02d}" for i in range(20)]
    df = pd.DataFrame({
        "max_hu_global": np.concatenate([
            rng.normal(387, 30, 10),  # focal
            rng.normal(744, 60, 10),  # diffuse
        ]),
    }, index=pids)
    labels = pd.Series(["focal"] * 10 + ["diffuse"] * 10, index=pids)
    info = assert_biological_sanity(
        df, labels, focal_label="focal", diffuse_label="diffuse",
        cohort="production_realistic",
    )
    # Focal at 387 HU is well above the 130 HU calcium floor -> gate passes
    assert info["passes"]
    # Ratio (~0.52) is below the warning threshold (0.5) is a close call;
    # ~0.52 should trip OR not trip depending on noise. The point of this
    # test is that the gate does NOT raise.


def test_assert_biological_sanity_raises_on_soft_tissue_focal():
    """Focal cluster below the 130 HU calcium floor -> gate raises."""
    rng = np.random.default_rng(0)
    pids = [f"p{i:02d}" for i in range(20)]
    df = pd.DataFrame({
        "max_hu_global": np.concatenate([
            rng.normal(50, 10, 10),    # focal: soft tissue
            rng.normal(420, 50, 10),   # diffuse: calcium
        ]),
    }, index=pids)
    labels = pd.Series(["focal"] * 10 + ["diffuse"] * 10, index=pids)
    with pytest.raises(ValueError, match=r"< 130(\.0)? HU"):
        assert_biological_sanity(
            df, labels, focal_label="focal", diffuse_label="diffuse",
            cohort="test",
        )


def test_assert_biological_sanity_raises_on_focal_just_below_floor():
    """Boundary: focal median at 129 HU just below the 130 HU floor."""
    pids = [f"p{i:02d}" for i in range(20)]
    df = pd.DataFrame({
        "max_hu_global": [129.0] * 10 + [500.0] * 10,
    }, index=pids)
    labels = pd.Series(["focal"] * 10 + ["diffuse"] * 10, index=pids)
    with pytest.raises(ValueError, match="IBSI calcium floor"):
        assert_biological_sanity(
            df, labels, focal_label="focal", diffuse_label="diffuse",
            cohort="test",
        )


def test_assert_biological_sanity_passes_at_floor_exactly():
    """Boundary: focal at exactly 130 HU passes (>= floor)."""
    pids = [f"p{i:02d}" for i in range(20)]
    df = pd.DataFrame({
        "max_hu_global": [130.0] * 10 + [500.0] * 10,
    }, index=pids)
    labels = pd.Series(["focal"] * 10 + ["diffuse"] * 10, index=pids)
    info = assert_biological_sanity(
        df, labels, focal_label="focal", diffuse_label="diffuse",
        cohort="test",
    )
    assert info["passes"]


def test_assert_biological_sanity_raises_on_negative_focal_median():
    """Negative HU values in the focal cluster (severe soft-tissue
    regression) hit the absolute floor."""
    pids = [f"p{i:02d}" for i in range(20)]
    df = pd.DataFrame({
        "max_hu_global": [-100.0] * 10 + [500.0] * 10,
    }, index=pids)
    labels = pd.Series(["focal"] * 10 + ["diffuse"] * 10, index=pids)
    with pytest.raises(ValueError, match="IBSI calcium floor"):
        assert_biological_sanity(
            df, labels, focal_label="focal", diffuse_label="diffuse",
            cohort="test",
        )


def test_assert_biological_sanity_warning_flag_set_on_low_ratio():
    """Focal at calcium levels, diffuse much higher: gate passes but the
    warning_low_ratio flag is set."""
    pids = [f"p{i:02d}" for i in range(20)]
    df = pd.DataFrame({
        "max_hu_global": [200.0] * 10 + [800.0] * 10,
    }, index=pids)
    labels = pd.Series(["focal"] * 10 + ["diffuse"] * 10, index=pids)
    info = assert_biological_sanity(
        df, labels, focal_label="focal", diffuse_label="diffuse",
        cohort="test",
    )
    assert info["passes"]
    # ratio = 200/800 = 0.25 < 0.5 -> warning
    assert info["warning_low_ratio"]
    assert abs(info["ratio"] - 0.25) < 1e-9


# ─────────────────────── label balance ───────────────────────


def test_assert_label_balance_passes_on_balanced():
    labels = pd.Series([0] * 60 + [1] * 40)
    info = assert_label_balance(labels, cohort="t", partition="t")
    assert info["passes"]
    assert info["minority_fraction"] == 0.40


def test_assert_label_balance_raises_on_imbalanced():
    labels = pd.Series([0] * 90 + [1] * 10)
    with pytest.raises(ValueError, match="too imbalanced"):
        assert_label_balance(labels, cohort="t", partition="t")


def test_assert_label_balance_raises_on_empty():
    labels = pd.Series([], dtype=int)
    with pytest.raises(ValueError, match="empty labels"):
        assert_label_balance(labels, cohort="t", partition="t")


def test_assert_label_balance_handles_three_clusters():
    # 60/30/10 split; minority is 10/100 = 0.10 < 0.15
    labels = pd.Series([0] * 60 + [1] * 30 + [2] * 10)
    with pytest.raises(ValueError, match="too imbalanced"):
        assert_label_balance(labels, cohort="t", partition="t")


# ─────────────────────── focal/diffuse mapping ───────────────────────


def test_determine_focal_diffuse_mapping_basic():
    df, labels = _two_clusters()
    # cluster 0 has n_calcified_arteries=1, cluster 1 has 4
    # So cluster 0 should be 'focal'
    mapping = determine_focal_diffuse_mapping(df, labels)
    assert mapping[0] == "focal"
    assert mapping[1] == "diffuse"


def test_determine_focal_diffuse_mapping_reversed():
    # Now cluster 0 has 4 arteries; cluster 1 has 1.
    pids = [f"p{i:02d}" for i in range(20)]
    df = pd.DataFrame({
        "n_calcified_arteries": np.concatenate([
            np.full(10, 4, dtype=float),
            np.full(10, 1, dtype=float),
        ]),
    }, index=pids)
    labels = pd.Series([0] * 10 + [1] * 10, index=pids)
    mapping = determine_focal_diffuse_mapping(df, labels)
    assert mapping[0] == "diffuse"
    assert mapping[1] == "focal"


def test_determine_focal_diffuse_mapping_raises_on_3_clusters():
    pids = [f"p{i:02d}" for i in range(30)]
    df = pd.DataFrame({
        "n_calcified_arteries": np.arange(30, dtype=float) % 4 + 1,
    }, index=pids)
    labels = pd.Series([0] * 10 + [1] * 10 + [2] * 10, index=pids)
    with pytest.raises(ValueError, match="expected exactly 2"):
        determine_focal_diffuse_mapping(df, labels)


def test_determine_focal_diffuse_mapping_raises_on_tied_medians():
    pids = [f"p{i:02d}" for i in range(20)]
    df = pd.DataFrame({
        "n_calcified_arteries": np.full(20, 2.0),
    }, index=pids)
    labels = pd.Series([0] * 10 + [1] * 10, index=pids)
    with pytest.raises(ValueError, match="identical median"):
        determine_focal_diffuse_mapping(df, labels)
