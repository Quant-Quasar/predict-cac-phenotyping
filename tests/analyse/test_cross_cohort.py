"""Tests for predict.analyse.cross_cohort (D027).

Coverage:
* consistency_table: each rule individually + combined criterion on
  synthetic per-cohort profile data
* robust_discriminator_count_summary aggregates correctly
* partition_ari_table reuses discover.validity.ari_on_shared_pids and
  applies the 0.80 cutoff
* empty input handled with explicit error message
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predict.analyse.cross_cohort import (
    ARI_PASS_THRESHOLD,
    consistency_table,
    partition_ari_table,
    robust_discriminator_count_summary,
)


def _profile_for(
    cohort: str,
    features: list[str],
    deltas: list[float],
    pvals: list[float],
) -> pd.DataFrame:
    """Build a minimal profile_df for one cohort with the given features."""
    return pd.DataFrame({
        "cohort": [cohort] * len(features),
        "partition": ["spatial_k2"] * len(features),
        "cluster": ["focal"] * len(features),
        "feature": features,
        "cliffs_delta": deltas,
        "fdr_bh_pval": pvals,
        "is_robust_discriminator": [
            (p < 0.05) and (abs(d) >= 0.20)
            for p, d in zip(pvals, deltas)
        ],
    })


# ─────────────────────── consistency rules individually ───────────────────────


def test_rule1_passes_on_consistent_direction():
    df = consistency_table({
        "full": _profile_for("full", ["f1"], [0.30], [0.001]),
        "qr": _profile_for("qr", ["f1"], [0.35], [0.001]),
        "i30": _profile_for("i30", ["f1"], [0.40], [0.001]),
    })
    assert df.iloc[0]["rule1_direction_consistent"]


def test_rule1_fails_on_mixed_direction():
    df = consistency_table({
        "full": _profile_for("full", ["f1"], [0.30], [0.001]),
        "qr": _profile_for("qr", ["f1"], [0.35], [0.001]),
        "i30": _profile_for("i30", ["f1"], [-0.40], [0.001]),
    })
    assert not df.iloc[0]["rule1_direction_consistent"]


def test_rule2_passes_with_2_of_3_significant():
    df = consistency_table({
        "full": _profile_for("full", ["f1"], [0.30], [0.001]),
        "qr": _profile_for("qr", ["f1"], [0.35], [0.01]),
        "i30": _profile_for("i30", ["f1"], [0.40], [0.20]),  # not sig
    })
    assert df.iloc[0]["rule2_significance_in_at_least_2_of_3"]


def test_rule2_fails_with_1_of_3_significant():
    df = consistency_table({
        "full": _profile_for("full", ["f1"], [0.30], [0.001]),
        "qr": _profile_for("qr", ["f1"], [0.35], [0.20]),
        "i30": _profile_for("i30", ["f1"], [0.40], [0.30]),
    })
    assert not df.iloc[0]["rule2_significance_in_at_least_2_of_3"]


def test_rule3_passes_when_all_above_threshold():
    df = consistency_table({
        "full": _profile_for("full", ["f1"], [0.20], [0.001]),
        "qr": _profile_for("qr", ["f1"], [0.25], [0.001]),
        "i30": _profile_for("i30", ["f1"], [0.30], [0.001]),
    })
    assert df.iloc[0]["rule3_min_effect_size_in_all_3"]


def test_rule3_fails_when_one_below_threshold():
    df = consistency_table({
        "full": _profile_for("full", ["f1"], [0.30], [0.001]),
        "qr": _profile_for("qr", ["f1"], [0.10], [0.001]),  # below 0.20
        "i30": _profile_for("i30", ["f1"], [0.40], [0.001]),
    })
    assert not df.iloc[0]["rule3_min_effect_size_in_all_3"]


# ─────────────────────── combined criterion ───────────────────────


def test_robust_discriminator_passes_all_three_rules():
    df = consistency_table({
        "full": _profile_for("full", ["f_robust"], [0.30], [0.001]),
        "qr": _profile_for("qr", ["f_robust"], [0.35], [0.001]),
        "i30": _profile_for("i30", ["f_robust"], [0.25], [0.04]),
    })
    assert df.iloc[0]["robust_discriminator"]


def test_robust_discriminator_fails_if_any_rule_fails():
    # Rule 1 fails: mixed direction
    df = consistency_table({
        "full": _profile_for("full", ["f1"], [0.30], [0.001]),
        "qr": _profile_for("qr", ["f1"], [-0.30], [0.001]),
        "i30": _profile_for("i30", ["f1"], [0.30], [0.001]),
    })
    assert not df.iloc[0]["robust_discriminator"]


def test_robust_discriminator_handles_significance_failure():
    # Rule 1, 3 pass, but rule 2 fails (1 of 3 significant)
    df = consistency_table({
        "full": _profile_for("full", ["f1"], [0.40], [0.001]),
        "qr": _profile_for("qr", ["f1"], [0.35], [0.20]),
        "i30": _profile_for("i30", ["f1"], [0.30], [0.30]),
    })
    assert df.iloc[0]["rule1_direction_consistent"]
    assert df.iloc[0]["rule3_min_effect_size_in_all_3"]
    assert not df.iloc[0]["rule2_significance_in_at_least_2_of_3"]
    assert not df.iloc[0]["robust_discriminator"]


# ─────────────────────── multiple features / bundles ───────────────────────


def test_consistency_table_preserves_all_features():
    features = [f"f{i}" for i in range(5)]
    deltas = [0.3] * 5
    pvals = [0.001] * 5
    df = consistency_table({
        "full": _profile_for("full", features, deltas, pvals),
        "qr": _profile_for("qr", features, deltas, pvals),
        "i30": _profile_for("i30", features, deltas, pvals),
    })
    assert len(df) == 5
    assert set(df["feature"]) == set(features)
    assert df["robust_discriminator"].all()


def test_consistency_table_columns_include_per_cohort():
    df = consistency_table({
        "full": _profile_for("full", ["f1"], [0.30], [0.001]),
        "qr": _profile_for("qr", ["f1"], [0.35], [0.01]),
        "i30": _profile_for("i30", ["f1"], [0.40], [0.04]),
    })
    for cohort in ("full", "qr", "i30"):
        assert f"sign_{cohort}" in df.columns
        assert f"delta_{cohort}" in df.columns
        assert f"fdr_pval_{cohort}" in df.columns


def test_consistency_table_raises_on_empty():
    with pytest.raises(ValueError, match="empty"):
        consistency_table({})


def test_consistency_table_handles_feature_present_in_some_cohorts_only():
    # f1 only in full + qr; f2 only in i30
    df = consistency_table({
        "full": _profile_for("full", ["f1"], [0.30], [0.001]),
        "qr": _profile_for("qr", ["f1"], [0.35], [0.001]),
        "i30": _profile_for("i30", ["f2"], [0.40], [0.001]),
    })
    # Both features should appear (outer merge); both should fail
    # rule 1 / rule 3 due to NaN in the missing cohort
    assert len(df) == 2
    for _, row in df.iterrows():
        assert not row["robust_discriminator"]


# ─────────────────────── count summary ───────────────────────


def test_count_summary_aggregates_correctly():
    df = pd.DataFrame({
        "feature": ["f1", "f2", "f3"],
        "partition": ["spatial_k2"] * 3,
        "cluster": ["focal"] * 3,
        "rule1_direction_consistent": [True, True, False],
        "rule2_significance_in_at_least_2_of_3": [True, False, True],
        "rule3_min_effect_size_in_all_3": [True, True, False],
        "robust_discriminator": [True, False, False],
    })
    summary = robust_discriminator_count_summary(df)
    row = summary.iloc[0]
    assert row["n_features"] == 3
    assert row["n_rule1_pass"] == 2
    assert row["n_rule2_pass"] == 2
    assert row["n_rule3_pass"] == 2
    assert row["n_robust_discriminators"] == 1


# ─────────────────────── partition ARI ───────────────────────


def test_partition_ari_table_perfect_agreement_passes():
    """Identical labels on shared pids -> ARI = 1.0 -> passes."""
    pids = [f"p{i:02d}" for i in range(20)]
    full = pd.Series([0] * 10 + [1] * 10, index=pids)
    strat = pd.Series([0] * 10 + [1] * 10, index=pids[:20])
    result = partition_ari_table(
        full, {"qr36d_2": strat}, partition="spatial_k2",
    )
    assert len(result) == 1
    assert result.iloc[0]["ari"] == 1.0
    assert result.iloc[0]["passes"]
    assert result.iloc[0]["n_shared_pids"] == 20


def test_partition_ari_table_random_labels_fails():
    """Random labels -> ARI near 0 -> fails."""
    pids = [f"p{i:02d}" for i in range(40)]
    rng = np.random.default_rng(0)
    full = pd.Series(rng.integers(0, 2, 40), index=pids)
    strat = pd.Series(rng.integers(0, 2, 40), index=pids)
    result = partition_ari_table(
        full, {"qr36d_2": strat}, partition="spatial_k2",
    )
    assert result.iloc[0]["ari"] < ARI_PASS_THRESHOLD
    assert not result.iloc[0]["passes"]


def test_partition_ari_table_partial_overlap_uses_intersection():
    pids_full = [f"p{i:02d}" for i in range(30)]
    pids_strat = [f"p{i:02d}" for i in range(15, 40)]  # overlap is 15..29
    full = pd.Series([0] * 15 + [1] * 15, index=pids_full)
    strat = pd.Series([1] * 15 + [0] * 10, index=pids_strat)
    result = partition_ari_table(
        full, {"qr36d_2": strat}, partition="spatial_k2",
    )
    # 15 shared pids; ARI on relabelled identity assignment is 1.0
    assert result.iloc[0]["n_shared_pids"] == 15


def test_partition_ari_table_multiple_strata():
    pids = [f"p{i:02d}" for i in range(20)]
    full = pd.Series([0] * 10 + [1] * 10, index=pids)
    qr = pd.Series([0] * 10 + [1] * 10, index=pids)
    i30 = pd.Series([1] * 10 + [0] * 10, index=pids)  # relabelled, same ARI
    result = partition_ari_table(
        full, {"qr36d_2": qr, "i30f_3": i30}, partition="spatial_k2",
    )
    assert len(result) == 2
    # Both should pass (ARI is invariant under label permutation)
    assert result["passes"].all()
