"""Tests for predict.analyse.hypotheses (D025).

Coverage:
* DIRECTIONAL_HYPOTHESES is a 6-tuple of the locked predictions
* run_directional_test: one-sided MW alternative, direction_match logic,
  NaN handling
* directional_hypotheses_table: end-to-end on synthetic with planted
  focal-vs-diffuse structure
* primary_pass at 3/6 and 4/6 boundaries
* secondary_pass on agree/disagree combinations
* overall_verdict on all 4 corner cases
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predict.analyse.hypotheses import (
    DIRECTIONAL_HYPOTHESES,
    PRIMARY_MIN_CONFIRMED,
    SECONDARY_MIN_DIRECTION_MATCH,
    directional_hypotheses_table,
    overall_verdict,
    primary_pass,
    run_directional_test,
    secondary_pass,
)


# ─────────────────────── pre-registration check ───────────────────────


def test_directional_hypotheses_are_six():
    assert len(DIRECTIONAL_HYPOTHESES) == 6


def test_directional_hypotheses_locked_list():
    """If you find yourself editing this test, ask whether you should be
    editing the hypothesis list. D025 is pre-registered."""
    expected = {
        ("lesion_count_lad", "focal>diffuse"),
        ("n_calcified_arteries", "focal<diffuse"),
        ("dist_from_top_max", "focal<diffuse"),
        ("gini_lesion_volume", "focal>diffuse"),
        ("vessel_burden_gini", "focal>diffuse"),
        ("first_to_last_dist_lad", "focal<diffuse"),
    }
    assert set(DIRECTIONAL_HYPOTHESES) == expected


# ─────────────────────── single-hypothesis tests ───────────────────────


def test_run_directional_one_sided_focal_greater_confirms():
    rng = np.random.default_rng(0)
    focal = rng.normal(10, 1, 30)
    diffuse = rng.normal(0, 1, 30)
    result = run_directional_test(focal, diffuse, "focal>diffuse")
    assert result["direction_match"]
    assert result["observed_sign"] == 1
    assert result["mannwhitney_u_pval_one_sided"] < 0.001
    # cliffs_delta should be near +1
    assert result["cliffs_delta"] > 0.9


def test_run_directional_one_sided_focal_less_confirms():
    rng = np.random.default_rng(0)
    focal = rng.normal(0, 1, 30)
    diffuse = rng.normal(10, 1, 30)
    result = run_directional_test(focal, diffuse, "focal<diffuse")
    assert result["direction_match"]
    assert result["observed_sign"] == -1
    assert result["mannwhitney_u_pval_one_sided"] < 0.001


def test_run_directional_wrong_direction_refutes():
    """If we predict focal > diffuse but focal is actually lower,
    direction_match should be False and the one-sided p should be near 1."""
    rng = np.random.default_rng(0)
    focal = rng.normal(0, 1, 30)
    diffuse = rng.normal(10, 1, 30)
    result = run_directional_test(focal, diffuse, "focal>diffuse")
    assert not result["direction_match"]
    assert result["observed_sign"] == -1
    # One-sided 'greater' on a 'less' result -> p near 1
    assert result["mannwhitney_u_pval_one_sided"] > 0.99


def test_run_directional_handles_nans():
    focal = np.array([np.nan, 5.0, 6.0, 7.0])
    diffuse = np.array([1.0, 2.0, np.nan, 3.0])
    result = run_directional_test(focal, diffuse, "focal>diffuse")
    # After NaN drop: focal=[5,6,7], diffuse=[1,2,3]; all focal > all diffuse
    assert result["direction_match"]
    assert result["cliffs_delta"] == 1.0


def test_run_directional_empty_after_nan_returns_nan_result():
    focal = np.array([np.nan, np.nan])
    diffuse = np.array([1.0, 2.0])
    result = run_directional_test(focal, diffuse, "focal>diffuse")
    assert not result["direction_match"]
    assert np.isnan(result["mannwhitney_u_pval_one_sided"])


# ─────────────────────── table builder ───────────────────────


def _synthetic_focal_diffuse(seed: int = 0, n_per: int = 50):
    """Build a cohort where all 6 hypotheses should confirm strongly."""
    rng = np.random.default_rng(seed)
    pids = [f"p{i:03d}" for i in range(2 * n_per)]
    df = pd.DataFrame({
        # focal>diffuse hypotheses (4 features)
        "lesion_count_lad": np.concatenate([
            rng.poisson(8, n_per).astype(float),
            rng.poisson(2, n_per).astype(float),
        ]),
        "gini_lesion_volume": np.concatenate([
            rng.uniform(0.6, 0.9, n_per),
            rng.uniform(0.1, 0.4, n_per),
        ]),
        "vessel_burden_gini": np.concatenate([
            rng.uniform(0.7, 0.95, n_per),
            rng.uniform(0.2, 0.5, n_per),
        ]),
        # focal<diffuse hypotheses (3 features)
        "n_calcified_arteries": np.concatenate([
            np.full(n_per, 1.0),
            np.full(n_per, 3.0),
        ]),
        "dist_from_top_max": np.concatenate([
            rng.uniform(10, 30, n_per),
            rng.uniform(50, 100, n_per),
        ]),
        "first_to_last_dist_lad": np.concatenate([
            rng.uniform(5, 15, n_per),
            rng.uniform(30, 60, n_per),
        ]),
    }, index=pids)
    labels = pd.Series(["focal"] * n_per + ["diffuse"] * n_per, index=pids)
    return df, labels


def test_directional_table_six_rows():
    df, labels = _synthetic_focal_diffuse()
    result = directional_hypotheses_table(df, labels, cohort="t")
    assert len(result) == 6
    assert set(result["feature"]) == {h[0] for h in DIRECTIONAL_HYPOTHESES}


def test_directional_table_all_confirm_on_strong_signal():
    df, labels = _synthetic_focal_diffuse()
    result = directional_hypotheses_table(df, labels, cohort="t")
    # All 6 should confirm with strong planted signal
    assert result["direction_match"].all()
    assert result["confirmed"].all()
    # FDR-adjusted p should be well below 0.05
    assert (result["fdr_bh_pval"] < 0.001).all()


def test_directional_table_handles_missing_feature():
    df, labels = _synthetic_focal_diffuse()
    df = df.drop(columns=["gini_lesion_volume"])
    result = directional_hypotheses_table(df, labels, cohort="t")
    # Missing feature row is present but flagged
    missing = result[result["feature"] == "gini_lesion_volume"].iloc[0]
    assert not missing["feature_present"]
    assert not missing["direction_match"]
    assert not missing["confirmed"]


def test_directional_table_raises_on_no_shared_pids():
    df = pd.DataFrame({"lesion_count_lad": [1.0]}, index=["a"])
    labels = pd.Series(["focal"], index=["b"])
    with pytest.raises(ValueError, match="no shared pids"):
        directional_hypotheses_table(df, labels, cohort="t")


# ─────────────────────── primary_pass ───────────────────────


def _make_table(confirmed_count: int, total: int = 6) -> pd.DataFrame:
    return pd.DataFrame({
        "feature": [f"f{i}" for i in range(total)],
        "direction_match": [i < confirmed_count for i in range(total)],
        "confirmed": [i < confirmed_count for i in range(total)],
        "fdr_bh_pval": [0.01 if i < confirmed_count else 0.5 for i in range(total)],
    })


def test_primary_pass_at_4_of_6():
    result = primary_pass(_make_table(confirmed_count=4))
    assert result["passes"]
    assert result["n_confirmed"] == 4


def test_primary_pass_fails_at_3_of_6():
    result = primary_pass(_make_table(confirmed_count=3))
    assert not result["passes"]
    assert result["n_confirmed"] == 3


def test_primary_pass_at_6_of_6():
    result = primary_pass(_make_table(confirmed_count=6))
    assert result["passes"]
    assert result["n_confirmed"] == 6


def test_primary_min_required_is_4():
    result = primary_pass(_make_table(confirmed_count=4))
    assert result["min_required"] == PRIMARY_MIN_CONFIRMED == 4


# ─────────────────────── secondary_pass ───────────────────────


def _make_direction_table(n_match: int, total: int = 6) -> pd.DataFrame:
    return pd.DataFrame({
        "feature": [f"f{i}" for i in range(total)],
        "direction_match": [i < n_match for i in range(total)],
    })


def test_secondary_pass_both_strata_pass():
    qr = _make_direction_table(n_match=5)
    i30 = _make_direction_table(n_match=4)
    result = secondary_pass(qr, i30)
    assert result["passes"]


def test_secondary_pass_one_stratum_fails():
    qr = _make_direction_table(n_match=5)
    i30 = _make_direction_table(n_match=3)
    result = secondary_pass(qr, i30)
    assert not result["passes"]
    assert result["qr36d_2_pass"]
    assert not result["i30f_3_pass"]


def test_secondary_pass_both_strata_fail():
    qr = _make_direction_table(n_match=2)
    i30 = _make_direction_table(n_match=1)
    result = secondary_pass(qr, i30)
    assert not result["passes"]


def test_secondary_min_required_is_4():
    qr = _make_direction_table(n_match=4)
    i30 = _make_direction_table(n_match=4)
    result = secondary_pass(qr, i30)
    assert result["min_required"] == SECONDARY_MIN_DIRECTION_MATCH == 4


# ─────────────────────── overall_verdict ───────────────────────


def test_overall_verdict_robust():
    primary = {"passes": True}
    secondary = {"passes": True}
    assert overall_verdict(primary, secondary) == "robust"


def test_overall_verdict_kernel_confounded():
    primary = {"passes": True}
    secondary = {"passes": False}
    assert overall_verdict(primary, secondary) == "kernel-confounded"


def test_overall_verdict_refuted_when_primary_fails():
    primary = {"passes": False}
    secondary = {"passes": True}
    # Even if secondary passes, primary failure means refuted
    assert overall_verdict(primary, secondary) == "refuted"


def test_overall_verdict_refuted_when_both_fail():
    primary = {"passes": False}
    secondary = {"passes": False}
    assert overall_verdict(primary, secondary) == "refuted"
