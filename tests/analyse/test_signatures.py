"""Tests for predict.analyse.signatures.

Coverage:
* top_n_signatures: row count, ranking, alphabetical tiebreak, direction
  column, only_robust filter, empty input
* signature_paragraph_for_paper: formatting + empty bundle behaviour
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from predict.analyse.signatures import (
    signature_paragraph_for_paper,
    top_n_signatures,
)


def _make_profile_df():
    """Build a small profile_df representing 1 cohort x 1 partition x 2
    clusters x 6 features (3 robust + 3 non-robust per cluster)."""
    rows = []
    # cluster 0: 3 robust features with descending |delta|
    cluster_0_features = [
        ("feature_a", 0.80, 0.001, True),
        ("feature_b", 0.60, 0.005, True),
        ("feature_c", -0.40, 0.02, True),
        ("feature_d", 0.15, 0.10, False),
        ("feature_e", 0.05, 0.30, False),
        ("feature_f", -0.10, 0.50, False),
    ]
    cluster_1_features = [
        ("feature_a", -0.80, 0.001, True),
        ("feature_b", -0.60, 0.005, True),
        ("feature_c", 0.40, 0.02, True),
        ("feature_d", -0.15, 0.10, False),
        ("feature_e", -0.05, 0.30, False),
        ("feature_f", 0.10, 0.50, False),
    ]
    for feature, delta, fdr_p, is_robust in cluster_0_features:
        rows.append({
            "cohort": "full", "partition": "spatial_k2", "cluster": "focal",
            "feature": feature, "cliffs_delta": delta, "fdr_bh_pval": fdr_p,
            "is_robust_discriminator": is_robust,
        })
    for feature, delta, fdr_p, is_robust in cluster_1_features:
        rows.append({
            "cohort": "full", "partition": "spatial_k2", "cluster": "diffuse",
            "feature": feature, "cliffs_delta": delta, "fdr_bh_pval": fdr_p,
            "is_robust_discriminator": is_robust,
        })
    return pd.DataFrame(rows)


def test_top_n_signatures_row_count():
    df = _make_profile_df()
    result = top_n_signatures(df, n=3, only_robust=True)
    # 2 clusters x 3 robust features per cluster = 6 rows
    assert len(result) == 6


def test_top_n_signatures_ranking_by_abs_delta():
    df = _make_profile_df()
    result = top_n_signatures(df, n=3, only_robust=True)
    focal = result[result["cluster"] == "focal"].sort_values("rank")
    # Rank 1 should be feature_a (|delta|=0.80)
    assert focal.iloc[0]["feature"] == "feature_a"
    assert focal.iloc[0]["abs_cliffs_delta"] == 0.80
    # Rank 2: feature_b (|delta|=0.60)
    assert focal.iloc[1]["feature"] == "feature_b"
    # Rank 3: feature_c (|delta|=0.40)
    assert focal.iloc[2]["feature"] == "feature_c"


def test_top_n_signatures_direction_column():
    df = _make_profile_df()
    result = top_n_signatures(df, n=3, only_robust=True)
    focal = result[result["cluster"] == "focal"]
    # feature_a in focal has delta=+0.80 -> "up"
    assert focal[focal["feature"] == "feature_a"].iloc[0]["direction"] == "up"
    # feature_c in focal has delta=-0.40 -> "down"
    assert focal[focal["feature"] == "feature_c"].iloc[0]["direction"] == "down"


def test_top_n_signatures_only_robust_filter():
    df = _make_profile_df()
    result_robust = top_n_signatures(df, n=10, only_robust=True)
    result_all = top_n_signatures(df, n=10, only_robust=False)
    # Robust: 2 clusters * 3 robust = 6
    assert len(result_robust) == 6
    # All: 2 clusters * 6 features = 12
    assert len(result_all) == 12


def test_top_n_signatures_alphabetical_tiebreak():
    # Two features with identical |delta| should be ranked alphabetically
    df = pd.DataFrame({
        "cohort": ["t"] * 4,
        "partition": ["p"] * 4,
        "cluster": ["c"] * 4,
        "feature": ["zeta", "alpha", "mu", "beta"],
        "cliffs_delta": [0.5, 0.5, 0.3, 0.3],
        "fdr_bh_pval": [0.001] * 4,
        "is_robust_discriminator": [True] * 4,
    })
    result = top_n_signatures(df, n=4, only_robust=True)
    # Rank 1 + 2 tie on 0.5 -> alpha, zeta (alphabetical)
    assert result.iloc[0]["feature"] == "alpha"
    assert result.iloc[1]["feature"] == "zeta"
    # Rank 3 + 4 tie on 0.3 -> beta, mu
    assert result.iloc[2]["feature"] == "beta"
    assert result.iloc[3]["feature"] == "mu"


def test_top_n_signatures_n_caps_output():
    df = _make_profile_df()
    result = top_n_signatures(df, n=2, only_robust=True)
    # 2 clusters * 2 = 4 rows
    assert len(result) == 4
    assert result["rank"].max() == 2


def test_top_n_signatures_empty_input():
    df = pd.DataFrame(columns=[
        "cohort", "partition", "cluster", "feature", "cliffs_delta",
        "fdr_bh_pval", "is_robust_discriminator",
    ])
    result = top_n_signatures(df)
    assert result.empty
    # Schema preserved even when empty
    assert "rank" in result.columns
    assert "direction" in result.columns


def test_top_n_signatures_no_robust_returns_empty_when_only_robust_true():
    df = pd.DataFrame({
        "cohort": ["t"], "partition": ["p"], "cluster": ["c"],
        "feature": ["f"], "cliffs_delta": [0.05],
        "fdr_bh_pval": [0.5], "is_robust_discriminator": [False],
    })
    result = top_n_signatures(df, n=5, only_robust=True)
    assert result.empty


def test_top_n_signatures_separate_bundles_dont_mix():
    rows = []
    for cohort in ("A", "B"):
        for cluster in ("0", "1"):
            for i in range(3):
                rows.append({
                    "cohort": cohort, "partition": "p", "cluster": cluster,
                    "feature": f"f{i}",
                    "cliffs_delta": 0.5 - i * 0.1,
                    "fdr_bh_pval": 0.001,
                    "is_robust_discriminator": True,
                })
    df = pd.DataFrame(rows)
    result = top_n_signatures(df, n=2, only_robust=True)
    # 2 cohorts * 2 clusters * 2 features = 8 rows
    assert len(result) == 8
    # Each (cohort, cluster) bundle has exactly 2 features
    for (cohort, cluster), bundle in result.groupby(["cohort", "cluster"]):
        assert len(bundle) == 2


# ─────────────────────── paragraph formatter ───────────────────────


def test_signature_paragraph_basic():
    df = _make_profile_df()
    sigs = top_n_signatures(df, n=5, only_robust=True)
    text = signature_paragraph_for_paper(
        sigs, cohort="full", partition="spatial_k2", cluster="focal",
        n_features=3,
    )
    assert "cluster=focal" in text
    assert "full" in text
    assert "feature_a" in text
    assert "up" in text


def test_signature_paragraph_empty_bundle_returns_empty_string():
    df = _make_profile_df()
    sigs = top_n_signatures(df, n=5, only_robust=True)
    text = signature_paragraph_for_paper(
        sigs, cohort="missing_cohort", partition="p", cluster="c",
    )
    assert text == ""


def test_signature_paragraph_includes_sign_for_positive_delta():
    df = pd.DataFrame({
        "cohort": ["t"], "partition": ["p"], "cluster": ["c"],
        "rank": [1], "feature": ["f"], "cliffs_delta": [0.45],
        "abs_cliffs_delta": [0.45], "direction": ["up"],
        "fdr_bh_pval": [0.001], "is_robust_discriminator": [True],
    })
    text = signature_paragraph_for_paper(df, "t", "p", "c", n_features=1)
    assert "+0.45" in text


def test_signature_paragraph_omits_explicit_plus_for_negative_delta():
    df = pd.DataFrame({
        "cohort": ["t"], "partition": ["p"], "cluster": ["c"],
        "rank": [1], "feature": ["f"], "cliffs_delta": [-0.45],
        "abs_cliffs_delta": [0.45], "direction": ["down"],
        "fdr_bh_pval": [0.001], "is_robust_discriminator": [True],
    })
    text = signature_paragraph_for_paper(df, "t", "p", "c", n_features=1)
    # The minus sign comes from the value itself; no explicit "+"
    assert "-0.45" in text
    assert "+-" not in text
