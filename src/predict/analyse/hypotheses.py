"""Stage 7 pre-specified directional hypothesis test (D025).

Six one-sided Mann-Whitney tests on the focal-vs-diffuse comparison,
pre-registered with the predicted direction before any stage 7 result was
observed. The list of hypotheses is locked in ``DIRECTIONAL_HYPOTHESES``.

Two-tier verdict:

    PRIMARY (full cohort):
      count(confirmed) >= 4 of 6 hypotheses
        confirmed = direction_match AND fdr_bh_pval < 0.05

    SECONDARY (kernel-stratified replication):
      Both Qr36d/2 stratum AND I30f/3 stratum independently must satisfy:
        count(direction_match) >= 4
      (significance NOT required)

    OVERALL VERDICT:
      primary + secondary       -> "robust"
      primary + not secondary   -> "kernel-confounded"
      not primary               -> "refuted"

Decisions referencing this module:
    D025 - directional hypothesis test
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from predict.analyse.profiles import cliffs_delta, mannwhitney_u_pval


Direction = Literal["focal>diffuse", "focal<diffuse"]


# ─────────────────────── pre-registered hypotheses ───────────────────────
#
# These six are LOCKED. They were specified in D025 before any stage 7
# numerical result was observed. Do not edit this list to match results.
DIRECTIONAL_HYPOTHESES: tuple[tuple[str, Direction], ...] = (
    ("lesion_count_lad",      "focal>diffuse"),  # focal concentrates in LAD
    ("n_calcified_arteries",  "focal<diffuse"),  # focal is single-vessel
    ("dist_from_top_max",     "focal<diffuse"),  # focal is proximal
    ("gini_lesion_volume",    "focal>diffuse"),  # focal is concentrated
    ("vessel_burden_gini",    "focal>diffuse"),  # focal concentrates burden
    ("first_to_last_dist_lad", "focal<diffuse"), # focal is compact along LAD
)


# Verdict thresholds (D025).
PRIMARY_MIN_CONFIRMED = 4
PRIMARY_ALPHA = 0.05
SECONDARY_MIN_DIRECTION_MATCH = 4


# ─────────────────────── per-hypothesis evaluation ───────────────────────


def run_directional_test(
    focal_vals: np.ndarray,
    diffuse_vals: np.ndarray,
    predicted_direction: Direction,
) -> dict:
    """Run one directional test on a single feature.

    Returns
    -------
    dict with: focal_median, diffuse_median, observed_sign (-1, 0, +1),
    predicted_sign (+1 or -1), direction_match (bool),
    mannwhitney_u_pval_one_sided, cliffs_delta.
    """
    focal = np.asarray(focal_vals, dtype=float)
    diffuse = np.asarray(diffuse_vals, dtype=float)
    focal_clean = focal[~np.isnan(focal)]
    diffuse_clean = diffuse[~np.isnan(diffuse)]

    if focal_clean.size == 0 or diffuse_clean.size == 0:
        return {
            "focal_median": float("nan"),
            "diffuse_median": float("nan"),
            "observed_sign": 0,
            "predicted_sign": 1 if predicted_direction == "focal>diffuse" else -1,
            "direction_match": False,
            "mannwhitney_u_pval_one_sided": float("nan"),
            "cliffs_delta": float("nan"),
        }

    focal_med = float(np.median(focal_clean))
    diffuse_med = float(np.median(diffuse_clean))
    observed_sign = int(np.sign(focal_med - diffuse_med))
    predicted_sign = 1 if predicted_direction == "focal>diffuse" else -1
    # One-sided alternative for scipy: 'greater' = focal > diffuse.
    alternative = "greater" if predicted_direction == "focal>diffuse" else "less"
    pval = mannwhitney_u_pval(focal_clean, diffuse_clean,
                              alternative=alternative)
    delta = cliffs_delta(focal_clean, diffuse_clean)

    return {
        "focal_median": focal_med,
        "diffuse_median": diffuse_med,
        "observed_sign": observed_sign,
        "predicted_sign": predicted_sign,
        "direction_match": bool(observed_sign == predicted_sign),
        "mannwhitney_u_pval_one_sided": pval,
        "cliffs_delta": delta,
    }


# ─────────────────────── batch evaluation ───────────────────────


def directional_hypotheses_table(
    raw_features: pd.DataFrame,
    spatial_labels: pd.Series,
    cohort: str,
    hypotheses: tuple[tuple[str, Direction], ...] = DIRECTIONAL_HYPOTHESES,
    focal_label: str = "focal",
    diffuse_label: str = "diffuse",
) -> pd.DataFrame:
    """Run all 6 (or however many in ``hypotheses``) directional tests.

    Returns a DataFrame with one row per hypothesis. Includes FDR-BH
    adjusted p-values across the family and a ``confirmed`` column
    (direction_match AND fdr_bh_pval < PRIMARY_ALPHA).
    """
    common = raw_features.index.intersection(spatial_labels.index)
    if len(common) == 0:
        raise ValueError(
            f"directional_hypotheses_table [{cohort}]: no shared pids"
        )
    raw_features = raw_features.loc[common]
    spatial_labels = spatial_labels.loc[common]

    focal_mask = spatial_labels == focal_label
    diffuse_mask = spatial_labels == diffuse_label

    rows: list[dict] = []
    for feature, predicted_direction in hypotheses:
        if feature not in raw_features.columns:
            rows.append({
                "cohort": cohort,
                "feature": feature,
                "predicted_direction": predicted_direction,
                "focal_median": float("nan"),
                "diffuse_median": float("nan"),
                "observed_sign": 0,
                "direction_match": False,
                "mannwhitney_u_pval_one_sided": float("nan"),
                "cliffs_delta": float("nan"),
                "feature_present": False,
            })
            continue
        result = run_directional_test(
            raw_features.loc[focal_mask, feature].to_numpy(dtype=float),
            raw_features.loc[diffuse_mask, feature].to_numpy(dtype=float),
            predicted_direction,
        )
        rows.append({
            "cohort": cohort,
            "feature": feature,
            "predicted_direction": predicted_direction,
            "focal_median": result["focal_median"],
            "diffuse_median": result["diffuse_median"],
            "observed_sign": result["observed_sign"],
            "direction_match": result["direction_match"],
            "mannwhitney_u_pval_one_sided": result["mannwhitney_u_pval_one_sided"],
            "cliffs_delta": result["cliffs_delta"],
            "feature_present": True,
        })

    df = pd.DataFrame(rows)
    # FDR-BH on the family of directional tests (only on rows with valid p).
    eligible = df[df["mannwhitney_u_pval_one_sided"].notna()]
    df["fdr_bh_pval"] = float("nan")
    if len(eligible) > 0:
        _, fdr_p, _, _ = multipletests(
            eligible["mannwhitney_u_pval_one_sided"].to_numpy(),
            alpha=PRIMARY_ALPHA, method="fdr_bh",
        )
        df.loc[eligible.index, "fdr_bh_pval"] = fdr_p
    df["confirmed"] = (
        df["direction_match"] & (df["fdr_bh_pval"] < PRIMARY_ALPHA)
    )
    return df


# ─────────────────────── verdict computation ───────────────────────


def primary_pass(full_cohort_table: pd.DataFrame) -> dict:
    """Compute primary verdict (full cohort, >= 4 of 6 confirmed)."""
    n_total = len(full_cohort_table)
    n_confirmed = int(full_cohort_table["confirmed"].sum())
    passes = n_confirmed >= PRIMARY_MIN_CONFIRMED
    return {
        "n_total": n_total,
        "n_confirmed": n_confirmed,
        "min_required": PRIMARY_MIN_CONFIRMED,
        "passes": passes,
    }


def secondary_pass(
    qr_table: pd.DataFrame,
    i30_table: pd.DataFrame,
) -> dict:
    """Compute secondary verdict (both strata >= 4 of 6 direction match,
    significance not required)."""
    qr_matches = int(qr_table["direction_match"].sum())
    i30_matches = int(i30_table["direction_match"].sum())
    qr_pass = qr_matches >= SECONDARY_MIN_DIRECTION_MATCH
    i30_pass = i30_matches >= SECONDARY_MIN_DIRECTION_MATCH
    passes = qr_pass and i30_pass
    return {
        "qr36d_2_match_count": qr_matches,
        "i30f_3_match_count": i30_matches,
        "min_required": SECONDARY_MIN_DIRECTION_MATCH,
        "qr36d_2_pass": qr_pass,
        "i30f_3_pass": i30_pass,
        "passes": passes,
    }


def overall_verdict(
    primary_result: dict,
    secondary_result: dict,
) -> str:
    """Combine primary + secondary into the 3-level overall verdict.

    Returns one of:
        "robust"            primary passes AND secondary passes
        "kernel-confounded" primary passes BUT secondary fails
        "refuted"           primary fails
    """
    if not primary_result["passes"]:
        return "refuted"
    if secondary_result["passes"]:
        return "robust"
    return "kernel-confounded"
