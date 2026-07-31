"""Stage 7 cross-cohort feature consistency + partition ARI (D027).

Two complementary replication checks:

1. **Feature-level 3-rule criterion**: for each (feature x partition x
   cluster) tuple, the discriminator is "robust cross-cohort" iff ALL
   three hold across the full / Qr36d/2 / I30f/3 cohort outputs:

       Rule 1  direction consistent
                 sign(cliffs_delta) is the same in all 3 cohorts
       Rule 2  significance in at least 2 of 3
                 fdr_bh_pval < 0.05 in at least 2 of the 3 cohorts
       Rule 3  minimum effect size in all 3
                 abs(cliffs_delta) >= 0.20 in all 3 cohorts

2. **Partition-level ARI on shared pids**: reuses
   ``predict.discover.validity.ari_on_shared_pids``. For each partition
   (spatial_k2, burden_k3), takes the intersection of pids between full
   and each stratified cohort and computes ARI between cluster labels on
   that shared set. PASS at ARI >= 0.80.

Decisions referencing this module:
    D027 - feature-level cross-cohort consistency criterion
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from predict.discover.validity import ari_on_shared_pids


# D027 thresholds.
RULE2_MIN_SIGNIFICANT_COHORTS = 2
RULE3_DELTA_THRESHOLD = 0.20
FDR_ALPHA = 0.05
ARI_PASS_THRESHOLD = 0.80


# ─────────────────────── feature-level consistency ───────────────────────


def consistency_table(
    profiles_by_cohort: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Apply the D027 three-rule criterion across cohorts.

    Parameters
    ----------
    profiles_by_cohort : dict mapping cohort label -> profile_df produced
                         by profiles.apply_fdr_bh. Each profile_df must
                         have columns: partition, cluster, feature,
                         cliffs_delta, fdr_bh_pval, is_robust_discriminator.

    Returns
    -------
    DataFrame with one row per (feature, partition, cluster) tuple,
    columns include:
      feature, partition, cluster,
      sign_<cohort>           sign(cliffs_delta) per cohort
      delta_<cohort>          abs(cliffs_delta) per cohort
      fdr_pval_<cohort>       FDR-adjusted p per cohort
      rule1_direction_consistent
      rule2_significance_in_at_least_2_of_3
      rule3_min_effect_size_in_all_3
      robust_discriminator     all 3 rules pass
    """
    if not profiles_by_cohort:
        raise ValueError(
            "consistency_table: profiles_by_cohort is empty"
        )

    cohorts = list(profiles_by_cohort.keys())

    # Build a per-(feature, partition, cluster) wide table, joining cohorts.
    pivot_columns: list[pd.DataFrame] = []
    for cohort, df in profiles_by_cohort.items():
        sub = df[["feature", "partition", "cluster",
                  "cliffs_delta", "fdr_bh_pval"]].copy()
        sub = sub.rename(columns={
            "cliffs_delta": f"cliffs_delta__{cohort}",
            "fdr_bh_pval": f"fdr_pval__{cohort}",
        })
        pivot_columns.append(sub)

    merged = pivot_columns[0]
    for next_df in pivot_columns[1:]:
        merged = merged.merge(
            next_df, on=["feature", "partition", "cluster"],
            how="outer",
        )

    rows: list[dict] = []
    for _, row in merged.iterrows():
        deltas = {c: row[f"cliffs_delta__{c}"] for c in cohorts}
        fdrs = {c: row[f"fdr_pval__{c}"] for c in cohorts}
        signs = {c: float(np.sign(d)) if not pd.isna(d) else float("nan")
                 for c, d in deltas.items()}
        abs_deltas = {c: abs(d) if not pd.isna(d) else float("nan")
                      for c, d in deltas.items()}

        # Rule 1: direction consistent across all 3 (and none NaN)
        all_signs_valid = all(not np.isnan(s) for s in signs.values())
        rule1 = (
            all_signs_valid
            and len(set(signs.values())) == 1
            and 0.0 not in set(signs.values())  # all sign != 0
        )

        # Rule 2: FDR < 0.05 in at least 2 of N (treating NaN p as not sig)
        n_significant = sum(
            1 for p in fdrs.values()
            if not pd.isna(p) and p < FDR_ALPHA
        )
        rule2 = n_significant >= RULE2_MIN_SIGNIFICANT_COHORTS

        # Rule 3: |delta| >= 0.20 in all 3
        rule3 = (
            all(not np.isnan(d) for d in abs_deltas.values())
            and all(d >= RULE3_DELTA_THRESHOLD for d in abs_deltas.values())
        )

        out_row = {
            "feature": row["feature"],
            "partition": row["partition"],
            "cluster": row["cluster"],
        }
        for cohort in cohorts:
            out_row[f"sign_{cohort}"] = signs[cohort]
            out_row[f"delta_{cohort}"] = abs_deltas[cohort]
            out_row[f"fdr_pval_{cohort}"] = fdrs[cohort]
        out_row["rule1_direction_consistent"] = bool(rule1)
        out_row["rule2_significance_in_at_least_2_of_3"] = bool(rule2)
        out_row["rule3_min_effect_size_in_all_3"] = bool(rule3)
        out_row["robust_discriminator"] = bool(rule1 and rule2 and rule3)
        rows.append(out_row)

    return pd.DataFrame(rows)


def robust_discriminator_count_summary(
    consistency_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate count of robust_discriminator features per
    (partition, cluster) bundle.

    Useful for the paper sentence "X features are robust discriminators
    of phenotype Y across all 3 cohorts".
    """
    return (
        consistency_df
        .groupby(["partition", "cluster"])
        .agg(
            n_features=("feature", "count"),
            n_rule1_pass=("rule1_direction_consistent", "sum"),
            n_rule2_pass=("rule2_significance_in_at_least_2_of_3", "sum"),
            n_rule3_pass=("rule3_min_effect_size_in_all_3", "sum"),
            n_robust_discriminators=("robust_discriminator", "sum"),
        )
        .reset_index()
    )


# ─────────────────────── partition-level ARI ───────────────────────


def partition_ari_table(
    full_labels: pd.Series,
    stratified_labels: dict[str, pd.Series],
    partition: str,
) -> pd.DataFrame:
    """Compute ARI on shared pids between the full cohort and each
    stratified cohort for one partition.

    Parameters
    ----------
    full_labels       : Series indexed by pid with cluster labels from
                        the full-cohort run
    stratified_labels : dict {cohort_label: Series} for each stratified
                        cohort
    partition         : "spatial_k2" or "burden_k3"; just a label

    Returns
    -------
    DataFrame with one row per (full vs each stratified) comparison.
    Columns: partition, stratum, n_shared_pids, ari, passes
    """
    rows: list[dict] = []
    for stratum_label, strat_labels in stratified_labels.items():
        full_pids = full_labels.index.tolist()
        strat_pids = strat_labels.index.tolist()
        ari, shared = ari_on_shared_pids(
            full_pids, full_labels.to_numpy(),
            strat_pids, strat_labels.to_numpy(),
        )
        rows.append({
            "partition": partition,
            "stratum": stratum_label,
            "n_shared_pids": len(shared),
            "ari": float(ari),
            "passes": bool(ari >= ARI_PASS_THRESHOLD),
        })
    return pd.DataFrame(rows)
