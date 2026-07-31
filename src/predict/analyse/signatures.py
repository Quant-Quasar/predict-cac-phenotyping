"""Stage 7 top-N signature feature ranking (D023).

Takes the cluster-profile dataframe produced by
``profiles.compute_cluster_profile`` + ``profiles.apply_fdr_bh`` and ranks
features within each (cohort x partition x cluster) bundle by absolute
Cliff's delta to produce the top-N "signature" features. Ranking is
restricted to features that passed the D023 ``is_robust_discriminator``
gate (FDR-adjusted p < 0.05 AND |Cliff's delta| >= 0.20) by default.

Used by the paper-table builder to populate the "top-3 distinguishing
features" column per phenotype cluster.

Ranking rule:
  primary   : absolute Cliff's delta, descending
  tiebreak  : feature name, ascending alphabetical (deterministic)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def top_n_signatures(
    profile_df: pd.DataFrame,
    n: int = 5,
    only_robust: bool = True,
) -> pd.DataFrame:
    """Per-cluster top-N signature features.

    Parameters
    ----------
    profile_df  : output of profiles.apply_fdr_bh, with columns including
                  cohort, partition, cluster, feature, cliffs_delta,
                  fdr_bh_pval, is_robust_discriminator
    n           : how many features per group; default 5
    only_robust : if True, restrict to is_robust_discriminator == True
                  (D023 gate). If False, rank all features regardless of
                  significance / effect size (useful for diagnostic
                  comparisons, NOT for the paper).

    Returns
    -------
    DataFrame with one row per (cohort, partition, cluster, rank) tuple,
    columns: cohort, partition, cluster, rank, feature, cliffs_delta,
    abs_cliffs_delta, direction (up / down), fdr_bh_pval,
    is_robust_discriminator.

    Empty input or groups with no robust features under
    ``only_robust=True`` yield no rows for that group (the bundle is
    simply absent from the output).
    """
    if profile_df.empty:
        return pd.DataFrame(columns=[
            "cohort", "partition", "cluster", "rank", "feature",
            "cliffs_delta", "abs_cliffs_delta", "direction",
            "fdr_bh_pval", "is_robust_discriminator",
        ])

    df = profile_df.copy()
    if only_robust:
        df = df[df["is_robust_discriminator"]].copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "cohort", "partition", "cluster", "rank", "feature",
            "cliffs_delta", "abs_cliffs_delta", "direction",
            "fdr_bh_pval", "is_robust_discriminator",
        ])

    df["abs_cliffs_delta"] = df["cliffs_delta"].abs()
    df["direction"] = np.where(df["cliffs_delta"] > 0, "up", "down")

    rows: list[dict] = []
    group_cols = ["cohort", "partition", "cluster"]
    for keys, group in df.groupby(group_cols, sort=False):
        # Sort: descending |delta|, ascending feature name (alphabetical
        # tiebreak for determinism).
        ranked = group.sort_values(
            ["abs_cliffs_delta", "feature"],
            ascending=[False, True],
        ).head(n).reset_index(drop=True)
        cohort, partition, cluster = keys
        for i, row in ranked.iterrows():
            rows.append({
                "cohort": cohort,
                "partition": partition,
                "cluster": cluster,
                "rank": i + 1,
                "feature": row["feature"],
                "cliffs_delta": row["cliffs_delta"],
                "abs_cliffs_delta": row["abs_cliffs_delta"],
                "direction": row["direction"],
                "fdr_bh_pval": row["fdr_bh_pval"],
                "is_robust_discriminator": row["is_robust_discriminator"],
            })
    return pd.DataFrame(rows)


def signature_paragraph_for_paper(
    signatures_df: pd.DataFrame,
    cohort: str,
    partition: str,
    cluster: str,
    n_features: int = 3,
) -> str:
    """Build a one-line text summary of the top-N signature for a single
    (cohort, partition, cluster) bundle, suitable for inclusion in the
    paper's phenotype description.

    Example output:
      "cluster=focal (full): lesion_count_lad (up, delta=+0.65),
       n_calcified_arteries (down, delta=-0.58), dist_from_top_max
       (down, delta=-0.41)"

    Returns an empty string if no robust signature exists.
    """
    bundle = signatures_df[
        (signatures_df["cohort"] == cohort)
        & (signatures_df["partition"] == partition)
        & (signatures_df["cluster"] == cluster)
    ].sort_values("rank").head(n_features)
    if len(bundle) == 0:
        return ""
    parts = []
    for _, row in bundle.iterrows():
        sign = "+" if row["cliffs_delta"] > 0 else ""
        parts.append(
            f"{row['feature']} ({row['direction']}, "
            f"delta={sign}{row['cliffs_delta']:.2f})"
        )
    return (
        f"cluster={cluster} ({cohort}, {partition}): "
        + ", ".join(parts)
    )
