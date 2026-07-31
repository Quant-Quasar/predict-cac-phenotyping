"""Stage 7 burden orthogonality protocol (D024).

Tests whether the spatial-only k=2 phenotype (focal vs diffuse) is
independent of total calcium burden via three statistics on agatston_total:

  * Mann-Whitney U for median (location) difference
  * Levene's test for variance (scale) difference
  * Cliff's delta as a rank-based effect size

Output adds a 3-level `interpretation` column:

    orthogonal  -> p > 0.05 AND |delta| < 0.20      (PASS, clear)
    marginal    -> exactly one arm crosses           (PASS with caveat)
    confounded  -> p < 0.05 AND |delta| >= 0.20     (FAIL)

Also produces burden-stratified spatial profiles: within each Agatston
tertile, recompute focal-vs-diffuse profiles for the 13 spatial features
plus the 6 directional-hypothesis features. The stratified PASS criterion
is: at least 4 of 6 directional hypotheses hold in the same predicted
direction in at least 2 of 3 burden tertiles.

Decisions referencing this module:
    D024 - burden orthogonality protocol
    D025 - directional hypotheses are reused here for stratified replication
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from predict.analyse.profiles import cliffs_delta, mannwhitney_u_pval


# Pass thresholds (D024).
ORTHOGONAL_P_THRESHOLD = 0.05
ORTHOGONAL_DELTA_THRESHOLD = 0.20


# ─────────────────────── orthogonality test ───────────────────────


@dataclass(frozen=True)
class BurdenOrthogonalityResult:
    cohort: str
    n_focal: int
    n_diffuse: int
    focal_median_agatston: float
    diffuse_median_agatston: float
    focal_iqr_agatston: tuple[float, float]
    diffuse_iqr_agatston: tuple[float, float]
    mannwhitney_pval: float
    levene_pval: float
    cliffs_delta_agatston: float
    interpretation: str  # "orthogonal" / "marginal" / "confounded"
    passes: bool

    def to_dict(self) -> dict:
        return {
            "cohort": self.cohort,
            "n_focal": self.n_focal,
            "n_diffuse": self.n_diffuse,
            "focal_median_agatston": self.focal_median_agatston,
            "diffuse_median_agatston": self.diffuse_median_agatston,
            "focal_iqr_lower": self.focal_iqr_agatston[0],
            "focal_iqr_upper": self.focal_iqr_agatston[1],
            "diffuse_iqr_lower": self.diffuse_iqr_agatston[0],
            "diffuse_iqr_upper": self.diffuse_iqr_agatston[1],
            "mannwhitney_pval": self.mannwhitney_pval,
            "levene_pval": self.levene_pval,
            "cliffs_delta_agatston": self.cliffs_delta_agatston,
            "interpretation": self.interpretation,
            "passes": self.passes,
        }


def _classify_interpretation(pval: float, delta: float) -> str:
    abs_delta = abs(delta)
    sig = pval < ORTHOGONAL_P_THRESHOLD
    big = abs_delta >= ORTHOGONAL_DELTA_THRESHOLD
    if not sig and not big:
        return "orthogonal"
    if sig and big:
        return "confounded"
    return "marginal"


def assess_burden_orthogonality(
    agatston_focal: np.ndarray,
    agatston_diffuse: np.ndarray,
    cohort: str,
) -> BurdenOrthogonalityResult:
    """Run the three-statistic burden orthogonality protocol.

    All inputs are RAW agatston_total values (NOT log-transformed; we test
    on the clinical scale). NaN values are dropped before computation.
    """
    a_focal = np.asarray(agatston_focal, dtype=float)
    a_diffuse = np.asarray(agatston_diffuse, dtype=float)
    a_focal = a_focal[~np.isnan(a_focal)]
    a_diffuse = a_diffuse[~np.isnan(a_diffuse)]

    if a_focal.size == 0 or a_diffuse.size == 0:
        raise ValueError(
            f"test_burden_orthogonality [{cohort}]: empty focal "
            f"(n={a_focal.size}) or diffuse (n={a_diffuse.size}) sample"
        )

    mw_p = mannwhitney_u_pval(a_focal, a_diffuse, alternative="two-sided")
    # Levene with center='median' is the Brown-Forsythe variant: robust to
    # non-normality, which is critical for heavy-tailed Agatston scores.
    try:
        levene_result = stats.levene(a_focal, a_diffuse, center="median")
        levene_p = float(levene_result.pvalue)
    except ValueError:
        levene_p = float("nan")
    delta = cliffs_delta(a_focal, a_diffuse)
    interpretation = _classify_interpretation(mw_p, delta)
    passes = interpretation in {"orthogonal", "marginal"}

    return BurdenOrthogonalityResult(
        cohort=cohort,
        n_focal=int(a_focal.size),
        n_diffuse=int(a_diffuse.size),
        focal_median_agatston=float(np.median(a_focal)),
        diffuse_median_agatston=float(np.median(a_diffuse)),
        focal_iqr_agatston=(
            float(np.percentile(a_focal, 25)),
            float(np.percentile(a_focal, 75)),
        ),
        diffuse_iqr_agatston=(
            float(np.percentile(a_diffuse, 25)),
            float(np.percentile(a_diffuse, 75)),
        ),
        mannwhitney_pval=mw_p,
        levene_pval=levene_p,
        cliffs_delta_agatston=delta,
        interpretation=interpretation,
        passes=passes,
    )


# ─────────────────────── burden-stratified profiles ───────────────────────


def burden_tertile_assignment(
    agatston: pd.Series,
    n_tertiles: int = 3,
) -> pd.Series:
    """Assign each patient to a burden tertile via pd.qcut.

    Returns a Series of integer tertile labels (0 = low, 1 = mid,
    2 = high). Patients whose agatston is NaN get NaN labels.

    `duplicates='drop'` is used because COCA's Agatston distribution has
    ties at the long left tail; if the boundary lands on a tied value,
    qcut would otherwise raise.
    """
    quantile_labels = pd.qcut(
        agatston, q=n_tertiles,
        labels=False,
        duplicates="drop",
    )
    quantile_labels.name = "burden_tertile"
    return quantile_labels


def burden_stratified_spatial_replication(
    raw_features: pd.DataFrame,
    spatial_labels: pd.Series,
    agatston: pd.Series,
    directional_features: list[tuple[str, str]],
    cohort: str,
    n_tertiles: int = 3,
) -> pd.DataFrame:
    """Within each burden tertile, recompute focal-vs-diffuse direction
    on the D025 directional hypothesis features.

    Parameters
    ----------
    raw_features         : DataFrame indexed by pid, has the columns named
                           in ``directional_features``
    spatial_labels       : Series of {"focal", "diffuse"} per pid (already
                           mapped from cluster ids via
                           profiles.determine_focal_diffuse_mapping)
    agatston             : Series of RAW agatston_total per pid
    directional_features : list of (feature_name, predicted_direction)
                           tuples where predicted_direction is
                           "focal>diffuse" or "focal<diffuse"
    cohort               : human-readable cohort label
    n_tertiles           : usually 3, matches D024

    Returns
    -------
    DataFrame with one row per (tertile x hypothesis), columns:
      cohort, tertile, feature, predicted_direction,
      n_focal_in_tertile, n_diffuse_in_tertile,
      focal_median, diffuse_median, direction_match
    """
    # Align all inputs by pid.
    common = raw_features.index.intersection(spatial_labels.index).intersection(agatston.index)
    if len(common) == 0:
        raise ValueError(
            f"burden_stratified_spatial_replication [{cohort}]: no shared pids"
        )
    raw_features = raw_features.loc[common]
    spatial_labels = spatial_labels.loc[common]
    agatston = agatston.loc[common]

    tertile_labels = burden_tertile_assignment(agatston, n_tertiles=n_tertiles)

    rows: list[dict] = []
    for tertile in sorted(tertile_labels.dropna().unique()):
        tertile_mask = tertile_labels == tertile
        focal_in_tertile = (spatial_labels == "focal") & tertile_mask
        diffuse_in_tertile = (spatial_labels == "diffuse") & tertile_mask
        for feature, predicted_direction in directional_features:
            if feature not in raw_features.columns:
                continue
            focal_vals = raw_features.loc[focal_in_tertile, feature].dropna()
            diffuse_vals = raw_features.loc[diffuse_in_tertile, feature].dropna()
            if len(focal_vals) == 0 or len(diffuse_vals) == 0:
                rows.append({
                    "cohort": cohort,
                    "tertile": int(tertile),
                    "feature": feature,
                    "predicted_direction": predicted_direction,
                    "n_focal_in_tertile": int(focal_in_tertile.sum()),
                    "n_diffuse_in_tertile": int(diffuse_in_tertile.sum()),
                    "focal_median": float("nan"),
                    "diffuse_median": float("nan"),
                    "direction_match": False,
                })
                continue
            f_med = float(np.median(focal_vals))
            d_med = float(np.median(diffuse_vals))
            observed_sign = np.sign(f_med - d_med)
            predicted_sign = 1.0 if predicted_direction == "focal>diffuse" else -1.0
            rows.append({
                "cohort": cohort,
                "tertile": int(tertile),
                "feature": feature,
                "predicted_direction": predicted_direction,
                "n_focal_in_tertile": int(focal_in_tertile.sum()),
                "n_diffuse_in_tertile": int(diffuse_in_tertile.sum()),
                "focal_median": f_med,
                "diffuse_median": d_med,
                "direction_match": bool(observed_sign == predicted_sign),
            })
    return pd.DataFrame(rows)


def burden_stratified_pass_verdict(
    stratified_df: pd.DataFrame,
    min_hypotheses_per_tertile: int = 4,
    min_tertiles: int = 2,
) -> dict:
    """Compute the D024 stratified replication PASS verdict.

    PASS = at least ``min_hypotheses_per_tertile`` directional hypotheses
    match in the predicted direction in at least ``min_tertiles`` of the
    burden tertiles.

    Returns a dict with per-tertile match counts and the boolean verdict.
    """
    per_tertile = stratified_df.groupby("tertile")["direction_match"].sum().to_dict()
    per_tertile = {int(k): int(v) for k, v in per_tertile.items()}
    tertiles_passing = sum(
        1 for matches in per_tertile.values()
        if matches >= min_hypotheses_per_tertile
    )
    return {
        "per_tertile_match_count": per_tertile,
        "min_hypotheses_per_tertile": min_hypotheses_per_tertile,
        "min_tertiles_required": min_tertiles,
        "tertiles_passing": int(tertiles_passing),
        "passes": tertiles_passing >= min_tertiles,
    }
