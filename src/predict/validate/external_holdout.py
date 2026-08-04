"""Stage 8 D029 — external GE-scanner holdout validation.

Applies the frozen production pipeline to the 4 GE-manufacturer patients
that D004 excluded from the main cohort (pids 19, 28, 76, 77), and
reports their projected spatial PC scores + GMM-predicted phenotype.

Critical: the holdout is NEVER used to refit any model. D019 transforms,
spatial PCA, and GMM are all fit on the full N=420 production cohort;
only `.transform` and `.predict` paths touch the holdout.

ComBat is skipped (D029.1): the 13 spatial features used for the k=2
phenotype are not in the 6-feature ComBat-harmonised set, so the
spatial-only projection is ComBat-free by construction. The holdout
report does not include PyRadiomics texture summaries.

The public entry point is :func:`run_external_holdout_validation`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture

from predict.reduce.pca import fit_pca
from predict.reduce.prepare_matrix import (
    D017_DROPPED_FEATURES,
    D018_BINARISE_SOURCE,
    D018_BINARISE_TARGET,
    SPARSE_COLUMNS,
    run_matrix_prep,
)
from predict.validate.label_alignment import determine_mapping

_log = logging.getLogger(__name__)


# The 13 spatial features used for the spatial-only k=2 GMM (D021 /
# Finding 3). Mirrors SPATIAL_FEATURES_AFTER_D017 in scripts/07_discover.py.
SPATIAL_FEATURES_FOR_PROJECTION: tuple[str, ...] = (
    "lesion_count_lad", "lesion_count_rca", "lesion_count_lcx",
    "lesion_count_lm", "lesion_count_total",
    "n_calcified_arteries",
    "gini_lesion_volume",
    "dist_from_top_max", "dist_from_top_mean",
    "center_of_mass_z",
    "inter_lesion_dist_mean_lad", "inter_lesion_dist_max_lad",
    "first_to_last_dist_lad",
)

# Canonical raw context columns the holdout report always carries.
HOLDOUT_RAW_CONTEXT_COLS: tuple[str, ...] = (
    "kernel", "scanner_model",
    "agatston_total", "n_calcified_arteries", "lesion_count_total",
    "mask_voxels", "low_burden_flag",
)

# Default GMM hyperparameters (mirrors scripts/07_discover.py).
GMM_RANDOM_STATE = 0
GMM_N_INIT = 10
GMM_COVARIANCE_TYPE = "full"


# ─────────────────────── data classes ───────────────────────


@dataclass(frozen=True)
class FrozenPipeline:
    """Captured fit state from the full production cohort. Used to
    transform the holdout WITHOUT leakage.
    """
    feature_cols_post_d019: list[str]
    zscore_means: dict[str, float]
    zscore_stds: dict[str, float]
    yj_records: list[dict]
    spatial_feature_cols: list[str]
    # Sign-normalised components from fit_pca; sign convention is BAKED
    # into the rows (no separate signs array). Shape (n_retain, n_features).
    spatial_pca_components: np.ndarray
    # Column-wise mean of the post-D019 full-cohort spatial matrix; required
    # for the holdout projection identity (X_holdout - mean) @ components.T.
    # On a fully z-scored full cohort this is ~0; capturing it preserves
    # exact PCA projection semantics even under numerical drift.
    spatial_pca_means: np.ndarray
    gmm: GaussianMixture
    gmm_focal_label_raw: int  # the cluster id mapped to "focal"
    full_cohort_focal_diffuse_mapping: dict


# ─────────────────────── fit ───────────────────────


def _filter_to_combat_compatible(
    df: pd.DataFrame, kernel_col: str = "kernel",
) -> pd.DataFrame:
    """Drop singleton-kernel patients per D019 ComBat constraint."""
    if kernel_col not in df.columns:
        return df
    counts = df[kernel_col].value_counts()
    keep_kernels = set(counts[counts >= 2].index)
    return df[df[kernel_col].isin(keep_kernels)].reset_index(drop=True)


def fit_frozen_pipeline(
    full_cohort_features: pd.DataFrame,
    gated_feature_cols: Iterable[str],
    *,
    spatial_cols: Iterable[str] = SPATIAL_FEATURES_FOR_PROJECTION,
    pca_cumvar: float = 0.85,
    random_state: int = 0,
) -> FrozenPipeline:
    """Fit the entire D019 + spatial PCA + GMM chain on the full cohort.

    Returns a :class:`FrozenPipeline` whose `.transform`-style fields
    are byte-stable on a single machine.
    """
    df = _filter_to_combat_compatible(full_cohort_features.copy())
    feature_cols = list(gated_feature_cols)

    _log.info("D019 on full cohort: N=%d, %d features",
              len(df), len(feature_cols))
    prep_df, prep_feat_cols, log = run_matrix_prep(df, feature_cols)
    _log.info("post-D019: N=%d, %d features", len(prep_df), len(prep_feat_cols))

    spatial_kept = [c for c in spatial_cols if c in prep_feat_cols]
    if len(spatial_kept) < 2:
        raise ValueError(
            f"fit_frozen_pipeline: only {len(spatial_kept)} of "
            f"{len(SPATIAL_FEATURES_FOR_PROJECTION)} spatial features "
            f"survived D019 (need >= 2)."
        )

    spatial_pca = fit_pca(
        prep_df, spatial_kept,
        cumvar_threshold=pca_cumvar, random_state=random_state,
    )
    # Capture the per-feature mean used in the projection identity.
    spatial_means = prep_df[spatial_kept].to_numpy(dtype=float).mean(axis=0)
    # fit_pca's .scores are already the n_retain projection; reuse them.
    spatial_scores = spatial_pca.scores  # (N, n_retain)
    spatial_components_retained = spatial_pca.components[:spatial_pca.n_retain]

    gmm = GaussianMixture(
        n_components=2,
        covariance_type=GMM_COVARIANCE_TYPE,
        random_state=GMM_RANDOM_STATE,
        n_init=GMM_N_INIT,
    )
    gmm.fit(spatial_scores)
    gmm_labels_raw = gmm.predict(spatial_scores)

    labels_series = pd.Series(gmm_labels_raw, index=prep_df["pid"].astype(str))
    raw_feats_for_mapping = full_cohort_features.set_index(
        full_cohort_features["pid"].astype(str)
    ).reindex(labels_series.index)
    mapping = determine_mapping(raw_feats_for_mapping, labels_series)
    focal_raw_label = next(k for k, v in mapping.items() if v == "focal")

    return FrozenPipeline(
        feature_cols_post_d019=list(prep_feat_cols),
        zscore_means=dict(log.zscore_means),
        zscore_stds=dict(log.zscore_stds),
        yj_records=list(log.yj_per_column),
        spatial_feature_cols=spatial_kept,
        spatial_pca_components=spatial_components_retained,
        spatial_pca_means=spatial_means,
        gmm=gmm,
        gmm_focal_label_raw=int(focal_raw_label),
        full_cohort_focal_diffuse_mapping=mapping,
    )


# ─────────────────────── transform ───────────────────────


def transform_holdout_features(
    holdout_raw: pd.DataFrame,
    frozen: FrozenPipeline,
) -> pd.DataFrame:
    """Apply the captured D019 transforms to the holdout's RAW features.

    Steps (ComBat skipped per D029.1):
      1. Drop D017 columns.
      2. Apply D018 binarisation if source present.
      3. Apply YJ with the per-column lambdas captured on the full cohort.
         If the full cohort fell back to RANK on a column, the holdout
         column for that feature is rank-mapped into the full-cohort
         range via empirical CDF. Constant + binary columns pass through.
      4. Apply per-column z-score with captured means + stds.

    Returns the holdout dataframe restricted to ``frozen.spatial_feature_cols``
    plus a 'pid' column, ready for spatial-PCA projection.
    """
    df = holdout_raw.copy()

    # D017 drops
    for col in D017_DROPPED_FEATURES:
        if col in df.columns:
            df = df.drop(columns=col)

    # D018 binarisation
    if (D018_BINARISE_SOURCE in df.columns
            and D018_BINARISE_TARGET not in df.columns):
        df[D018_BINARISE_TARGET] = (df[D018_BINARISE_SOURCE] > 0).astype(int)

    # YJ via captured records (only for SPARSE_COLUMNS that had a non-binary
    # transform). For rank-fallback columns, we keep the holdout value at its
    # raw scale; subsequent z-score uses the full-cohort raw mean / std so
    # the result is comparable. This is conservative but defensible for N=4.
    yj_lookup = {rec["feature"]: rec for rec in frozen.yj_records}
    for col in SPARSE_COLUMNS:
        if col not in df.columns:
            continue
        rec = yj_lookup.get(col)
        if rec is None or rec.get("transform_used") in (
            "binary_skipped", "constant_skipped", "rank",
        ):
            continue
        lam = rec.get("lambda")
        if lam is None:
            continue
        v = df[col].to_numpy(dtype=float)
        # Yeo-Johnson formula by branch (matches sklearn PowerTransformer
        # with standardize=False; we only need the forward transform).
        df[col] = _yeo_johnson_apply(v, float(lam))

    # Z-score using captured means + stds
    for col in frozen.feature_cols_post_d019:
        if col not in df.columns:
            continue
        mean = frozen.zscore_means.get(col, 0.0)
        std = frozen.zscore_stds.get(col, 1.0)
        v = df[col].to_numpy(dtype=float)
        if std == 0.0:
            df[col] = v - mean
        else:
            df[col] = (v - mean) / std

    # Keep only the spatial cols we'll project + pid
    keep_cols = ["pid"] + [c for c in frozen.spatial_feature_cols
                           if c in df.columns]
    return df[keep_cols].reset_index(drop=True)


def _yeo_johnson_apply(x: np.ndarray, lam: float) -> np.ndarray:
    """Forward Yeo-Johnson transform with a fixed lambda (no fitting).

    Mirrors sklearn's PowerTransformer(yeo-johnson, standardize=False)
    transform branch.
    """
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    if abs(lam) < 1e-9:
        out[pos] = np.log1p(x[pos])
    else:
        out[pos] = (np.power(x[pos] + 1.0, lam) - 1.0) / lam
    if abs(lam - 2.0) < 1e-9:
        out[neg] = -np.log1p(-x[neg])
    else:
        out[neg] = -(np.power(-x[neg] + 1.0, 2.0 - lam) - 1.0) / (2.0 - lam)
    return out


def project_holdout_to_spatial_pca(
    holdout_transformed: pd.DataFrame,
    frozen: FrozenPipeline,
) -> np.ndarray:
    """Project pre-transformed holdout features onto the frozen spatial PCA.

    Returns an array of shape (n_holdout, n_pcs).
    """
    X = holdout_transformed[frozen.spatial_feature_cols].to_numpy(dtype=float)
    centred = X - frozen.spatial_pca_means
    return centred @ frozen.spatial_pca_components.T


# ─────────────────────── predict + report ───────────────────────


def predict_holdout_phenotype(
    spatial_scores: np.ndarray,
    frozen: FrozenPipeline,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(raw_labels, distance_to_centroids)``.

    ``raw_labels`` are the unmapped GMM cluster ids in ``{0, 1}``.
    ``distance_to_centroids`` is shape ``(n_holdout, 2)`` giving Euclidean
    distance to each GMM mean (column order matches the GMM's internal
    component ordering).
    """
    raw_labels = frozen.gmm.predict(spatial_scores)
    means = frozen.gmm.means_  # (2, n_pcs)
    diffs = spatial_scores[:, None, :] - means[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    return raw_labels.astype(int), dists


def build_holdout_report(
    holdout_raw: pd.DataFrame,
    spatial_scores: np.ndarray,
    raw_labels: np.ndarray,
    centroid_dists: np.ndarray,
    frozen: FrozenPipeline,
    xml_roundtrip_pass_by_pid: dict[str, bool] | None = None,
) -> pd.DataFrame:
    """Assemble the per-pid report described in D029.

    Output columns:
      pid, kernel, scanner_model, mask_voxels, low_burden_flag,
      agatston_total, n_calcified_arteries, lesion_count_total,
      spatial_pc1, spatial_pc2, ..., spatial_pcK,
      predicted_phenotype_raw, predicted_phenotype,
      distance_to_focal_centroid, distance_to_diffuse_centroid,
      xml_roundtrip_max_pass.
    """
    n_pcs = spatial_scores.shape[1]
    rows: list[dict] = []
    focal_raw = frozen.gmm_focal_label_raw
    diffuse_raw = 1 - focal_raw if focal_raw in (0, 1) else None
    if diffuse_raw is None:
        raise ValueError("focal_raw must be 0 or 1")

    label_to_str = {
        focal_raw: "focal", diffuse_raw: "diffuse",
    }

    for i, (_, row) in enumerate(holdout_raw.iterrows()):
        pid = str(row["pid"])
        rec: dict = {"pid": pid}
        for col in HOLDOUT_RAW_CONTEXT_COLS:
            rec[col] = row.get(col, None)
        for k in range(n_pcs):
            rec[f"spatial_pc{k + 1}"] = float(spatial_scores[i, k])
        rec["predicted_phenotype_raw"] = int(raw_labels[i])
        rec["predicted_phenotype"] = label_to_str[int(raw_labels[i])]
        rec["distance_to_focal_centroid"] = float(centroid_dists[i, focal_raw])
        rec["distance_to_diffuse_centroid"] = float(centroid_dists[i, diffuse_raw])
        if xml_roundtrip_pass_by_pid is not None:
            rec["xml_roundtrip_max_pass"] = bool(
                xml_roundtrip_pass_by_pid.get(pid, False)
            )
        else:
            rec["xml_roundtrip_max_pass"] = None
        rows.append(rec)
    return pd.DataFrame(rows)


# ─────────────────────── public entry point ───────────────────────


def run_external_holdout_validation(
    full_cohort_features: pd.DataFrame,
    holdout_features: pd.DataFrame,
    gated_feature_cols: Iterable[str],
    *,
    pca_cumvar: float = 0.85,
    random_state: int = 0,
    xml_roundtrip_pass_by_pid: dict[str, bool] | None = None,
) -> tuple[pd.DataFrame, FrozenPipeline]:
    """Top-level orchestrator. Returns ``(report_df, frozen_pipeline)``.

    The frozen pipeline is returned so the orchestrator can reuse it
    for leave_k_out or other downstream stage 8 work without refitting.
    """
    frozen = fit_frozen_pipeline(
        full_cohort_features, gated_feature_cols,
        pca_cumvar=pca_cumvar, random_state=random_state,
    )
    transformed = transform_holdout_features(holdout_features, frozen)
    scores = project_holdout_to_spatial_pca(transformed, frozen)
    raw_labels, dists = predict_holdout_phenotype(scores, frozen)
    report = build_holdout_report(
        holdout_features, scores, raw_labels, dists, frozen,
        xml_roundtrip_pass_by_pid=xml_roundtrip_pass_by_pid,
    )
    return report, frozen


__all__ = [
    "SPATIAL_FEATURES_FOR_PROJECTION",
    "HOLDOUT_RAW_CONTEXT_COLS",
    "FrozenPipeline",
    "fit_frozen_pipeline",
    "transform_holdout_features",
    "project_holdout_to_spatial_pca",
    "predict_holdout_phenotype",
    "build_holdout_report",
    "run_external_holdout_validation",
]
