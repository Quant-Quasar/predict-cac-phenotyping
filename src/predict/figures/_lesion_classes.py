"""Dynamic discovery of lesion-morphology broad classes and the C8-like
phenotype, based on actual cluster medians instead of hard-coded labels.

Background: k-means cluster labels are arbitrary across reruns. The
"cluster labelled 8" in one run is not necessarily the same morphology
as "cluster labelled 8" in another run. Figures that depend on cluster
identity must DISCOVER the broad class / C8-like cluster from each
rerun's cluster_profiles.csv, not bake in a label-to-class dict.

Rules locked here (matching ``experiments/lesion_morphology/findings.md``):

  Broad classes:
    dense_plaques     : max_hu_median  >= 500
    soft_microspots   : max_hu_median  <= 230 AND volume_mm3_median <= 15
    moderate_nodules  : everything else

  C8-like signature (the RCA-dominant massive sheet plaque cluster):
    lm_obs_over_exp == 0      (strict LM exclusion)
    volume_mm3_median > 100   (large)
    max_hu_median > 500       (dense)
    rca_obs_over_exp > 1.5    (RCA-biased)
    n_rois_median >= 5        (multi-slice / sheet)
  If multiple candidates match, pick the one with highest rca_obs_over_exp.
"""
from __future__ import annotations

import pandas as pd


def discover_broad_classes(profiles: pd.DataFrame) -> dict[int, str]:
    """Map cluster id -> broad class based on per-cluster medians."""
    required = {"cluster", "max_hu_median", "volume_mm3_median"}
    missing = required - set(profiles.columns)
    if missing:
        raise ValueError(
            f"discover_broad_classes: profiles missing columns {missing}"
        )
    out: dict[int, str] = {}
    for _, row in profiles.iterrows():
        cid = int(row["cluster"])
        max_hu = float(row["max_hu_median"])
        vol = float(row["volume_mm3_median"])
        if max_hu >= 500:
            out[cid] = "dense_plaques"
        elif max_hu <= 230 and vol <= 15:
            out[cid] = "soft_microspots"
        else:
            out[cid] = "moderate_nodules"
    return out


def discover_c8_like_cluster(
    profiles: pd.DataFrame,
    vessel_chi_sq: pd.DataFrame,
) -> int | None:
    """Find the cluster matching the C8 morphology signature.

    Returns the cluster id, or None if no cluster qualifies on this rerun.
    """
    required_p = {"cluster", "volume_mm3_median", "max_hu_median",
                  "n_rois_median"}
    required_v = {"cluster", "rca_obs_over_exp", "lm_obs_over_exp"}
    if missing := (required_p - set(profiles.columns)):
        raise ValueError(f"profiles missing columns {missing}")
    if missing := (required_v - set(vessel_chi_sq.columns)):
        raise ValueError(f"vessel_chi_sq missing columns {missing}")

    merged = profiles.merge(
        vessel_chi_sq[list(required_v)], on="cluster", how="left",
    )
    cand = merged[
        (merged["lm_obs_over_exp"] == 0)
        & (merged["volume_mm3_median"] > 100)
        & (merged["max_hu_median"] > 500)
        & (merged["rca_obs_over_exp"] > 1.5)
        & (merged["n_rois_median"] >= 5)
    ]
    if len(cand) == 0:
        return None
    return int(
        cand.sort_values("rca_obs_over_exp", ascending=False).iloc[0]["cluster"]
    )
