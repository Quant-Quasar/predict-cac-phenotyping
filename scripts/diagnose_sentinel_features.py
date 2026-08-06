#!/usr/bin/env python
"""Sentinel-value diagnostic for per-vessel distance-style features.

A feature has a 'sentinel value' problem when a large fraction of patients
land on the exact same numerical value for non-phenotypic reasons (e.g. 0
lesions or 1 lesion in a vessel makes the distance feature degenerate).
This is a common silent killer of unsupervised analyses: PCA aligns its
first component with the sentinel split rather than any biological axis.

This script examines the 4 vessels x 4 distance-style features = 16 columns
in ``outputs/03_features/features.csv``:

  per-vessel:
    - inter_lesion_dist_mean_{vessel}   (sentinel 0.0 when N < 2)
    - inter_lesion_dist_max_{vessel}    (sentinel 0.0 when N < 2)
    - first_to_last_dist_{vessel}       (sentinel 0.0 when N < 2)
    - diffusivity_{vessel}              (sentinel 0.0 when N = 0, 1.0 when N = 1
                                         per D016; continuous N/d otherwise)

For each one it reports:

  - the fraction at the dominant sentinel value
  - whether that sentinel correlates with low_burden_flag and Agatston category
  - the n_unique non-sentinel values (continuous diffusivity / distance regime)

Read-only; safe to run anytime.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from predict.config import load_config


VESSELS: tuple[str, ...] = ("lad", "rca", "lcx", "lm")

DISTANCE_FEATURES: tuple[str, ...] = (
    "inter_lesion_dist_mean",
    "inter_lesion_dist_max",
    "first_to_last_dist",
)


def _section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def _pct(part: int, whole: int) -> str:
    return f"{part}/{whole} ({100.0 * part / whole:.1f}%)" if whole else "0/0"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    features_csv = cfg.paths.outputs / "03_features" / "features.csv"
    df = pd.read_csv(features_csv, dtype={"pid": str})
    n = len(df)
    print(f"Loaded {features_csv}: {n} rows.")

    # ── Per-vessel distance-style sentinel audit ─────────────────────────
    _section("1. Per-vessel sentinel rates (0.0 means N < 2 lesions in this vessel)")
    rows = []
    for vessel in VESSELS:
        lesion_count = f"lesion_count_{vessel}"
        for stem in DISTANCE_FEATURES:
            col = f"{stem}_{vessel}"
            sentinel_mask = (df[col] == 0.0)
            n_sentinel = int(sentinel_mask.sum())
            # Confirm sentinel iff lesion_count < 2.
            n_lt2 = int((df[lesion_count] < 2).sum())
            consistent = int(((sentinel_mask) == (df[lesion_count] < 2)).sum())
            n_nonsentinel_unique = int(df.loc[~sentinel_mask, col].nunique())
            rows.append({
                "feature": col,
                "n_sentinel_zero": n_sentinel,
                "pct_sentinel": f"{100.0 * n_sentinel / n:.1f}%",
                "n_patients_lt2_lesions": n_lt2,
                "sentinel_matches_lt2_lesions": _pct(consistent, n),
                "n_unique_nonsentinel": n_nonsentinel_unique,
            })
    cols = ["feature", "n_sentinel_zero", "pct_sentinel",
            "n_patients_lt2_lesions", "sentinel_matches_lt2_lesions",
            "n_unique_nonsentinel"]
    sentinel_df = pd.DataFrame(rows, columns=cols)
    print(sentinel_df.to_string(index=False))

    # ── Diffusivity, special case (1.0 sentinel for N = 1) ───────────────
    _section("2. Diffusivity: 0.0 (N=0) vs 1.0 (N=1) vs continuous (N>=2)")
    rows = []
    for vessel in VESSELS:
        lesion_count = f"lesion_count_{vessel}"
        diff = f"diffusivity_{vessel}"
        n_zero = int((df[diff] == 0.0).sum())
        n_one = int((df[diff] == 1.0).sum())
        n_continuous = n - n_zero - n_one
        n_lc0 = int((df[lesion_count] == 0).sum())
        n_lc1 = int((df[lesion_count] == 1).sum())
        n_lcge2 = int((df[lesion_count] >= 2).sum())
        rows.append({
            "feature": diff,
            "n_zero": f"{n_zero} ({100*n_zero/n:.1f}%)",
            "n_one":  f"{n_one} ({100*n_one/n:.1f}%)",
            "n_continuous": f"{n_continuous} ({100*n_continuous/n:.1f}%)",
            "expected_zero_from_lc0": n_lc0,
            "expected_one_from_lc1": n_lc1,
            "expected_continuous_lc_ge2": n_lcge2,
        })
    cols = ["feature", "n_zero", "n_one", "n_continuous",
            "expected_zero_from_lc0", "expected_one_from_lc1",
            "expected_continuous_lc_ge2"]
    print(pd.DataFrame(rows, columns=cols).to_string(index=False))

    # ── Joint pattern across all 4 vessels ───────────────────────────────
    _section("3. Joint diffusivity patterns across all 4 vessels")
    diffs = df[[f"diffusivity_{v}" for v in VESSELS]]
    all_zero = ((diffs == 0.0).all(axis=1)).sum()
    all_one = ((diffs == 1.0).all(axis=1)).sum()
    all_sentinel = (((diffs == 0.0) | (diffs == 1.0)).all(axis=1)).sum()
    any_continuous = (~(((diffs == 0.0) | (diffs == 1.0)).all(axis=1))).sum()
    print(f"Patients with diffusivity == 0.0 on ALL 4 vessels: {all_zero}/{n} "
          f"({100*all_zero/n:.1f}%)  (these have 0 lesions cohort-wide, should be impossible "
          f"since every COCA patient has calcium)")
    print(f"Patients with diffusivity == 1.0 on ALL 4 vessels: {all_one}/{n} "
          f"({100*all_one/n:.1f}%)  (one lesion per vessel, multi-vessel single-lesion case)")
    print(f"Patients with diffusivity in {{0.0, 1.0}} on ALL 4 vessels: {all_sentinel}/{n} "
          f"({100*all_sentinel/n:.1f}%)  <- 'degenerate corner' patients")
    print(f"Patients with at least one vessel diffusivity in continuous regime: "
          f"{any_continuous}/{n} ({100*any_continuous/n:.1f}%)")

    # Sentinel rate per vessel.
    print()
    print("Per-vessel breakdown of diffusivity sentinel rate (0.0 OR 1.0):")
    for vessel in VESSELS:
        diff = f"diffusivity_{vessel}"
        n_sent = int(((df[diff] == 0.0) | (df[diff] == 1.0)).sum())
        print(f"  {diff:<22}  sentinel: {_pct(n_sent, n)}")

    # ── Cross-tab with low_burden_flag and Agatston category ─────────────
    _section("4. Degenerate corner vs low_burden_flag and Agatston category")
    if "low_burden_flag" in df.columns:
        lb_col = df["low_burden_flag"].astype(bool)
        deg_mask = ((diffs == 0.0) | (diffs == 1.0)).all(axis=1)
        ct = pd.crosstab(deg_mask, lb_col, rownames=["degenerate_all_4"],
                         colnames=["low_burden_flag"], margins=True)
        print("low_burden_flag vs all-4-vessel-diffusivity-sentinel:")
        print(ct.to_string())
        print()
        # Conditional probabilities.
        deg_and_lb = int((deg_mask & lb_col).sum())
        deg_total = int(deg_mask.sum())
        lb_total = int(lb_col.sum())
        if deg_total:
            print(f"P(low_burden | degenerate) = {deg_and_lb}/{deg_total} = "
                  f"{100*deg_and_lb/deg_total:.1f}%")
        if lb_total:
            print(f"P(degenerate | low_burden) = {deg_and_lb}/{lb_total} = "
                  f"{100*deg_and_lb/lb_total:.1f}%")

    if "category" in df.columns:
        print()
        print("Agatston category vs all-4-vessel-diffusivity-sentinel:")
        ct = pd.crosstab(deg_mask, df["category"], rownames=["degenerate"],
                         colnames=["category"], margins=True)
        print(ct.to_string())

    # ── Per-vessel diffusivity correlation with lesion_count_{vessel} ────
    _section("5. Diffusivity correlation with (lesion_count == 1) indicator")
    print("If diffusivity carries more information than 'is lesion_count == 1', "
          "the correlation should be moderate, not near 1.0.")
    print()
    for vessel in VESSELS:
        diff_col = f"diffusivity_{vessel}"
        is_one_lesion = (df[f"lesion_count_{vessel}"] == 1).astype(float)
        is_diff_1 = (df[diff_col] == 1.0).astype(float)
        # Point-biserial-style: correlation between continuous diffusivity
        # and the indicator (lesion_count_{vessel} == 1).
        corr_full = np.corrcoef(df[diff_col], is_one_lesion)[0, 1]
        # Also: exact match rate between (diffusivity == 1.0) and (lesion_count == 1).
        match = float(((is_diff_1 == 1) == (is_one_lesion == 1)).mean())
        print(f"  {diff_col:<22}  corr(diff, 1{{count==1}}) = {corr_full:+.3f}   "
              f"P((diff==1) == (count==1)) = {match:.3f}")

    # ── Variance breakdown ───────────────────────────────────────────────
    _section("6. Variance contribution: sentinel-dominated columns")
    print("Variance from sentinel rows alone vs from continuous rows alone.")
    print("If the sentinel rows dominate the variance, the feature is mostly a "
          "discrete indicator with cosmetic noise.")
    print()
    for vessel in VESSELS:
        for stem in DISTANCE_FEATURES + ("diffusivity",):
            col = f"{stem}_{vessel}"
            sentinel_mask = (df[col] == 0.0) | (
                (stem == "diffusivity") & (df[col] == 1.0)
            )
            n_sent = int(sentinel_mask.sum())
            n_cont = n - n_sent
            sd_full = float(df[col].std(ddof=0))
            sd_cont = float(df.loc[~sentinel_mask, col].std(ddof=0)) if n_cont > 1 else 0.0
            print(f"  {col:<32}  n_sentinel={n_sent:>3}  sd_full={sd_full:>8.3f}  "
                  f"sd_continuous_only={sd_cont:>8.3f}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
