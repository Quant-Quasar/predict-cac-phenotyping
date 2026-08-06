#!/usr/bin/env python
"""Sparsity diagnostic for per-vessel density-tier features + dense_calcium_count.

The 16 per-vessel density-tier counts (``n_rois_d{1..4}_{lad,rca,lcx,lm}``)
plus ``dense_calcium_count`` are integer-valued, lower-bounded at zero, and
likely heavily zero-inflated for low-burden patients. Same publication-risk
pattern as the per-vessel distance / diffusivity features (D017): if most
patients sit at zero in a column, PCA aligns its first component with the
"is this bin populated" indicator rather than any phenotype axis.

This script characterises the 17 columns the same way D017's diagnostic did
for distance / diffusivity:

  1. Per-column zero rate + nonzero distribution (1, 2, 3+).
  2. Cross-tab with low_burden_flag and Agatston category.
  3. How many of the 16 tier bins each patient populates (sparsity profile).
  4. Per-tier prevalence (which tier x vessel combinations are nearly empty).

Read-only; safe to run anytime after stage 3.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from predict.config import load_config


VESSELS: tuple[str, ...] = ("lad", "rca", "lcx", "lm")
TIERS: tuple[str, ...] = ("d1", "d2", "d3", "d4")


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

    tier_cols = [f"n_rois_{t}_{v}" for v in VESSELS for t in TIERS]
    all_cols = tier_cols + ["dense_calcium_count"]

    # ── (1) Per-column zero rate + nonzero distribution ───────────────────
    _section("1. Per-column zero rate + value distribution")
    rows = []
    for col in all_cols:
        vals = df[col]
        n_zero = int((vals == 0).sum())
        n_one = int((vals == 1).sum())
        n_two = int((vals == 2).sum())
        n_three_plus = int((vals >= 3).sum())
        max_val = int(vals.max())
        mean_nonzero = float(vals[vals > 0].mean()) if (vals > 0).any() else 0.0
        rows.append({
            "feature": col,
            "zero_pct": f"{100*n_zero/n:.1f}%",
            "one_pct":  f"{100*n_one/n:.1f}%",
            "two_pct":  f"{100*n_two/n:.1f}%",
            "ge3_pct":  f"{100*n_three_plus/n:.1f}%",
            "max":      max_val,
            "mean_nonzero": f"{mean_nonzero:.2f}",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # ── (2) Per-tier across vessels (which bins are nearly empty) ─────────
    _section("2. Per-tier x vessel prevalence (fraction of patients with >= 1 ROI)")
    grid_rows = []
    for tier in TIERS:
        row = {"tier": tier}
        for v in VESSELS:
            col = f"n_rois_{tier}_{v}"
            pop_rate = float((df[col] > 0).mean())
            row[v.upper()] = f"{100*pop_rate:.1f}%"
        grid_rows.append(row)
    print(pd.DataFrame(grid_rows).to_string(index=False))

    # ── (3) Per-patient sparsity: how many of the 16 bins are nonzero ─────
    _section("3. Per-patient sparsity: count of populated tier-bins (out of 16)")
    nonzero_per_patient = (df[tier_cols] > 0).sum(axis=1)
    sparsity = nonzero_per_patient.value_counts().sort_index()
    print(f"{'n_populated':>12}  {'n_patients':>11}  {'pct':>6}")
    print("-" * 35)
    cum = 0
    for k, count in sparsity.items():
        cum += int(count)
        print(f"{int(k):>12}  {int(count):>11}  {100*count/n:>5.1f}%")
    print(f"\nmean populated bins/patient: {nonzero_per_patient.mean():.2f} / 16")
    print(f"median populated bins/patient: {int(nonzero_per_patient.median())} / 16")
    print(f"patients with <= 1 populated bin: "
          f"{_pct(int((nonzero_per_patient <= 1).sum()), n)}")
    print(f"patients with <= 3 populated bins: "
          f"{_pct(int((nonzero_per_patient <= 3).sum()), n)}")

    # ── (4) Cross-tab with low_burden_flag and Agatston category ──────────
    _section("4. Sparse-tier patients vs low_burden_flag and Agatston category")
    sparse_mask = nonzero_per_patient <= 1   # 0 or 1 populated bin
    if "low_burden_flag" in df.columns:
        lb = df["low_burden_flag"].astype(bool)
        ct = pd.crosstab(sparse_mask, lb,
                         rownames=["<=1_populated_bin"],
                         colnames=["low_burden_flag"], margins=True)
        print("<=1 populated bin vs low_burden_flag:")
        print(ct.to_string())
        print()
        sparse_total = int(sparse_mask.sum())
        lb_total = int(lb.sum())
        if sparse_total:
            sparse_and_lb = int((sparse_mask & lb).sum())
            print(f"P(low_burden | <=1 bin) = "
                  f"{sparse_and_lb}/{sparse_total} = {100*sparse_and_lb/sparse_total:.1f}%")

    if "category" in df.columns:
        print()
        print("<=1 populated bin vs Agatston category:")
        ct = pd.crosstab(sparse_mask, df["category"],
                         rownames=["<=1_populated_bin"],
                         colnames=["category"], margins=True)
        print(ct.to_string())

    # ── (5) dense_calcium_count specifically ──────────────────────────────
    _section("5. dense_calcium_count")
    col = df["dense_calcium_count"]
    n_zero = int((col == 0).sum())
    n_pos = int((col > 0).sum())
    print(f"zero: {_pct(n_zero, n)}")
    print(f"non-zero: {_pct(n_pos, n)}")
    print(f"max: {int(col.max())}")
    if (col > 0).any():
        print(f"mean of non-zero: {col[col > 0].mean():.2f}")
        print(f"distribution of non-zero values:")
        nonzero_dist = col[col > 0].value_counts().sort_index()
        for v, c in nonzero_dist.items():
            print(f"  value {int(v):>3}: {int(c)} patients")

    # ── (6) Variance contribution (full vs nonzero-only) ──────────────────
    _section("6. Variance contribution (full vs nonzero rows only)")
    print("If sd_full >> sd_nonzero, the variance is dominated by the zero point mass.")
    print()
    rows = []
    for col in all_cols:
        vals = df[col].astype(float)
        sd_full = float(vals.std(ddof=0))
        nonzero = vals[vals > 0]
        sd_nonzero = float(nonzero.std(ddof=0)) if len(nonzero) > 1 else 0.0
        rows.append({
            "feature": col,
            "sd_full": f"{sd_full:.3f}",
            "sd_nonzero": f"{sd_nonzero:.3f}",
            "ratio_sd_nonzero_to_full":
                f"{sd_nonzero / sd_full:.2f}" if sd_full > 0 else "n/a",
            "n_nonzero": int((vals > 0).sum()),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
