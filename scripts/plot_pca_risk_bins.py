#!/usr/bin/env python
"""Exploratory: PCA scatter of the patient feature space coloured by
the 5-tier classical Agatston risk strata.

NOT a paper figure. F1 is locked at qcut tertiles. This script is a
sanity-check view requested for teammate intuition: where do the
clinical-threshold bins fall on PC1 vs PC2?

Bins (user-specified, classical Rumberger / Agatston):

    <  10        : Low Risk
    10 -    99   : Medium Risk
    100 -   399  : High Risk
    400 -   999  : Very High Risk
    >= 1000      : Extreme Risk

Reads the same seam files F1 reads:
    outputs/06_reduce/pca_scores.npy
    outputs/06_reduce/pca_scores_pid_order.csv
    outputs/06_reduce/cohort_metadata.csv  (column: agatston_total)

Writes:
    outputs/exploratory/pca_risk_bins/pca_pc1_pc2_by_risk.png
    outputs/exploratory/pca_risk_bins/pca_pc1_pc2_by_risk.pdf
    outputs/exploratory/pca_risk_bins/bin_counts.csv

Run from repo root on the remote:
    conda activate predict_env
    python scripts/plot_pca_risk_bins.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RISK_BINS = [
    ("Low Risk",        0.0,       10.0,    "#4C9F70"),   # green
    ("Medium Risk",     10.0,      100.0,   "#E8C547"),   # yellow
    ("High Risk",       100.0,     400.0,   "#E08E45"),   # orange
    ("Very High Risk",  400.0,     1000.0,  "#D7263D"),   # red
    ("Extreme Risk",    1000.0,    float("inf"), "#5A1A39"),  # dark purple
]


def assign_risk_label(score: float) -> str:
    """5-tier classical Agatston risk strata."""
    if score < 10:
        return "Low Risk"
    if score < 100:
        return "Medium Risk"
    if score < 400:
        return "High Risk"
    if score < 1000:
        return "Very High Risk"
    return "Extreme Risk"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reduce-dir", type=Path,
        default=Path("outputs/06_reduce"),
        help="Directory with pca_scores.npy + cohort_metadata.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/exploratory/pca_risk_bins"),
        help="Where to write the figures and the bin-count CSV",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load seam (byte-identical to F1's load path) ──────────────────
    pca_scores = np.load(args.reduce_dir / "pca_scores.npy")
    pid_order = pd.read_csv(
        args.reduce_dir / "pca_scores_pid_order.csv", dtype={"pid": str}
    )["pid"].tolist()
    meta = pd.read_csv(
        args.reduce_dir / "cohort_metadata.csv", dtype={"pid": str}
    ).set_index("pid")
    agatston = meta.loc[pid_order, "agatston_total"].astype(float).to_numpy()

    # ── Assign label per patient ──────────────────────────────────────
    labels = np.array([assign_risk_label(float(s)) for s in agatston])

    # ── Bin counts (sanity-check table) ───────────────────────────────
    counts = pd.DataFrame({
        "tier":  [b[0] for b in RISK_BINS],
        "lower": [b[1] for b in RISK_BINS],
        "upper": [b[2] for b in RISK_BINS],
        "n":     [int((labels == b[0]).sum()) for b in RISK_BINS],
    })
    counts["pct"] = (counts["n"] / counts["n"].sum() * 100).round(1)
    counts.to_csv(args.output_dir / "bin_counts.csv", index=False)

    # ── Scatter ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for name, lo, hi, color in RISK_BINS:
        m = labels == name
        n = int(m.sum())
        if n == 0:
            continue
        rng = f"[{int(lo)}, {int(hi) if hi != float('inf') else '∞'})"
        ax.scatter(
            pca_scores[m, 0], pca_scores[m, 1],
            c=color, s=26, alpha=0.75,
            edgecolors="white", linewidths=0.4,
            label=f"{name}  Agatston {rng}  (n={n})",
        )

    ax.set_xlabel("PC1  (burden axis, 35% variance)")
    ax.set_ylabel("PC2  (location axis, 11% variance)")
    ax.set_title(
        "Patient feature space (multi-block PCA) coloured by "
        "5-tier Agatston risk\n"
        f"N = {len(labels)} patients  (stage 5 cohort)"
    )
    ax.axhline(0, c="black", lw=0.4, alpha=0.4)
    ax.axvline(0, c="black", lw=0.4, alpha=0.4)
    ax.legend(loc="best", fontsize=9, framealpha=0.85)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / f"pca_pc1_pc2_by_risk.{ext}", dpi=200)

    print(f"Wrote: {args.output_dir / 'pca_pc1_pc2_by_risk.png'}")
    print(f"Wrote: {args.output_dir / 'pca_pc1_pc2_by_risk.pdf'}")
    print(f"Wrote: {args.output_dir / 'bin_counts.csv'}")
    print("\nBin counts:")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
