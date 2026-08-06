#!/usr/bin/env python
"""Stage 4 diagnostic dump.

Reads ``outputs/05_icc/icc_report.csv`` + ``icc_summary.json`` and prints:

  1. Overall summary from the JSON.
  2. Shape-feature ICCs (regression check: must all be ~1.0 under our
     fixed-mask design; anything below 0.99 means the perturbation pipeline
     is accidentally moving the mask).
  3. All empirical features that passed the gate, sorted by ICC desc.
  4. The worst empirical features, sorted by ICC asc.
  5. ``n_subjects`` distribution across empirical features (catches features
     where listwise NaN deletion shrank the cohort below 422).
  6. Per-PyRadiomics-family pass-rate breakdown.

Pure read-only; safe to run anytime after ``05_icc_gate.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from predict.config import load_config


PYRADIOMICS_FAMILIES: tuple[str, ...] = (
    "shape", "firstorder", "glcm", "glszm", "glrlm", "ngtdm", "gldm",
)


def _fmt(x: float, ndigits: int = 4) -> str:
    if isinstance(x, float) and np.isnan(x):
        return "NaN"
    return f"{x:.{ndigits}f}"


def _section(title: str) -> None:
    print()
    print("=" * 76)
    print(title)
    print("=" * 76)


def _print_table(df: pd.DataFrame, cols: list[str]) -> None:
    """Print a small dataframe with monospaced columns."""
    if df.empty:
        print("(empty)")
        return
    widths = {c: max(len(c), df[c].astype(str).str.len().max()) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("  ".join("-" * widths[c] for c in cols))
    for _, row in df.iterrows():
        print("  ".join(str(row[c]).ljust(widths[c]) for c in cols))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--top", type=int, default=25,
                        help="Show top-N passing empirical features (default 25).")
    parser.add_argument("--bottom", type=int, default=15,
                        help="Show bottom-N empirical features (default 15).")
    parser.add_argument("--shape-tolerance", type=float, default=0.01,
                        help="Warn if any shape ICC is more than this below 1.0.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    icc_dir = cfg.paths.outputs / "05_icc"
    report = pd.read_csv(icc_dir / "icc_report.csv")
    with open(icc_dir / "icc_summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    # Coerce ICC column to float (script may have written "NaN" string for NaNs).
    report["icc"] = pd.to_numeric(report["icc"], errors="coerce")

    # ── (1) Summary ──────────────────────────────────────────────
    _section("1. icc_summary.json")
    print(json.dumps(summary, indent=2))

    # ── (2) Shape-feature regression check ───────────────────────
    _section("2. Shape ICCs (regression check: should all be ~1.0)")
    shape_mask = report["feature"].str.startswith("original_shape_")
    shape_df = report.loc[shape_mask, ["feature", "icc", "n_subjects", "passes_gate"]].copy()
    shape_df["icc"] = shape_df["icc"].apply(_fmt)
    _print_table(shape_df, ["feature", "icc", "n_subjects", "passes_gate"])

    shape_iccs = pd.to_numeric(report.loc[shape_mask, "icc"], errors="coerce")
    n_shape = int(shape_mask.sum())
    n_shape_high = int((shape_iccs >= 1.0 - args.shape_tolerance).sum())
    print()
    if n_shape_high == n_shape:
        print(f"OK: all {n_shape} shape features have ICC >= {1.0 - args.shape_tolerance:.3f}.")
        print("    Perturbation pipeline is mask-stable (mask is genuinely held fixed).")
    else:
        print(f"WARNING: only {n_shape_high}/{n_shape} shape features have ICC "
              f">= {1.0 - args.shape_tolerance:.3f}.")
        print("         The mask may be moving under perturbation; investigate "
              "apply_perturbation in src/predict/stability/perturbations.py.")

    # ── (3) Passing empirical features ───────────────────────────
    _section(f"3. Empirical features that PASS the gate (sorted ICC desc, top {args.top})")
    emp_pass = report[(report["icc_source"] == "empirical") &
                      (report["passes_gate"] == True)].copy()  # noqa: E712
    emp_pass = emp_pass.sort_values("icc", ascending=False).head(args.top)
    emp_pass["icc"] = emp_pass["icc"].apply(_fmt)
    print(f"Total empirical passing: {(report['icc_source'] == 'empirical').pipe(lambda s: report.loc[s, 'passes_gate'].sum())}")
    print()
    _print_table(emp_pass, ["feature", "icc", "n_subjects", "n_raters"])

    # ── (4) Worst empirical features ─────────────────────────────
    _section(f"4. Worst empirical features (sorted ICC asc, bottom {args.bottom})")
    emp = report[report["icc_source"] == "empirical"].copy()
    emp_sorted = emp.sort_values("icc", ascending=True, na_position="first").head(args.bottom)
    emp_sorted["icc"] = emp_sorted["icc"].apply(_fmt)
    _print_table(emp_sorted, ["feature", "icc", "n_subjects", "passes_gate"])

    # ── (5) n_subjects distribution ──────────────────────────────
    _section("5. n_subjects distribution across empirical features")
    n_used = emp["n_subjects"].value_counts().sort_index()
    print(f"{'n_subjects':>12}  {'n_features':>10}")
    print("-" * 28)
    for n, count in n_used.items():
        marker = "" if n == 422 else "  <- below full eligible cohort"
        print(f"{int(n):>12}  {int(count):>10}{marker}")
    print()
    full = int((emp["n_subjects"] == 422).sum())
    print(f"Features at full cohort (n=422): {full} / {len(emp)}")
    if full < len(emp):
        print("Features whose ICC was computed on < 422 patients due to listwise NaN deletion:")
        below = emp[emp["n_subjects"] < 422][["feature", "n_subjects"]].sort_values("n_subjects")
        _print_table(below, ["feature", "n_subjects"])

    # ── (6) Per-family pass rate ─────────────────────────────────
    _section("6. Per-PyRadiomics-family pass-rate breakdown")
    rows = []
    for fam in PYRADIOMICS_FAMILIES:
        prefix = f"original_{fam}_"
        in_fam = emp["feature"].str.startswith(prefix)
        total = int(in_fam.sum())
        passing = int((in_fam & (emp["passes_gate"] == True)).sum())  # noqa: E712
        fam_iccs = pd.to_numeric(emp.loc[in_fam, "icc"], errors="coerce")
        rows.append({
            "family": fam,
            "total": total,
            "passing": passing,
            "pass_pct": f"{(100.0 * passing / total) if total else 0:.0f}%",
            "icc_min": _fmt(float(fam_iccs.min()) if total else float("nan")),
            "icc_median": _fmt(float(fam_iccs.median()) if total else float("nan")),
            "icc_max": _fmt(float(fam_iccs.max()) if total else float("nan")),
        })
    fam_df = pd.DataFrame(rows)
    _print_table(fam_df, ["family", "total", "passing", "pass_pct",
                          "icc_min", "icc_median", "icc_max"])

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
