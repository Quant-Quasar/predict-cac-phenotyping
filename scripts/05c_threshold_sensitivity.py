#!/usr/bin/env python
"""Stage 4c, threshold sensitivity table for the ICC gate.

Reads ``outputs/05_icc/icc_report.csv`` (produced by ``05_icc_gate.py``) and
re-applies several plausible threshold values to count how many features pass
at each. No ICCs are recomputed; this script is purely a counting exercise on
the existing report.

The D013-locked production threshold is 0.75. The other thresholds in this
table are for transparency only and must not be used to retroactively select
a different gate (D013 explicitly locks 0.75 before any data is examined).

Outputs:

  - ``outputs/05_icc/threshold_sensitivity.csv``  one row per (threshold,
    source) pair with pass / total counts.
  - prints the table to stdout for quick inspection.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from predict.config import load_config


DEFAULT_THRESHOLDS: tuple[float, ...] = (0.50, 0.60, 0.75, 0.85)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS),
        help="Thresholds to evaluate (default 0.50 0.60 0.75 0.85).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    icc_dir = cfg.paths.outputs / "05_icc"
    report = pd.read_csv(icc_dir / "icc_report.csv")
    report["icc"] = pd.to_numeric(report["icc"], errors="coerce")

    bypass_mask = report["icc_source"] == "invariant_by_construction"
    empirical_mask = report["icc_source"] == "empirical"

    n_bypass = int(bypass_mask.sum())
    n_empirical = int(empirical_mask.sum())

    rows: list[dict] = []
    for t in args.thresholds:
        # Bypass always passes regardless of threshold; ICC = 1.0 by construction.
        bypass_passing = n_bypass

        emp_icc = report.loc[empirical_mask, "icc"]
        emp_passing = int(((~emp_icc.isna()) & (emp_icc >= t)).sum())

        total_passing = bypass_passing + emp_passing
        rows.append({
            "threshold": t,
            "bypass_total": n_bypass,
            "bypass_passing": bypass_passing,
            "empirical_total": n_empirical,
            "empirical_passing": emp_passing,
            "total_passing": total_passing,
            "total_pct_of_175": round(100.0 * total_passing / (n_bypass + n_empirical), 1),
            "is_production_threshold": (abs(t - cfg.stability.icc_threshold) < 1e-9),
        })

    out_df = pd.DataFrame(rows)
    out_csv = icc_dir / "threshold_sensitivity.csv"
    out_df.to_csv(out_csv, index=False)

    # Pretty stdout.
    print()
    print("Threshold sensitivity (D013 production threshold marked with *):")
    print()
    cols = ["threshold", "bypass_passing", "empirical_passing",
            "total_passing", "total_pct_of_175"]
    widths = {c: max(len(c), out_df[c].astype(str).str.len().max()) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols) + "  is_locked"
    print(header)
    print("  ".join("-" * widths[c] for c in cols) + "  " + "-" * 9)
    for _, r in out_df.iterrows():
        mark = "*" if r["is_production_threshold"] else " "
        line = "  ".join(str(r[c]).ljust(widths[c]) for c in cols)
        print(f"{line}    {mark}")
    print()
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
