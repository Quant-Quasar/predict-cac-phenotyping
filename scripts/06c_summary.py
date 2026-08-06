#!/usr/bin/env python
"""Stage 5 cross-cohort summary report.

Reads ``outputs/06_reduce/`` (full cohort) and the two ``stratified_*``
subdirectories (Qr36d/2 and I30f/3), then prints:

  1. Per-cohort Hopkins statistic + verdict
  2. Per-cohort gap-statistic selected k across 3 algorithms x 3 feature spaces
  3. Per-cohort Hennig stability at forced k=3 (full feature space) and
     at the spatial-only x GMM x k=2 secondary probe
  4. Per-cohort kernel chi-square p-values (skipped for stratified cohorts
     because there is only one kernel by construction)
  5. Cross-cohort consistency check: does the continuum signature
     (gap monotonically rising) replicate within each kernel stratum?

Read-only; safe to run any time after the three production runs finish.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from predict.config import load_config


COHORT_LABELS: tuple[tuple[str, str], ...] = (
    ("full", ""),
    ("Qr36d/2", "stratified_Qr36d_2"),
    ("I30f/3", "stratified_I30f_3"),
)


def _section(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def _cohort_dir(base: Path, subpath: str) -> Path:
    return base if not subpath else base / subpath


def _load_json(path: Path) -> Optional[dict | list]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


# ─────────────────── per-cohort accessors ───────────────────


def hopkins_row(name: str, cdir: Path) -> dict:
    payload = _load_json(cdir / "hopkins.json")
    if payload is None:
        return {"cohort": name, "H": None, "verdict": "n/a",
                "n_total": None, "n_features": None}
    return {
        "cohort": name,
        "H": round(payload["H"], 4),
        "verdict": payload["verdict"],
        "n_total": payload["n_total"],
        "n_features": payload["n_features"],
    }


def gap_summary_long(name: str, cdir: Path) -> pd.DataFrame:
    df = _load_csv(cdir / "gap_statistic_summary.csv")
    if df is None:
        return pd.DataFrame()
    out = df[["feature_space", "algorithm", "selected_k"]].copy()
    out.insert(0, "cohort", name)
    return out


def gap_curve(name: str, cdir: Path) -> pd.DataFrame:
    """Wide-form gap curve: columns gap_k1..gap_k12 from the summary CSV."""
    df = _load_csv(cdir / "gap_statistic_summary.csv")
    if df is None:
        return pd.DataFrame()
    df = df.copy()
    df.insert(0, "cohort", name)
    return df


def hennig_table(name: str, cdir: Path) -> pd.DataFrame:
    df = _load_csv(cdir / "validity_checks.csv")
    if df is None:
        return pd.DataFrame()
    sub = df[df["test"] == "hennig_clusterboot"].copy()
    if sub.empty:
        return sub
    sub.insert(0, "cohort", name)
    return sub


def kernel_chi_table(name: str, cdir: Path) -> pd.DataFrame:
    df = _load_csv(cdir / "validity_checks.csv")
    if df is None:
        return pd.DataFrame()
    sub = df[df["test"] == "kernel_chi_square"].copy()
    if sub.empty:
        return sub
    sub.insert(0, "cohort", name)
    return sub


def consensus_summary(name: str, cdir: Path) -> pd.DataFrame:
    payload = _load_json(cdir / "consensus_summary.json")
    if payload is None:
        return pd.DataFrame()
    out = pd.DataFrame(payload)
    out.insert(0, "cohort", name)
    return out


# ─────────────────── continuum signature check ───────────────────


def is_continuum(gap_row: pd.Series, monotonic_tol: float = 1e-6) -> bool:
    """A monotonically (non-decreasing) gap curve with no clear plateau is the
    empirical signature of continuum structure. We define 'plateau' as a
    drop of more than ``monotonic_tol`` between consecutive k values."""
    gap_cols = [c for c in gap_row.index if c.startswith("gap_k")]
    if not gap_cols:
        return False
    gap_cols.sort(key=lambda c: int(c[len("gap_k"):]))
    values = gap_row[gap_cols].astype(float).to_numpy()
    diffs = np.diff(values)
    return bool(np.all(diffs > -monotonic_tol))


# ─────────────────── main ───────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    base = cfg.paths.outputs / "06_reduce"
    if not base.exists():
        print(f"ERROR: {base} does not exist; run scripts/06_reduce.py first.")
        return 2

    # ── 1. Hopkins per cohort ────────────────────────────────────
    _section("1. Hopkins clusterability (per cohort)")
    rows = [hopkins_row(name, _cohort_dir(base, sub))
            for name, sub in COHORT_LABELS]
    print(pd.DataFrame(rows).to_string(index=False))

    # ── 2. Gap selected k per (cohort, feature_space, algorithm) ─
    _section("2. Gap-statistic selected k (per cohort x feature_space x algorithm)")
    long = pd.concat([gap_summary_long(name, _cohort_dir(base, sub))
                       for name, sub in COHORT_LABELS], ignore_index=True)
    if not long.empty:
        pivot = long.pivot_table(index=["feature_space", "algorithm"],
                                  columns="cohort",
                                  values="selected_k",
                                  aggfunc="first")
        # Reorder columns by COHORT_LABELS.
        wanted = [name for name, _ in COHORT_LABELS if name in pivot.columns]
        pivot = pivot[wanted]
        print(pivot.to_string())
    else:
        print("(no gap-statistic data found)")

    # ── 3. Continuum signature per (cohort, feature_space, algorithm) ─
    _section("3. Continuum signature (monotonic gap rise => continuum=True)")
    all_curves = pd.concat([gap_curve(name, _cohort_dir(base, sub))
                             for name, sub in COHORT_LABELS], ignore_index=True)
    if not all_curves.empty:
        continuum_records = []
        for _, row in all_curves.iterrows():
            continuum_records.append({
                "cohort": row["cohort"],
                "feature_space": row["feature_space"],
                "algorithm": row["algorithm"],
                "selected_k": int(row["selected_k"]),
                "continuum_curve": is_continuum(row),
            })
        cont_df = pd.DataFrame(continuum_records)
        print(cont_df.to_string(index=False))
        print()
        n_continuum = int(cont_df["continuum_curve"].sum())
        n_total = len(cont_df)
        print(f"Monotonic continuum signature: {n_continuum} / {n_total} runs")
    else:
        print("(no gap-statistic curves found)")

    # ── 4. Kernel chi-square at forced k=3 ───────────────────────
    _section("4. Kernel chi-square at forced k=3 (full cohort only)")
    chi = kernel_chi_table("full", _cohort_dir(base, ""))
    if not chi.empty:
        keep_cols = [c for c in ("algorithm", "chi2", "pval", "dof", "passes")
                     if c in chi.columns]
        print(chi[keep_cols].to_string(index=False))
    else:
        print("(no chi-square records found)")
    print("\nStratified cohorts contain only one kernel by construction; "
          "chi-square not meaningful there.")

    # ── 5. Hennig stability ──────────────────────────────────────
    _section("5. Hennig clusterboot stability per cohort")
    hen = pd.concat([hennig_table(name, _cohort_dir(base, sub))
                      for name, sub in COHORT_LABELS], ignore_index=True)
    if hen.empty:
        print("(no Hennig records found)")
    else:
        keep_cols = [c for c in (
            "cohort", "algorithm", "k", "feature_space",
            "cluster_id", "jaccard_median", "jaccard_mean", "stable",
        ) if c in hen.columns]
        # If feature_space column is missing (older outputs), inject "full".
        if "feature_space" not in hen.columns:
            hen["feature_space"] = "full"
            keep_cols.insert(3, "feature_space")
        # Forced k=3 block (full space).
        forced_k3 = hen[(hen["k"] == 3) & (hen["feature_space"] == "full")]
        if not forced_k3.empty:
            print("--- forced k=3 (full feature space) ---")
            print(forced_k3[keep_cols].to_string(index=False))
        # Spatial-only x GMM x k=2 secondary probe.
        spatial = hen[(hen["k"] == 2) & (hen["feature_space"] == "spatial_only")]
        if not spatial.empty:
            print()
            print("--- spatial-only x GMM x k=2 (secondary finding probe) ---")
            print(spatial[keep_cols].to_string(index=False))

    # ── 6. Consensus PAC ─────────────────────────────────────────
    _section("6. Monti consensus PAC at gap-selected k (full feature space)")
    cons = pd.concat([consensus_summary(name, _cohort_dir(base, sub))
                       for name, sub in COHORT_LABELS], ignore_index=True)
    if cons.empty:
        print("(no consensus records found)")
    else:
        keep_cols = [c for c in (
            "cohort", "algorithm", "k", "pac_score",
            "n_subsamples", "subsample_frac",
        ) if c in cons.columns]
        print(cons[keep_cols].to_string(index=False))

    # ── 7. Headline summary ──────────────────────────────────────
    _section("7. Headline summary")
    h_full = next((r for r in rows if r["cohort"] == "full"), None)
    h_qr = next((r for r in rows if r["cohort"] == "Qr36d/2"), None)
    h_i30 = next((r for r in rows if r["cohort"] == "I30f/3"), None)
    if h_full:
        print(f"  full cohort Hopkins H = {h_full['H']} ({h_full['verdict']})")
    if h_qr:
        print(f"  Qr36d/2 stratum H     = {h_qr['H']} ({h_qr['verdict']})")
    if h_i30:
        print(f"  I30f/3 stratum H      = {h_i30['H']} ({h_i30['verdict']})")
    if not all_curves.empty:
        full_curves = all_curves[all_curves["cohort"] == "full"]
        qr_curves = all_curves[all_curves["cohort"] == "Qr36d/2"]
        i30_curves = all_curves[all_curves["cohort"] == "I30f/3"]

        def _mono_frac(df):
            if df.empty:
                return None
            n = len(df)
            mono = sum(1 for _, r in df.iterrows() if is_continuum(r))
            return f"{mono}/{n}"

        print(f"  Monotonic gap rises (continuum signature):")
        print(f"    full cohort:   {_mono_frac(full_curves)}")
        print(f"    Qr36d/2 stratum: {_mono_frac(qr_curves)}")
        print(f"    I30f/3 stratum:  {_mono_frac(i30_curves)}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
