#!/usr/bin/env python
"""Stage 5 robust-feature deliverable.

Reads the representative-feature lists and the redundancy-cluster audits
from the full cohort plus the two kernel-stratified subcohorts, then prints:

  1. Per-cohort representative count + family breakdown
  2. The full-cohort 23 representatives with annotations:
       feature, family, icc_source, icc value, cluster_size, decided_by
  3. Cross-cohort intersection: features that appear in all three cohorts
     (the publication-grade "kernel-independent robust features")
  4. Per-cohort exclusive features
  5. The 11 spatial-only features used in finding-3
     (spatial-only x GMM x k=2 partition that replicated across cohorts)

This is read-only; it expects ``scripts/06_reduce.py`` to have run on the
full cohort plus the two kernel-stratified cohorts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from predict.config import load_config
from predict.reduce.pca import assign_family


COHORTS: tuple[tuple[str, str], ...] = (
    ("full", ""),
    ("Qr36d/2", "stratified_Qr36d_2"),
    ("I30f/3", "stratified_I30f_3"),
)


# The 11 spatial-distribution features used in the D021 finding-3
# spatial-only feature space (after D017 drops, before any PCA).
SPATIAL_FEATURES_FOR_FINDING3: tuple[str, ...] = (
    "lesion_count_lad", "lesion_count_rca", "lesion_count_lcx", "lesion_count_lm",
    "lesion_count_total",
    "n_calcified_arteries",
    "gini_lesion_volume",
    "dist_from_top_max", "dist_from_top_mean",
    "center_of_mass_z",
    "inter_lesion_dist_mean_lad", "inter_lesion_dist_max_lad",
    "first_to_last_dist_lad",
)


def _section(title: str) -> None:
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


def _cohort_dir(base: Path, sub: str) -> Path:
    return base if not sub else base / sub


def _load_reps(cdir: Path) -> list[str]:
    p = cdir / "representative_features.csv"
    if not p.exists():
        return []
    return pd.read_csv(p)["feature"].tolist()


def _load_clusters(cdir: Path) -> pd.DataFrame:
    """Read either the multi-block assignments (D022 primary) or the
    single-matrix cluster assignments (D020 sensitivity), whichever exists.

    Multi-block has the extra `block` column; single-matrix does not.
    """
    multi = cdir / "multi_block_assignments.csv"
    single = cdir / "redundancy_clusters.csv"
    if multi.exists():
        return pd.read_csv(multi)
    if single.exists():
        return pd.read_csv(single)
    return pd.DataFrame()


def _family_of(name: str) -> str:
    return assign_family(name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    base = cfg.paths.outputs / "06_reduce"
    if not base.exists():
        print(f"ERROR: {base} does not exist.")
        return 2

    reps_per_cohort: dict[str, list[str]] = {}
    clusters_per_cohort: dict[str, pd.DataFrame] = {}
    for name, sub in COHORTS:
        cdir = _cohort_dir(base, sub)
        reps_per_cohort[name] = _load_reps(cdir)
        clusters_per_cohort[name] = _load_clusters(cdir)

    # ── 1. Per-cohort representative count + family breakdown ────────────
    _section("1. Representative-feature counts per cohort")
    summary_rows = []
    for name, reps in reps_per_cohort.items():
        fam_counts: dict[str, int] = {}
        for f in reps:
            fam_counts.setdefault(_family_of(f), 0)
            fam_counts[_family_of(f)] += 1
        row = {"cohort": name, "n_representatives": len(reps)}
        row.update(fam_counts)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).fillna(0).set_index("cohort")
    # Keep stable column order: count first, then families sorted alphabetically.
    cols = ["n_representatives"] + sorted(
        c for c in summary_df.columns if c != "n_representatives"
    )
    print(summary_df[cols].astype({"n_representatives": int}).to_string())

    # ── 2. Full-cohort representatives with annotations ───────────────
    _section("2. Full-cohort representatives with annotation "
             "(feature, block, family, icc_source, icc, cluster_size, decided_by)")
    full_clusters = clusters_per_cohort["full"]
    if not full_clusters.empty:
        rep_rows = full_clusters[full_clusters["is_representative"] == True].copy()
        rep_rows["family"] = rep_rows["feature"].map(_family_of)
        keep = ["feature", "block", "family", "icc_source", "icc",
                "cluster_size", "decided_by", "cluster_id"]
        keep = [c for c in keep if c in rep_rows.columns]
        rep_rows = rep_rows[keep]
        sort_cols = [c for c in ("block", "family", "feature") if c in rep_rows.columns]
        rep_rows = rep_rows.sort_values(sort_cols)
        print(rep_rows.to_string(index=False))
    else:
        print("(no full-cohort redundancy_clusters.csv or "
              "multi_block_assignments.csv found)")

    # ── 3. Cross-cohort intersection ─────────────────────────────────────
    _section("3. Cross-cohort intersection (features representative in ALL three cohorts)")
    sets = [set(reps) for reps in reps_per_cohort.values()]
    if all(sets):
        intersection = sorted(set.intersection(*sets))
        print(f"Count: {len(intersection)}")
        if intersection:
            rows = [{"feature": f, "family": _family_of(f)} for f in intersection]
            print(pd.DataFrame(rows).to_string(index=False))
    else:
        print("(at least one cohort has no representatives)")

    # ── 4. Per-cohort exclusive features ─────────────────────────────────
    _section("4. Per-cohort exclusive features (only in that cohort's representative set)")
    for name, reps in reps_per_cohort.items():
        others = set()
        for other_name, other_reps in reps_per_cohort.items():
            if other_name != name:
                others |= set(other_reps)
        exclusive = sorted(set(reps) - others)
        print(f"--- {name} exclusive ({len(exclusive)}) ---")
        if exclusive:
            for f in exclusive:
                print(f"  {f}  ({_family_of(f)})")
        else:
            print("  (none)")

    # ── 5. Spatial features used in finding-3 ─────────────────────────────
    _section("5. Spatial-distribution features used in finding-3 "
             "(spatial-only x GMM x k=2, replicated across all three cohorts)")
    rows = []
    for f in SPATIAL_FEATURES_FOR_FINDING3:
        appearances = [name for name, reps in reps_per_cohort.items()
                       if f in reps]
        rows.append({
            "feature": f,
            "family": _family_of(f),
            "in_full_reps": "full" in appearances,
            "in_qr36d_reps": "Qr36d/2" in appearances,
            "in_i30f_reps": "I30f/3" in appearances,
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print("Note: the spatial-only feature space is the 11 features above used "
          "as RAW INPUTS to a separate PCA before clustering at the spatial-only")
    print("axis (finding-3). They are not all required to appear in the full-")
    print("matrix representative set; some get absorbed into the same Spearman")
    print("r^2 cluster as a non-spatial feature.")

    # ── 6. Headline summary ──────────────────────────────────────────────
    _section("6. Headline summary")
    print(f"  Full cohort:     {len(reps_per_cohort['full'])} robust representatives")
    print(f"  Qr36d/2 stratum: {len(reps_per_cohort['Qr36d/2'])} robust representatives")
    print(f"  I30f/3 stratum:  {len(reps_per_cohort['I30f/3'])} robust representatives")
    if all(sets):
        print(f"  Kernel-independent (all 3 cohorts): "
              f"{len(set.intersection(*sets))} features")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
