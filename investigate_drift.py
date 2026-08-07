#!/usr/bin/env python
"""Drift investigation for the post-rebuild rerun.

The verification pass showed asymmetric Hopkins drift across the three
cohorts (full +0.003, Qr36d/2 -0.058, I30f/3 +0.017). Pure BLAS noise
should be symmetric and scale with 1/sqrt(N); the Qr36d/2 drift is
20x the full-cohort drift and opposite in sign to I30f/3. That rules
out pure noise -- something specific changed for the smaller strata.

This script runs 6 diagnostics, cheap first, to localise the cause:

  1. cohort N + pid sets per stratum (catches different patients)
  2. SHA-256 of representative_features.csv per stratum
     (catches different multi-block representatives picked)
  3. SHA-256 of prepared_matrix.csv columns per stratum
     (catches D019 / ComBat / YJ value drift)
  4. ComBat post-correction R^2 per stratum + texture column
     (catches ComBat convergence drift)
  5. PCA top-3 eigenvalues per stratum (catches eigenvalue tie flips)
  6. INTRA-MACHINE determinism: rerun stage 5 on Qr36d/2 a SECOND time,
     SHA pca_scores.npy, compare to the first run. If different, stage 5
     is non-deterministic on this machine -- a real bug.

Each section prints a verdict + the values it found. Run after a full
pipeline run has completed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from predict.config import load_config


def _sha16(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _file_sha16(path: Path) -> str:
    if not path.exists():
        return "missing"
    return _sha16(path.read_bytes())


def _section(title: str) -> None:
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)


COHORT_DIRS = {
    "full":    "",
    "Qr36d/2": "stratified_Qr36d_2",
    "I30f/3":  "stratified_I30f_3",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--skip-determinism", action="store_true",
                        help="skip the stage 5 rerun (saves ~1 minute)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    reduce_root = cfg.paths.outputs / "06_reduce"
    repo_root = cfg.paths.outputs.parent

    # ── 1. Cohort N + pid sets ────────────────────────────────────
    _section("1. Cohort N + pid sets")
    cohort_pids: dict[str, list[str]] = {}
    for cohort, sub in COHORT_DIRS.items():
        cdir = reduce_root if sub == "" else reduce_root / sub
        meta_csv = cdir / "cohort_metadata.csv"
        if not meta_csv.exists():
            print(f"  [{cohort}] cohort_metadata.csv missing")
            continue
        meta = pd.read_csv(meta_csv, dtype={"pid": str})
        cohort_pids[cohort] = sorted(meta["pid"].tolist())
        pid_sha = _sha16("|".join(cohort_pids[cohort]).encode())
        print(f"  [{cohort}] N = {len(cohort_pids[cohort])}, "
              f"pid_set sha = {pid_sha}")

    # ── 2. Representative features per stratum ────────────────────
    _section("2. SHA of representative_features.csv per stratum")
    rep_shas: dict[str, str] = {}
    for cohort, sub in COHORT_DIRS.items():
        cdir = reduce_root if sub == "" else reduce_root / sub
        reps_csv = cdir / "representative_features.csv"
        if not reps_csv.exists():
            print(f"  [{cohort}] missing")
            continue
        reps = pd.read_csv(reps_csv)
        # SHA on the SORTED feature list, so order doesn't matter
        sorted_feats = "|".join(sorted(reps["feature"].astype(str).tolist()))
        rep_shas[cohort] = _sha16(sorted_feats.encode())
        print(f"  [{cohort}] {len(reps)} features, "
              f"sorted_set sha = {rep_shas[cohort]}")

    if "full" in cohort_pids and "Qr36d/2" in cohort_pids:
        # Show which representatives differ between Qr36d/2 and full
        # cohort (these are the cohort-exclusive ones)
        try:
            full_reps = set(pd.read_csv(reduce_root / "representative_features.csv")["feature"])
            qr_reps = set(pd.read_csv(reduce_root / "stratified_Qr36d_2/representative_features.csv")["feature"])
            i30_reps = set(pd.read_csv(reduce_root / "stratified_I30f_3/representative_features.csv")["feature"])
            in_qr_only = sorted(qr_reps - full_reps - i30_reps)
            in_i30_only = sorted(i30_reps - full_reps - qr_reps)
            print(f"  features exclusively in Qr36d/2 set: {in_qr_only}")
            print(f"  features exclusively in I30f/3  set: {in_i30_only}")
        except Exception as e:
            print(f"  (could not compute exclusive sets: {e})")

    # ── 3. Prepared matrix column SHAs ───────────────────────────
    _section("3. SHA of prepared_matrix.csv content per stratum")
    for cohort, sub in COHORT_DIRS.items():
        cdir = reduce_root if sub == "" else reduce_root / sub
        prep_csv = cdir / "prepared_matrix.csv"
        if not prep_csv.exists():
            print(f"  [{cohort}] missing")
            continue
        prep = pd.read_csv(prep_csv, dtype={"pid": str})
        # SHA on the matrix in canonical form: sorted by pid, then by
        # column name, then the float values to 9 sig figs
        sorted_pids = sorted(prep["pid"].tolist())
        prep = prep.set_index("pid").loc[sorted_pids].reset_index()
        prep = prep[sorted(prep.columns)]
        canonical = prep.to_csv(index=False, float_format="%.9g").encode()
        sha = _sha16(canonical)
        print(f"  [{cohort}] shape={prep.shape}, "
              f"canonical_csv sha = {sha}")

    # ── 4. ComBat audit per stratum ──────────────────────────────
    _section("4. ComBat post-correction R^2 per stratum")
    for cohort, sub in COHORT_DIRS.items():
        cdir = reduce_root if sub == "" else reduce_root / sub
        combat_csv = cdir / "combat_audit.csv"
        if not combat_csv.exists():
            print(f"  [{cohort}] no combat_audit.csv (single-kernel stratum?)")
            continue
        c = pd.read_csv(combat_csv)
        cols = [col for col in
                ["feature", "kernel_r2_pre", "kernel_r2_post"]
                if col in c.columns]
        print(f"  [{cohort}]")
        print(c[cols].to_string(index=False))

    # ── 5. PCA eigenvalues per stratum ───────────────────────────
    _section("5. PCA top-3 eigenvalues per stratum (catch eigenvalue ties)")
    for cohort, sub in COHORT_DIRS.items():
        cdir = reduce_root if sub == "" else reduce_root / sub
        ev_csv = cdir / "pca_explained_variance.csv"
        if not ev_csv.exists():
            print(f"  [{cohort}] missing pca_explained_variance.csv")
            continue
        ev = pd.read_csv(ev_csv)
        # Look for an eigenvalue / variance column
        val_col = None
        for cand in ["eigenvalue", "explained_variance", "variance",
                     "explained_variance_ratio"]:
            if cand in ev.columns:
                val_col = cand
                break
        if val_col is None:
            print(f"  [{cohort}] could not find variance column in "
                  f"{list(ev.columns)}")
            continue
        top3 = ev[val_col].sort_values(ascending=False).head(3).tolist()
        gap_1_2 = top3[0] - top3[1] if len(top3) >= 2 else float("nan")
        gap_2_3 = top3[1] - top3[2] if len(top3) >= 3 else float("nan")
        warn = ""
        # A small relative gap between eigenvalues 1-2 or 2-3 is the
        # classic precondition for cross-machine PC reordering.
        if len(top3) >= 2 and top3[0] > 0:
            rel_gap_1_2 = gap_1_2 / top3[0]
            if rel_gap_1_2 < 0.05:
                warn = "  WARN: PC1/PC2 within 5% -- vulnerable to reordering"
        print(f"  [{cohort}] top-3 ({val_col}) = {top3[0]:.4f}, "
              f"{top3[1]:.4f}, {top3[2]:.4f}; gaps = {gap_1_2:.4f}, "
              f"{gap_2_3:.4f}{warn}")

    # ── 5b. ALL retained PC eigenvalues per stratum + adjacent gaps ──
    # If any adjacent (PC_i, PC_{i+1}) gap is < 5% of PC_i, BLAS-level
    # eigendecomposition drift can swap the order across machines, which
    # in turn shifts nearest-neighbour distances in the retained PC space
    # and propagates into Hopkins. Below 1% is highly vulnerable.
    _section("5b. ALL retained PC eigenvalues + adjacent gaps per stratum")
    print("  flags: ! rel_gap < 5% (vulnerable), !! rel_gap < 1% (highly)")
    print()
    for cohort, sub in COHORT_DIRS.items():
        cdir = reduce_root if sub == "" else reduce_root / sub
        ev_csv = cdir / "pca_explained_variance.csv"
        npy = cdir / "pca_scores.npy"
        if not ev_csv.exists() or not npy.exists():
            print(f"  [{cohort}] missing inputs"); continue
        ev = pd.read_csv(ev_csv)
        val_col = next((c for c in ["eigenvalue", "explained_variance",
                                      "variance", "explained_variance_ratio"]
                        if c in ev.columns), None)
        if val_col is None:
            continue
        n_retain = int(np.load(npy).shape[1])
        evs = ev[val_col].sort_values(ascending=False).head(n_retain).tolist()
        n_flag_5 = 0; n_flag_1 = 0
        print(f"  [{cohort}] n_retain={n_retain} retained PC eigenvalues:")
        for i, e in enumerate(evs):
            if i + 1 < len(evs):
                gap = e - evs[i + 1]
                rel = gap / e if e > 0 else 0.0
                mark = "  "
                if rel < 0.01: mark = "!!"; n_flag_1 += 1
                elif rel < 0.05: mark = "! "; n_flag_5 += 1
                print(f"    PC{i+1:>2}={e:8.4f}   gap_to_PC{i+2}={gap:7.4f}   "
                      f"rel={rel*100:5.2f}%  {mark}")
            else:
                print(f"    PC{i+1:>2}={e:8.4f}")
        print(f"    summary: {n_flag_5} gap(s) < 5% (!), "
              f"{n_flag_1} gap(s) < 1% (!!)")
        print()

    # ── 6. Determinism: rerun stage 5 on Qr36d/2 and compare SHAs ─
    _section("6. INTRA-MACHINE determinism on Qr36d/2 (catches a real bug)")
    if args.skip_determinism:
        print("  skipped (--skip-determinism)")
    else:
        primary_npy = reduce_root / "stratified_Qr36d_2" / "pca_scores.npy"
        if not primary_npy.exists():
            print("  cannot run determinism check: primary stratified Qr36d/2 "
                  "outputs missing")
        else:
            primary_sha = _file_sha16(primary_npy)
            primary_arr = np.load(primary_npy)
            primary_byte_sha = hashlib.sha256(
                np.ascontiguousarray(primary_arr, dtype=np.float64).tobytes()
            ).hexdigest()[:16]
            print(f"  primary Qr36d/2 pca_scores.npy file sha = {primary_sha}")
            print(f"  primary Qr36d/2 pca_scores.npy float64-bytes sha = {primary_byte_sha}")

            # Run stage 5 to a temp output dir, then SHA again
            tmp_out = repo_root / "outputs_drift_check"
            print(f"  rerunning 06_reduce.py --kernel-filter 'Qr36d/2' "
                  f"with paths.outputs={tmp_out} ...")
            tmp_out.mkdir(parents=True, exist_ok=True)
            # Temporarily redirect outputs via a config override env var.
            # We exploit the standard --config flag by writing a tiny YAML
            # that inherits the real config but overrides paths.outputs.
            real_cfg = repo_root / "configs" / "default.yaml"
            override_yaml = tmp_out / "drift_check.yaml"
            override_yaml.write_text(
                real_cfg.read_text().replace(
                    "outputs:", f"outputs: {tmp_out.as_posix()}\n_original_outputs:"
                )
            )
            # Simpler & more robust: copy stage-3 features and stage-5 inputs
            # into the tmp tree (so 06_reduce.py finds them under the new
            # paths.outputs root), then run with --config override.
            for sub in ("03_features", "05_icc"):
                src = cfg.paths.outputs / sub
                dst = tmp_out / sub
                if dst.exists():
                    continue
                if src.exists():
                    shutil.copytree(src, dst)
            try:
                r = subprocess.run(
                    [sys.executable, "scripts/06_reduce.py",
                     "--config", str(override_yaml),
                     "--kernel-filter", "Qr36d/2",
                     "--n-jobs", "16"],
                    cwd=repo_root, capture_output=True, text=True,
                    timeout=240,
                )
                if r.returncode != 0:
                    print("  rerun FAILED:")
                    print(r.stdout[-2000:])
                    print(r.stderr[-2000:])
                else:
                    rerun_npy = tmp_out / "06_reduce" / "stratified_Qr36d_2" / "pca_scores.npy"
                    if not rerun_npy.exists():
                        print(f"  rerun did not produce {rerun_npy}")
                    else:
                        rerun_arr = np.load(rerun_npy)
                        rerun_byte_sha = hashlib.sha256(
                            np.ascontiguousarray(rerun_arr, dtype=np.float64).tobytes()
                        ).hexdigest()[:16]
                        print(f"  rerun Qr36d/2 pca_scores.npy float64-bytes sha = {rerun_byte_sha}")
                        if rerun_byte_sha == primary_byte_sha:
                            print("  VERDICT: DETERMINISTIC. Stage 5 produces "
                                  "identical bytes on identical inputs on this "
                                  "machine. The drift vs the original box is "
                                  "purely cross-machine BLAS / linker.")
                        else:
                            print("  VERDICT: NON-DETERMINISTIC ON THIS "
                                  "MACHINE. Stage 5 gives different bytes on "
                                  "identical inputs in successive runs. This "
                                  "is a real bug -- some RNG seed or iteration "
                                  "order is not pinned.")
                            # Quantify the divergence
                            diff = np.abs(primary_arr - rerun_arr).max()
                            print(f"  max |primary - rerun| = {diff:.6g}")
            finally:
                # leave tmp_out around for forensic inspection; the user
                # can rm -rf it after looking
                print(f"  (tmp outputs left at {tmp_out} for inspection; "
                      f"`rm -rf {tmp_out}` to clean up)")

    # ── Summary ──────────────────────────────────────────────────
    _section("Summary")
    print("Read each section above. The cheap signals (1, 2, 3) localise the")
    print("drift to a stratum-specific cause; section 4 (ComBat) and section")
    print("5 (eigenvalue gap) explain the mechanism if any of them shows an")
    print("anomaly. Section 6 is the bug-detector: if the SHAs match in")
    print("section 6, the rerun is intra-machine deterministic and the cross-")
    print("machine drift is purely BLAS / linker. If they differ, we have a")
    print("real determinism bug to track down.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
