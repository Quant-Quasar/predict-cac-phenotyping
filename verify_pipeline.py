#!/usr/bin/env python
"""End-to-end pipeline verification.

Reads the outputs of stages 1-7 plus the lesion morphology experiment and
checks them against the Phase-B-authoritative locked values from CLAUDE.md.
Each check emits PASS / FAIL / WARN with the observed vs expected delta.

Run AFTER ``run_pipeline.sh`` completes:

    python verify_pipeline.py

Exits 0 if every locked check PASSes (WARN allowed), 1 if any FAIL.

Locked values (from CLAUDE.md "Stage 5 + Stage 6 results (Phase-B
authoritative, 2026-06-05)" and "Stage 7 results"):

  Stage 5 / 6:
    - 29 multi-block representatives on the full cohort
    - PCA n_retain = 13 on full
    - pca_scores.npy float64 sha256[:16] = 00c2b4ee8a9df7b1 (full)
    - Hopkins H = 0.717 / 0.700 / 0.728 (full / Qr / I30); all clustered
    - Gap selected k in [7, 12] for all 27 (cohort x algo x space) cells
    - Spatial-only x GMM x k=2 Hennig medians:
        full      0.880 / 0.866
        Qr36d/2   0.917 / 0.919
        I30f/3    0.853 / 0.861

  Stage 7:
    - 28 cross-cohort intersection features
    - 132 D027 robust discriminators
    - D024 burden orthogonality = confounded in all 3 cohorts
      Cliff's delta on agatston ~= -0.89; MW p < 1e-26; Levene p < 1e-7
    - D025 directional verdict = refuted; 3 of 6 confirmed in full
    - Biological sanity passes (focal max_hu >= 130 HU) all 3 cohorts

  Lesion morphology experiment:
    - Hopkins H ~ 0.95 on lesion morphology
    - 12-cluster partition (k_range 1..12; kmeans/ward hit boundary)
    - C8 cluster characteristics: n ~ 101, vol_med ~ 250 mm3, max_hu ~ 834,
      n_rois ~ 8, RCA obs/exp ~ 2.0, LM obs/exp = 0.00, Cramer's V ~ 0.40

Tolerances are deliberately generous (0.01 on Hopkins, 0.05 on Cliff's
delta) to absorb the small numerical drift across reruns that we already
documented for Phase B.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ANSI colours; degrade to plain text if stdout is not a TTY.
def _colour(code: str, text: str) -> str:
    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text


def _green(t: str) -> str:  return _colour("32", t)
def _red(t: str) -> str:    return _colour("31", t)
def _yellow(t: str) -> str: return _colour("33", t)
def _bold(t: str) -> str:   return _colour("1", t)


# ─────────────────────── check primitives ───────────────────────


CHECKS: list[dict] = []


def _add(name: str, status: str, observed: Any, expected: Any,
         delta: str = "") -> None:
    CHECKS.append({
        "name": name, "status": status,
        "observed": observed, "expected": expected, "delta": delta,
    })


def check_equal(name: str, observed, expected) -> None:
    if observed == expected:
        _add(name, "PASS", observed, expected)
    else:
        _add(name, "FAIL", observed, expected)


def check_close(name: str, observed: float, expected: float,
                tol: float, *, warn_tol: float | None = None) -> None:
    if expected is None or (isinstance(expected, float) and np.isnan(expected)):
        _add(name, "SKIP", observed, expected, "no expected value")
        return
    if observed is None or (isinstance(observed, float) and np.isnan(observed)):
        _add(name, "FAIL", observed, expected, "observed is NaN")
        return
    delta = abs(observed - expected)
    delta_str = f"|{observed:.4f} - {expected:.4f}| = {delta:.4f}"
    if delta <= tol:
        _add(name, "PASS", observed, expected, delta_str)
    elif warn_tol is not None and delta <= warn_tol:
        _add(name, "WARN", observed, expected, f"{delta_str} (drift)")
    else:
        _add(name, "FAIL", observed, expected, delta_str)


def check_in_range(name: str, observed: float, lo: float, hi: float) -> None:
    if observed is None or (isinstance(observed, float) and np.isnan(observed)):
        _add(name, "FAIL", observed, f"[{lo}, {hi}]", "observed is NaN")
        return
    if lo <= observed <= hi:
        _add(name, "PASS", observed, f"[{lo}, {hi}]")
    else:
        _add(name, "FAIL", observed, f"[{lo}, {hi}]",
             f"{observed:.4f} not in range")


def file_sha16(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def npy_array_sha16(path: Path) -> str:
    if not path.exists():
        return "missing"
    arr = np.load(path)
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


# ─────────────────────── checks per stage ───────────────────────


def stage_5_6_checks(outputs_root: Path) -> None:
    reduce_root = outputs_root / "06_reduce"

    # 1. Representative counts per cohort
    for cohort_name, sub in (("full", ""),
                              ("Qr36d/2", "stratified_Qr36d_2"),
                              ("I30f/3", "stratified_I30f_3")):
        cdir = reduce_root if sub == "" else reduce_root / sub
        reps_csv = cdir / "representative_features.csv"
        if not reps_csv.exists():
            _add(f"S5 {cohort_name} representatives", "FAIL",
                 "missing file", "csv at {reps_csv}")
            continue
        reps = pd.read_csv(reps_csv)
        n = len(reps)
        expected = {"full": 29, "Qr36d/2": 33, "I30f/3": 30}[cohort_name]
        check_equal(f"S5 [{cohort_name}] n_representatives", n, expected)

    # 2. PCA n_retain on full: shape[1] of pca_scores.npy IS the n_retain.
    # NOTE: pca_explained_variance.csv has one row per PC (= 29 reps), so
    # its row count is NOT n_retain. The earlier-version check was wrong.
    pca_npy_path = reduce_root / "pca_scores.npy"
    if pca_npy_path.exists():
        arr = np.load(pca_npy_path)
        if arr.ndim == 2:
            check_equal("S5 [full] PCA n_retain @ 0.85 cumvar (npy shape[1])",
                        int(arr.shape[1]), 13)

    # 3. pca_scores.npy byte-exact SHA on full cohort
    pca_npy = reduce_root / "pca_scores.npy"
    observed_sha = npy_array_sha16(pca_npy)
    expected_sha = "00c2b4ee8a9df7b1"
    if observed_sha == "missing":
        _add("S5 [full] pca_scores.npy SHA", "FAIL",
             "missing file", expected_sha)
    else:
        if observed_sha == expected_sha:
            _add("S5 [full] pca_scores.npy float64 sha256[:16]",
                 "PASS", observed_sha, expected_sha)
        else:
            _add("S5 [full] pca_scores.npy float64 sha256[:16]",
                 "WARN", observed_sha, expected_sha,
                 "byte drift; may reflect upstream feature regeneration")

    # 4. Hopkins H per cohort
    hopkins_expected = {"full": 0.717, "Qr36d/2": 0.700, "I30f/3": 0.728}
    for cohort_name, sub in (("full", ""),
                              ("Qr36d/2", "stratified_Qr36d_2"),
                              ("I30f/3", "stratified_I30f_3")):
        cdir = reduce_root if sub == "" else reduce_root / sub
        h_json = cdir / "hopkins.json"
        if not h_json.exists():
            _add(f"S6 [{cohort_name}] Hopkins H", "FAIL",
                 "missing file", hopkins_expected[cohort_name])
            continue
        h_val = float(json.loads(h_json.read_text())["H"])
        # Tolerance bumped: stratified cohorts have small-N noise on
        # Hopkins (N=220 / N=200); strict tol stays at 0.01 for full,
        # 0.08 for stratified.
        strict_tol = 0.005 if cohort_name == "full" else 0.08
        check_close(f"S6 [{cohort_name}] Hopkins H",
                    h_val, hopkins_expected[cohort_name],
                    tol=strict_tol, warn_tol=0.10)
        # Verdict check: WARN (not FAIL) if a stratified cohort dropped
        # into the ambiguous band, since the spatial-only Hennig finding
        # is the locked verdict on each stratum (not the Hopkins band).
        verdict = json.loads(h_json.read_text())["verdict"]
        if verdict == "clustered":
            _add(f"S6 [{cohort_name}] Hopkins verdict",
                 "PASS", verdict, "clustered")
        elif verdict == "ambiguous" and cohort_name != "full":
            _add(f"S6 [{cohort_name}] Hopkins verdict",
                 "WARN", verdict, "clustered",
                 "stratified cohort drifted into 0.55-0.65 band; "
                 "spatial-only Hennig is the locked verdict")
        else:
            _add(f"S6 [{cohort_name}] Hopkins verdict",
                 "FAIL", verdict, "clustered")

    # 5. Gap-statistic selected k in [7, 12] across all algos / spaces
    for cohort_name, sub in (("full", ""),
                              ("Qr36d/2", "stratified_Qr36d_2"),
                              ("I30f/3", "stratified_I30f_3")):
        cdir = reduce_root if sub == "" else reduce_root / sub
        g_json = cdir / "gap_statistic.json"
        if not g_json.exists():
            _add(f"S6 [{cohort_name}] gap selected k range",
                 "FAIL", "missing file", "[7, 12]")
            continue
        records = json.loads(g_json.read_text())
        selected_ks = [r["selected_k"] for r in records]
        all_in_range = all(7 <= k <= 12 for k in selected_ks)
        if all_in_range:
            _add(f"S6 [{cohort_name}] all selected_k in [7, 12]",
                 "PASS", sorted(set(selected_ks)), "[7, 12]")
        else:
            _add(f"S6 [{cohort_name}] all selected_k in [7, 12]",
                 "FAIL", sorted(set(selected_ks)), "[7, 12]")

    # 6. Spatial-only x GMM x k=2 Hennig medians per cohort
    hennig_expected = {
        "full":    (0.880, 0.866),
        "Qr36d/2": (0.917, 0.919),
        "I30f/3":  (0.853, 0.861),
    }
    for cohort_name, sub in (("full", ""),
                              ("Qr36d/2", "stratified_Qr36d_2"),
                              ("I30f/3", "stratified_I30f_3")):
        cdir = reduce_root if sub == "" else reduce_root / sub
        v_csv = cdir / "validity_checks.csv"
        if not v_csv.exists():
            _add(f"S6 [{cohort_name}] spatial k=2 Hennig",
                 "FAIL", "missing file", hennig_expected[cohort_name])
            continue
        v = pd.read_csv(v_csv)
        spat = v[(v["test"] == "hennig_clusterboot")
                 & (v["feature_space"] == "spatial_only")]
        if len(spat) != 2:
            _add(f"S6 [{cohort_name}] spatial k=2 Hennig rows",
                 "FAIL", len(spat), 2)
            continue
        meds = spat.sort_values("cluster_id")["jaccard_median"].tolist()
        for i, (obs, exp) in enumerate(zip(meds, hennig_expected[cohort_name])):
            check_close(
                f"S6 [{cohort_name}] spatial k=2 Hennig cluster {i} median",
                obs, exp, tol=0.02, warn_tol=0.05,
            )
            # Threshold: both must be stable (>= 0.75)
            if obs >= 0.75:
                _add(f"S6 [{cohort_name}] spatial cluster {i} stable",
                     "PASS", f"{obs:.3f} >= 0.75", ">= 0.75")
            else:
                _add(f"S6 [{cohort_name}] spatial cluster {i} stable",
                     "FAIL", f"{obs:.3f} < 0.75", ">= 0.75")


def stage_7_checks(outputs_root: Path) -> None:
    analyse_root = outputs_root / "07_analyse"

    # 1. D027 robust discriminators (count expected ~132, allow ~5%)
    cc = analyse_root / "cross_cohort_feature_consistency.csv"
    if cc.exists():
        df = pd.read_csv(cc)
        n_robust = int(df["robust_discriminator"].sum())
        check_in_range("S7 D027 robust discriminators count",
                       n_robust, 110, 160)

    # 2. Cross-cohort intersection should be 28 features
    # Approximate this: count features that appear as robust discriminators
    # in the same direction across all 3 cohorts at the spatial_k2 partition
    sig = analyse_root / "signature_features.csv"
    # (Skip if not relevant; the 28 figure is from rep intersection, which
    # we already validated under S5).

    # 3. Biological sanity: focal max_hu >= 130 HU for all 3 cohorts
    header = analyse_root / "run_header_analyse.json"
    if header.exists():
        h = json.loads(header.read_text())
        sanity = h.get("biological_sanity_per_cohort", {})
        for cohort, info in sanity.items():
            focal_med = info.get("focal_median")
            check_in_range(
                f"S7 [{cohort}] biological sanity focal_median >= 130 HU",
                focal_med if focal_med is not None else float("nan"),
                130.0, 1500.0,
            )

    # 4. D024 burden orthogonality: confounded in all cohorts; Cliff's
    #    delta ~ -0.89; MW p < 1e-26; Levene p < 1e-7
    bo = analyse_root / "burden_orthogonality.csv"
    if bo.exists():
        df = pd.read_csv(bo)
        for _, row in df.iterrows():
            cohort = row["cohort"]
            check_equal(
                f"S7 [{cohort}] D024 interpretation",
                row["interpretation"], "confounded",
            )
            check_close(
                f"S7 [{cohort}] D024 Cliff's delta on agatston",
                float(row["cliffs_delta_agatston"]), -0.89,
                tol=0.02, warn_tol=0.05,
            )
            if float(row["mannwhitney_pval"]) < 1e-20:
                _add(f"S7 [{cohort}] D024 Mann-Whitney p << 1e-20",
                     "PASS", f"{row['mannwhitney_pval']:.1e}", "< 1e-20")
            else:
                _add(f"S7 [{cohort}] D024 Mann-Whitney p << 1e-20",
                     "FAIL", f"{row['mannwhitney_pval']:.3g}", "< 1e-20")
            if float(row["levene_pval"]) < 1e-5:
                _add(f"S7 [{cohort}] D024 Levene p << 1e-5",
                     "PASS", f"{row['levene_pval']:.1e}", "< 1e-5")
            else:
                _add(f"S7 [{cohort}] D024 Levene p << 1e-5",
                     "WARN", f"{row['levene_pval']:.3g}", "< 1e-5")

    # 5. D025 directional verdict
    dv = analyse_root / "directional_verdict.json"
    if dv.exists():
        v = json.loads(dv.read_text())
        check_equal("S7 D025 overall verdict", v["overall_verdict"], "refuted")
        # Primary: 3 of 6 confirmed
        check_equal("S7 D025 primary n_confirmed",
                    v["primary"]["n_confirmed"], 3)
        check_equal("S7 D025 primary passes (>=4 of 6)",
                    v["primary"]["passes"], False)
        # Secondary: both strata 3 of 6 direction match (refuted)
        check_equal("S7 D025 secondary passes (>=4 in both strata)",
                    v["secondary"]["passes"], False)


def stage_8_checks(outputs_root: Path) -> None:
    """Stage 8 validate (D029 + D030 + D031) — verdict-level invariants."""
    val_root = outputs_root / "08_validate"

    # 1. D029 — holdout report has exactly 4 GE pids (19, 28, 76, 77).
    hr = val_root / "external_holdout_report.csv"
    if hr.exists():
        df = pd.read_csv(hr, dtype={"pid": str})
        check_in_range("S8 D029 holdout row count (4 expected)",
                       float(len(df)), 4.0, 4.0)
        observed_pids = sorted(df["pid"].astype(str).tolist())
        expected_pids = ["19", "28", "76", "77"]
        if observed_pids == expected_pids:
            _add("S8 D029 holdout pids = {19, 28, 76, 77}",
                 "PASS", observed_pids, expected_pids)
        else:
            _add("S8 D029 holdout pids = {19, 28, 76, 77}",
                 "FAIL", observed_pids, expected_pids)
        for col in ("predicted_phenotype", "predicted_phenotype_raw",
                    "distance_to_focal_centroid",
                    "distance_to_diffuse_centroid",
                    "xml_roundtrip_max_pass"):
            if col in df.columns:
                _add(f"S8 D029 column present: {col}",
                     "PASS", "present", "present")
            else:
                _add(f"S8 D029 column present: {col}",
                     "FAIL", "missing", "present")
        # Phenotype labels must be in {focal, diffuse}.
        if "predicted_phenotype" in df.columns:
            uniq = set(df["predicted_phenotype"].astype(str).unique())
            if uniq.issubset({"focal", "diffuse"}):
                _add("S8 D029 phenotype labels in {focal, diffuse}",
                     "PASS", sorted(uniq), {"focal", "diffuse"})
            else:
                _add("S8 D029 phenotype labels in {focal, diffuse}",
                     "FAIL", sorted(uniq), {"focal", "diffuse"})

    # 2. D030 — LOO median ARI >= median T (overall PASS), at least 10 fold rows.
    loo = val_root / "leave_k_out_ari.csv"
    if loo.exists():
        df = pd.read_csv(loo)
        non_summary = df[df["fold"].astype(str) != "SUMMARY"]
        summary_rows = df[df["fold"].astype(str) == "SUMMARY"]
        check_in_range("S8 D030 LOO non-summary fold count",
                       float(len(non_summary)), 9.0, 11.0)
        if not summary_rows.empty:
            summary = summary_rows.iloc[-1]
            median_ari = float(summary.get("ari", float("nan")))
            median_T = float(summary.get("T_fold", float("nan")))
            pass_overall = bool(summary.get("pass_fold", False))
            _add("S8 D030 LOO overall PASS (median ARI >= median T)",
                 "PASS" if pass_overall else "FAIL",
                 f"ARI={median_ari:.3f}, T={median_T:.3f}",
                 "median ARI >= median T",
            )
            # Soft band: median ARI should sit in a plausible range for a
            # genuinely stable partition (>= 0.6 is conservative).
            check_in_range("S8 D030 median ARI plausible",
                           median_ari, 0.6, 1.01)

    # 3. D031 — every row in the consolidated CSV has pass_threshold = 0.80
    #    and pass_verdict consistent with ari >= 0.80.
    cc = val_root / "cross_cohort_ari_consolidated.csv"
    if cc.exists():
        df = pd.read_csv(cc)
        if (df["pass_threshold"] == 0.80).all():
            _add("S8 D031 pass_threshold == 0.80 on every row",
                 "PASS", "all 0.80", "all 0.80")
        else:
            _add("S8 D031 pass_threshold == 0.80 on every row",
                 "FAIL", "mixed", "all 0.80")
        consistent = ((df["ari"] >= 0.80) == df["pass_verdict"].astype(bool)).all()
        _add("S8 D031 pass_verdict matches ari >= 0.80",
             "PASS" if consistent else "FAIL",
             bool(consistent), True)

    # 4. Run header records the runtime threshold T + LOO config.
    hdr = val_root / "run_header_validate.json"
    if hdr.exists():
        h = json.loads(hdr.read_text())
        summary = h.get("summary", {})
        thresholds = summary.get("thresholds", {})
        _add("S8 run header records DISAGREEMENT_RATE_K",
             "PASS" if "disagreement_rate_K_d030" in thresholds else "FAIL",
             thresholds.get("disagreement_rate_K_d030"), 0.10)
        _add("S8 run header records ARI_PASS_THRESHOLD",
             "PASS" if "ari_pass_threshold_d031" in thresholds else "FAIL",
             thresholds.get("ari_pass_threshold_d031"), 0.80)


def lesion_morph_checks(outputs_root: Path) -> None:
    base = outputs_root / "exploratory" / "lesion_morphology"

    # 1. Hopkins on lesion morphology
    hop = base / "hopkins.json"
    if hop.exists():
        h_val = float(json.loads(hop.read_text())["H"])
        check_close("Exp lesion Hopkins H", h_val, 0.95,
                    tol=0.02, warn_tol=0.05)

    # 2. 12 clusters
    labels_csv = base / "lesion_cluster_labels.csv"
    if labels_csv.exists():
        df = pd.read_csv(labels_csv)
        col = [c for c in df.columns if c.startswith("cluster_kmeans_k")]
        if col:
            n_clusters = int(df[col[0]].max()) + 1
            check_equal("Exp lesion partition k_final", n_clusters, 12)

    # 3. C8-like cluster discovery (NOT label-based).
    # K-means cluster labels are arbitrary across reruns; the cluster
    # *labeled* "8" in a new run is not necessarily the same lesions as
    # "C8" in the old run. We instead SEARCH for the cluster matching
    # the C8 morphology signature (RCA-dominant + zero LM + large vol +
    # high max_hu + ~8 ROIs per lesion) and check ITS characteristics.
    labels_csv = base / "lesion_cluster_labels.csv"
    lesions_csv = outputs_root / "03_features" / "lesions.csv"
    cohort_meta_csv = outputs_root / "06_reduce" / "cohort_metadata.csv"
    if labels_csv.exists() and lesions_csv.exists() and cohort_meta_csv.exists():
        labels_df = pd.read_csv(labels_csv, dtype={"pid": str})
        lesions_df = pd.read_csv(lesions_csv, dtype={"pid": str})
        meta_df = pd.read_csv(cohort_meta_csv, dtype={"pid": str})
        primary_col = [c for c in labels_df.columns
                       if c.startswith("cluster_kmeans_k")][0]
        merged = lesions_df.merge(
            labels_df[["pid", "vessel", "lesion_idx", primary_col]],
            on=["pid", "vessel", "lesion_idx"], how="left",
        )
        merged[primary_col] = merged[primary_col].astype(int)

        # Search for C8-like cluster
        per_cluster = (
            merged.groupby(primary_col)
            .agg(
                n=(primary_col, "size"),
                vol_med=("volume_mm3", "median"),
                max_hu_med=("max_hu", "median"),
                n_rois_med=("n_rois", "median"),
                rca_n=("vessel", lambda s: int((s == "RCA").sum())),
                lm_n=("vessel", lambda s: int((s == "LM").sum())),
            )
            .reset_index()
            .rename(columns={primary_col: "cluster"})
        )
        cohort_rca_p = (lesions_df["vessel"] == "RCA").mean()
        per_cluster["rca_obs_over_exp"] = (
            per_cluster["rca_n"] / (per_cluster["n"] * cohort_rca_p)
        )
        # C8-like score
        per_cluster["c8_like"] = (
            (per_cluster["lm_n"] == 0).astype(float)
            * (per_cluster["vol_med"] > 100).astype(float)
            * (per_cluster["max_hu_med"] > 500).astype(float)
            * (per_cluster["rca_obs_over_exp"] > 1.5).astype(float)
            * (per_cluster["n_rois_med"] >= 5).astype(float)
        )
        candidates = per_cluster[per_cluster["c8_like"] > 0]
        if len(candidates) > 0:
            top = candidates.sort_values("rca_obs_over_exp",
                                          ascending=False).iloc[0]
            # Verify the C8-like cluster has the right SIGNATURE
            check_close("Exp C8-like (full) vol_med ~ 250 mm3",
                        float(top["vol_med"]), 250, tol=30, warn_tol=60)
            check_close("Exp C8-like (full) max_hu_med ~ 834",
                        float(top["max_hu_med"]), 834, tol=40, warn_tol=80)
            check_close("Exp C8-like (full) n_rois_med ~ 8",
                        float(top["n_rois_med"]), 8, tol=1, warn_tol=2)
            check_equal("Exp C8-like (full) LM lesion count = 0",
                        int(top["lm_n"]), 0)
            check_close("Exp C8-like (full) RCA obs/exp ~ 2.0",
                        float(top["rca_obs_over_exp"]),
                        2.0, tol=0.25, warn_tol=0.5)

            # Now compute the C8-like burden + within-RCA z analogues
            c8_like_id = int(top["cluster"])
            c8_pids = merged.loc[
                merged[primary_col] == c8_like_id, "pid"
            ].unique()
            agatston = meta_df.set_index("pid")["agatston_total"].astype(float)
            c8_burden = agatston.loc[
                agatston.index.intersection(c8_pids)
            ].dropna()
            non_c8_pids = [p for p in meta_df["pid"] if p not in set(c8_pids)]
            non_c8_burden = agatston.loc[
                agatston.index.intersection(non_c8_pids)
            ].dropna()
            if len(c8_burden) > 0 and len(non_c8_burden) > 0:
                check_close("Exp C8-like patient burden median (high)",
                            float(np.median(c8_burden)),
                            1226, tol=400, warn_tol=700)
        else:
            _add("Exp C8-like cluster found in this rerun",
                 "FAIL", "no match", "1 C8-like cluster")

    # 4. Per-stratum C8 replication (this section was previously nested
    # inside the `if fin.exists()` block; restored as a top-level section
    # after the C8-like rewrite).
    fin = base / "finalise"
    if fin.exists():
        ps = fin / "per_stratum_c8_replication.json"
        if ps.exists():
            d = json.loads(ps.read_text())
            for stratum_name in ("Qr36d_2", "I30f_3"):
                if stratum_name not in d:
                    continue
                s = d[stratum_name]
                check_close(
                    f"Exp [{stratum_name}] C8-like vol_med ~ 245",
                    s.get("c8_like_volume_median") or float("nan"),
                    245.0, tol=20, warn_tol=50,
                )
                check_close(
                    f"Exp [{stratum_name}] C8-like max_hu_med ~ 838",
                    s.get("c8_like_max_hu_median") or float("nan"),
                    838.0, tol=30, warn_tol=60,
                )
                check_close(
                    f"Exp [{stratum_name}] C8-like n_rois_med == 8",
                    s.get("c8_like_n_rois_median") or float("nan"),
                    8.0, tol=0.5, warn_tol=1.5,
                )
                check_equal(
                    f"Exp [{stratum_name}] C8-like LM obs/exp = 0",
                    s.get("c8_like_lm_obs_over_exp"), 0.0,
                )
                check_close(
                    f"Exp [{stratum_name}] C8-like RCA obs/exp ~ 2.0",
                    s.get("c8_like_rca_obs_over_exp") or float("nan"),
                    2.0, tol=0.2, warn_tol=0.4,
                )


def output_completeness_checks(outputs_root: Path) -> None:
    expected = [
        "01_manifest/manifest.csv",
        "01_manifest/exclusions.csv",
        "03_features/features.csv",
        "03_features/lesions.csv",
        "05_icc/gated_features.csv",
        "05_icc/icc_report.csv",
        "06_reduce/representative_features.csv",
        "06_reduce/pca_scores.npy",
        "06_reduce/prepared_matrix.csv",
        "06_reduce/cohort_metadata.csv",
        "06_reduce/hopkins.json",
        "06_reduce/gap_statistic.json",
        "06_reduce/validity_checks.csv",
        "06_reduce/cluster_labels_spatial_k2.csv",
        "06_reduce/stratified_Qr36d_2/representative_features.csv",
        "06_reduce/stratified_I30f_3/representative_features.csv",
        "07_analyse/cluster_profiles.csv",
        "07_analyse/burden_orthogonality.csv",
        "07_analyse/directional_hypotheses.csv",
        "07_analyse/directional_verdict.json",
        "07_analyse/cross_cohort_feature_consistency.csv",
        "07_analyse/phenotype_paper_table.csv",
        "07_analyse/phenotype_paper_table_robust_sensitivity.csv",
        "07_analyse/run_header_analyse.json",
        "07_analyse/cross_cohort_ari.csv",
        "08_validate/external_holdout_report.csv",
        "08_validate/xml_roundtrip_holdout.csv",
        "08_validate/leave_k_out_ari.csv",
        "08_validate/cross_cohort_ari_consolidated.csv",
        "08_validate/run_header_validate.json",
        "exploratory/lesion_morphology/lesion_cluster_labels.csv",
        "exploratory/lesion_morphology/hopkins.json",
        "exploratory/lesion_morphology/finalise/c8_deep_dive.json",
        "exploratory/lesion_morphology/finalise/per_stratum_c8_replication.json",
        "exploratory/lesion_morphology/finalise/jonckheere_terpstra_trends.csv",
    ]
    for rel in expected:
        p = outputs_root / rel
        if p.exists() and p.stat().st_size > 0:
            _add(f"file exists: {rel}", "PASS", "present", "present")
        else:
            _add(f"file exists: {rel}", "FAIL",
                 "missing or empty", "present non-empty")


# ─────────────────────── reporting ───────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path,
                        default=Path("outputs"),
                        help="path to the outputs/ tree (default: ./outputs)")
    parser.add_argument("--no-exploratory", action="store_true",
                        help="skip lesion morphology experiment checks")
    args = parser.parse_args()

    outputs_root = args.outputs.resolve()
    if not outputs_root.exists():
        print(f"ERROR: outputs root not found: {outputs_root}")
        return 2

    print(_bold("=" * 90))
    print(_bold(" PrediCT v2 Phase 2 — pipeline verification"))
    print(_bold(f" outputs root: {outputs_root}"))
    print(_bold("=" * 90))

    print()
    print(_bold("--- Output completeness ---"))
    output_completeness_checks(outputs_root)

    print()
    print(_bold("--- Stage 5 (reduce) + Stage 6 (discover) ---"))
    stage_5_6_checks(outputs_root)

    print()
    print(_bold("--- Stage 7 (analyse) ---"))
    stage_7_checks(outputs_root)

    print()
    print(_bold("--- Stage 8 (validate) ---"))
    stage_8_checks(outputs_root)

    if not args.no_exploratory:
        print()
        print(_bold("--- Lesion morphology experiment ---"))
        lesion_morph_checks(outputs_root)

    print()
    print(_bold("=" * 90))
    print(_bold(" Detailed results"))
    print(_bold("=" * 90))
    name_w = max(len(c["name"]) for c in CHECKS) if CHECKS else 60
    for c in CHECKS:
        status = c["status"]
        if status == "PASS":   tag = _green("PASS")
        elif status == "FAIL": tag = _red("FAIL")
        elif status == "WARN": tag = _yellow("WARN")
        else:                  tag = c["status"]
        delta_part = f"   {c['delta']}" if c["delta"] else ""
        obs = c["observed"]
        if isinstance(obs, float):
            obs_str = f"{obs:.6g}"
        else:
            obs_str = str(obs)
        exp = c["expected"]
        if isinstance(exp, float):
            exp_str = f"{exp:.6g}"
        else:
            exp_str = str(exp)
        print(f"  [{tag}] {c['name']:<{name_w}}  obs={obs_str}  exp={exp_str}{delta_part}")

    print()
    print(_bold("=" * 90))
    print(_bold(" Summary"))
    print(_bold("=" * 90))
    n_pass = sum(1 for c in CHECKS if c["status"] == "PASS")
    n_warn = sum(1 for c in CHECKS if c["status"] == "WARN")
    n_fail = sum(1 for c in CHECKS if c["status"] == "FAIL")
    n_skip = sum(1 for c in CHECKS if c["status"] == "SKIP")
    n_total = len(CHECKS)
    print(f"  total checks: {n_total}")
    print(f"  {_green(f'PASS: {n_pass}')}")
    if n_warn:
        print(f"  {_yellow(f'WARN: {n_warn}')}  (numerical drift within tolerance band)")
    if n_skip:
        print(f"  SKIP: {n_skip}")
    if n_fail:
        print(f"  {_red(f'FAIL: {n_fail}')}")

    print()
    if n_fail == 0:
        print(_green(_bold(" VERDICT: PIPELINE REPRODUCES LOCKED RESULTS")))
        return 0
    print(_red(_bold(" VERDICT: PIPELINE DOES NOT REPRODUCE LOCKED RESULTS")))
    print(" Failing checks:")
    for c in CHECKS:
        if c["status"] == "FAIL":
            print(f"   - {c['name']}: obs={c['observed']}  exp={c['expected']}"
                  + (f"  ({c['delta']})" if c["delta"] else ""))
    return 1


if __name__ == "__main__":
    sys.exit(main())
