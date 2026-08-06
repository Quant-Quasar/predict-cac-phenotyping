#!/usr/bin/env python
"""Stage 5 orchestrator (D019, D020, D022).

End-to-end pipeline for the dimensionality-reduction stage:

  1. Load stage-3 features.csv + stage-4 icc_report.csv
  2. Filter to eligible cohort (radiomics_status == "ok"; D015 N=422)
  3. Restrict to the gated 88-feature set from stage 4 (gated_features.csv)
  4. Apply ComBat kernel filter (drop singleton-kernel patients; D019);
     optional `--kernel-filter` for the kernel-stratified sensitivity rerun
  5. Run matrix preparation (D019): D017 drops + D018 binarisation +
     derived features + variance filter + ComBat + Yeo-Johnson + z-score
  6. Spearman r^2 redundancy clustering: multi-block by default (D022),
     single-matrix sensitivity via --block-mode single (D020)
  7. PCA on the representative feature subset, sign-normalised (D020)
  8. Persist everything under outputs/06_reduce/ (or stratified subdir)

Cluster discovery, gap statistic, Monti consensus, forced-k characterisation,
and Hennig stability checks (D021) moved to ``scripts/07_discover.py`` on
2026-06-05 (Phase B). This script writes the seam files that 07_discover.py
consumes:

  * pca_scores.csv            -> X_full (PC matrix)
  * prepared_matrix.csv       -> post-D019 features (for spatial PCA)
  * cohort_metadata.csv       -> RAW kernel, low_burden_flag, category,
                                 agatston_total (for burden residualisation
                                 and forced-k crosstabs)
  * representative_features.csv, multi_block_assignments.csv,
    pca_loadings.csv, pca_top_loadings.csv, pca_explained_variance.csv,
    pc_agatston_correlation.csv, combat_audit.csv, matrix_prep_log.json,
    run_header.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from predict.config import Config, load_config
from predict.discover.clusterability import hopkins_statistic
from predict.features.feature_schema import feature_names
from predict.reduce.pca import (
    explained_variance_table,
    fit_pca,
    pc_external_correlation,
    top_loadings_table,
)
from predict.reduce.prepare_matrix import run_matrix_prep
from predict.reduce.redundancy import (
    DEFAULT_BLOCKS,
    build_icc_lookup,
    run_multi_block_redundancy_clustering,
    run_redundancy_clustering,
)


DERIVED_FEATURE_CANDIDATES: tuple[str, ...] = (
    "high_density_fraction", "vessel_burden_gini",
)

# Raw metadata columns to persist for downstream stages (07_discover.py
# needs raw agatston_total for burden residualisation; the z-scored feature
# version is destroyed by D019).
METADATA_COLS_FOR_DOWNSTREAM: tuple[str, ...] = (
    "pid", "kernel", "scanner_model", "mask_voxels", "low_burden_flag",
    "roundtrip_quality", "category", "radiomics_status", "radiomics_reason",
    "agatston_total",
)


# ─────────────────────── reproducibility breadcrumbs ───────────────────────


def _git_hash(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root), capture_output=True, text=True, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _file_sha(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def build_run_header(
    repo_root: Path, cfg: Config, args: argparse.Namespace,
) -> dict:
    info: dict = {
        "stage": "reduce",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_hash(repo_root),
        "config_yaml_sha": _file_sha(repo_root / "configs" / "default.yaml"),
        "python_version": sys.version.split()[0],
        "args": vars(args),
    }
    for mod in ("numpy", "pandas", "scipy", "sklearn"):
        try:
            m = __import__(mod)
            info[f"{mod}_version"] = getattr(m, "__version__", "unknown")
        except Exception:
            info[f"{mod}_version"] = "n/a"
    return info


# ─────────────────────── helpers ───────────────────────


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _load_inputs(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (eligible features dataframe, ICC report, gated feature list)."""
    features_csv = cfg.paths.outputs / "03_features" / "features.csv"
    icc_csv = cfg.paths.outputs / "05_icc" / "icc_report.csv"
    gated_csv = cfg.paths.outputs / "05_icc" / "gated_features.csv"

    if not features_csv.exists():
        raise FileNotFoundError(f"missing {features_csv}")
    if not icc_csv.exists():
        raise FileNotFoundError(f"missing {icc_csv}")
    if not gated_csv.exists():
        raise FileNotFoundError(f"missing {gated_csv}")

    features = pd.read_csv(features_csv, dtype={"pid": str})
    icc = pd.read_csv(icc_csv)
    gated = pd.read_csv(gated_csv)["feature"].tolist()

    eligible = features[features["radiomics_status"] == "ok"].copy().reset_index(drop=True)
    return eligible, icc, gated


# ─────────────────────── main ───────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--kernel-filter", type=str, default=None,
                        help="Restrict the cohort to a single kernel for the "
                             "D021 kernel-stratified sensitivity rerun. "
                             "Outputs land in outputs/06_reduce/stratified_<kernel>/.")
    parser.add_argument("--n-jobs", type=int, default=16,
                        help="Reserved for future parallel sections; reduce "
                             "stage itself is fast and single-threaded.")
    parser.add_argument("--block-mode", choices=("multi", "single"),
                        default="multi",
                        help="D022: multi-block redundancy clustering is the "
                             "primary (default). 'single' runs the D020 "
                             "single-matrix variant as a sensitivity check.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log = logging.getLogger("reduce")

    cfg = load_config(args.config)
    base_out = cfg.paths.outputs / "06_reduce"
    if args.kernel_filter is not None:
        kernel_slug = args.kernel_filter.replace("/", "_")
        out_dir = _ensure_dir(base_out / f"stratified_{kernel_slug}")
    else:
        out_dir = _ensure_dir(base_out)
    repo_root = cfg.paths.outputs.parent

    # ── Run header ────────────────────────────────────────────────
    header = build_run_header(repo_root, cfg, args)
    _save_json(out_dir / "run_header.json", header)
    log.info("run_header: git=%s, config_sha=%s",
             header["git_commit"], header["config_yaml_sha"])

    # ── 1 / 2. Load inputs + eligible cohort ──────────────────────
    eligible, icc_report, gated_features = _load_inputs(cfg)
    log.info("eligible cohort: N=%d (D015)", len(eligible))
    log.info("gated features from stage 4: %d", len(gated_features))

    # Restrict columns to gated + metadata.
    metadata_cols = [c for c in
                     ("pid", "kernel", "scanner_model", "mask_voxels",
                      "low_burden_flag", "roundtrip_quality", "category",
                      "radiomics_status", "radiomics_reason")
                     if c in eligible.columns]
    keep_cols = metadata_cols + [c for c in gated_features if c in eligible.columns]
    eligible = eligible[keep_cols].copy()

    # Stash RAW agatston_total BEFORE D019 z-scores it. 07_discover.py needs
    # the raw value for burden residualisation.
    if "agatston_total" in eligible.columns:
        raw_agatston = eligible[["pid", "agatston_total"]].copy()
    else:
        raw_agatston = None
        log.warning("agatston_total not in eligible columns; downstream "
                    "burden residualisation will be unavailable")

    # ── ComBat kernel filter (D019) ──────────────────────────────
    # Patients with rare kernels (singleton-kernel groups in COCA: B35f and
    # I36f/3 with one patient each) are dropped because ComBat cannot estimate
    # within-batch variance on a single sample. The --kernel-filter flag
    # further restricts to one kernel for the D021 stratified rerun.
    combat_groups = set(cfg.raw["reduce"].get("combat_kernel_groups", []))
    if args.kernel_filter is not None:
        target = {args.kernel_filter}
        log.info("kernel-stratified rerun: restricting cohort to kernel %s",
                 args.kernel_filter)
    elif combat_groups:
        target = combat_groups
    else:
        target = None

    if target is not None:
        kernel_counts = eligible["kernel"].value_counts().to_dict()
        log.info("kernel distribution in eligible cohort: %s", kernel_counts)
        before = len(eligible)
        eligible = eligible[eligible["kernel"].isin(target)].copy().reset_index(drop=True)
        dropped = before - len(eligible)
        if dropped > 0:
            log.info(
                "dropped %d patient(s) with kernel outside %s",
                dropped, sorted(target),
            )
        log.info("post-kernel-filter cohort: N=%d", len(eligible))
        if len(eligible) < 2:
            raise SystemExit(
                f"after --kernel-filter {args.kernel_filter}, only {len(eligible)} "
                f"patients remain; cannot run stage 5"
            )

    # ── 3. D019 matrix prep ───────────────────────────────────────
    log.info("D019 matrix preparation...")
    prep_df, prep_features, prep_log = run_matrix_prep(
        eligible, [c for c in gated_features if c in eligible.columns],
        variance_threshold=cfg.reduce.variance_threshold,
        yj_skew_threshold=cfg.raw["reduce"].get("yeo_johnson_skewness_fallback", 1.0),
        combat_max_post_r2=0.02,
        add_derived=True,
    )
    _save_json(out_dir / "matrix_prep_log.json", prep_log.to_dict())
    if prep_log.combat_audit:
        pd.DataFrame(prep_log.combat_audit).to_csv(
            out_dir / "combat_audit.csv", index=False,
        )
    log.info("prep: %d -> %d features (D019 complete)",
             prep_log.n_features_in, prep_log.n_features_out)

    # ── 4. D020 / D022 redundancy clustering ──────────────────────
    log.info("D020/D022 redundancy clustering (mode=%s)...", args.block_mode)
    icc_lookup = build_icc_lookup(
        icc_report, derived_names=DERIVED_FEATURE_CANDIDATES,
    )
    canonical_set = set(feature_names()) | set(DERIVED_FEATURE_CANDIDATES)

    if args.block_mode == "multi":
        # D022 multi-block (primary).
        multi = run_multi_block_redundancy_clustering(
            prep_df, prep_features, icc_lookup, canonical_set,
            blocks=DEFAULT_BLOCKS,
            primary_method=cfg.reduce.r2_linkage,
            sensitivity_methods=(),
            min_gap=cfg.reduce.r2_elbow_min_gap,
            fallback_distance=cfg.reduce.r2_fallback_distance,
        )
        representatives = multi.representatives
        multi.assignments_dataframe().to_csv(
            out_dir / "multi_block_assignments.csv", index=False,
        )
        per_block_summary = {
            name: {
                "n_input_features": len(result.feature_order),
                "n_clusters": result.n_clusters(),
                "cut_threshold": result.cut_threshold,
                "representatives": list(result.representatives),
            }
            for name, result in multi.per_block.items()
        }
        _save_json(out_dir / "multi_block_summary.json", {
            "block_partition": {
                b.name: {
                    "prefixes": list(b.prefixes),
                    "exact_names": list(b.exact_names),
                }
                for b in DEFAULT_BLOCKS
            },
            "per_block": per_block_summary,
            "unassigned_features": list(multi.unassigned),
            "total_representatives": len(representatives),
        })
        pd.DataFrame({"feature": representatives}).to_csv(
            out_dir / "representative_features.csv", index=False,
        )
        log.info("multi-block redundancy: %d features -> %d representatives "
                 "across %d non-empty blocks",
                 len(prep_features), len(representatives),
                 len(multi.per_block))
        for name, info in per_block_summary.items():
            log.info("  block[%s]: %d -> %d", name,
                     info["n_input_features"], info["n_clusters"])
    else:
        # D020 single-matrix (sensitivity).
        redundancy = run_redundancy_clustering(
            prep_df, prep_features, icc_lookup, canonical_set,
            primary_method=cfg.reduce.r2_linkage,
            sensitivity_methods=("ward", "complete"),
            min_gap=cfg.reduce.r2_elbow_min_gap,
            fallback_distance=cfg.reduce.r2_fallback_distance,
        )
        representatives = redundancy.representatives
        redundancy.cluster_assignments.to_csv(
            out_dir / "redundancy_clusters.csv", index=False,
        )
        pd.DataFrame({"feature": representatives}).to_csv(
            out_dir / "representative_features.csv", index=False,
        )
        _save_json(out_dir / "redundancy_sensitivity.json", {
            "primary": {
                "method": cfg.reduce.r2_linkage,
                "n_clusters": redundancy.n_clusters(),
                "cut_threshold": redundancy.cut_threshold,
                "representatives": redundancy.representatives,
            },
            "sensitivity_per_method": redundancy.sensitivity_per_method,
        })
        log.info("single-matrix redundancy: %d features -> %d representatives",
                 len(prep_features), len(representatives))

    # ── 5. D020 PCA on representatives ────────────────────────────
    log.info("D020 PCA...")
    pca = fit_pca(prep_df, representatives,
                  cumvar_threshold=cfg.reduce.pca_cumvar,
                  random_state=args.random_state)
    log.info("PCA: n_retain=%d at %.0f%% cumvar",
             pca.n_retain, 100 * cfg.reduce.pca_cumvar)

    # ── Diagnostic: in-memory Hopkins on pca.scores ────────────────
    # If this matches 07_discover.py's reported H exactly, the NPY/CSV seam
    # is byte-exact. If it differs from Phase A's locked value, the drift
    # is upstream (D019 or PCA), not the seam.
    diag_H = hopkins_statistic(
        pca.scores,
        sample_size=max(2, int(cfg.raw["hopkins"]["sample_frac"] * pca.scores.shape[0])),
        random_state=args.random_state,
    )
    log.info("DIAG: in-memory Hopkins on pca.scores = %.6f (N=%d, n_retain=%d)",
             diag_H, pca.scores.shape[0], pca.n_retain)

    # Audit hash of pca.scores in-memory bytes (helps confirm npy seam
    # roundtrip is lossless).
    pca_scores_sha = hashlib.sha256(
        np.ascontiguousarray(pca.scores, dtype=np.float64).tobytes()
    ).hexdigest()[:16]
    log.info("DIAG: pca.scores float64-bytes sha256[:16] = %s", pca_scores_sha)

    explained_variance_table(pca).to_csv(
        out_dir / "pca_explained_variance.csv", index=False,
    )
    top_loadings_table(pca).to_csv(
        out_dir / "pca_top_loadings.csv", index=False,
    )
    pd.DataFrame(
        pca.components,
        index=[f"PC{i + 1}" for i in range(pca.components.shape[0])],
        columns=pca.feature_names,
    ).to_csv(out_dir / "pca_loadings.csv")
    pd.DataFrame(
        pca.scores,
        columns=[f"PC{i + 1}" for i in range(pca.n_retain)],
        index=pca.pid_order,
    ).rename_axis("pid").to_csv(out_dir / "pca_scores.csv")
    # Byte-exact seam for 07_discover.py. CSV roundtrips lose ~12-digit
    # float precision which is enough to shift Hopkins by ~0.03 on the
    # full cohort. NPY preserves float64 exactly so the downstream
    # discover stage is reproducible against the in-memory pca.scores.
    np.save(out_dir / "pca_scores.npy", pca.scores)
    pd.Series(pca.pid_order, name="pid").to_csv(
        out_dir / "pca_scores_pid_order.csv", index=False,
    )

    # PC vs Agatston external sanity check.
    if raw_agatston is not None:
        agatston_aligned = (raw_agatston.set_index("pid")
                            .loc[pca.pid_order, "agatston_total"])
        pc_external_correlation(pca, agatston_aligned, name="agatston_total").to_csv(
            out_dir / "pc_agatston_correlation.csv", index=False,
        )

    # ── 6. Persist seam files for 07_discover.py ──────────────────
    # prepared_matrix.csv: post-D019 prep_df (z-scored features + metadata).
    # 07_discover.py reads this for the spatial-only PCA reconstruction.
    prep_df.to_csv(out_dir / "prepared_matrix.csv", index=False)

    # cohort_metadata.csv: RAW values 07_discover.py needs (agatston_total
    # before z-scoring; kernel / low_burden_flag / category for crosstabs).
    # Rebuild from features.csv to get raw agatston_total.
    raw_features_csv = cfg.paths.outputs / "03_features" / "features.csv"
    raw_features = pd.read_csv(raw_features_csv, dtype={"pid": str})
    meta_cols_present = [c for c in METADATA_COLS_FOR_DOWNSTREAM
                         if c in raw_features.columns]
    cohort_meta = (raw_features[meta_cols_present]
                   .set_index("pid")
                   .loc[prep_df["pid"].tolist()]
                   .reset_index())
    cohort_meta.to_csv(out_dir / "cohort_metadata.csv", index=False)
    log.info("seam files written: prepared_matrix.csv (%d rows x %d cols), "
             "cohort_metadata.csv (%d rows x %d cols)",
             len(prep_df), prep_df.shape[1],
             len(cohort_meta), cohort_meta.shape[1])

    log.info("stage 5 (reduce) complete. outputs in %s", out_dir)
    log.info("next: python scripts/07_discover.py --cohort-dir %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
