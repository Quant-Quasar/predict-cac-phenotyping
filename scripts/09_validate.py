#!/usr/bin/env python
"""Stage 8 — validate (D029 + D030 + D031).

Orchestrator for the three stage-8 deliverables:

  D029  external GE-scanner holdout: project pids {19, 28, 76, 77} onto
        the frozen production spatial-only k=2 phenotype. Descriptive
        only (N=4). Writes external_holdout_report.csv +
        xml_roundtrip_holdout.csv.

  D030  leave-k-out cross-validation: 10-fold kernel-stratified;
        full per-fold refit; per-fold ARI vs full-cohort reference;
        runtime PASS threshold T = 5th-percentile of K=10% disagreement
        simulation. Writes leave_k_out_ari.csv (10 fold rows + SUMMARY).

  D031  cross-cohort ARI consolidation: pure re-export of
        outputs/07_analyse/cross_cohort_ari.csv with PASS columns.
        Writes cross_cohort_ari_consolidated.csv.

All outputs land in ``outputs/08_validate/``. The run header
``run_header_validate.json`` records seam SHAs + the computed LOO
threshold T + per-deliverable verdicts.

Usage::

    python scripts/09_validate.py
    python scripts/09_validate.py --skip-holdout --skip-loo
    python scripts/09_validate.py --n-jobs 16
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from predict.config import Config, load_config
from predict.io import load_patient_metadata
from predict.io.spacing import load_spacing_metadata
from predict.validate.cross_cohort_ari import (
    ARI_PASS_THRESHOLD,
    write as write_cross_cohort_ari,
)
from predict.validate.external_holdout import (
    SPATIAL_FEATURES_FOR_PROJECTION,
    run_external_holdout_validation,
)
from predict.validate.leave_k_out import (
    DISAGREEMENT_RATE_DEFAULT,
    N_SIM_DEFAULT,
    N_SPLITS_DEFAULT,
    PERCENTILE_DEFAULT,
    SEED_DEFAULT,
    attach_summary_row,
    run_leave_k_out,
)


# ─────────────────────── helpers ───────────────────────


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


def _save_json(path: Path, payload: dict | list) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8",
    )


def _load_script_module(path: Path, mod_name: str):
    """Import a numerically-named script under scripts/ as a module."""
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _discover_holdout_pids(cfg: Config) -> list[str]:
    """Read GE holdout pids from outputs/01_manifest/exclusions.csv.

    Falls back to the documented set if the audit is missing (a fresh
    pipeline must run stage 1 first; but stage 8 should not crash if
    the user is doing a partial rerun).
    """
    excl = cfg.paths.outputs / "01_manifest" / "exclusions.csv"
    if excl.exists():
        df = pd.read_csv(excl, dtype={"pid": str})
        ge = df[df["reason"] == "ge_scanner"]["pid"].astype(str).tolist()
        if ge:
            return sorted(ge, key=lambda s: int(s) if s.isdigit() else s)
    # Documented fallback from CLAUDE.md
    return ["19", "28", "76", "77"]


# ─────────────────────── D029: holdout pipeline ───────────────────────


def _preprocess_holdout(
    pids: list[str],
    cfg: Config,
    out_dir: Path,
    log: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run scripts/02_preprocess._process_one on each holdout pid.

    Returns ``(preprocess_report_df, xml_roundtrip_df)`` for the holdout.
    """
    preprocess_mod = _load_script_module(
        Path(__file__).resolve().parent / "02_preprocess.py",
        "scripts_02_preprocess",
    )
    data_raw_root = str(cfg.paths.data_raw)
    xml_dir = str(cfg.paths.data_xml)
    target_spacing = cfg.resample.target_spacing
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save the spacing.json so the stage 3 reader can pick it up.
    from predict.io.spacing import save_spacing_metadata
    save_spacing_metadata(out_dir / "spacing.json", target_spacing)

    rows: list[dict] = []
    trips_all: list[dict] = []
    for pid in pids:
        log.info("D029 preprocess: pid %s", pid)
        result = preprocess_mod._process_one(
            pid, data_raw_root, xml_dir, str(out_dir),
            target_spacing,
            cfg.hu.clip_min, cfg.hu.clip_max,
            cfg.hu.metal_artifact_threshold,
            cfg.resample.slice_mm / 2.0,
        )
        if result["status"] != "ok":
            log.error("preprocess failed for %s: %s", pid, result.get("error"))
            continue
        trips = result.pop("trips", [])
        rows.append({k: v for k, v in result.items() if k != "trips"})
        for t in trips:
            t["pid"] = pid
            trips_all.append(t)
    return pd.DataFrame(rows), pd.DataFrame(trips_all)


def _extract_holdout_features(
    pids: list[str],
    cfg: Config,
    holdout_preproc_dir: Path,
    quality_by_pid: dict[str, str],
    excluded_by_pid: dict[str, set],
    log: logging.Logger,
) -> pd.DataFrame:
    """Run scripts/03_features._process_one (sans per-vessel PyRadiomics)
    on each holdout pid. Returns a features dataframe."""
    features_mod = _load_script_module(
        Path(__file__).resolve().parent / "03_features.py",
        "scripts_03_features",
    )
    target_spacing = (
        load_spacing_metadata(holdout_preproc_dir / "spacing.json")
        if (holdout_preproc_dir / "spacing.json").exists()
        else cfg.resample.target_spacing
    )

    extractor_holder: dict = {}
    rows: list[dict] = []
    for pid in pids:
        log.info("D029 features: pid %s", pid)
        result = features_mod._process_one(
            pid, str(cfg.paths.data_raw.parent), str(holdout_preproc_dir),
            excluded_by_pid.get(pid, set()),
            target_spacing,
            quality_by_pid.get(pid, "ok"),
            False,                          # no per-vessel PyRadiomics
            extractor_holder,
        )
        if result["status"] != "ok":
            log.error("features failed for %s: %s",
                      pid, result.get("error"))
            continue
        rows.append(result["row"])
    return pd.DataFrame(rows)


def _xml_pass_by_pid_from_trips(trips_df: pd.DataFrame) -> dict[str, bool]:
    """D029.3: per-pid D002 PASS = all non-dirty ROIs pass Max-exact.

    Pids whose ROIs are ALL ``matched_via="dirty"`` have nothing to gate
    on; they PASS by default (consistent with stage 2's behaviour for the
    same condition).
    """
    if trips_df.empty:
        return {}
    out: dict[str, bool] = {}
    for pid, grp in trips_df.groupby("pid"):
        non_dirty = grp[grp["matched_via"] != "dirty"]
        if non_dirty.empty:
            out[str(pid)] = True
        else:
            out[str(pid)] = bool(non_dirty["passes"].astype(bool).all())
    return out


def _gated_feature_cols(cfg: Config) -> list[str]:
    gated_csv = cfg.paths.outputs / "05_icc" / "gated_features.csv"
    return pd.read_csv(gated_csv)["feature"].tolist()


def _full_cohort_eligible(cfg: Config) -> pd.DataFrame:
    features_csv = cfg.paths.outputs / "03_features" / "features.csv"
    df = pd.read_csv(features_csv, dtype={"pid": str})
    eligible = df[df["radiomics_status"] == "ok"].copy().reset_index(drop=True)
    return eligible


def run_d029_holdout(cfg: Config, out_dir: Path,
                     log: logging.Logger) -> dict:
    """Run the D029 external GE-holdout pipeline. Returns a summary dict."""
    holdout_pids = _discover_holdout_pids(cfg)
    log.info("D029 holdout pids: %s", holdout_pids)

    holdout_preproc_dir = out_dir / "holdout_preprocessed"
    pre_report, trips = _preprocess_holdout(
        holdout_pids, cfg, holdout_preproc_dir, log,
    )
    if trips.empty:
        log.warning("D029: no preprocess trips; holdout aborted")
        return {"verdict": "aborted", "reason": "no_preprocess"}

    # D002 XML round-trip audit on the holdout.
    trips.to_csv(out_dir / "xml_roundtrip_holdout.csv", index=False)
    pre_report.to_csv(out_dir / "holdout_preprocess_report.csv", index=False)
    xml_pass_by_pid = _xml_pass_by_pid_from_trips(trips)

    # quality + excluded ROIs for stage 3.
    quality_by_pid = dict(zip(
        pre_report["pid"].astype(str), pre_report["roundtrip_quality"]
    )) if not pre_report.empty else {}
    fails = trips[
        (~trips["passes"].astype(bool)) & (trips["matched_via"] != "dirty")
    ]
    excluded_by_pid: dict[str, set] = {}
    for _, r in fails.iterrows():
        excluded_by_pid.setdefault(str(r["pid"]), set()).add(
            (int(r["image_index"]), int(r["roi_idx_in_slice"]))
        )

    holdout_features = _extract_holdout_features(
        holdout_pids, cfg, holdout_preproc_dir,
        quality_by_pid, excluded_by_pid, log,
    )
    holdout_features.to_csv(
        out_dir / "holdout_features.csv", index=False,
    )
    if holdout_features.empty:
        log.warning("D029: no holdout features extracted")
        return {"verdict": "aborted", "reason": "no_features"}

    # Fit frozen pipeline on production cohort, project + predict holdout.
    full_cohort = _full_cohort_eligible(cfg)
    gated_cols = _gated_feature_cols(cfg)
    report, _frozen = run_external_holdout_validation(
        full_cohort, holdout_features, gated_cols,
        xml_roundtrip_pass_by_pid=xml_pass_by_pid,
    )
    report.to_csv(out_dir / "external_holdout_report.csv", index=False)
    log.info("D029 wrote %d holdout rows", len(report))

    return {
        "verdict": "ok",
        "n_holdout": int(len(report)),
        "holdout_pids": holdout_pids,
        "xml_pass_by_pid": xml_pass_by_pid,
        "phenotype_counts": (
            report["predicted_phenotype"].value_counts().to_dict()
        ),
    }


# ─────────────────────── D030: leave-k-out ───────────────────────


def _load_reference_labels(cfg: Config) -> pd.Series:
    """Load full-cohort spatial-k=2 labels from stage 6 seam."""
    p = (cfg.paths.outputs / "06_reduce"
         / "cluster_labels_spatial_k2.csv")
    df = pd.read_csv(p, dtype={"pid": str})
    # The seam writes a single label column; pick the GMM one.
    label_col = next(
        (c for c in df.columns if c.startswith("spatial_k2") or "gmm" in c.lower()),
        None,
    )
    if label_col is None:
        candidates = [c for c in df.columns if c != "pid"]
        if len(candidates) != 1:
            raise ValueError(
                f"D030: cannot infer label column in {p}; got {candidates}"
            )
        label_col = candidates[0]
    return pd.Series(
        df[label_col].astype(int).to_numpy(),
        index=df["pid"].astype(str),
        name="spatial_k2_label",
    )


def run_d030_loo(
    cfg: Config, out_dir: Path, log: logging.Logger,
    n_splits: int, n_simulations: int, seed: int,
) -> dict:
    """Run the D030 LOO CV. Returns a summary dict including threshold T."""
    full_cohort = _full_cohort_eligible(cfg)
    gated_cols = _gated_feature_cols(cfg)
    reference = _load_reference_labels(cfg)

    log.info("D030 LOO: N_cohort=%d, n_splits=%d, n_sim=%d",
             len(full_cohort), n_splits, n_simulations)
    per_fold, summary = run_leave_k_out(
        full_cohort, gated_cols, reference,
        n_splits=n_splits, seed=seed,
        n_simulations=n_simulations,
        disagreement_rate=DISAGREEMENT_RATE_DEFAULT,
    )
    out_df = attach_summary_row(per_fold, summary)
    out_df.to_csv(out_dir / "leave_k_out_ari.csv", index=False)
    log.info("D030 wrote %d fold rows + 1 summary", len(per_fold))
    log.info("D030 median ARI = %.3f, median T = %.3f, PASS = %s",
             summary["median_ari"], summary["median_T_fold"],
             summary["overall_pass"])
    return summary


# ─────────────────────── D031: ARI consolidation ───────────────────────


def run_d031_consolidation(
    cfg: Config, out_dir: Path, log: logging.Logger,
) -> dict:
    src = cfg.paths.outputs / "07_analyse" / "cross_cohort_ari.csv"
    if not src.exists():
        log.warning("D031: %s missing; skipping consolidation", src)
        return {"verdict": "skipped", "reason": "stage7_missing"}
    out_path = out_dir / "cross_cohort_ari_consolidated.csv"
    df = write_cross_cohort_ari(src, out_path)
    log.info("D031 wrote %d rows", len(df))
    return {
        "verdict": "ok",
        "n_rows": int(len(df)),
        "pass_threshold": ARI_PASS_THRESHOLD,
        "n_passing": int(df["pass_verdict"].sum()),
    }


# ─────────────────────── main ───────────────────────


def build_run_header(repo_root: Path, cfg: Config,
                     args: argparse.Namespace) -> dict:
    out = cfg.paths.outputs
    info: dict = {
        "stage": "validate",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_hash(repo_root),
        "config_yaml_sha": _file_sha(repo_root / "configs" / "default.yaml"),
        "features_csv_sha": _file_sha(out / "03_features" / "features.csv"),
        "gated_features_csv_sha": _file_sha(
            out / "05_icc" / "gated_features.csv"
        ),
        "cluster_labels_spatial_k2_sha": _file_sha(
            out / "06_reduce" / "cluster_labels_spatial_k2.csv"
        ),
        "cross_cohort_ari_sha": _file_sha(
            out / "07_analyse" / "cross_cohort_ari.csv"
        ),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--skip-holdout", action="store_true",
                        help="Skip D029 external holdout.")
    parser.add_argument("--skip-loo", action="store_true",
                        help="Skip D030 leave-k-out (slow ~5 min).")
    parser.add_argument("--skip-ari-consolidation", action="store_true",
                        help="Skip D031 cross-cohort ARI consolidation.")
    parser.add_argument("--n-splits", type=int, default=N_SPLITS_DEFAULT)
    parser.add_argument("--n-simulations", type=int, default=N_SIM_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("validate")

    cfg = load_config(args.config)
    out_dir = cfg.paths.outputs / "08_validate"
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_root = cfg.paths.outputs.parent
    t_overall = time.perf_counter()

    summary: dict = {
        "stage": "validate",
        "config": dict(vars(args)),
        "thresholds": {
            "ari_pass_threshold_d031": ARI_PASS_THRESHOLD,
            "disagreement_rate_K_d030": DISAGREEMENT_RATE_DEFAULT,
            "percentile_d030": PERCENTILE_DEFAULT,
            "n_simulations_d030": int(args.n_simulations),
        },
    }

    if not args.skip_holdout:
        t0 = time.perf_counter()
        summary["d029"] = run_d029_holdout(cfg, out_dir, log)
        summary["d029"]["runtime_sec"] = round(time.perf_counter() - t0, 2)
    else:
        log.info("D029 skipped by flag")
        summary["d029"] = {"verdict": "skipped"}

    if not args.skip_loo:
        t0 = time.perf_counter()
        summary["d030"] = run_d030_loo(
            cfg, out_dir, log,
            n_splits=args.n_splits,
            n_simulations=args.n_simulations,
            seed=args.seed,
        )
        summary["d030"]["runtime_sec"] = round(time.perf_counter() - t0, 2)
    else:
        log.info("D030 skipped by flag")
        summary["d030"] = {"verdict": "skipped"}

    if not args.skip_ari_consolidation:
        t0 = time.perf_counter()
        summary["d031"] = run_d031_consolidation(cfg, out_dir, log)
        summary["d031"]["runtime_sec"] = round(time.perf_counter() - t0, 2)
    else:
        log.info("D031 skipped by flag")
        summary["d031"] = {"verdict": "skipped"}

    header = build_run_header(repo_root, cfg, args)
    header["summary"] = summary
    header["total_runtime_sec"] = round(time.perf_counter() - t_overall, 2)
    _save_json(out_dir / "run_header_validate.json", header)
    log.info("stage 8 complete in %.1f s; outputs in %s",
             header["total_runtime_sec"], out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
