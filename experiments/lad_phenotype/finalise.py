#!/usr/bin/env python
"""Step 5 of the LAD-phenotype experiment.

Reads the full-cohort + per-stratum outputs and:
  - Bundles cross-stratum replication into one JSON.
  - Writes a human-readable report.txt summary.

Pre-registered replication PASS criterion (per plan.md):
  Each verdict must hold (PASS) in BOTH strata independently, OR each
  stratum's effect-size estimate must be within +/-0.10 of the
  full-cohort estimate.

Run AFTER:
  python experiments/lad_phenotype/run.py
  python experiments/lad_phenotype/run.py --stratum Qr36d/2
  python experiments/lad_phenotype/run.py --stratum I30f/3
  python experiments/lad_phenotype/matched.py
  python experiments/lad_phenotype/matched.py --stratum Qr36d/2
  python experiments/lad_phenotype/matched.py --stratum I30f/3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from predict.config import load_config

REPLICATION_DELTA_TOL = 0.10
STRATA = ("Qr36d_2", "I30f_3")

_log = logging.getLogger("lad_phenotype.finalise")


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8",
    )


def _gather(out_root: Path) -> dict:
    base = {
        "full": {
            "signature": _load_json(out_root / "lad_cluster_signature.json"),
            "axial": _load_json(out_root / "axial_summary.json"),
            "carrier": _load_json(out_root / "carrier_summary.json"),
            "match_diagnostics": _load_json(out_root / "matched_diagnostics.json"),
            "match_comparison": _load_json(out_root / "matched_comparison_summary.json"),
        },
    }
    for s in STRATA:
        sdir = out_root / f"stratified_{s}"
        base[s] = {
            "signature": _load_json(sdir / "lad_cluster_signature.json"),
            "axial": _load_json(sdir / "axial_summary.json"),
            "carrier": _load_json(sdir / "carrier_summary.json"),
            "match_diagnostics": _load_json(sdir / "matched_diagnostics.json"),
            "match_comparison": _load_json(sdir / "matched_comparison_summary.json"),
        }
    return base


def _compare_replication(full: dict, strata: dict) -> dict:
    """Cross-stratum replication verdict per plan.md."""
    out: dict = {}

    # Signature: same cluster ids in each stratum.
    if full.get("signature"):
        full_ids = set(full["signature"].get("lad_dominant_cluster_ids", []))
        per_stratum = {}
        for s in STRATA:
            sig = strata.get(s, {}).get("signature")
            if sig:
                per_stratum[s] = sig.get("lad_dominant_cluster_ids", [])
            else:
                per_stratum[s] = None
        out["signature_replication"] = {
            "full_ids": sorted(full_ids),
            "per_stratum": per_stratum,
            "replicates": all(
                per_stratum[s] is not None
                and set(per_stratum[s]) == full_ids
                for s in STRATA
            ),
        }

    # Axial verdict.
    if full.get("axial"):
        full_pass = bool(full["axial"].get("passes"))
        full_diff = float(full["axial"].get("median_diff") or 0.0)
        per_stratum = {}
        for s in STRATA:
            ax = strata.get(s, {}).get("axial")
            if ax is None:
                per_stratum[s] = None
                continue
            per_stratum[s] = {
                "passes": bool(ax.get("passes")),
                "median_diff": float(ax.get("median_diff") or 0.0),
                "within_tol": abs(
                    float(ax.get("median_diff") or 0.0) - full_diff
                ) <= REPLICATION_DELTA_TOL,
            }
        out["axial_replication"] = {
            "full_passes": full_pass,
            "full_median_diff": full_diff,
            "per_stratum": per_stratum,
            "replicates": all(
                per_stratum[s] is not None and (
                    per_stratum[s]["passes"]
                    or per_stratum[s]["within_tol"]
                )
                for s in STRATA
            ),
        }

    # Matched comparison verdict.
    if full.get("match_comparison"):
        full_overall = bool(full["match_comparison"].get("overall_pass"))
        per_stratum = {}
        for s in STRATA:
            mc = strata.get(s, {}).get("match_comparison")
            if mc is None:
                per_stratum[s] = None
                continue
            per_stratum[s] = {
                "overall_pass": bool(mc.get("overall_pass")),
                "interpretation": mc.get("interpretation"),
            }
        out["matched_replication"] = {
            "full_overall_pass": full_overall,
            "per_stratum": per_stratum,
            "replicates": all(
                per_stratum[s] is not None
                and per_stratum[s]["overall_pass"]
                for s in STRATA
            ),
        }

    out["overall_replicates"] = all(
        sub.get("replicates", False)
        for key, sub in out.items()
        if isinstance(sub, dict) and "replicates" in sub
    )
    return out


def _report_text(bundle: dict, replication: dict) -> str:
    lines: list[str] = []
    lines.append("LAD-phenotype experiment - final report")
    lines.append("=" * 60)
    lines.append("")
    full = bundle.get("full", {})

    sig = full.get("signature")
    if sig is None:
        lines.append("STEP 1: signature discovery NOT RUN")
    else:
        ids = sig.get("lad_dominant_cluster_ids", [])
        lines.append(f"STEP 1: LAD-dominant cluster ids (pre-reg) = {ids}")
        if not ids:
            lines.append("  Termination: no cluster matches pre-registered "
                          "signature. Experiment terminates per plan.md.")
            return "\n".join(lines)

    ax = full.get("axial")
    if ax:
        lines.append("")
        lines.append("STEP 2: within-LAD axial localisation")
        lines.append(f"  rel-z median (LAD-cluster lesions) = "
                     f"{ax.get('median_in_cluster')}")
        lines.append(f"  rel-z median (other LAD lesions)   = "
                     f"{ax.get('median_other')}")
        lines.append(f"  median diff = {ax.get('median_diff')}")
        lines.append(f"  Mann-Whitney p (one-sided, less) = "
                     f"{ax.get('mw_p_one_sided_less')}")
        lines.append(f"  PASSES: {ax.get('passes')}")

    cs = full.get("carrier")
    if cs:
        lines.append("")
        lines.append("STEP 3: carrier patient signature")
        lines.append(f"  directional confirmations: "
                     f"{cs.get('n_passing_directional')}"
                     f"/{cs.get('n_total_directional')}")
        lines.append(f"  overall PASSES: {cs.get('overall_passes')}")

    md = full.get("match_diagnostics")
    if md:
        lines.append("")
        lines.append("STEP 4: burden-propensity match diagnostics")
        lines.append(f"  verdict: {md.get('verdict')}")
        lines.append(f"  yield = {md.get('match_yield')}")
        lines.append(f"  SMD pre = {md.get('smd_pre')}, "
                     f"post = {md.get('smd_post')}")

    mc = full.get("match_comparison")
    if mc:
        lines.append("")
        lines.append("STEP 4 (cont.): post-match tests")
        lines.append(f"  {mc.get('n_passing')}/{mc.get('n_total')} pass")
        lines.append(f"  interpretation: {mc.get('interpretation')}")

    lines.append("")
    lines.append("STEP 5: cross-stratum replication")
    lines.append(f"  overall replicates: {replication.get('overall_replicates')}")
    for key, sub in replication.items():
        if isinstance(sub, dict) and "replicates" in sub:
            lines.append(f"  {key}: replicates={sub['replicates']}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load_config(args.config)
    out_root = cfg.paths.outputs / "exploratory" / "lad_phenotype"
    if not out_root.exists():
        _log.error("missing %s; nothing to finalise.", out_root)
        return 2

    bundle = _gather(out_root)
    strata = {s: bundle.get(s, {}) for s in STRATA}
    replication = _compare_replication(bundle.get("full", {}), strata)
    _save_json(out_root / "cross_stratum_replication.json", replication)

    report = _report_text(bundle, replication)
    (out_root / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
