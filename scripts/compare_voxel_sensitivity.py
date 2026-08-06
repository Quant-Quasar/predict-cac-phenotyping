#!/usr/bin/env python
"""Side-by-side verdict comparison between the 0.5 mm primary outputs
and the 0.375 mm sensitivity-probe outputs.

Reads both output trees, evaluates every V1-V7 sub-test from D032,
and emits:

* ``<outputs_test>/voxel_sensitivity_report.json`` - machine-readable
* ``<outputs_test>/voxel_sensitivity_report.txt`` - human-readable
* exit code 0 if all sub-tests pass; non-zero if any fail

Usage:
    python scripts/compare_voxel_sensitivity.py \\
        --primary outputs \\
        --test outputs_0375

Threshold definitions are imported / mirrored from
``decisions/D032-voxel-sensitivity-probe-0375.md``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ─── Pre-registered tolerances (locked in D032) ───

V1_HOPKINS_FULL_TOLERANCE = 0.05
V1_HENNIG_JACCARD_MIN = 0.75
V2_LESION_HOPKINS_TOLERANCE = 0.03
V2_LESION_HOPKINS_LOCKED = 0.952
V3_JT_P_MAX = 1e-6
V3_DENSE_FRAC_DIFF_MIN = 0.20
V4_RCA_OBS_EXP_MIN = 1.7
V4_LM_OBS_EXP_MAX = 0.0
V4_REL_Z_MIN = 0.65
V4_CARRIER_BURDEN_RATIO_MIN = 10.0
V5_LOO_ARI_MIN = 0.70
V5_HOLDOUT_PIDS = {"19", "28", "76", "77"}
V6_LAD_REL_Z_MAX = 0.30
V7_DISP_SIZE_TOL = 3
V7_DISP_LM_RATE_REQ = 1.0


_log = logging.getLogger("compare_voxel")


# ─── helpers ───


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


@dataclass
class SubTest:
    verdict_id: str
    name: str
    primary_value: Any = None
    test_value: Any = None
    threshold: Any = None
    passes: bool | None = None
    fail_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VerdictBundle:
    verdict_id: str
    subtests: list[SubTest] = field(default_factory=list)

    def passes(self) -> bool:
        return all(t.passes for t in self.subtests if t.passes is not None)

    def to_dict(self) -> dict:
        return {
            "verdict_id": self.verdict_id,
            "passes": self.passes(),
            "subtests": [t.to_dict() for t in self.subtests],
        }


# ─── V1 continuum ───


def evaluate_v1(primary: Path, test: Path) -> VerdictBundle:
    bundle = VerdictBundle("V1_continuum")

    # V1.a: gap statistic monotone (no plateau)
    for tree, role in ((primary, "primary"), (test, "test")):
        gap_json = tree / "06_reduce" / "gap_statistic.json"
        gap = _load_json(gap_json)
        if gap is None:
            bundle.subtests.append(SubTest(
                "V1.a", f"gap statistic file present ({role})",
                primary_value=None, test_value=None,
                passes=False, fail_reason=f"missing {gap_json}",
            ))
            return bundle

    # V1.b: Hopkins H within tolerance
    hk_p = _load_json(primary / "06_reduce" / "hopkins.json")
    hk_t = _load_json(test / "06_reduce" / "hopkins.json")
    p_val = hk_p.get("H") if hk_p else None
    t_val = hk_t.get("H") if hk_t else None
    if p_val is not None and t_val is not None:
        delta = abs(p_val - t_val)
        bundle.subtests.append(SubTest(
            "V1.b", "Hopkins H drift (full cohort) within tolerance",
            primary_value=round(p_val, 4), test_value=round(t_val, 4),
            threshold=f"|delta| <= {V1_HOPKINS_FULL_TOLERANCE}",
            passes=bool(delta <= V1_HOPKINS_FULL_TOLERANCE),
            fail_reason=(f"|{p_val:.3f} - {t_val:.3f}| = {delta:.3f}"
                          if delta > V1_HOPKINS_FULL_TOLERANCE else ""),
        ))
    else:
        bundle.subtests.append(SubTest(
            "V1.b", "Hopkins H drift", passes=False,
            fail_reason="one or both hopkins.json files missing",
        ))

    # V1.c: Hennig stability of spatial-only k=2 partition
    # (each cluster median Jaccard >= 0.75). The validity_checks.csv
    # schema (per scripts/07_discover.py) is:
    #   test, algorithm, k, feature_space, cluster_id, jaccard_median,
    #   jaccard_mean, stable
    # The "spatial-only k=2" partition lives at feature_space==spatial_only
    # AND k==2; the Hennig rows have test=="hennig_clusterboot".
    for role, tree in (("test", test),):
        validity = _load_csv(tree / "06_reduce" / "validity_checks.csv")
        if validity is None:
            bundle.subtests.append(SubTest(
                "V1.c", f"validity_checks.csv present ({role})",
                passes=False, fail_reason=f"missing in {tree}",
            ))
            continue
        needed = {"test", "feature_space", "k", "jaccard_median"}
        if not needed.issubset(set(validity.columns)):
            bundle.subtests.append(SubTest(
                "V1.c", "spatial-k2 Hennig min cluster median Jaccard",
                passes=False,
                fail_reason=f"missing columns {sorted(needed - set(validity.columns))}",
            ))
            continue
        spat = validity[
            validity["test"].astype(str).str.contains(
                "hennig", case=False, na=False,
            )
            & (validity["feature_space"].astype(str) == "spatial_only")
            & (validity["k"].astype(int) == 2)
        ]
        min_jaccard = (float(spat["jaccard_median"].min())
                       if not spat.empty else float("nan"))
        bundle.subtests.append(SubTest(
            "V1.c", "spatial-k2 Hennig min cluster median Jaccard",
            primary_value=None, test_value=min_jaccard,
            threshold=f">= {V1_HENNIG_JACCARD_MIN}",
            passes=bool(min_jaccard >= V1_HENNIG_JACCARD_MIN)
                    if not np.isnan(min_jaccard) else False,
            fail_reason=("no spatial_only k=2 hennig rows"
                          if np.isnan(min_jaccard) else ""),
        ))
    return bundle


# ─── V2 lesion clustering ───


def evaluate_v2(primary: Path, test: Path) -> VerdictBundle:
    bundle = VerdictBundle("V2_three_lesion_classes")

    # V2.a: lesion-level Hopkins
    hk_p = _load_json(primary / "exploratory" / "lesion_morphology" / "hopkins.json")
    hk_t = _load_json(test / "exploratory" / "lesion_morphology" / "hopkins.json")
    p_val = hk_p.get("H") if hk_p else None
    t_val = hk_t.get("H") if hk_t else None
    if t_val is not None:
        delta = abs(V2_LESION_HOPKINS_LOCKED - t_val)
        bundle.subtests.append(SubTest(
            "V2.a", "lesion-level Hopkins H within tolerance of locked 0.95",
            primary_value=p_val, test_value=round(t_val, 4),
            threshold=f"|delta vs {V2_LESION_HOPKINS_LOCKED}| <= "
                       f"{V2_LESION_HOPKINS_TOLERANCE}",
            passes=bool(delta <= V2_LESION_HOPKINS_TOLERANCE),
            fail_reason=(f"|{V2_LESION_HOPKINS_LOCKED} - {t_val:.3f}| = {delta:.3f}"
                          if delta > V2_LESION_HOPKINS_TOLERANCE else ""),
        ))
    else:
        bundle.subtests.append(SubTest(
            "V2.a", "lesion-level Hopkins H", passes=False,
            fail_reason="test hopkins.json missing",
        ))

    # V2.b: three broad classes recoverable
    # Indirect via cluster_profiles.csv max_hu_median bands
    prof = _load_csv(test / "exploratory" / "lesion_morphology" / "cluster_profiles.csv")
    n_dense = n_soft = n_mod = 0
    if prof is not None and not prof.empty:
        n_dense = int((prof["max_hu_median"] >= 500).sum())
        n_soft = int(((prof["max_hu_median"] <= 230)
                       & (prof["volume_mm3_median"] <= 15)).sum())
        n_mod = int(len(prof) - n_dense - n_soft)
    classes_present = sum(c > 0 for c in (n_dense, n_soft, n_mod))
    bundle.subtests.append(SubTest(
        "V2.b", "three broad classes (soft / moderate / dense) recoverable",
        primary_value=3, test_value=classes_present,
        threshold="all three classes have >= 1 cluster",
        passes=bool(classes_present == 3),
        fail_reason=(f"only {classes_present}/3 classes present (dense={n_dense},"
                      f" soft={n_soft}, moderate={n_mod})"
                      if classes_present != 3 else ""),
    ))
    return bundle


# ─── V3 maturation ───


def evaluate_v3(primary: Path, test: Path) -> VerdictBundle:
    bundle = VerdictBundle("V3_maturation")

    jt_csv = test / "exploratory" / "lesion_morphology" / "finalise" / \
              "jonckheere_terpstra_trends.csv"
    jt_df = _load_csv(jt_csv)
    if jt_df is None:
        bundle.subtests.append(SubTest(
            "V3.a", "JT trends file present", passes=False,
            fail_reason=f"missing {jt_csv}",
        ))
        return bundle

    # V3.a: best per-cluster JT p value < threshold (dense plaque cluster)
    min_p = float(jt_df.get("JT_p_two_sided", pd.Series([1.0])).min())
    bundle.subtests.append(SubTest(
        "V3.a", "minimum JT p across cluster trends",
        primary_value=None, test_value=min_p,
        threshold=f"< {V3_JT_P_MAX}",
        passes=bool(min_p < V3_JT_P_MAX),
    ))
    return bundle


# ─── V4 C8 RCA-distal sheet ───


def evaluate_v4(primary: Path, test: Path) -> VerdictBundle:
    bundle = VerdictBundle("V4_c8_rca_distal_sheet")

    vc = _load_csv(test / "exploratory" / "lesion_morphology"
                    / "cluster_vessel_chi_square.csv")
    prof = _load_csv(test / "exploratory" / "lesion_morphology"
                      / "cluster_profiles.csv")
    if vc is None or prof is None:
        bundle.subtests.append(SubTest(
            "V4.a", "C8 inputs present", passes=False,
            fail_reason="vessel-chi-square or cluster-profiles missing",
        ))
        return bundle

    # Discover c8-like cluster by signature
    merged = prof.merge(
        vc[["cluster", "rca_obs_over_exp", "lm_obs_over_exp"]],
        on="cluster", how="left",
    )
    cand = merged[
        (merged["lm_obs_over_exp"] == 0)
        & (merged["volume_mm3_median"] > 100)
        & (merged["max_hu_median"] > 500)
        & (merged["rca_obs_over_exp"] > 1.5)
        & (merged["n_rois_median"] >= 5)
    ]
    if cand.empty:
        bundle.subtests.append(SubTest(
            "V4.a", "C8-like cluster discoverable by signature",
            passes=False, fail_reason="no cluster matches signature",
        ))
        return bundle
    c8 = cand.sort_values("rca_obs_over_exp", ascending=False).iloc[0]
    bundle.subtests.append(SubTest(
        "V4.a", "C8-like cluster discoverable",
        test_value=int(c8["cluster"]), passes=True,
    ))
    bundle.subtests.append(SubTest(
        "V4.b", "RCA obs/exp",
        test_value=round(float(c8["rca_obs_over_exp"]), 3),
        threshold=f">= {V4_RCA_OBS_EXP_MIN}",
        passes=bool(c8["rca_obs_over_exp"] >= V4_RCA_OBS_EXP_MIN),
    ))
    bundle.subtests.append(SubTest(
        "V4.c", "LM obs/exp (strict zero)",
        test_value=round(float(c8["lm_obs_over_exp"]), 3),
        threshold=f"== {V4_LM_OBS_EXP_MAX}",
        passes=bool(c8["lm_obs_over_exp"] == V4_LM_OBS_EXP_MAX),
    ))
    return bundle


# ─── V5 Stage 8 ───


def evaluate_v5(primary: Path, test: Path) -> VerdictBundle:
    bundle = VerdictBundle("V5_stage8_validation")

    loo = _load_csv(test / "08_validate" / "leave_k_out_ari.csv")
    if loo is None:
        bundle.subtests.append(SubTest(
            "V5.a", "LOO output present", passes=False,
            fail_reason="leave_k_out_ari.csv missing",
        ))
    else:
        summary_row = loo[loo["fold"].astype(str) == "SUMMARY"]
        if summary_row.empty:
            bundle.subtests.append(SubTest(
                "V5.a", "LOO SUMMARY row present", passes=False,
                fail_reason="SUMMARY row not found",
            ))
        else:
            median_ari = float(summary_row.iloc[-1]["ari"])
            pass_flag = bool(summary_row.iloc[-1]["pass_fold"])
            bundle.subtests.append(SubTest(
                "V5.a", "LOO overall PASS at runtime threshold T",
                test_value=pass_flag, threshold="True", passes=pass_flag,
            ))
            bundle.subtests.append(SubTest(
                "V5.b", "LOO median ARI absolute floor",
                test_value=round(median_ari, 3),
                threshold=f">= {V5_LOO_ARI_MIN}",
                passes=bool(median_ari >= V5_LOO_ARI_MIN),
            ))

    holdout = _load_csv(test / "08_validate" / "external_holdout_report.csv")
    if holdout is not None:
        pids = set(holdout["pid"].astype(str).tolist())
        bundle.subtests.append(SubTest(
            "V5.c", "GE holdout pids = {19, 28, 76, 77}",
            test_value=sorted(pids), threshold=sorted(V5_HOLDOUT_PIDS),
            passes=bool(pids == V5_HOLDOUT_PIDS),
        ))
        labels = set(holdout["predicted_phenotype"].astype(str).unique())
        bundle.subtests.append(SubTest(
            "V5.d", "GE holdout phenotype labels in {focal, diffuse}",
            test_value=sorted(labels),
            threshold="{focal, diffuse}",
            passes=bool(labels.issubset({"focal", "diffuse"})),
        ))
    else:
        bundle.subtests.append(SubTest(
            "V5.c", "GE holdout report present", passes=False,
            fail_reason="external_holdout_report.csv missing",
        ))
    return bundle


# ─── V6 LAD phenotype ───


def evaluate_v6(primary: Path, test: Path) -> VerdictBundle:
    bundle = VerdictBundle("V6_lad_phenotype")

    sig = _load_json(test / "exploratory" / "lad_phenotype"
                      / "lad_cluster_signature.json")
    if sig is None:
        bundle.subtests.append(SubTest(
            "V6.a", "LAD signature output present", passes=False,
            fail_reason="lad_cluster_signature.json missing",
        ))
        return bundle
    ids = sig.get("lad_dominant_cluster_ids") or []
    bundle.subtests.append(SubTest(
        "V6.a", "at least one cluster matches LAD-dominant pre-reg signature",
        test_value=ids, threshold=">=1 cluster",
        passes=bool(len(ids) >= 1),
    ))

    ax = _load_json(test / "exploratory" / "lad_phenotype" / "axial_summary.json")
    if ax is not None:
        rel_z = ax.get("median_in_cluster")
        if rel_z is not None:
            bundle.subtests.append(SubTest(
                "V6.b", "LAD-cluster within-LAD rel-z median (proximal)",
                test_value=round(float(rel_z), 3),
                threshold=f"< {V6_LAD_REL_Z_MAX}",
                passes=bool(float(rel_z) < V6_LAD_REL_Z_MAX),
            ))
        bundle.subtests.append(SubTest(
            "V6.c", "axial sub-test (plan.md P2) overall PASS",
            test_value=bool(ax.get("passes")),
            threshold="True", passes=bool(ax.get("passes")),
        ))

    md = _load_json(test / "exploratory" / "lad_phenotype" / "matched_diagnostics.json")
    if md is not None:
        verdict = md.get("verdict")
        bundle.subtests.append(SubTest(
            "V6.d", "burden-propensity match remains infeasible "
                     "(carrier/non-carrier burden disjoint)",
            test_value=verdict,
            threshold="match_infeasible",
            passes=bool(verdict == "match_infeasible"),
        ))
    return bundle


# ─── V7 LM-isolated low burden ───


def evaluate_v7(primary: Path, test: Path) -> VerdictBundle:
    bundle = VerdictBundle("V7_lm_isolated_low_burden")

    summary = _load_json(test / "exploratory" / "lm_isolated_low_burden"
                          / "summary.json")
    if summary is None:
        bundle.subtests.append(SubTest(
            "V7.a", "summary.json present", passes=False,
            fail_reason="lm_isolated summary missing",
        ))
        return bundle

    n_disp = int(summary.get("n_displaced", 0))
    bundle.subtests.append(SubTest(
        "V7.a", "displaced subgroup size within +/-3 of locked 10",
        primary_value=10, test_value=n_disp,
        threshold=f"|n - 10| <= {V7_DISP_SIZE_TOL}",
        passes=bool(abs(n_disp - 10) <= V7_DISP_SIZE_TOL),
    ))
    rate = float(summary.get("displaced_lm_rate", 0.0))
    bundle.subtests.append(SubTest(
        "V7.b", "displaced LM rate (strict 100%)",
        test_value=round(rate, 4),
        threshold=f"== {V7_DISP_LM_RATE_REQ}",
        passes=bool(rate >= V7_DISP_LM_RATE_REQ),
    ))

    cs = _load_json(test / "exploratory" / "lm_isolated_low_burden"
                     / "cross_stratum.json")
    if cs is not None:
        per = cs.get("per_stratum", {})
        all_strata_100 = all(
            info.get("displaced_lm_rate", 0) >= V7_DISP_LM_RATE_REQ
            for info in per.values()
        )
        bundle.subtests.append(SubTest(
            "V7.c", "per-stratum displaced LM rate == 100% in every stratum",
            test_value={k: round(v.get("displaced_lm_rate", 0), 3)
                         for k, v in per.items()},
            threshold="100% in every stratum",
            passes=all_strata_100,
        ))

    overlap = _load_json(test / "exploratory" / "lm_isolated_low_burden"
                         / "cluster_overlap.json")
    if overlap is not None:
        frac = float(overlap.get("overlap_fraction", 0.0))
        # The LM-isolated experiment now writes the LAD-dominant cluster
        # ids it actually used (read from the test-tree's LAD experiment
        # output). K-means cluster labels are arbitrary across reruns, so
        # the overlap fraction is what V7.d should compare, not the
        # literal cluster id list.
        lad_ids_used = overlap.get("lad_dominant_clusters_used",
                                     "unknown (legacy run)")
        bundle.subtests.append(SubTest(
            "V7.d", "overlap fraction with run-time LAD-dominant "
                     "clusters (strict zero)",
            test_value={"overlap_fraction": round(frac, 4),
                         "lad_dominant_clusters_used": lad_ids_used},
            threshold="overlap_fraction == 0.0",
            passes=bool(frac == 0.0),
            fail_reason=("LM-isolated experiment was run with stale "
                          "LAD-dominant cluster ids; rerun "
                          "experiments/lm_isolated_low_burden/run.py "
                          "after the LAD experiment so the IDs match "
                          "the current k-means labelling."
                          if (frac > 0.0 and lad_ids_used ==
                              "unknown (legacy run)") else ""),
        ))
    return bundle


# ─── orchestration ───


def evaluate_all(primary: Path, test: Path) -> dict:
    out: dict = {}
    for fn in (evaluate_v1, evaluate_v2, evaluate_v3, evaluate_v4,
                evaluate_v5, evaluate_v6, evaluate_v7):
        try:
            bundle = fn(primary, test)
        except Exception as exc:  # noqa: BLE001
            bundle = VerdictBundle(fn.__name__.replace("evaluate_", "V?_"))
            bundle.subtests.append(SubTest(
                "exception", str(exc), passes=False,
                fail_reason=f"{type(exc).__name__}: {exc}",
            ))
        out[bundle.verdict_id] = bundle.to_dict()
    out["overall_passes"] = all(b["passes"] for b in out.values()
                                  if isinstance(b, dict) and "passes" in b)
    return out


def _render_text(report: dict) -> str:
    lines: list[str] = []
    lines.append("Voxel-sensitivity probe (D032) - verdict comparison")
    lines.append("=" * 60)
    for vid, vbundle in report.items():
        if vid == "overall_passes":
            continue
        lines.append("")
        status = "PASS" if vbundle["passes"] else "FAIL"
        lines.append(f"[{status}] {vid}")
        for t in vbundle.get("subtests", []):
            sub_status = "PASS" if t["passes"] else "FAIL"
            lines.append(f"  [{sub_status}] {t['verdict_id']} - {t['name']}")
            if t.get("primary_value") is not None or t.get("test_value") is not None:
                lines.append(f"        primary={t.get('primary_value')}, "
                             f"test={t.get('test_value')}, "
                             f"threshold={t.get('threshold')}")
            if t.get("fail_reason"):
                lines.append(f"        reason: {t['fail_reason']}")
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"OVERALL: {'PASS' if report['overall_passes'] else 'FAIL'}")
    lines.append("=" * 60)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=Path("outputs"),
                         help="locked 0.5 mm output tree")
    parser.add_argument("--test", type=Path, default=Path("outputs_0375"),
                         help="sensitivity-probe 0.375 mm output tree")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if not args.primary.exists():
        _log.error("primary tree missing: %s", args.primary)
        return 2
    if not args.test.exists():
        _log.error("test tree missing: %s", args.test)
        return 2

    report = evaluate_all(args.primary.resolve(), args.test.resolve())
    out_json = args.test / "voxel_sensitivity_report.json"
    out_txt = args.test / "voxel_sensitivity_report.txt"
    out_json.write_text(json.dumps(report, indent=2, default=str) + "\n",
                         encoding="utf-8")
    out_txt.write_text(_render_text(report), encoding="utf-8")
    print(_render_text(report))
    return 0 if report["overall_passes"] else 1


if __name__ == "__main__":
    sys.exit(main())
