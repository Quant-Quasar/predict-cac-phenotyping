"""F5 — C8 anatomical specificity (RCA dominance, zero LM, distal z).

Visualises Finding L3: the C8-like lesion cluster (massive sheet
plaques) is a sharp anatomical phenotype with strict LM exclusion and
distal-RCA localisation. Carrier patients are extreme high-burden.

All three panels DISCOVER the C8-like cluster from this rerun's
cluster medians + vessel chi-square (NOT the literal label "8"), then
recompute the burden distribution and within-RCA z-position on the
discovered cluster's lesions. K-means cluster labels are arbitrary
across reruns; the morphology signature is not.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from predict.figures._lesion_classes import discover_c8_like_cluster
from predict.figures.style import PALETTE


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    diffs = a[:, None] - b[None, :]
    return (float((diffs > 0).sum()) - float((diffs < 0).sum())) / (a.size * b.size)


def build(
    lesion_morph_dir: Path,
    lesions_csv: Path,
    cohort_metadata_csv: Path,
) -> "plt.Figure":
    vc = pd.read_csv(lesion_morph_dir / "cluster_vessel_chi_square.csv")
    profiles = pd.read_csv(lesion_morph_dir / "cluster_profiles.csv")
    c8_id = discover_c8_like_cluster(profiles, vc)
    if c8_id is None:
        # Fallback: highest RCA o/e among any cluster
        c8_id = int(vc.loc[vc["rca_obs_over_exp"].idxmax(), "cluster"])
    c8_row = vc[vc["cluster"] == c8_id].iloc[0]

    # ── Recompute panels (b) + (c) for THIS cluster id ────────────
    lesions = pd.read_csv(lesions_csv, dtype={"pid": str})
    labels_df = pd.read_csv(
        lesion_morph_dir / "lesion_cluster_labels.csv", dtype={"pid": str},
    )
    primary_col = next(c for c in labels_df.columns
                        if c.startswith("cluster_kmeans_k"))
    lesions = lesions.merge(
        labels_df[["pid", "vessel", "lesion_idx", primary_col]],
        on=["pid", "vessel", "lesion_idx"], how="left",
    )
    lesions[primary_col] = lesions[primary_col].astype("Int64")
    c8_lesions = lesions[lesions[primary_col] == c8_id]
    c8_pids = set(c8_lesions["pid"].unique())

    meta = pd.read_csv(cohort_metadata_csv, dtype={"pid": str})
    agatston = meta.set_index("pid")["agatston_total"].astype(float)
    c8_burden = agatston.loc[agatston.index.intersection(c8_pids)].dropna()
    non_c8_burden = agatston.loc[
        agatston.index.difference(c8_pids)
    ].dropna()
    burden_p = float(stats.mannwhitneyu(
        c8_burden.to_numpy(), non_c8_burden.to_numpy(), alternative="greater",
    ).pvalue) if (len(c8_burden) and len(non_c8_burden)) else float("nan")
    burden_delta = _cliffs_delta(c8_burden.to_numpy(), non_c8_burden.to_numpy())

    # Within-RCA relative z (per patient): for each patient with RCA
    # lesions, compute (z - z_min) / (z_max - z_min) per RCA lesion.
    rca = lesions[lesions["vessel"] == "RCA"]
    rel_z_records = []
    for pid, grp in rca.groupby("pid"):
        if len(grp) < 2:
            continue
        zmin = grp["centroid_z_mm"].min()
        zmax = grp["centroid_z_mm"].max()
        if zmax == zmin:
            continue
        for _, row in grp.iterrows():
            rel = (row["centroid_z_mm"] - zmin) / (zmax - zmin)
            rel_z_records.append({
                "in_c8": bool(row[primary_col] == c8_id),
                "relative_z_within_RCA": float(rel),
            })
    rel_z = pd.DataFrame(rel_z_records)
    c8_rel_z = rel_z.loc[rel_z["in_c8"], "relative_z_within_RCA"].to_numpy()
    other_rel_z = rel_z.loc[~rel_z["in_c8"], "relative_z_within_RCA"].to_numpy()

    vessels = ("LAD", "RCA", "LCx", "LM")
    obs_over_exp = [float(c8_row[f"{v.lower()}_obs_over_exp"]) for v in vessels]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ax_vessel, ax_z, ax_burden = axes

    # (a) Vessel observed/expected ratio
    colours = [PALETTE[v] for v in vessels]
    bars = ax_vessel.bar(vessels, obs_over_exp,
                          color=colours, edgecolor="white", linewidth=1)
    ax_vessel.axhline(1.0, c="black", lw=0.7, linestyle="--",
                       alpha=0.7, label="cohort baseline")
    for v, r, bar in zip(vessels, obs_over_exp, bars):
        if v == "LM" and r == 0:
            ax_vessel.text(bar.get_x() + bar.get_width() / 2, 0.05,
                            "ZERO LM\nlesions",
                            ha="center", va="bottom", fontsize=9,
                            color=PALETTE["refuted"], weight="bold")
        else:
            ax_vessel.text(bar.get_x() + bar.get_width() / 2,
                            r + 0.05, f"{r:.2f}",
                            ha="center", va="bottom", fontsize=9)
    ax_vessel.set_ylabel("observed / expected ratio")
    ax_vessel.set_title(
        f"(a) Vessel distribution\n"
        f"cluster id (this rerun) = {c8_id}, n = {int(c8_row['n_lesions'])}\n"
        f"$\\chi^2$ p = {c8_row['chi2_p']:.1e};  "
        f"Cramer's V = {c8_row['cramers_v']:.2f}"
    )
    ax_vessel.set_ylim(0, max(2.5, max(obs_over_exp) + 0.4))
    ax_vessel.legend(loc="upper left")

    # (b) Within-RCA relative-z histogram (recomputed for discovered cluster)
    bins = np.linspace(0, 1, 21)
    if c8_rel_z.size > 0 and other_rel_z.size > 0:
        ax_z.hist(other_rel_z, bins=bins, alpha=0.55, color="#888888",
                   label="other RCA lesions",
                   edgecolor="white", linewidth=0.4, density=True)
        ax_z.hist(c8_rel_z, bins=bins, alpha=0.85,
                   color=PALETTE["refuted"], label="C8-like lesions",
                   edgecolor="white", linewidth=0.4, density=True)
        ax_z.set_title(
            "(b) C8-like localises within RCA\n"
            f"median rel-z = {np.median(c8_rel_z):.2f} "
            f"vs {np.median(other_rel_z):.2f} for other RCA lesions"
        )
        ax_z.legend(loc="upper center")
    else:
        ax_z.text(0.5, 0.5,
                  "insufficient RCA lesions in C8-like cluster",
                  ha="center", va="center", transform=ax_z.transAxes)
    ax_z.set_xlabel("relative z within patient's RCA\n"
                     "(0 = most proximal, 1 = most distal)")
    ax_z.set_ylabel("density")

    # (c) Burden distribution: C8 patients vs non-C8 (recomputed)
    c8_burden_med = float(np.median(c8_burden)) if len(c8_burden) else float("nan")
    non_c8_burden_med = float(np.median(non_c8_burden)) if len(non_c8_burden) else float("nan")
    ax_burden.barh(
        ["non-C8-like patients", "C8-like patients"],
        [non_c8_burden_med, c8_burden_med],
        color=[PALETTE["neutral_blue"], PALETTE["refuted"]],
        edgecolor="white", linewidth=1,
    )
    for i, val in enumerate([non_c8_burden_med, c8_burden_med]):
        ax_burden.text(val + max(c8_burden_med, 50) * 0.03, i, f"{val:.0f}",
                        va="center", ha="left", fontsize=9)
    ax_burden.set_xlabel("median Agatston total")
    ax_burden.set_title(
        f"(c) C8-like patients are extreme high-burden\n"
        f"Cliff's $\\delta$ = {burden_delta:.2f}; MW p = {burden_p:.1e}\n"
        f"(N_C8 = {len(c8_burden)}, N_other = {len(non_c8_burden)})"
    )

    fig.suptitle(
        "Figure 5. The C8-like lesion cluster is a sharp anatomical phenotype:\n"
        "RCA-dominant, zero LM representation, distal-RCA localisation, "
        "extreme high-burden patients.\n"
        "(cluster discovered by morphology signature on each rerun; not by "
        "literal label)",
        y=1.07, fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    return fig
