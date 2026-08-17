# Exploratory: LAD-specific morphological phenotype

**Status**: exploratory; outside the production pipeline. Does not produce
any seam file consumed by stages 1-8. Does not modify any decision.
No CLAUDE.md tracking. No verify_pipeline.py coverage.

## Question

The C8 finding (RCA-dominant coalescent sheet plaque, replicated in both
kernel strata, Cramer's V = 0.40) raises the obvious parallel question:
does the LAD have its own characteristic morphology cluster, and if so
how does it differ from C8 in location within the vessel, density
profile, and carrier-patient burden signature?

The widowmaker clinical motivation is non-trivial:

```
LAD calcium burden       -> LAD plaque burden indicator
LAD plaque burden        -> LAD occlusion risk
Proximal LAD occlusion   -> anterior STEMI (highest acute mortality)
```

The biological hypothesis is that haemodynamic differences between the
LAD (high-shear, straight, anteriorly-running) and the RCA (lower-shear,
curved, inferiorly-running) produce different calcification morphologies
at high burden. If the lesion-level cluster taxonomy already separates
these vessels (preliminary numbers show C6 LAD obs/exp = 1.37, C8 RCA
obs/exp = 1.99), formalising the contrast adds a paper-grade finding.

## What this experiment IS

1. A pre-registered, signature-based discovery of which lesion clusters
   match an LAD-dominant morphology profile (locked in `plan.md` before
   inspecting any cluster).
2. A within-LAD axial localisation analysis: do those clusters
   concentrate proximally (clinically dangerous) or distally?
3. A carrier-patient profile: do LAD-cluster carriers have a different
   burden / vessel-distribution signature than C8 carriers?
4. **A burden-propensity-matched comparison**: after matching on total
   Agatston, does the LAD bias survive? This is the same independence
   test waiting for the C8 RCA bias question, so the infrastructure
   built here is dual-purpose.
5. Cross-stratum (Qr36d/2 + I30f/3) replication of every quantitative
   verdict.

## What this experiment is NOT

* NOT a new pipeline stage.
* NOT a decision (no D### number).
* NOT something CLAUDE.md or the main test suite track.
* NOT a re-design of stages 5, 6, 7, or 8.
* NOT a publication claim on its own. If results replicate at the
  pre-registered thresholds and survive propensity matching, it
  becomes a paper-2 lead analysis. Otherwise it stays here as
  documented exploratory work.

## Inputs (all read-only)

| File | Source | Used for |
|---|---|---|
| `outputs/03_features/lesions.csv` | stage 3 audit | per-lesion centroid + vessel labels |
| `outputs/03_features/features.csv` | stage 3 | per-patient agatston_total + per-vessel agatston |
| `outputs/exploratory/lesion_morphology/lesion_cluster_labels.csv` | lesion experiment | per-lesion cluster id |
| `outputs/exploratory/lesion_morphology/cluster_profiles.csv` | lesion experiment | per-cluster median morphology |
| `outputs/exploratory/lesion_morphology/cluster_vessel_chi_square.csv` | lesion experiment | per-cluster vessel obs/exp |
| `outputs/06_reduce/cohort_metadata.csv` | stage 5 | per-patient kernel + low_burden_flag |

## Outputs

All under `outputs/exploratory/lad_phenotype/`:

| File | Content |
|---|---|
| `lad_cluster_signature.json` | which clusters match LAD-dominant signature + their carrier counts |
| `axial_within_lad.csv` | per-lesion relative-z within each patient's LAD z-range, with cluster + carrier labels |
| `axial_summary.json` | median rel-z by cluster + Mann-Whitney vs other LAD lesions |
| `carrier_profile.csv` | per-pid total Agatston + per-vessel burden + carrier flag |
| `carrier_summary.json` | Cliff's delta + MW p for carriers vs non-carriers (burden, n_calc_arteries, vessel breakdown) |
| `matched_pairs.csv` | propensity-matched case-control pairs (case=carrier, controls=non-carriers at same burden) |
| `matched_diagnostics.json` | post-match standardised mean difference per covariate (must be < 0.1) |
| `matched_comparison.csv` | after matching: vessel-distribution + morphology + axial differences |
| `cross_stratum_replication.json` | Qr36d/2 vs I30f/3 independent reruns of all of the above |
| `report.txt` | human-readable summary printed by finalise.py |
| `run_header.json` | git hash + libraries + CLI args + seam SHAs |

## How to run

```bash
# Step 1-3 (signature discovery + axial + carrier profile)
python experiments/lad_phenotype/run.py

# Step 4 (burden-matched comparison)
python experiments/lad_phenotype/matched.py

# Step 5 + report (cross-stratum replication + bundle + report.txt)
python experiments/lad_phenotype/finalise.py
```

Wall-clock at production cohort size: ~3 minutes total.

## Methodological consistency

This experiment reuses the production-tested helpers from
`predict.analyse`:

* `cliffs_delta` and `mannwhitney_u_pval` for effect sizes
* `apply_fdr_bh` for multiple-comparisons correction across vessel
  contrasts

The propensity-matching helper (`propensity.py`) is new infrastructure
written for this experiment but designed to be reusable by the C8
RCA-burden-independence follow-up. Tests in `tests/test_propensity.py`.

## Discipline notes

* Every threshold and hypothesis direction is locked in `plan.md`
  BEFORE running. Read `plan.md` first.
* Do not iterate on thresholds after seeing which clusters match. That
  is hidden p-hacking and undoes the pre-registration.
* `findings.md` stays empty until the experiment runs to completion
  with all PASS criteria evaluated.
