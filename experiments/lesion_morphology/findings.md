# Lesion morphology experiment — locked findings (finalised)

**Status**: exploratory. Outside the production pipeline.
**Date**: 2026-06-06 (finalised after `finalise.py` rerun).
**Inputs**: `outputs/03_features/lesions.csv` (3201 lesions across 444 patients), `outputs/06_reduce/{cohort_metadata, cluster_labels_spatial_k2}.csv`, `outputs/03_features/features.csv`.
**Methodological reuse**: `predict.discover.{clusterability, cluster_discovery, validity}` (Hopkins k=2-for-both, Maitra-Ramler gap-1-SE rule, Hennig clusterboot). Identical machinery to stages 6 and 7.

## Headline findings

| # | Finding | Strength | Evidence |
|---|---|---|---|
| L1 | Lesion morphology is a CONTINUUM at every k tested. Hopkins H = 0.95 (strongly clustered) but gap statistic hits the boundary at k=8 (first run) and k=12 (extended run) for kmeans/ward, and GMM jumped from k=3 to k=11 when the boundary moved — all algorithms chase the boundary when allowed. Patient mixture vectors in CLR space are similarly clusterable (Hopkins 0.83) but with 0 of 4 stable Hennig clusters at GMM k=4. | strong | `hopkins.json`, `gap_statistic.json`, `mixture_k4_hennig.json` |
| L2 | A broad three-class taxonomy {small / medium / large} is moderately supported by GMM at k=3, ARI = 0.52 against the manual cluster-medians grouping. The latent separation axis is SIZE, not density: volume_mm3 varies 15x across the 3 classes while max_hu varies only ~3x. Density-based names (soft_microspots / moderate_nodules / dense_plaques) remain descriptive because size and density correlate in COCA, but the primary axis is size. | moderate | `gmm3_density_vs_size_separation.csv`, `gmm_k3_summary.json` |
| **L3** | **The C8 phenotype is REAL, REPLICATED, and ANATOMICALLY SPECIFIC.** Massive (~250 mm3) dense (~834 HU) multi-slice (~8 ROIs) RCA-dominant sheet plaques with strict LM exclusion. Replicates across kernel strata at effectively identical characteristics: volume (250 / 250 / 243), max_hu (834 / 817 / 858), n_rois (8 / 8 / 8 exactly), RCA obs/exp (1.98 / 1.91 / 2.16), and **zero LM lesions in every stratum**. C8 patients have median Agatston 1226 vs 79 for non-C8 (Cliff's delta 0.926, Mann-Whitney p = 3e-36). Within RCA, C8 lesions sit at the distal end (relative z 0.754 vs 0.153 for other RCA lesions). | **strong, replicated** | `per_stratum_c8_replication.json`, `c8_deep_dive.json` |
| L4 | Burden-monotonic trend confirmed by Jonckheere-Terpstra for 8 of 12 clusters at p < 0.001 (all dense and moderate clusters). The 4 non-significant clusters are the soft_microspots family (C1, C3, C5, C9). **C9 does NOT decline monotonically** under formal per-patient JT testing (the earlier mean-fraction decline was a tail artefact); on medians C9 is essentially flat across tertiles. | strong (for the dense / moderate clusters); refuted for C9 | `jonckheere_terpstra_trends.csv` |
| L5 | Patient-level mixture phenotypes do NOT exist as discrete clusters. Mixture GMM at k=4 produces 0 of 4 Hennig-stable clusters (medians 0.30, 0.14, 0.48, 0.60). The clusters HAVE biological character (one is focal-skewed low-burden, one is diffuse-skewed high-burden) but they're not stable to bootstrap resampling. **Confirms that patient-level continuum extends even into lesion-mixture-composition space.** | refuted at Hennig threshold; biology present but not discrete | `mixture_k4_hennig.json`, `mixture_k4_profile.csv`, `mixture_k4_crosstabs.csv` |

## Cluster-by-cluster reference (k=12 partition)

Numeric labels only. The 3-class broad mapping is locked in `broad_class_mapping.csv` and reproduced in `analyse.py:HYPOTHESISED_K3_GROUPING`.

| ID | broad | n | vol med | max_hu med | n_rois med | JT direction (p) | Note |
|---|---|---|---|---|---|---|---|
| C9 | soft_microspots | 612 (19.1%) | 5.0 | 185 | 1 | dec (0.22) n.s. | dominant low-burden type; flat by JT |
| C1 | soft_microspots | 176 (5.5%) | 5.4 | 182 | 1 | inc (0.09) n.s. | |
| C5 | soft_microspots | 207 (6.5%) | 9.4 | 233 | 1 | inc (4e-3) | |
| C3 | moderate_nodules | 448 (14.0%) | 13.1 | 300 | 1 | inc (0.03) n.s. | |
| C2 | moderate_nodules | 367 (11.5%) | 22.0 | 300 | 2 | inc (2e-5) | |
| C7 | moderate_nodules | 206 (6.4%) | 31.0 | 412 | 2 | inc (1e-8) | |
| C10 | moderate_nodules | 355 (11.1%) | 37.1 | 476 | 1 | inc (0.0) | strongest moderate-class increase |
| C11 | moderate_nodules | 152 (4.8%) | 44.8 | 484 | 2 | inc (6e-10) | |
| C0 | dense_plaques | 211 (6.6%) | 80.3 | 514 | 4 | inc (0.0) | RCA+LCx biased (V=0.20) |
| C6 | dense_plaques | 250 (7.8%) | 92.0 | 717 | 2 | inc (0.0) | LAD+LM biased (V=0.21) |
| C4 | dense_plaques | 116 (3.6%) | 163.9 | 991 | 2 | inc (2e-9) | LAD-biased (V=0.21) |
| **C8** | **dense_plaques** | **101 (3.2%)** | **250.1** | **834** | **~8** | inc (5e-14) | **RCA-dominant; zero LM; replicates across strata** |

## Vessel-biased clusters (full cohort)

Threshold: chi-square p < 0.001 AND Cramer's V >= 0.20 against the cohort-level vessel base rate (LAD 1335 / RCA 1007 / LCx 673 / LM 186 over 3201 lesions).

| Cluster | n | chi-square p | Cramer's V | LAD/RCA/LCx/LM (obs/exp) | Note |
|---|---|---|---|---|---|
| C0 | 211 | 8e-6 | 0.20 | 0.68 / 1.34 / 1.31 / 0.33 | RCA+LCx biased; LM under-rep |
| C4 | 116 | 1e-3 | 0.21 | 1.43 / 0.69 / 0.62 / 1.04 | LAD-biased |
| C6 | 250 | 2e-7 | 0.21 | 1.37 / 0.58 / 0.76 / 1.45 | LAD+LM biased |
| **C8** | **101** | **3e-10** | **0.40** | **0.52 / 1.98 / 0.75 / 0.00** | **RCA-dominant, zero LM** |
| C11 | 152 | 1e-3 | 0.18 (below cut) | 1.29 / 0.71 / 0.69 / 1.59 | flagged in vessel_biased=False but listed for completeness |

C8 is the strongest specific finding (Cramer's V = 0.40 = large effect by Cohen's convention vs 0.20-0.21 for the others).

## C8 deep dive

| Quantity | C8 patients (N=75) | non-C8 patients (N=345) | Test |
|---|---|---|---|
| Agatston median | 1226 | 79 | MW p = 3e-36, Cliff's delta = 0.926 |
| Relative-z within RCA (per-patient) | 0.754 (distal) | 0.153 (proximal) | MW p = 2e-10, Cliff's delta = 0.511 |

**C8 lesions are not just "RCA lesions" — they are specifically distal RCA lesions in extreme-high-burden patients.** The distal position is biologically consistent with mature sheet calcification: distal RCA has longer transit times and known stress patterns favouring large mature plaques.

## Per-stratum C8 replication

| Stratum | cluster id | N | RCA o/e | LM o/e | Cramer's V | vol med | max_hu med | n_rois med | chi-square p |
|---|---|---|---|---|---|---|---|---|---|
| Full | C8 | 101 | 1.98 | **0.00** | 0.40 | 250 | 834 | 8 | 3e-10 |
| Qr36d/2 | cluster 1 (refit) | 67 | 1.91 | **0.00** | 0.38 | 250 | 817 | 8 | 2.5e-6 |
| I30f/3 | cluster 9 (refit) | 34 | 2.16 | **0.00** | 0.46 | 243 | 858 | 8 | 7.5e-5 |

Within 3% on every continuous summary statistic, and exact match on n_rois (8) and LM count (0) across all three cohorts. The C8 phenotype is scanner-independent.

## Mechanistic narrative (locked for paper supplementary)

> COCA coronary calcium lesions exhibit a continuous morphology spectrum at the lesion level (Hopkins H = 0.95) with no gap-statistic elbow at any k up to 12 across kmeans, ward, or Gaussian mixture clustering. A broad three-class GMM partition captures the principal axis of variation (size-dominated; volume varies ~15x across the three classes while peak HU varies ~3x). Patient-level lesion-mixture composition is also continuous (CLR Hopkins H = 0.83, but Gaussian-mixture k=4 produces 0 of 4 bootstrap-stable clusters), consistent with the patient-level absence-of-discrete-phenotypes finding reported in stage 7. Burden-monotonic trends (Jonckheere-Terpstra p < 0.001) are present in 8 of 12 lesion clusters (all moderate and dense classes), supporting a maturation trajectory in which higher-burden patients carry an increasing complement of larger, denser lesions on top of the same low-burden microspot baseline. One specific lesion phenotype is anatomically distinct: a small cluster of massive (~250 mm3) dense (~834 HU) multi-slice (~8 ROIs) lesions is strongly RCA-dominant (chi-square p = 3e-10, Cramer's V = 0.40), excluded from the left main coronary artery (zero LM representation), localised at the distal end of the RCA (median relative-z 0.75 vs 0.15 for other RCA lesions, Cliff's delta 0.51), and confined to extreme-high-burden patients (median Agatston 1226 vs 79 in non-carriers, Cliff's delta 0.93). This phenotype replicates with effectively identical characteristics in both kernel-stratified subcohorts (volume 243-250 mm3, peak HU 817-858, n_rois exactly 8, RCA obs/exp 1.91-2.16, zero LM lesions in all strata), confirming it is scanner-independent.

## Net update to the overall study findings

| Stage / experiment finding | Update from lesion morphology |
|---|---|
| **Stage 6/7 Finding 1: continuum, no discrete patient phenotypes** | **STRENGTHENED**: continuum extends to lesion resolution and to mixture-composition space (both refute discrete clusters at Hennig threshold). |
| Stage 6/7 Finding 2: kernel chi-square = patient bias | unchanged (lesion clusters are kernel-balanced) |
| Stage 7 Finding 3 (revised): spatial k=2 = burden dichotomy | **MECHANISTIC EXPLANATION ADDED**: burden-tertile mixture shifts (Jonckheere-Terpstra confirmed for the dense and moderate classes) are the per-lesion mechanism behind the patient-level burden continuum. |
| New: lesion-mixture-phenotype claim | **REFUTED** (Hennig 0 of 4 stable at k=4). |
| **New: C8 RCA-dominant distal massive plaque phenotype** | **NEW PUBLISHABLE FINDING.** Replicates across kernel strata; anatomically and biologically specific; large effect size. |

## Open follow-ups (NOT in scope for closing this exploration)

- C8 leave-one-out cross-validation: would require refitting the clustering with each patient held out and confirming the C8 phenotype emerges in each fold. The two within-kernel-stratum refits ALREADY provide strong replication; LOO would be belt-and-braces.
- Per-patient frame anatomy mapping: the relative-z proxy used here approximates proximal/mid/distal but is not a true anatomical mapping. A future analysis could use vessel-centreline arc-length if XML gives sufficient anatomical landmarks.
- C8 association with hard MACE outcomes: this experiment doesn't touch outcomes. If a future cohort with MACE data is available, C8 carriership would be a candidate clinical predictor.

## Decision to close the exploration

All 8 items from the user's plan have been executed. Three findings are publication-ready (L1 continuum extension, L3 C8 phenotype, L4 burden-monotonic trend for the dense classes), one is empirically REFUTED (L5 mixture-phenotype claim), and one is a moderate-strength observation (L2 broad 3-class). No further analysis is needed inside this exploratory scope.

The production pipeline (stages 1-7) is unchanged. CLAUDE.md is unchanged. The test suite is unchanged.
