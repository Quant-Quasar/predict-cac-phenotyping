# D032 Voxel-sensitivity probe at native-median in-plane spacing (0.375 mm)

**Date**: 2026-06-15
**Stage**: methodology sensitivity (cross-stage; does not modify any stage)
**Status**: Active (sensitivity probe; does NOT supersede D005)
**Outputs**: `outputs_0375/` (parallel tree to `outputs/`)
**Comparison report**: `outputs_0375/voxel_sensitivity_report.json` + `.txt`

## Decision

Rerun the full production pipeline (stages 2 through 8) plus the
three exploratory experiments (lesion_morphology, lad_phenotype,
lm_isolated_low_burden) at a resampling target of
**0.375 x 0.375 x 3.0 mm** in-plane, vs the locked D005 target of
0.5 x 0.5 x 3.0 mm. Native cohort median in-plane spacing is
0.377 mm, so 0.375 mm matches native almost exactly and tests
whether the slight downsampling implicit in D005 (0.5 mm above
median) introduced any verdict-altering bias.

D005 stays as the **primary** for the main paper. D032 is the
**sensitivity probe** documented in the paper's supplementary
methods. The outcome is binary: every locked verdict either
replicates at the new voxel target, or it does not.

## Pre-registered comparison criteria

Each verdict below has a locked PASS / FAIL condition. The condition
is evaluated against the 0.375 mm rerun outputs after the 0.5 mm
outputs have been frozen. The conditions are NOT relaxed if the
0.375 mm result fails - a failure is documented as a real
finding, not as a methodological problem.

### V1. Continuum verdict (Finding 1)

| Sub-test | PASS condition |
|---|---|
| All 27 gap statistic curves monotone, no plateau | Same outcome at 0.375 mm |
| Selected k at boundary of search range (>= k_max - 2) in every run | Same outcome at 0.375 mm |
| Hopkins H on full cohort | within +/-0.05 of locked 0.72 |
| Spatial-only k=2 Hennig stability | each cluster median Jaccard >= 0.75 at 0.375 mm |

### V2. Three-lesion-class verdict (Finding 2)

| Sub-test | PASS condition |
|---|---|
| Lesion-level Hopkins H | within +/-0.03 of locked 0.95 |
| Broad-class signature discovery (via `_lesion_classes.discover_broad_classes`) | exactly three broad classes recovered (soft microspots, moderate nodules, dense plaques) |

### V3. Maturation verdict (Finding 3)

| Sub-test | PASS condition |
|---|---|
| Jonckheere-Terpstra p on dense-plaque fraction across burden tertiles | p < 10^-6 at 0.375 mm |
| Dense-plaque fraction in HIGH tertile - LOW tertile | >= 0.20 (locked: ~0.27) |

### V4. C8 RCA-distal sheet plaque verdict (Finding 4)

| Sub-test | PASS condition |
|---|---|
| C8-like cluster discoverable via signature (`discover_c8_like_cluster`) | not None |
| RCA obs/exp ratio | >= 1.7 (locked: ~2.0) |
| LM obs/exp ratio | == 0 (strict) |
| Within-RCA median rel-z | >= 0.65 (locked: ~0.75) |
| Carrier median Agatston / non-carrier median Agatston | >= 10x (locked: ~15x) |

### V5. Stage 8 LOO + GE holdout verdicts (Finding 5)

| Sub-test | PASS condition |
|---|---|
| LOO median ARI >= median T (overall PASS) | True at 0.375 mm |
| LOO median ARI absolute | >= 0.70 (locked: 0.859; tolerance for voxel drift) |
| GE holdout row count | == 4 (pids 19, 28, 76, 77) |
| GE holdout phenotype labels | subset of {focal, diffuse} |

### V6. LAD-phenotype experiment (paper-2 lead)

| Sub-test | PASS condition |
|---|---|
| At least one cluster matches pre-reg LAD-dominant signature | True at 0.375 mm |
| Within-LAD axial localisation: rel_z median (LAD-cluster) | < 0.30 (locked: ~0.15) |
| Within-LAD axial: median diff and MW p one-sided | meet plan.md P2 thresholds |
| Burden-propensity match infeasible (per plan.md) | match remains infeasible at 0.375 mm |

### V7. LM-isolated low-burden experiment (supplementary)

| Sub-test | PASS condition |
|---|---|
| Displaced subgroup size | within +/-3 of locked 10 |
| Displaced LM rate | == 100% (every displaced patient LM-positive) |
| Cross-stratum LM rate (each stratum) | == 100% |
| Cluster 10/11 overlap fraction | == 0% (strict; no exceptions) |
| Density profile W3/W4 dominant | True at 0.375 mm |

## Expected drifts (not failures)

These quantities are EXPECTED to shift between 0.5 mm and 0.375 mm
because voxel size mechanically changes them. The numbers below are
NOT verdict-determining; they are diagnostic of the voxel change
itself.

| Quantity | Expected drift | Direction |
|---|---|---|
| `mask_voxels` per patient | ~1.8x larger | UP |
| Patients with `low_burden_flag = True` (mask < 100 voxels) | DOWN from 142 | DOWN |
| PyRadiomics-skipped patients (mask < 14 voxels) | DOWN from 22 | DOWN |
| ICC-passing features (currently 88) | shift +/-5 features | mixed |
| Cohort-level Hopkins H (Qr36d/2 stratum) | up to +/-0.05 drift | both directions |
| Cluster 10/11 volume_mm3_median | up to +/-15% | mixed |

## Verdict-level termination conditions

| Condition | Outcome |
|---|---|
| Any sub-test in V1-V7 fails its PASS condition | Documented as a real finding in the paper's supplementary methods. D032 reports "voxel-dependent verdict at sub-test X." |
| All sub-tests pass | Supplementary methods reports "all locked verdicts replicate at 0.375 mm in-plane resampling," strengthening the paper. |
| Comparison script (`scripts/compare_voxel_sensitivity.py`) fails to run on either output tree | Pipeline incomplete; D032 cannot be evaluated. |

## What this decision does NOT do

* Does NOT supersede D005. The 0.5 mm grid stays the primary cohort.
* Does NOT modify the locked findings in `findings.md` files.
* Does NOT change any other decision document.
* Does NOT add to CLAUDE.md's stage status table (the sensitivity
  probe is documented here only).
* Does NOT extend `verify_pipeline.py`'s expected-value table; that
  remains calibrated to the 0.5 mm outputs.

## Why the pre-registration matters

The temptation when running a sensitivity probe is to relax a
threshold after seeing the new number. Pre-registering the
acceptance bands BEFORE running the rerun prevents this. If a
verdict fails its band, that is the data telling us the verdict
was voxel-dependent, and the paper acknowledges it. If the verdict
passes its band, the paper's claim is strengthened.

The bands above were chosen with deliberate slack relative to the
0.5 mm locked numbers:
* Hopkins H tolerance +/-0.05 covers the known cross-machine
  BLAS-drift band (max observed ~0.06 on stratified cohorts)
* Lesion-level Hopkins tolerance +/-0.03 reflects the tighter
  lesion-level stability
* LOO ARI >= 0.70 leaves ~0.16 of room below the locked 0.859
* C8 RCA obs/exp >= 1.7 leaves room from 2.0 down to 1.7
* Cluster 10/11 overlap fraction strict 0% because any non-zero
  overlap reframes the LM-isolated finding

## Verified by

* `scripts/compare_voxel_sensitivity.py` (one-shot comparison tool)
* Human review of the auto-generated `voxel_sensitivity_report.txt`
* The 19 unit tests + the 723 production tests + the 6 + 12 + 19
  experiment-tests all continue to pass on both output trees
