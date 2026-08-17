# Decision Log

One markdown file per decision: `D###-short-slug.md`. Numbers are monotonic; superseded decisions are kept but marked **Superseded by D###**. Each decision file follows the template below.

## Template

```
# D### — Short title

**Date**: YYYY-MM-DD
**Stage**: io | preprocess | features | stability | reduce | discover | analyse | validate
**Status**: Active | Superseded by D### | Deferred
**Module**: src/predict/<stage>/<module>.py

## Decision
One paragraph stating the rule, threshold, or convention.

## Rationale
Why this and not the alternatives. Cite dataset evidence or literature.

## Alternatives considered
- Option A — rejected because …
- Option B — rejected because …

## Verified by
Test file/function or empirical check.
```

## Index

| ID | Title | Stage | Status |
|----|-------|-------|--------|
| [D001](D001-z-coordinate-matching.md) | Z-coordinate matching as primary ROI→slice mapping | io / preprocess | Active |
| [D002](D002-xml-stat-roundtrip.md) | XML stat round-trip as preprocessing correctness gate | preprocess / validate | Active |
| [D003](D003-no-display-hu-window.md) | No display HU window in pipeline outputs | preprocess | Active |
| [D004](D004-cohort-exclusions-at-discovery.md) | Cohort exclusions applied at patient discovery | io | Active |
| [D005](D005-target-voxel-grid.md) | Target voxel grid 0.5 × 0.5 × 3.0 mm | preprocess | Active |
| [D006](D006-multi-series-selection.md) | Multi-series DICOM folder reduction rule (largest slice count) | io | Active (supersedes v1 D012) |
| [D007](D007-lesion-grouping-rule.md) | Lesion grouping rule (BFS, ≤5 mm in-plane, gap=1) | features | Active |
| [D008](D008-per-artery-mask.md) | Per-artery masks via filter-then-rasterise | features | Active |
| [D009](D009-pyradiomics-config.md) | PyRadiomics extractor configuration | features | Active |
| [D010](D010-low-burden-flag.md) | `low_burden_flag` for very-small-mask patients | features | Active |
| [D011](D011-agatston-thickness-helper.md) | Single Agatston thickness-correction helper | features | Active (supersedes v1 D018) |
| [D012](D012-excluded-roi-ids-input.md) | `excluded_roi_ids` as explicit input to every feature path | features | Active |
| [D013](D013-icc-formulation-and-threshold.md) | ICC(3,1) absolute agreement, threshold 0.75 | stability | Active |
| [D014](D014-perturbation-set.md) | 14-perturbation set (rotations, translations, noise) | stability | Active |
| [D015](D015-cohort-subset-for-stability.md) | Full eligible cohort (N=422) for ICC computation | stability | Active |
| [D016](D016-geometric-feature-bypass.md) | Geometric features bypass empirical gate with ICC=1.0, tagged via `icc_source` | stability | Active |
| [D017](D017-sentinel-prone-feature-exclusion.md) | Drop 13 zero-inflated per-vessel features at stage 5 ingestion (88 -> 75) | reduce | Active |
| [D018](D018-density-tier-and-dense-calcium-handling.md) | Keep 16 density tiers (robust scaling at stage 5); binarise dense_calcium_count -> has_dense_calcium | reduce | Active |
| [D019](D019-stage5-preprocessing.md) | Stage 5 preprocessing: variance filter, derived-feature R^2 gate, ComBat on 6 texture, Yeo-Johnson + rank fallback on 19 sparse, global z-score | reduce | Active |
| [D020](D020-stage5-redundancy-and-pca.md) | Spearman r^2 > 0.75 hierarchical clustering (average linkage primary; ward / complete sensitivity); ICC-first representative selection; global PCA at 0.85 cumvar | reduce | Active |
| [D021](D021-stage5-cluster-discovery-and-validity.md) | Hopkins -> gap statistic 500 boots x 3 runs (full / burden-residualised / spatial-only) -> consensus -> forced k=3 -> kernel/bootstrap validity -> 280-patient robust + 2 kernel-stratified sensitivity reruns | discover | Active (re-tagged from `reduce` on 2026-06-05 Phase B; number unchanged) |
| [D022](D022-multi-block-redundancy-clustering.md) | Multi-block Spearman r² > 0.75 redundancy clustering across 6 prospective blocks (burden, HU stats, density tier, spatial, texture, shape); single-matrix variant retained as sensitivity | reduce | Active (supersedes single-matrix portion of D020 for primary analysis) |
| [D023](D023-stage7-per-cluster-characterisation.md) | Per-cluster characterisation: median + IQR + Cliff's delta + Mann-Whitney with FDR-BH across 41 features; biological sanity check (focal max_hu >= 0.9 * diffuse); label balance check (minority class > 15%) | analyse | Active |
| [D024](D024-stage7-burden-orthogonality.md) | Burden orthogonality on agatston_total: Mann-Whitney + Levene + Cliff's delta with 3-level interpretation column (orthogonal / marginal / confounded); burden-stratified spatial profiles within each Agatston tertile | analyse | Active |
| [D025](D025-stage7-directional-hypotheses.md) | Six pre-specified one-sided Mann-Whitney directional hypotheses for focal vs diffuse; primary (>= 4 of 6 at FDR p < 0.05 in full) + secondary (>= 4 of 6 same direction in BOTH strata, direction only) verdict | analyse | Active |
| [D026](D026-stage7-monotonicity-classification.md) | Spearman rho + Kendall tau vs agatston_total for each of the 28 robust features; classify into burden_tracking (\|rho\| >= 0.5) / structure_tracking / spatial_tracking / mixed | analyse | Active |
| [D027](D027-stage7-cross-cohort-feature-consistency.md) | Feature-level cross-cohort consistency: direction in all 3 + FDR p < 0.05 in >= 2 of 3 + \|Cliff's delta\| >= 0.20 in all 3; partition ARI on shared pids preserved as separate evidence pillar (>= 0.80) | analyse | Active |
| [D028](D028-stage7-low-burden-sensitivity-output.md) | Robust-cohort (low_burden_flag = False, N ~ 280) sensitivity as a separate output file, not extra rows in the main paper table | analyse | Active |
| [D029](D029-stage8-external-ge-holdout.md) | External GE-scanner holdout (pids 19, 28, 76, 77): apply frozen pipeline; skip ComBat (spatial-only phenotype is ComBat-free); descriptive only (N=4); D023 mapping on predicted_phenotype | validate | Active |
| [D030](D030-stage8-leave-k-out-cv.md) | 10-fold kernel-stratified LOO with full per-fold refit (D019 + D022 + D020 + D021 + GMM); PASS = median ARI ≥ T where T = 5th-percentile simulation at K=10% disagreement; raw GMM labels into ARI; D023 mapping on interpretive columns only | validate | Active |
| [D031](D031-stage8-cross-cohort-ari-consolidation.md) | Re-export stage 7 cross-cohort ARI with PASS verdict columns (ARI ≥ 0.80); pure bookkeeping for the stage 8 deliverable triad | validate | Active |
| [D032](D032-voxel-sensitivity-probe-0375.md) | Sensitivity probe: rerun full pipeline at 0.375 mm in-plane (native cohort median) vs the D005-locked 0.5 mm primary; pre-registered V1-V7 verdict comparison criteria; does NOT supersede D005 | sensitivity (cross-stage) | Active |
