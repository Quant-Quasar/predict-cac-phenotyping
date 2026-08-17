# D017 Drop per-vessel sentinel-prone features at stage 5 entry

**Date**: 2026-06-03
**Stage**: reduce (stage 5)
**Status**: Active
**Module**: `scripts/06_reduce.py` (when built; consumes the gated feature list and drops the 13 features below before downstream analysis)

## Decision

Stage 5 reads `outputs/05_icc/gated_features.csv` (88 features that passed the ICC gate) and **excludes 13 per-vessel sentinel-prone features** before constructing the analysis matrix. The remaining **75 features x 422 patients** is the stage 5 input.

Dropped (13):

- `diffusivity_lad`, `diffusivity_rca`, `diffusivity_lcx`, `diffusivity_lm` (4)
- `inter_lesion_dist_mean_{rca, lcx, lm}` (3)
- `inter_lesion_dist_max_{rca, lcx, lm}` (3)
- `first_to_last_dist_{rca, lcx, lm}` (3)

Kept (the three LAD-only distance features are retained despite a 35% sentinel rate; rationale below):

- `inter_lesion_dist_mean_lad`
- `inter_lesion_dist_max_lad`
- `first_to_last_dist_lad`

The full features.csv (184 cols) and the gated set (88 features) are unchanged. The drop happens only when constructing the stage 5 analysis matrix, so the historical record remains intact and the decision is reversible at any later stage.

## Rationale

Empirical audit on the full 444-patient features.csv (`scripts/diagnose_sentinel_features.py`, 2026-06-03):

### 1. Diffusivity is mathematically degenerate

For every vessel:

| feature | corr(value, 1{count==1}) | P((diff==1) == (count==1)) | continuous regime |
|---|---|---|---|
| diffusivity_lad | +0.967 | 1.000 | 64.9% |
| diffusivity_rca | +0.985 | 1.000 | 43.7% |
| diffusivity_lcx | +0.976 | 1.000 | 38.7% |
| diffusivity_lm  | +0.986 | 1.000 | 7.0% |

`diffusivity_{vessel} == 1.0` is identically equal to `lesion_count_{vessel} == 1`. The diffusivity column carries zero additional information over the lesion-count column for the 0-or-1-lesion regime, which covers 35% to 93% of patients depending on vessel. The continuous regime (N >= 2) adds real signal (N divided by first-to-last distance), but the categorical + continuous mixture is bad for any distance-based clustering algorithm and worse for PCA.

### 2. RCA / LCx / LM distance features are sentinel-dominated

Per-vessel sentinel rates for the distance features (`inter_lesion_dist_mean`, `inter_lesion_dist_max`, `first_to_last_dist`):

| vessel | sentinel rate | sd_full / sd_continuous (mean inter-lesion dist) |
|---|---|---|
| LAD | 35.1% | 9.6 / 7.3 |
| RCA | 56.3% | 12.5 / 9.2 |
| LCx | 61.3% | 9.8 / 8.3 |
| LM  | 93.0% | 2.5 / 3.4 |

For RCA/LCx the sentinel rate exceeds 55% and the standard deviation in the continuous regime is materially lower than the full-population standard deviation, indicating the point mass at zero contributes most of the variance. For LM the situation is extreme: 93% of patients are at the sentinel, only 31 patients (7%) carry a continuous value, and the continuous-regime sd exceeds the full-population sd because the 413 zeros pull the full sd downward, making any global statistic meaningless.

A zero-inflated continuous feature with these rates produces a point mass that dominates the first principal component. The first PC then encodes "does this patient have at least 2 lesions in this vessel" instead of any phenotype axis. Standard PCA, k-means, Ward linkage, and Gaussian mixture models all assume continuous distributions and break in characteristic ways on zero-inflated columns (Hartigan and Wong 1979 k-means assumes within-cluster Gaussianity; PCA assumes a continuous covariance structure; both Tobit and Box's analyses warn against using point-mass-inflated columns directly in standard regression).

### 3. LAD distance features are borderline but kept

LAD distance features have a 35.1% sentinel rate, which is still problematic. The argument for keeping them:

- LAD is the most prevalent vessel (88.7% of patients have LAD lesions).
- The continuous regime covers 288 patients with a meaningful inter-lesion-distance distribution; the continuous-regime sd (7.25 mm) is close to the full-population sd (9.59 mm), so the point mass at zero is not the dominant variance contributor.
- Dropping these would leave us with no per-vessel distance information at all (only the global `dist_from_top_*` and `center_of_mass_z` survive at the patient level).
- The 156 LAD-singleton patients align almost exactly with low_burden_flag = True (see section 4 below), and stage 5 will sensitivity-exclude that subset anyway.

Stage 5 must still apply a robust scaling transform (e.g. rank-based, Yeo-Johnson, or median / IQR) to the LAD distance features rather than naive z-score to limit the influence of the residual point mass on PCA.

### 4. The degenerate corner aligns with low burden

Cross-tabulation of "all 4 vessels at diffusivity sentinel" vs `low_burden_flag`:

```
                  low_burden_flag
degenerate_all_4    False    True   All
False                284      38   322
True                  18     104   122
```

P(low_burden | degenerate) = 85.2%. P(degenerate | low_burden) = 73.2%.

Cross-tabulation against Agatston category:

```
                  category
degenerate          1-99   100-399   400+
False                100       102    120
True                 117         4      1
```

122 of 444 patients sit in the degenerate corner; 117 of them are Agatston 1-99 and only 5 are Agatston 100+. The degeneracy concentrates almost entirely in the low-burden / clinically uncertain segment that stage 5 was already planning to sensitivity-exclude. This is a small mercy: the corruption is contained, not spread.

## Alternatives considered

- **Drop only diffusivity (4 features), keep all 12 distance features**. The LM 93% sentinel rate would still produce a PCA artefact; would require documenting and downstream sensitivity-handling. Rejected because LM distance carries essentially no information at this sentinel rate.
- **Drop all 16 per-vessel distance-style features**. Removes residual LAD signal (65% continuous regime is genuinely informative for the high-burden subset). Cleanest matrix but throws away the only per-vessel distance information that survives the sentinel test. Rejected as too aggressive.
- **Hurdle-style decomposition** (binary indicator + continuous component with NaN). Doubles per-vessel distance features from 12 to 24, adds NaN-handling complexity, and the binary indicators are linear functions of `lesion_count_{vessel} >= 2` so they duplicate information already in the count features. Rejected as redundant engineering.
- **Replace with global aggregates only** (e.g. weighted mean across vessels). Loses per-vessel resolution. Rejected because LAD distance has enough density to be useful at the per-vessel level.

## Verified by

- `scripts/diagnose_sentinel_features.py` (read-only) produces all numbers cited above on demand. Rerun anytime after stage 3.
- `outputs/03_features/features.csv` is unchanged; the drop is a stage 5 ingestion-time filter, not a feature recomputation.
- Stage 5 design will include unit tests asserting that the analysis matrix has exactly 75 columns and that none of the 13 dropped feature names are present.
