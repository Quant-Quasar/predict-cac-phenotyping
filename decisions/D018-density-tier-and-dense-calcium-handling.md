# D018 Density-tier retention and dense_calcium_count binarisation

**Date**: 2026-06-03
**Stage**: reduce (stage 5)
**Status**: Active
**Module**: `scripts/06_reduce.py` (when built; consumes the gated feature list and applies the transforms below before downstream analysis)

## Decision

At stage 5 ingestion (after D017's drops), the 17 remaining sparsity-prone features in the canonical bypass list are handled as follows:

1. **All 16 per-vessel density-tier features kept**: `n_rois_d{1..4}_{lad,rca,lcx,lm}`. Stage 5 applies a robust per-column scaling transform (rank-based or median/IQR, not z-score) before PCA so that the zero point mass does not dominate the first principal components.

2. **`dense_calcium_count` is binarised** at stage 5 ingestion to a new feature `has_dense_calcium = (dense_calcium_count > 0)`. The continuous count is dropped from the analysis matrix; the binary indicator replaces it 1-for-1.

The stage 5 analysis matrix size is unchanged at **75 features x 422 patients** (1 column replaced, not removed; density tiers unchanged).

## Rationale

Empirical audit (`scripts/diagnose_density_tier_sparsity.py`, 2026-06-03) on the full 444-patient features.csv shows the two feature classes look superficially similar (both are sparse, low-burden-aligned) but diverge on the three criteria that mattered for D017.

### Density tiers, three reasons to keep all 16

#### 1. No mathematical degeneracy

Unlike diffusivity (D017, where `(diffusivity == 1.0) == (lesion_count == 1)` exactly), the density tier counts are independent integer features with clinical semantics. `n_rois_d2_lad = 3` says exactly "3 ROIs in Agatston tier 2 (200-300 HU) on LAD", and that interpretation does not collapse to any single-bit indicator already present in the schema.

#### 2. Variance is not point-mass-dominated

The diagnostic measured `sd_nonzero / sd_full` across all 16 bins. For D017's distance features the ratio ran 0.26 to 1.00, meaning zeros frequently dominated the variance. For density tiers it runs **0.84 to 1.48**, mostly at or above 1.0:

| feature | sd_full | sd_nonzero | ratio |
|---|---|---|---|
| n_rois_d1_lad | 1.733 | 1.740 | 1.00 |
| n_rois_d2_lad | 1.476 | 1.379 | 0.93 |
| n_rois_d3_lad | 1.202 | 1.171 | 0.97 |
| n_rois_d4_lad | 2.437 | 2.600 | 1.07 |
| n_rois_d1_rca | 1.723 | 1.833 | 1.06 |
| ... | ... | ... | ... |
| n_rois_d4_lcx | 2.085 | 3.092 | 1.48 |

A ratio at or above 1.0 means that when we restrict to nonzero rows the standard deviation is **higher** than over the full population. The reason: adding zeros pulls the mean down, which inflates the residual at the nonzero rows for fixed sd, so sd_full underestimates the dispersion that PCA actually sees in the nonzero rows. The zero rows contribute almost no variance because they all sit at the same value. So the PCA signal in these columns comes from the nonzero distribution, not from a sentinel split.

#### 3. Sparsity is biological and clinically interpretable

The 16 bins encode the 4 Agatston density factors (130-200, 200-300, 300-400, >400 HU) across 4 vessels. The Agatston density factors are the historical basis of CAC scoring and the clinical interpretation of these counts is direct. Per-patient sparsity (mean 5.7, median 5 populated bins of 16) is what we expect for a clinically realistic cohort: not every patient has calcium across all vessels and all density tiers; the distribution of which bins are populated is itself a phenotype.

Cross-tab with low burden:

```
                  low_burden_flag
<=1_populated_bin   False  True   All
False                297     70   367
True                   5     72    77
```

P(low_burden | <=1 bin) = 93.5%. 75 of 77 sparse-tier patients are Agatston 1-99; none are 400+. The sparsity concentrates in the same low-burden segment that stage 5 will sensitivity-exclude anyway, so the contaminating effect on cluster discovery is bounded.

### dense_calcium_count, three reasons to binarise

#### 1. Effective binarity

- 89.9% of patients have `dense_calcium_count = 0`
- 31 of the 45 nonzero patients have value exactly 1
- The remaining 14 patients are spread across values 2, 3, 4, 5, 6, 7, 10 with at most 5 patients at any value
- `sd_nonzero / sd_full = 2.24`, the highest ratio in the audit and a clean indicator that the nonzero distribution is qualitatively different from the zero majority

#### 2. The count semantics are weak

The clinical justification for `dense_calcium_count` is Hoori 2024's "stable plaque marker": calcium denser than 1000 HU correlates with more stable plaque morphology. The binary "has any dense calcium" carries this clinical signal directly. The exact count (1 vs 3 vs 10) has no published clinical interpretation and the long tail in our cohort is sparse to the point of being uninformative (single patients at 4, 5, 6, 7, 10 HU above 1000).

#### 3. Binarisation preserves the gate test

`has_dense_calcium = (dense_calcium_count > 0)` is a strict function of the original column. The ICC gate already passed `dense_calcium_count` with ICC = 1.0 (canonical bypass, D016), and a strict function of an ICC=1.0 column is also ICC=1.0. So the binarisation does not re-open the stability question.

## Stage 5 scaling note

For the 16 density tier features (and for the 3 LAD distance features kept under D017), stage 5 must apply a robust scaling transform before PCA. Standard z-score scaling is inadequate because the zero rows would shift the column mean to a non-meaningful value and the column sd would be inflated by the zero-vs-nonzero split, producing a first principal component that captures "bin populated yes/no" instead of phenotype.

Two acceptable transforms (final choice deferred to the stage 5 design pass):

- **Rank transform** (per-column): replace each value by its rank divided by N. Removes the point mass entirely and produces a uniform marginal that PCA handles cleanly. Loses the absolute count magnitude.
- **Median / IQR scaling**: subtract the column median, divide by the IQR. Less aggressive than rank; preserves count magnitudes for the nonzero subset; handles the zero point mass robustly.

The binary `has_dense_calcium` column does not need this treatment; it goes into PCA as is.

## Alternatives considered

- **Drop the 4 LM density tier bins (83-90% zero)**. LM is the rarest vessel and its bins are the most sparse. Rejected because LM tier counts still pass criterion 2 (variance not point-mass-dominated) and removing the LM dimension entirely would prevent any LM-specific phenotype discovery in stage 6.
- **Drop all 16 density tier features**. Rejected: removes a clinically interpretable feature class that mirrors the Agatston density-factor decomposition. Would leave only `agatston_{vessel}` totals as the per-vessel density information, losing the within-vessel HU distribution.
- **Keep dense_calcium_count as continuous with robust scaling**. Rejected because the count semantics are weak past 1 and the tail is single-patient-dominated.
- **Drop dense_calcium_count entirely**. Rejected because the binary "has dense calcium" carries real Hoori 2024 signal and binarising is a cheap way to keep it.

## Verified by

- `scripts/diagnose_density_tier_sparsity.py` (read-only). Reproduces all numbers cited above on demand.
- Stage 5 design will include tests asserting: the analysis matrix has exactly 75 columns; `dense_calcium_count` is absent; `has_dense_calcium` is present and is integer-valued in {0, 1}; the 16 density tier columns are present and have the original numeric values.
