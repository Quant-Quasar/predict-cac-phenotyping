# Stage 5, reduce (matrix preparation, redundancy clustering, PCA)

## What this stage does

Transforms the 88 stage-4-gated features (per D013/D016) into a low-dimensional
PC matrix on the 420 ComBat-eligible patients, in three steps:

1. **D019 matrix preparation**: D017 sentinel drops (88 to 75), D018 dense-calcium
   binarisation, conditional derived features (`high_density_fraction`,
   `vessel_burden_gini`) with an R^2 < 0.95 non-redundancy gate, variance filter
   (sd >= 0.01), ComBat on the 6 PyRadiomics texture columns with kernel
   covariate (target post-ComBat scanner R^2 < 0.02), Yeo-Johnson with
   rank-transform fallback on 19 sparse columns, global z-score.
2. **D020 / D022 redundancy clustering**: Spearman r^2 > 0.75 clustering. The
   primary path (D022) runs multi-block over 6 prospectively-defined blocks
   (burden / HU statistics / density tier / spatial / texture / shape). The
   single-matrix variant (D020) is preserved as a sensitivity probe via
   `--block-mode single`. Representatives are chosen per cluster by ICC-first,
   canonical-over-PyRadiomics, alphabetical tiebreak.
3. **D020 PCA**: sklearn `PCA` (`svd_solver='full'`) on the representatives,
   sign-normalised so the dominant-loading feature is non-negative per PC,
   retained at 0.85 cumulative variance.

Cluster discovery, gap statistic, consensus clustering, forced-k characterisation,
and Hennig validity were in this script until 2026-06-05. They moved to the
**discover** stage during Phase B; see `docs/modules/discover.md`.

## Module list

- `src/predict/reduce/prepare_matrix.py`  D019 pipeline: drops, derived
  features, variance filter, ComBat, Yeo-Johnson + rank fallback, z-score.
  `MatrixPrepLog` records every step for audit.
- `src/predict/reduce/redundancy.py`  D020 single-matrix and D022 multi-block
  Spearman r^2 clustering, representative selection. 6 prospective
  `FeatureBlock` partitions in `DEFAULT_BLOCKS`.
- `src/predict/reduce/pca.py`  PCA with sign normalisation (`normalise_pc_signs`),
  family-tagged loadings table, `PcaResult` dataclass.
- `scripts/06_reduce.py`  the orchestrator. Reads stage 3 features + stage 4
  ICC, runs D019/D020/D022/PCA, writes the seam files for `07_discover.py`.

## Why these design choices

### D019 ComBat on texture only (not on canonical features)

- Canonical features (Agatston, mass, volume, HU stats, density tiers, spatial,
  count, shape) read from XML's frozen `Max`/`Mean` plus the polygon vertices.
  No CT array indexing, so kernel cannot bias them. ComBat would be a no-op
  with overhead.
- The 6 PyRadiomics texture survivors (glrlm, gldm, glszm, firstorder Range) DO
  index into the CT array and DO exhibit pre-ComBat scanner R^2 of 0.002 to
  0.024. ComBat reduces this to 0.0001 to 0.0024 (well below the 0.02 cut).

### D019 Yeo-Johnson with rank fallback

- 35-93% sentinel rates on LM density-tier bins make a pure z-score produce a
  non-Gaussian column that PCA misinterprets.
- Pure rank transform is too aggressive: it discards count magnitude in the
  continuous regime.
- Yeo-Johnson with per-column post-transform skewness check (abs > 1.0
  triggers rank fallback) gets the best of both. 15 of 19 sparse columns
  succeed under YJ; 4 fall back to rank (all LM density tiers).

### D022 multi-block over D020 single-matrix

The single-matrix Spearman cluster produces 23 representatives. The composition
is the problem, not the count: all 6 PyRadiomics texture features and most
spatial features are absorbed by the burden cluster (size 29 includes mass_*,
volume_*, glrlm/gldm/glszm). 16 of 23 reps end up burden-flavoured.

The multi-block partition runs Spearman clustering INDEPENDENTLY within 6
prospective blocks (burden, hu_statistics, density_tier, spatial, texture,
shape). The block boundaries are defined a priori from PyRadiomics IBSI
taxonomy, canonical feature provenance, and Lin 2022 / Hoori 2024 / v1
hypotheses, NOT from any stage-5 finding. Result on the full cohort: 29
representatives, balanced (4 HU / 2 burden / 1 count / 1 dense / 14 density-tier
/ 1 derived-density / 3 spatial / 2 shape / 2 texture). Cross-cohort
intersection rises from 12 (single-matrix) to 28 (multi-block).

The single-matrix variant is retained as a sensitivity probe via
`--block-mode single`. The composition argument, not just the rep count, is
the reason multi-block is primary.

### D020 PC sign normalisation

PCA eigenvectors are sign-arbitrary. Without normalisation, "high PC1" can
mean opposite things on two runs of the same code on the same data, and
downstream clustering then assigns opposite labels. The D034-from-v1
convention: for each PC, flip the sign so the feature with the largest
absolute loading has a non-negative loading. Tested by
`test_fit_pca_sign_invariant_under_input_sign_flip`.

### Single-kernel cohort exclusion (ComBat prerequisite)

COCA has 4 kernels: Qr36d/2 (220), I30f/3 (200), B35f (1), I36f/3 (1).
neuroCombat with a singleton kernel produces NaN because within-batch variance
is undefined. The orchestrator filters to {Qr36d/2, I30f/3} before matrix
prep. Stage 5 analysis cohort is therefore N=420, not 422. Defense-in-depth
assertion in `combat_harmonise` raises if any singleton slips through.

## Module contracts

```python
# prepare_matrix.run_matrix_prep
def run_matrix_prep(
    df: pd.DataFrame,              # eligible cohort, pid + metadata + features
    feature_cols: list[str],       # gated 88 from stage 4
    variance_threshold: float = 0.01,
    yj_skew_threshold: float = 1.0,
    combat_max_post_r2: float = 0.02,
    add_derived: bool = True,
) -> tuple[pd.DataFrame, list[str], MatrixPrepLog]:
    """Returns prepared dataframe, list of surviving feature columns, audit log."""

# redundancy.run_multi_block_redundancy_clustering
def run_multi_block_redundancy_clustering(
    df: pd.DataFrame,
    feature_cols: list[str],
    icc_lookup: dict[str, IccInfo],
    canonical_set: set[str],
    blocks: tuple[FeatureBlock, ...] = DEFAULT_BLOCKS,
    primary_method: LinkageMethod = "average",
    sensitivity_methods: tuple[LinkageMethod, ...] = (),
    min_gap: float = 0.05,
    fallback_distance: float = 0.25,
) -> MultiBlockResult:
    """Returns representatives (union over blocks), per-block clustering, audit."""

# pca.fit_pca
def fit_pca(
    df: pd.DataFrame,
    feature_cols: list[str],       # the representatives
    cumvar_threshold: float = 0.85,
    random_state: int = 42,
) -> PcaResult:
    """Returns components, scores (N x n_retain), pid_order, sign-normalised."""
```

## Decisions

- D017 - sentinel-prone per-vessel feature exclusion (88 to 75 at stage 5 ingestion)
- D018 - density tier and dense_calcium binarisation
- D019 - preprocessing pipeline (variance filter, ComBat on 6 texture, YJ + rank)
- D020 - redundancy clustering and PCA
- D022 - multi-block redundancy clustering (primary, supersedes single-matrix portion of D020)

## Tests

`tests/reduce/`, all passing (as of Phase B2 close, 2026-06-05):

| Module | Test file | Count |
|---|---|---|
| prepare_matrix | `test_prepare_matrix.py` | 57 |
| redundancy (single + multi-block) | `test_redundancy.py` | 53 |
| pca | `test_pca.py` | 65 |
| **Total** | | **175** |

(Plus 89 tests under `tests/discover/` for the cluster-discovery half, covered
in `docs/modules/discover.md`.)

## Empirical results (full cohort, N=420, Phase-B production rerun)

- D019: 88 to 77 features. Variance filter dropped 0. Derived features both
  accepted (high_density_fraction R^2 = 0.72, vessel_burden_gini R^2 = 0.86).
- ComBat audit on the 6 PyRadiomics textures: post-correction kernel R^2 of
  0.0001 to 0.0024 on every column (pre-correction 0.002 to 0.024).
- Yeo-Johnson + rank fallback: 15 of 19 sparse columns YJ-transformed cleanly;
  4 LM density tiers fell back to rank.
- D022 multi-block: 77 to 29 representatives across 6 non-empty blocks
  (burden 15 to 2, hu_statistics 11 to 4, density_tier 18 to 16, spatial
  14 to 3, texture 5 to 2, shape 14 to 2).
- D020 single-matrix sensitivity: 77 to 23 representatives. Burden cluster of
  size 29 absorbs all 6 PyRadiomics texture features and most spatial features.
- PCA: n_retain = 13 at 85% cumvar.
- Numerical reproducibility: `pca.scores` sha256[:16] = `00c2b4ee8a9df7b1`.
  The seam file `pca_scores.npy` is byte-exact to the in-memory `pca.scores`.

## Outputs (seam to discover stage)

Written under `outputs/06_reduce/` (full cohort) or
`outputs/06_reduce/stratified_<kernel>/` (kernel-stratified sensitivity):

| File | Content |
|---|---|
| `pca_scores.npy` + `pca_scores_pid_order.csv` | N x n_retain PC matrix, byte-exact seam |
| `pca_scores.csv` | same data as the NPY, lossy CSV form for human inspection |
| `prepared_matrix.csv` | post-D019 features (pid + metadata + 77 transformed) |
| `cohort_metadata.csv` | RAW kernel, agatston_total, low_burden_flag, category |
| `representative_features.csv` | 29 multi-block representatives (full cohort) |
| `multi_block_assignments.csv` | per-feature: block, cluster id, is_representative, decided_by |
| `multi_block_summary.json` | per-block cluster count + cut threshold + reps |
| `pca_explained_variance.csv` | per-PC variance and cumulative variance |
| `pca_top_loadings.csv` | top 10 absolute loadings per PC with family tag |
| `pca_loadings.csv` | full per-feature loadings |
| `pc_agatston_correlation.csv` | Spearman rho between each PC and agatston_total |
| `combat_audit.csv` | pre / post kernel R^2 on each ComBat target |
| `matrix_prep_log.json` | D019 step-by-step audit |
| `run_header.json` | git hash, library versions, CLI args, timestamp |

## How to run

```bash
# Production full cohort (~1 min)
python scripts/06_reduce.py --n-jobs 80

# D022 sensitivity probe (single-matrix Spearman)
python scripts/06_reduce.py --block-mode single --n-jobs 80

# Kernel-stratified reruns (D021 sensitivity cohorts)
python scripts/06_reduce.py --kernel-filter "Qr36d/2" --n-jobs 80
python scripts/06_reduce.py --kernel-filter "I30f/3" --n-jobs 80
```

Each command writes the seam files for the downstream `07_discover.py`.

## Known limitations

- ComBat fits on 2 kernels with N=220 / 200. Convergence is fine but
  N=200 is on the lower end of ComBat's well-tested regime. The post-correction
  R^2 < 0.0025 confirms it worked, but the theoretical justification weakens
  on smaller batches.
- PCA at 0.85 cumvar gives n_retain = 13 on the full cohort, but this varies
  by stratum. Downstream comparisons across cohorts must be on PC scores, not
  on PC component identities.
- The composition argument for multi-block (D022) is prospective but somewhat
  judgement-based. Sensitivity vs `--block-mode single` is the empirical
  defence.
