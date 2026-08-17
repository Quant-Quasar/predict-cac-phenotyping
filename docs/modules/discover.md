# Stage 6, discover (cluster discovery, validity, sensitivity)

## What this stage does

Reads the seam files produced by `scripts/06_reduce.py` and runs all of D021
on the PC matrix:

1. **Hopkins clusterability** on `X_full` (verdict: clustered, random,
   ambiguous).
2. **Gap statistic** on 3 algorithms x 3 feature spaces, 500 bootstraps each,
   with the Maitra-Ramler 2010 1-SE-from-argmax k-selection rule.
   - feature spaces: full PC matrix; burden-residualised PC matrix (against
     log(agatston_total + 1)); spatial-only feature subset (re-PCA'd from
     the 13 spatial features in `prepared_matrix.csv`).
   - algorithms: k-means, Ward, Gaussian mixture.
3. **Monti consensus** at gap-selected k for each algorithm (100 subsamples,
   80% fraction).
4. **Forced k=3 characterisation**: cluster sizes per algorithm + crosstabs
   against burden tertile, kernel, low_burden_flag, category. Descriptive
   only, NOT a discovery claim.
5. **Validity checks**:
   - Kernel chi-square confounder test (HARD FAIL at p < 0.05).
   - Hennig clusterboot bootstrap stability at forced k=3 (median Jaccard
     per cluster).
   - Hennig stability on spatial-only x GMM x k=2 (the Finding-3 phenotype
     probe).
6. **ARI cross-algorithm** at forced k=3.

Split out from the reduce stage on 2026-06-05 (Phase B). The script seam is
`outputs/06_reduce/{pca_scores.npy, prepared_matrix.csv, cohort_metadata.csv}`
to `scripts/07_discover.py`.

## Module list

- `src/predict/discover/clusterability.py`  Hopkins statistic with pyclustertend
  k=2-for-both convention; `assess_clusterability` returns a verdict.
- `src/predict/discover/cluster_discovery.py`  `fit_cluster` (kmeans / ward / gmm),
  `gap_statistic` with 1-SE-from-argmax k selection, `monti_consensus`,
  `burden_residualise`, within-cluster dispersion `W`. joblib-parallel in
  the gap and consensus loops.
- `src/predict/discover/validity.py`  `kernel_chi_square`, `hennig_clusterboot`,
  `ari`, `ari_on_shared_pids`.
- `scripts/07_discover.py`  the orchestrator. Reads the seam, runs all of D021,
  writes outputs back into the same cohort directory.

## Why these design choices

### Hopkins with k=2 for both queries (pyclustertend convention)

Hopkins compares nearest-neighbour distances on the real data to those of
uniform-random points in the bounding box. Two failure modes I hit and fixed:

- **Wrong**: plain distances with k=1 for uniform queries and k=2 for real
  queries. Gives H near 0 on uniform random data because the uniform point
  finds an accidentally-close real point with k=1 while the real point
  necessarily skips self with k=2.
- **Wrong**: d-th power distances per Hopkins 1954. At d >= 5 the
  finite-sample asymmetry blows up; H collapses to 0 or 1.
- **Right**: plain distances with k=2 for BOTH queries. Real query skips self
  at distance 0; uniform query skips its accidentally-close 1-NN. H near 0.5
  on uniform random data, H >> 0.5 on clusters. Tests pin H_uniform in
  [0.4, 0.6] and H_clustered > 0.7.

### Gap statistic with 1-SE-from-argmax (Maitra-Ramler 2010), NOT Tibshirani 2001

Tibshirani's classic rule "smallest k where gap(k) >= gap(k+1) - sk(k+1)"
fails on data with strong separation: at low k the gap is negative (within-
cluster dispersion exceeds uniform reference). gap(1) and gap(2) are both
negative and roughly equal, so the rule picks k=1 even when the true elbow
is at k=3. The R factoextra default (Maitra-Ramler 1-SE-from-argmax) is
robust to this; it picks k=true_k on clustered data and k=1 on uniform
random (whole curve flat within 1 SE).

### k_range extended to 1..12

Initial smoke runs at k=1..8 had 7 of 9 algorithm x space combinations
selecting k=8 (the boundary). Extending to k=1..12 still has selected_k
in [9, 12] on every cohort. The gap curve is monotonic with no plateau
through k=12. This is the gap-statistic signature for a continuum, not
a discrete-phenotype structure.

### Algorithms (kmeans + Ward + GMM, no spectral)

Spectral clustering was in the v1 design but requires a similarity-kernel
bandwidth that is not theoretically justifiable on this feature space.
v2 D021 drops spectral and uses three algorithms whose hyperparameters are
unambiguous (just k).

### Spatial-only k=2 as the Finding-3 probe

After observing the gap-statistic continuum on the full PC space, the spatial-
only subspace (13 features describing per-vessel lesion count and inter-lesion
geometry) showed a different signal at GMM k=2: two robust modes with Hennig
median Jaccard 0.85 to 0.92 across all three cohorts (full, Qr36d/2, I30f/3).
This is the focal-vs-diffuse spatial topology phenotype, identified
independently of total burden. Each 07_discover.py run writes the spatial
GMM k=2 labels to `cluster_labels_spatial_k2.csv` for downstream stage 7.

### Forced k=3 is descriptive, NOT discovery

Forced k=3 partitions are useful for sanity (e.g., kernel chi-square at the
same k across algorithms, ARI between algorithms at a common k). They are
NOT a phenotype claim, since the gap statistic does not select k=3 on any
cohort. The script's forced_k_crosstabs.csv exists to characterise the
forced partition against burden / kernel / category covariates, not to
support phenotype existence.

## Module contracts

```python
# clusterability.assess_clusterability
def assess_clusterability(
    X: np.ndarray,
    sample_frac: float = 0.10,
    threshold: float = 0.65,
    ambiguous_band: tuple[float, float] = (0.55, 0.65),
    random_state: int = 42,
) -> HopkinsResult:
    """Returns H, verdict in {clustered, ambiguous, random}, sample size, seed."""

# cluster_discovery.gap_statistic
def gap_statistic(
    X: np.ndarray,
    algorithm: ClusterAlgorithm,
    k_range: tuple[int, ...] = (1, 2, ..., 12),
    n_bootstrap: int = 500,
    null_reference: str = "pca_bounding_box",
    random_state: int = 42,
    n_jobs: int = 1,
) -> GapStatisticResult:
    """Returns log_Wk_observed, log_Wk_ref_mean/std, gap_values, sk_values,
    selected_k via Maitra-Ramler 1-SE-from-argmax."""

# cluster_discovery.monti_consensus
def monti_consensus(
    X: np.ndarray,
    k: int,
    algorithm: ClusterAlgorithm,
    n_subsamples: int = 100,
    subsample_frac: float = 0.80,
    random_state: int = 42,
    n_jobs: int = 1,
) -> ConsensusResult:
    """Returns (n, n) consensus matrix + PAC score."""

# validity.hennig_clusterboot
def hennig_clusterboot(
    X: np.ndarray, labels: np.ndarray, k: int, algorithm: ClusterAlgorithm,
    n_bootstrap: int = 100, threshold: float = 0.75, random_state: int = 42,
    n_jobs: int = 1,
) -> HennigStabilityResult:
    """Returns per-cluster jaccard_median, jaccard_mean, stable_mask."""
```

## Decisions

- D021 - cluster discovery, validity, and sensitivity cohorts (stage tag
  changed from `reduce` to `discover` on 2026-06-05 Phase B; decision
  number unchanged).

## Tests

`tests/discover/`, all passing (as of Phase B2 close, 2026-06-05):

| Module | Test file | Count |
|---|---|---|
| clusterability | `test_clusterability.py` | 20 |
| cluster_discovery (gap, monti, fit_cluster, burden_residualise, W) | `test_cluster_discovery.py` | 38 |
| validity (chi-square, Hennig, ARI, Jaccard) | `test_validity.py` | 31 |
| **Total** | | **89** |

Key tests: parallelism byte-identity (gap, monti, Hennig produce identical
results at n_jobs=1 vs n_jobs=4), PC sign invariance (already in pca tests
but the seam relies on it), Hopkins H in [0.4, 0.6] on uniform random and
> 0.7 on three-blob synthetic.

## Empirical results (full cohort, N=420, Phase-B production rerun)

Numerical reproducibility: `X_full` (loaded from `pca_scores.npy`) has
sha256[:16] = `00c2b4ee8a9df7b1`. All three cohort numbers below come from
identical-seed, full-bootstrap (500/100/100) runs.

### Hopkins

| Cohort | H (range observed across reruns) | Verdict |
|---|---|---|
| full (N=420) | ~0.72 (0.717-0.720) | clustered |
| Qr36d/2 (N=220) | 0.64-0.70 | clustered / ambiguous (boundary) |
| I30f/3 (N=200) | 0.73-0.75 | clustered |

Full and I30f/3 land consistently in the clustered band (> 0.65). Qr36d/2
is the sensitive stratum (smaller N=220, smaller signal margin) and has
been observed at both 0.700 (original Phase-B run) and 0.642 (rebuilt-
machine rerun 2026-06-08). The Hopkins point estimate is therefore
treated as a Methods-section robustness check, not a published headline
number. The locked cluster-tendency claim is the spatial-only k=2 Hennig
median Jaccard (0.85-0.92 across cohorts), which replicates across reruns.

### Gap statistic

All 27 combinations (3 cohorts x 3 algorithms x 3 spaces) select selected_k
in [7, 12]. No elbow at any k. Gap curves monotonically non-decreasing
through k=12. This is the locked signature for a continuum (no discrete
phenotype structure).

### Kernel chi-square at forced k=3 (full cohort)

- kmeans p = 0.048 (fails)
- ward p ~ 0.02 band
- gmm p ~ 0.03 band

All three fail at p < 0.05. Combined with post-ComBat texture R^2 < 0.0025
(D019 audit), the chi-square failure is attributed to patient-population
bias by scanner subcohort, NOT residual technical bias.

### Hennig spatial-only x GMM x k=2 (the Finding-3 phenotype)

| Cohort | cluster 0 median Jaccard | cluster 1 median Jaccard | both stable? |
|---|---|---|---|
| full | 0.880 | 0.866 | yes |
| Qr36d/2 | 0.917 | 0.919 | yes |
| I30f/3 | 0.853 | 0.861 | yes |

All 6 cluster-medians above the 0.75 stable threshold. Replicates within
each kernel stratum independently, ruling out scanner artefact.

## Outputs

Written back into `--cohort-dir` alongside the reduce-stage outputs:

| File | Content |
|---|---|
| `hopkins.json` | H, verdict, sample size, seed |
| `gap_statistic.json` | full gap curves per (space, algo) |
| `gap_statistic_summary.csv` | selected k per (space, algo), plus gap values |
| `consensus_matrices.npz` | per-algorithm (n, n) consensus matrices |
| `consensus_summary.json` | PAC score per algorithm |
| `forced_k_characterisation.json` | cluster sizes per algorithm at forced k |
| `cluster_labels_forced.csv` | per-pid labels per algorithm at forced k |
| `cluster_labels_spatial_k2.csv` | per-pid GMM k=2 labels on spatial-only |
| `forced_k_crosstabs.csv` | cluster x covariate counts |
| `validity_checks.csv` | chi-square + Hennig (forced k AND spatial k=2) |
| `ari_across_algorithms.csv` | pairwise ARI at forced k=3 |
| `run_header_discover.json` | git hash, library versions, CLI args, SHA of seam files |

## How to run

```bash
# Production discovery on the full cohort
python scripts/07_discover.py --cohort-dir outputs/06_reduce/ --n-jobs 80

# Kernel-stratified reruns (matching the strata produced by 06_reduce.py)
python scripts/07_discover.py --cohort-dir outputs/06_reduce/stratified_Qr36d_2/ --n-jobs 80
python scripts/07_discover.py --cohort-dir outputs/06_reduce/stratified_I30f_3/ --n-jobs 80

# Smoke (small bootstraps for ~2 min runtime; numerically noisy but
# verdicts should be consistent)
python scripts/07_discover.py --cohort-dir outputs/06_reduce/ \
    --gap-bootstraps 20 --consensus-subsamples 20 --hennig-bootstraps 20 \
    --n-jobs 80
```

Wall-clock at production parameters: ~3-5 min per cohort on the 80-core
remote box.

## Known limitations

- Forced k=3 is descriptive characterisation only; do NOT cite it as a
  discovery claim. The gap statistic explicitly does not support k=3.
- The Hopkins verdict is "clustered" because H > 0.65, but a clustered
  verdict combined with gap-curve monotonicity to k=12 is the empirical
  signature of a continuum (cluster tendency dominated by burden as the
  principal axis), NOT of discrete phenotypes. Reading Hopkins alone
  without the gap context would be misleading.
- The spatial-only Finding-3 partition uses GMM-specific k=2. It is robust
  across cohorts at this k, but the published paper should clarify that
  k=2 was identified by gap-statistic ambiguity on the spatial subspace
  plus Hennig stability, not by gap-elbow selection.
- Numerical reproducibility note (updated 2026-06-08 after machine rebuild):
  Hopkins shows machine-dependent drift driven by BLAS / numpy / scipy
  stack differences. Observed values across reruns:
    full     0.717 to 0.720  (no threshold drift; consistently clustered)
    Qr36d/2  0.642 to 0.700  (sensitive stratum; crosses 0.65 boundary)
    I30f/3   0.728 to 0.745  (consistently clustered)
  The drift originates from `pca_scores.npy` float64 bits being machine-
  dependent (sha256[:16] differs across rebuilds even with identical
  input data); Hopkins on nearest-neighbour distances is sensitive to
  these bit-level differences in the smaller stratified cohorts. The
  paper should cite the spatial-only k=2 Hennig medians (0.85-0.92,
  range-stable across reruns) as the cluster-tendency headline, and
  treat Hopkins as a supporting range, not a point estimate. All
  verdict-level findings (gap monotonic, kernel chi-square fails, spatial
  k=2 stable, D024 confounded, D025 refuted, 132 D027 robust
  discriminators) are unchanged across the rebuild.
