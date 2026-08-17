# D030 Stage 8 leave-k-out cross-validation for phenotype robustness

**Date**: 2026-06-08
**Stage**: validate (stage 8)
**Status**: Active
**Module**: `src/predict/validate/leave_k_out.py`, `src/predict/validate/label_alignment.py`, `scripts/09_validate.py`

## Decision

The spatial-only k=2 phenotype (Finding 3) is validated via 10-fold
cross-validation with full per-fold pipeline refitting. For each fold:

1. Split N=420 patients into ~378 train and ~42 held out, **kernel-stratified**, seed=42.
2. Refit on training set: D019 (variance / derived / ComBat / YJ / z-score)
   + D022 (multi-block redundancy) + D020 (PCA) + D021 spatial-only PCA +
   GMM k=2 on spatial PC scores.
3. Project the held-out patients' raw features onto the train-fold's
   frozen spatial PCA.
4. Predict held-out phenotype via `GMM.predict` on projected scores.
5. Compute ARI between predicted held-out labels and the **full-cohort
   spatial k=2 labels** (`outputs/06_reduce/cluster_labels_spatial_k2.csv`)
   restricted to that fold's held-out pids.
6. Persist fold-level (median ARI, IQR, per-fold N, class balance).

PASS criterion (D030.4 below) is computed at runtime, not baked.

Output: `outputs/08_validate/leave_k_out_ari.csv` (10 rows + summary row).

## Spec points (D030.1 - D030.5)

### D030.1: Full per-fold refit (rigorous, no leakage)

Every stage 5-6 fit is redone on the training set per fold:

- D019 `prepare_matrix.fit_transform` (variance filter, derived feature
  construction with R² < 0.95 gate, ComBat kernel harmonisation, YJ +
  rank fallback, z-score).
- D022 `multi_block_redundancy` clustering and representative selection.
- D020 PCA (cumulative variance 0.85; D034 PC sign normalisation).
- D021 spatial-only PCA on the 13 fixed spatial features.
- GMM k=2 on spatial PC scores, `random_state=0`, `n_init=10`,
  `covariance_type="full"`.

Held-out patients are then `transform`-only:

- Apply train-fold's YJ lambdas and z-score means/stds.
- Apply train-fold's ComBat batch corrections to the held-out kernel
  (the held-out fold is guaranteed to contain at least one patient per
  kernel under D030.2 stratification).
- Project onto train-fold's spatial PCA components.
- `GMM.predict` to get raw cluster label `in {0, 1}`.

Rejected alternative: "fix representatives + PCA loadings to full-cohort
fits and only refit ComBat / GMM" (~30 s vs ~5 min per fold). Rejected
because the question being validated is robustness of the entire
phenotype-discovery pipeline; fixing reps would beg the question.

### D030.2: Kernel-stratified 10-fold, seed=42

`sklearn.model_selection.StratifiedKFold(n_splits=10, shuffle=True,
random_state=42)` stratified by `kernel` (Qr36d/2 vs I30f/3 for the
N=420 production cohort).

Required because:

- ComBat needs ≥2 samples per batch (D019 invariant).
- Random splits could place all of one kernel into a single fold,
  breaking ComBat on the remaining training folds.

With `n_splits=10` and class proportions ≈ 220/200, each fold has ~22
Qr36d/2 + ~20 I30f/3 held out and ~198 Qr36d/2 + ~180 I30f/3 in
training; both kernels guaranteed in every training and test fold.

### D030.3: ARI reference = full-cohort labels restricted to held-out pids

For each fold, ARI is computed between:

- `predicted_labels`: per-fold GMM.predict output on the held-out pids
  (raw, unmapped, in `{0, 1}`)
- `reference_labels`: full-cohort `cluster_labels_spatial_k2.csv` labels
  restricted to that fold's held-out pids (raw GMM labels from the
  production run on N=420)

Rejected alternative: hold-out-vs-hold-out (each patient predicted once
across all 10 folds, then ARI on the full N=420 prediction vector vs
the full-cohort reference). This is a different question (cross-fold
consistency vs train-vs-test agreement) and is reported as a secondary
metric in the summary row only.

### D030.4: PASS = median ARI ≥ T, T computed at runtime

The PASS threshold T is the 5th percentile of a 10 000-iteration null
simulation:

```
for each fold:
  N_fold = len(held_out_pids)            # ~42
  p_focal = (reference_labels == focal_id).mean()
  # Simulate K=10% per-patient disagreement:
  for _ in range(10000):
    perfect = reference_labels.copy()
    flip_idx = np.random.choice(N_fold, size=int(0.10*N_fold), replace=False)
    perturbed = perfect.copy()
    perturbed[flip_idx] = 1 - perturbed[flip_idx]
    aris.append(adjusted_rand_score(perfect, perturbed))
  T_fold = np.percentile(aris, 5)

T = median(T_fold for fold in folds)     # locked at run time
```

Why empirical, not conventional:

- N=42 per fold means ARI sampling variance is non-trivial; a 0.85
  conventional threshold can reject genuinely stable folds where 2-3 of
  42 patients flip due to model refit noise (not real instability).
- The threshold inherits the same class-imbalance bias as the actual
  per-fold ARI computation.
- K=10% (≤4 of 42 flipping) is the operational definition of "stable"
  for this project; choice pre-committed in this decision doc, not
  data-driven.

T is persisted to `outputs/08_validate/run_header_validate.json` along
with the simulation parameters, so the threshold is reproducible.

### D030.5: D023 focal/diffuse mapping NOT applied before ARI

ARI is permutation-invariant. Applying the focal/diffuse mapping before
computing ARI is redundant and would mask any sign-flip bug in the
mapping itself. The raw GMM labels (in `{0, 1}`) are passed to
`adjusted_rand_score` directly.

The mapping IS applied to all interpretive columns in the per-fold
output (`predicted_phenotype`, per-cluster medians, class balance
columns). The shared helper `predict.validate.label_alignment` enforces
this everywhere.

## Output schema

`outputs/08_validate/leave_k_out_ari.csv` columns:

| Column | Type | Description |
|---|---|---|
| `fold` | int | 0-9 (10 folds) |
| `n_train` | int | training set size |
| `n_test` | int | held-out set size |
| `n_test_qr36d2`, `n_test_i30f3` | int | per-kernel test breakdown |
| `n_test_focal`, `n_test_diffuse` | int | reference-label class breakdown on held-out set |
| `ari` | float | per-fold ARI (raw GMM labels vs full-cohort reference labels, no mapping) |
| `T_fold` | float | per-fold simulated 5th-percentile threshold at K=10% disagreement |
| `pass_fold` | bool | `ari >= T_fold` |
| `n_pyradiomics_dropped` | int | held-out patients dropped from ComBat transform if singleton kernel (should be 0 under D030.2) |

Plus a summary row:

| Column | Value |
|---|---|
| `fold` | "SUMMARY" |
| `ari` | median of per-fold ARI |
| `T_fold` | median of per-fold T (= overall PASS threshold T) |
| `pass_fold` | overall PASS verdict (median ARI ≥ T) |

## Out of scope

- Cross-validating the burden / forced k=3 partitions (D021 found these
  inferior to spatial-only k=2; LOO only validates the locked phenotype).
- Stratification by Agatston tertile (already handled by spatial-only
  feature set per D021).
- Bootstrap CI on ARI (median + IQR is sufficient for 10 folds).

## Verified by

- `tests/validate/test_leave_k_out.py`:
  - kernel-stratified split: every fold contains ≥1 patient of each
    kernel
  - per-fold refit determinism: re-running the same fold twice produces
    byte-identical PCA components on the same machine
  - ARI computation: synthetic test where `predicted == reference` gives
    ARI=1.0 and `predicted == 1 - reference` (perfect inversion) also
    gives ARI=1.0 (permutation invariance)
  - mapping NOT applied to ARI input (regression: if a future contributor
    "fixes" the per-fold prediction with focal/diffuse mapping before
    ARI, this test fails)
  - threshold T_fold computation: synthetic balanced case at K=10%
    reproduces published percentile within ±0.02
  - seed=42 reproducibility: run twice, identical fold assignments
- `tests/validate/test_label_alignment.py`:
  - D023 mapping rule applied correctly: cluster with lower median
    `n_calcified_arteries` mapped to 0 (focal)
  - mapping is idempotent
  - mapping handles ties (defined fallback)
