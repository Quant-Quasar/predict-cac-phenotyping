# Stage 7, analyse (phenotype characterisation)

## What this stage does

Takes the partitions produced by stage 6 (spatial-only k=2 and forced k=3
in the PC space, plus an agatston tertile partition computed at stage 7
entry) and characterises each phenotype cluster with publication-grade
statistical rigour. It does NOT re-cluster, NOT re-select features, and
NOT validate externally; that is the job of stage 6 and stage 8.

The analysis runs in 13 ordered steps in the orchestrator
(`scripts/08_analyse.py`):

  0. Augment raw features with the two D019 derived features
     (`high_density_fraction`, `vessel_burden_gini`) using the
     byte-identical formulas from `predict.reduce.prepare_matrix`.
  1. Label balance check on every spatial_k2 partition (D023).
  2. Focal / diffuse mapping (D023): lower median `n_calcified_arteries`
     = focal.
  3. Biological sanity check (D023): focal median `max_hu_global` must
     be >= 130 HU (the IBSI / Agatston calcium definition). Fail loud
     on a soft-tissue-mask regression.
  4. Per-cohort cluster profiles (D023): median + IQR + Cliff's delta +
     Mann-Whitney + FDR-BH across the 41-feature bundle (28 cross-cohort
     robust + 13 spatial PCA inputs, deduplicated).
  5. Top-N signatures per (cohort x partition x cluster).
  6. Burden orthogonality (D024): Mann-Whitney + Levene + Cliff's delta
     on `agatston_total` between focal and diffuse, with 3-level
     `interpretation` column (orthogonal / marginal / confounded).
  7. Burden-stratified spatial replication (D024 part 2): within each
     Agatston tertile, re-check the 6 directional hypotheses.
  8. Directional hypothesis test (D025): 6 pre-registered one-sided
     Mann-Whitney tests + FDR-BH on the family + two-tier verdict
     (primary 4/6 confirmed; secondary 4/6 direction-match in both strata).
  9. Monotonicity classification (D026): Spearman + Kendall vs
     `agatston_total` for each of the 28 robust features; classify into
     burden_tracking / structure_tracking / spatial_tracking / mixed.
 10. Cross-cohort feature consistency (D027): three-rule criterion
     (direction consistent + significance in >= 2 of 3 + |delta| >= 0.20
     in all 3).
 11. Partition ARI on shared pids (D027 complementary check).
 12. Main paper table (D028: 15 rows = 3 cohorts x 5 phenotype clusters).
 13. Robust sensitivity table (D028: 5 rows, low_burden_flag = False
     subset of the full cohort).
 14. `run_header_analyse.json` with SHA of every seam file consumed.

## Module list

| Module | Concern |
|---|---|
| `predict/analyse/derived_features.py` | Re-derive D019 features (`high_density_fraction`, `vessel_burden_gini`) at raw clinical scale; byte-identical to `predict.reduce.prepare_matrix.compute_*`. |
| `predict/analyse/profiles.py` | Cliff's delta + Mann-Whitney + FDR-BH; biological sanity (130 HU floor); label balance (>15% minority); focal/diffuse mapping rule. |
| `predict/analyse/orthogonality.py` | Burden orthogonality (Mann-Whitney + Levene + Cliff's delta on agatston) with 3-level interpretation; burden-stratified spatial replication. |
| `predict/analyse/hypotheses.py` | Pre-registered 6 directional hypotheses; one-sided Mann-Whitney; primary + secondary + overall verdict. |
| `predict/analyse/monotonicity.py` | Spearman + Kendall vs agatston; burden / structure / spatial / mixed classification. |
| `predict/analyse/signatures.py` | Top-N ranking by FDR-adjusted Cliff's delta per (cohort x partition x cluster). |
| `predict/analyse/cross_cohort.py` | 3-rule feature-level consistency + partition ARI on shared pids. |
| `predict/analyse/paper_table.py` | 15-row main paper table + 5-row robust sensitivity. |
| `scripts/08_analyse.py` | Orchestrator. |
| `scripts/08d_stage7_results.py` | Read-only presentation report. |
| `scripts/08e_verify_focal_diffuse.py` | Independent re-derivation harness for the focal/diffuse chain. |

## Why these design choices

### D023: Cliff's delta over Cohen's d

Cliff's delta is rank-based, dimensionless, and robust to outliers and
skew. Cohen's d on z-scored data is the radiomics-publication convention
but assumes Gaussian inputs. On 93%-sparse LM density-tier bins, Cohen's
d explodes to physically meaningless values, and on heavy-tailed Agatston
scores it under-weights the long tail. Romano 2006 thresholds: 0.147
(small), 0.330 (medium), 0.474 (large). We adopt |delta| >= 0.20 as the
robust-discriminator floor.

### D023: biological sanity = absolute 130 HU floor (not a relative ratio)

The earlier draft used `focal max_hu >= 0.9 * diffuse max_hu`. It fired
on the real cohort (production COCA ratio is ~0.52; focal calcium is
legitimately lower-peak than diffuse). The correct gate is the IBSI
calcium definition: focal cluster median max_hu_global must be >= 130 HU
(the Agatston calcium threshold). A pipeline regression that puts
soft-tissue voxels in the focal cluster would push max_hu to near-zero,
far below 130. The ratio is logged as a WARNING when below 0.5, not
raised.

### D024: 3-level interpretation column on burden orthogonality

A binary pass/fail collapses two qualitatively different scientific
situations: (a) p < 0.05 with |delta| < 0.20 (significant but trivial
effect in a large cohort) is a PASS, but the trivial-effect nature must
be visible; (b) p >= 0.05 with |delta| >= 0.20 (underpowered with
visible effect in a small cohort) is a PASS, but the underpowered-but-
visible nature must be visible. The orthogonal / marginal / confounded
3-level column lets the paper sentence read "orthogonal in cohort A,
marginal in cohort B, confounded in cohort C" cleanly.

### D025: pre-registered 6 directional hypotheses, two-tier verdict

The 6 hypotheses are LOCKED. They were specified in D025 before any
stage 7 numerical result was observed. Primary requires >= 4 of 6
confirmed at FDR p < 0.05 in the full cohort. Secondary requires >= 4
of 6 direction-match in BOTH strata (direction only; significance not
required). The secondary check protects against full-cohort pass driven
by kernel imbalance.

### D026: classification rule uses both rho AND block

Spearman |rho| >= 0.5 -> burden_tracking regardless of block (the
correlation strength dominates). For |rho| < 0.3, the block tag becomes
the tiebreaker (spatial block -> spatial_tracking; hu / texture /
shape -> structure_tracking; otherwise mixed). The intermediate band
0.3 <= |rho| < 0.5 is always mixed. This prevents the procedure from
re-classifying a structurally spatial feature as burden-tracking on
weak-rho evidence.

### D027: three-rule cross-cohort criterion + complementary ARI

ARI alone tests partition identity (same patient -> same cluster across
cohort runs). Feature-level rules test feature discriminator identity
(same feature distinguishes the same way across cohorts). Both belong
in the paper as separate evidence pillars. The three rules together
form a stable replication-grade discriminator definition: direction
consistent (rule 1), significant in at least 2/3 cohorts (rule 2),
moderate effect in ALL 3 cohorts (rule 3, the safety net against
effect-size dilution).

### D028: robust cohort sensitivity as a SEPARATE file

Adding the robust cohort as a 4th row block in the main paper table
would conflate cross-kernel replication (D027) with low-burden
sensitivity (D028). The separate file lets the paper's Methods section
reference it explicitly: "as a sensitivity analysis, we re-ran the
characterisation on the 280 patients with low_burden_flag = False
(Supplementary Table X)".

## Module contracts

```python
# profiles.compute_cluster_profile
def compute_cluster_profile(
    raw_features: pd.DataFrame,
    labels: pd.Series,
    feature_names: list[str],
    cohort: str,
    partition: str,
) -> pd.DataFrame:
    """One row per (cluster, feature). Insufficient_data excluded from FDR-BH."""

# profiles.apply_fdr_bh
def apply_fdr_bh(
    profile_df: pd.DataFrame,
    p_column: str = "mannwhitney_u_pval",
    out_column: str = "fdr_bh_pval",
    delta_threshold: float = 0.20,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """BH adjustment within each (cohort, partition) bundle; adds
    is_robust_discriminator column."""

# orthogonality.assess_burden_orthogonality
def assess_burden_orthogonality(
    agatston_focal: np.ndarray, agatston_diffuse: np.ndarray, cohort: str,
) -> BurdenOrthogonalityResult:
    """Mann-Whitney + Levene + Cliff's delta + 3-level interpretation."""

# hypotheses.directional_hypotheses_table
def directional_hypotheses_table(
    raw_features: pd.DataFrame,
    spatial_labels: pd.Series,
    cohort: str,
    hypotheses: tuple[tuple[str, Direction], ...] = DIRECTIONAL_HYPOTHESES,
) -> pd.DataFrame:
    """One row per hypothesis with one-sided MW + FDR-BH + confirmed."""

# monotonicity.compute_monotonicity
def compute_monotonicity(
    raw_features: pd.DataFrame, feature_names: Iterable[str],
    agatston: pd.Series, cohort: str, block_lookup: dict[str, str],
) -> pd.DataFrame:
    """Spearman + Kendall + 4-class classification per feature."""

# cross_cohort.consistency_table
def consistency_table(
    profiles_by_cohort: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """3-rule criterion per (feature, partition, cluster) tuple."""

# derived_features.augment_raw_with_derived
def augment_raw_with_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add high_density_fraction + vessel_burden_gini columns (byte-
    identical to predict.reduce.prepare_matrix.compute_*) if their input
    columns are present."""
```

## Decisions

- D023 - per-cluster characterisation (Cliff's delta + FDR-BH; 130 HU
  absolute biological sanity floor; 15% minority label balance)
- D024 - burden orthogonality protocol (Mann-Whitney + Levene + delta;
  3-level interpretation; burden-stratified replication)
- D025 - directional hypothesis test (6 pre-registered + two-tier verdict)
- D026 - monotonicity classification (Spearman / Kendall; 4-class rule)
- D027 - cross-cohort feature consistency (3-rule + ARI complementary)
- D028 - low-burden sensitivity (separate output file)

## Tests

`tests/analyse/`, all passing as of Phase B2 + stage 7 close 2026-06-06:

| Module | Tests |
|---|---|
| derived_features (byte-identity vs stage 5) | 16 |
| profiles (Cliff + Mann-Whitney + FDR + sanity + balance + focal map) | 37 |
| orthogonality (3-level interp + Levene + tertile replication) | 16 |
| hypotheses (6 directional + primary + secondary + overall) | 23 |
| monotonicity (Spearman + Kendall + classification + edge cases) | 23 |
| signatures (top-N + alphabetical tiebreak + paragraph formatter) | 13 |
| cross_cohort (3-rule + ARI on shared pids) | 18 |
| paper_table (15 rows + sensitivity + Hennig lookup + tertile balance) | 18 |
| **Total** | **165** |

## Empirical results on the COCA cohort (Phase B2 production, N=420)

### Biological sanity

| Cohort | focal median max_hu | diffuse median max_hu | ratio | gate |
|---|---|---|---|---|
| full | 387 | 744 | 0.52 | PASS (warning) |
| Qr36d/2 | 439 | 765 | 0.57 | PASS (warning) |
| I30f/3 | 370 | 720 | 0.51 | PASS (warning) |

All three exceed the 130 HU calcium floor. The ratio warning fires
(consistent with focal disease being earlier-stage / softer-plaque).

### Burden orthogonality (D024)

| Cohort | n_focal | n_diffuse | focal med agat | diffuse med agat | Cliff's delta | MW p | Levene p | interpretation |
|---|---|---|---|---|---|---|---|---|
| full | 195 | 225 | 30 | 425 | -0.889 | 9.8e-56 | 5.2e-18 | confounded |
| Qr36d/2 | 111 | 109 | 49 | 675 | -0.886 | 7.2e-30 | 6.2e-13 | confounded |
| I30f/3 | 93 | 107 | 26 | 319 | -0.887 | 3.0e-27 | 1.2e-7 | confounded |

The two spatial-only k=2 clusters are **NOT orthogonal to total
calcium burden**. Cliff's delta ~ -0.89 in every cohort.

### Directional hypothesis verdict (D025)

| Hypothesis | full | Qr36d/2 | I30f/3 |
|---|---|---|---|
| `lesion_count_lad` focal > diffuse | REFUTED | REFUTED | REFUTED |
| `n_calcified_arteries` focal < diffuse | confirmed | confirmed | confirmed |
| `dist_from_top_max` focal < diffuse | confirmed | confirmed | confirmed |
| `gini_lesion_volume` focal > diffuse | REFUTED | REFUTED | REFUTED |
| `vessel_burden_gini` focal > diffuse | REFUTED | REFUTED | REFUTED |
| `first_to_last_dist_lad` focal < diffuse | confirmed | confirmed | confirmed |

Primary: 3 of 6 confirmed in full (refuted at < 4).
Secondary: 3 of 6 direction-match in BOTH Qr36d/2 and I30f/3 strata.
**Overall verdict: refuted.**

### Cross-cohort feature consistency (D027)

132 robust discriminators across all 3 cohorts (out of ~570 feature x
partition x cluster comparisons). The high count is dominated by
burden-driven discriminators (the same features that move with agatston
move with the focal/diffuse split).

## Publication-grade interpretation (revised Finding 3)

The Hennig-stable spatial k=2 partition observed in stage 6 is real but
is empirically a **low-burden-vs-high-burden dichotomy**, not a
focal-vs-distributed topology phenotype. Three pieces of evidence:

1. Burden orthogonality = confounded in all 3 cohorts (Cliff's delta on
   agatston = -0.89; Mann-Whitney p < 1e-26; Levene p < 1e-7).
2. Directional hypothesis test = refuted in all 3 cohorts (3 of 6
   pre-registered hypotheses confirmed; the 3 that confirm are
   biology-of-fewer-vessels which is mechanical to lower burden; the
   3 that refute are the concentration-pattern hypotheses that would
   have indicated a true focal-vs-diffuse topology).
3. Cluster biology: focal patients have median 1 calcified artery, 1-2
   lesions total, agatston ~ 30; diffuse have 3 calcified arteries,
   9-13 lesions, agatston ~ 425.

The reproducibility of the partition (Hennig median Jaccard 0.85 to
0.92 across cohorts) is preserved. The partition itself is best
described as a stable burden dichotomy projected through the spatial
subspace.

## Outputs (seam to stage 8)

Written under `outputs/07_analyse/`:

| File | Content |
|---|---|
| `cluster_profiles.csv` | per (cohort x partition x cluster x feature): median, IQR, Cliff's delta, MW p, FDR-BH p, is_robust_discriminator |
| `signature_features.csv` | top-N per cluster from FDR-adjusted Cliff's delta |
| `burden_orthogonality.csv` | per cohort: MW p, Levene p, Cliff's delta on agatston, 3-level interpretation, passes |
| `burden_stratified_spatial.csv` | per (cohort x tertile x feature): direction match within tertile |
| `directional_hypotheses.csv` | per (cohort x hypothesis): focal/diffuse median, observed sign, one-sided MW p, FDR-BH p, confirmed |
| `directional_verdict.json` | primary + secondary + overall verdict |
| `monotonicity_classification.csv` | per (cohort x feature): Spearman, Kendall, classification |
| `monotonicity_summary.csv` | per cohort: count per classification |
| `cross_cohort_feature_consistency.csv` | per (feature x partition x cluster): 3-rule + robust_discriminator |
| `cross_cohort_robust_counts.csv` | aggregated robust discriminator counts |
| `cross_cohort_ari.csv` | partition ARI on shared pids per (partition x stratum) |
| `phenotype_paper_table.csv` | 15-row main paper table |
| `phenotype_paper_table_robust_sensitivity.csv` | 5-row robust sensitivity table |
| `run_header_analyse.json` | git hash + library versions + CLI args + seam SHA per cohort + biological sanity per cohort + verdict |

## How to run

```bash
# Production stage 7 across all 3 cohorts
python scripts/08_analyse.py

# Read-only presentation report
python scripts/08d_stage7_results.py

# Independent verification of the focal/diffuse chain (debugging)
python scripts/08e_verify_focal_diffuse.py
```

Wall-clock at production scale: ~1 second total. Stage 7 has no
clustering, no bootstrapping, no parallelism.

## Known limitations

- The orchestrator reads RAW values from `features.csv`; the two D019
  derived features (`high_density_fraction`, `vessel_burden_gini`) are
  re-derived at raw scale via `augment_raw_with_derived` because they
  do not exist in `features.csv`. Byte-identity to the stage-5 formulas
  is enforced by `tests/analyse/test_derived_features.py`.
- Forced k=3 burden tertiles are computed at stage 7 entry via
  `pd.qcut(agatston_total, q=3)`, NOT taken from the stage-6 forced k=3
  PC-space partition. The qcut tertiles are the conservative,
  distribution-defined choice; the PC-space forced k=3 is reported by
  stage 6 only for ARI cross-algorithm comparison.
- The 13 spatial features used as RAW inputs to the spatial PCA differ
  from the 28 cross-cohort representatives. The 41-feature analysis
  bundle is the union (39 after dedup). One PyRadiomics feature occasionally
  falls out of the per-cluster profile when it has all-NaN values for
  the 22 mask < 14 voxel patients (D010).
- Stage 7 is descriptive, not predictive. No MACE association, no
  demographic adjustment (demographics absent in COCA), no external
  cohort validation. Those are stage 8 (or beyond) concerns.
