# D029 Stage 8 external GE-scanner holdout validation

**Date**: 2026-06-08
**Stage**: validate (stage 8)
**Status**: Active
**Module**: `src/predict/validate/external_holdout.py`, `scripts/09_validate.py`

## Decision

The 4 GE-manufacturer patients (pids 19, 28, 76, 77) excluded from the main
cohort by D004 are processed at stage 8 as an **external descriptive
holdout**. They are NOT used to refit any model; instead, the frozen
production-cohort pipeline is applied to project each holdout patient onto
the locked spatial-only PCA and predict the spatial k=2 phenotype via the
frozen GMM.

Output: `outputs/08_validate/external_holdout_report.csv` with one row per
holdout pid containing:

| Column | Source |
|---|---|
| `pid` | manifest |
| `manufacturer`, `scanner_model`, `kernel` | DICOM headers |
| `agatston_total` | raw Stage 3 feature |
| `n_calcified_arteries` | raw Stage 3 feature |
| `lesion_count_total` | raw Stage 3 feature |
| `spatial_pc1`, `spatial_pc2`, ... | projection onto frozen spatial-only PCA components |
| `predicted_phenotype_raw` | GMM.predict on spatial PC scores; arbitrary {0, 1} |
| `predicted_phenotype` | D023 focal/diffuse mapping applied (via `predict.validate.label_alignment`) |
| `distance_to_focal_centroid` | Euclidean distance in spatial-PC space |
| `distance_to_diffuse_centroid` | Euclidean distance in spatial-PC space |
| `xml_roundtrip_max_pass` | D002 Max-exact gate per holdout pid |

Plus `outputs/08_validate/xml_roundtrip_holdout.csv` (per-ROI D002 audit on
the 4 holdout patients).

## Scope: descriptive only

**N=4 is too small for statistics.** No p-values, no effect sizes, no
inferential claims. The report exists to:

1. Demonstrate the production pipeline is technically reproducible on
   genuinely out-of-distribution scanner data (GE vs Siemens).
2. Surface any catastrophic failure mode (e.g., the spatial PCA projection
   producing wildly out-of-range PC scores that would indicate the holdout
   patients are off-manifold).
3. Provide per-patient evidence the cohort's phenotype labels are
   meaningful in a scanner-shift setting (qualitative only).

## Spec points (D029.1 - D029.4)

### D029.1: GE holdout skips ComBat

ComBat (D019) is fit on the production cohort's kernels `{Qr36d/2, I30f/3}`.
GE-scanner DICOMs use kernels NOT in this set. Applying ComBat to GE data
would require either:

- (a) including a "GE" batch in the ComBat fit (statistically unsupported
  with N=4 patients per kernel);
- (b) treating the GE kernel as one of the Siemens batches (wrong: it is
  not from the same distribution);
- (c) skipping ComBat entirely for the holdout.

**(c) is chosen.** The spatial-only k=2 phenotype (Finding 3) is built
from 13 SPATIAL features (lesion_count_*, n_calcified_arteries,
gini_lesion_volume, dist_from_top_*, inter_lesion_dist_*,
first_to_last_dist_lad, center_of_mass_z). NONE of these features are in
the set of 6 PyRadiomics texture features that ComBat harmonises. The
spatial PCA is therefore projectable onto the GE holdout without any
ComBat dependency.

The trade-off: the holdout report cannot include PyRadiomics texture
summaries. This is acceptable; the phenotype-of-record (D021/Finding 3)
is purely spatial.

### D029.2: ICC gate inherited

The stage-4 ICC gate (D013, threshold 0.75) was estimated on N=422
production patients. Re-estimating ICC on N=4 holdout patients is
meaningless. The holdout inherits the 88-feature gated set; no per-holdout
ICC computation.

### D029.3: D002 XML round-trip applied

The D002 Max-exact round-trip check applies identically to holdout
patients. Failures (per-ROI Max disagreement vs XML's frozen Max field)
are reported in `xml_roundtrip_holdout.csv` and a boolean
`xml_roundtrip_max_pass` per pid is propagated to the main holdout report.

A holdout patient that FAILS D002 is still reported but flagged; this is
descriptive output, not a hard gate.

### D029.4: D023 focal/diffuse mapping applied to predicted phenotype

The raw GMM.predict output is in `{0, 1}` and is arbitrary (label
assignment depends on GMM init). The `predicted_phenotype` column applies
the D023 rule (lower median `n_calcified_arteries` among the production
cohort training labels = focal) so that "focal" and "diffuse" mean the
same thing across the holdout report, the LOO report, and the full-cohort
labels. The raw label is kept in `predicted_phenotype_raw` for audit.

## Out of scope

- Outcomes / MACE association (no outcomes in COCA).
- Demographic adjustment (no demographics in COCA).
- Refitting any model on holdout data.
- Statistical inference (N=4).

## Verified by

- `tests/validate/test_external_holdout.py`:
  - holdout pid set is exactly `{19, 28, 76, 77}`
  - projection no-leakage: PCA components used for projection equal the
    full-cohort frozen PCA on disk (byte-identity)
  - GMM.predict path: predictions in `{0, 1}` for each pid
  - focal/diffuse mapping path: `predicted_phenotype` is consistent with
    `predicted_phenotype_raw` modulo the D023 rule
  - ComBat-skip path: no ComBat columns in the projection feature set
- `tests/validate/test_orchestrator.py`:
  - `external_holdout_report.csv` has exactly 4 rows; required columns
    present
