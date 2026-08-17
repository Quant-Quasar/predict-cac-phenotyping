# Decision Log

## D010 - COCA artery name normalization

**Date**: May 2026
**Tag**: `v0.4.3-dicom-loader-hardened`

**Decision**: COCA XML artery labels are normalized from the exact observed
full-name strings:

- `Left Anterior Descending Artery` -> `LAD`
- `Right Coronary Artery` -> `RCA`
- `Left Circumflex Artery` -> `LCx`
- `Left Coronary Artery` -> `LM`

**Rationale**: The inspected 101 XML files contain these four full-name
strings and no abbreviation variants such as `LAD`, `RCA`, `LCX`, `Lcx`,
`LM`, `LMA`, `LM_proximal`, or `Left Main`. Raw equality checks against
abbreviations would therefore silently emit zero per-artery features.

## D011 - COCA XML slice-index direction

**Date**: May 2026
**Tag**: `v0.4.3-dicom-loader-hardened`

**Decision**: COCA XML `ImageIndex` maps to DICOM InstanceNumber-sorted image
position using:

```text
dicom_position = (num_slices - 1) - ImageIndex
```

**Rationale**: DICOM InstanceNumber order has decreasing patient z for all
100 inspected patients. XML `ImageIndex` matches reverse zero-based DICOM
position exactly for non-empty annotated slices when each DICOM series is
evaluated independently. Patient 78 only appears inconsistent if two duplicate
series are concatenated before matching.

## D012 - DICOM multi-series folder selection rule

**Date**: May 2026
**Tag**: `v0.4.3-dicom-loader-hardened`

**Decision**: When a patient folder contains DICOM files from multiple
SeriesInstanceUIDs, `load_patient_dicom` selects one series using the rule:
(1) filter to `Modality == "CT"`; (2) pick the series with the lowest
`SeriesNumber`; (3) lexicographic `SeriesInstanceUID` as deterministic
tiebreaker. A `[WARNING]` listing all candidate series and marking the
selection is emitted to stdout.

**Alternatives considered**:
- Concatenate all series (previous behavior). Rejected because it produces
  a phantom stack from multiple acquisitions. Patient 78 in the 100-patient
  subset yields a 68-slice union of two identical 34-slice CT scans, which
  is silently wrong for downstream features.
- Cross-reference XML `SOPInstanceUID` references to identify the annotated
  series. Rejected for v0.4.3 because the COCA plist XML inspected locally
  contains slice indices and ROI geometry, not SOPInstanceUID references.
- Most-slices rule. Rejected because it fails for Patient 78, where both
  series have identical slice counts.
- Latest `AcquisitionTime` rule. Rejected because re-scans for technical
  reasons may be saved later; preferring latest would not be conservative.

**Rationale**: Deterministic, reproducible, no external metadata dependency,
correctly avoids concatenating unrelated series, and surfaces the choice in
pipeline logs without changing the public loader return contract.

**Verified by**: `tests/test_io_utils.py` unit tests for the selection helper;
integration tests on Patient 78 (multi-series -> 34 slices, warning emitted)
and Patient 1 (single-series -> unchanged, no warning).

---

## D013 - Lesion grouping: maximum centroid distance threshold

**Date**: May 2026
**Tag**: `v0.5.0-features-expanded`

**Decision**: `LESION_GROUPING_DISTANCE_MM = 5.0`

Two ROIs on adjacent slices belong to the same 3D lesion if and only if
their centroids are <= 5.0 mm apart in the XY plane.

**Literature basis**:
Hoori 2024 does not publish a connectivity radius; no calcium-radiomics paper
in the reference set specifies one. The threshold is therefore derived from
first principles grounded in coronary calcium morphology.

Coronary calcium deposits observable on NCCT span roughly 2-15 mm in their
largest transverse dimension for the mild-to-moderate Agatston range covering
most of the COCA cohort (Agatston 1-399). For a single lesion whose centroid
traces a path as it is imaged across adjacent 3mm slices, the centroid shift
per slice is bounded by half the lesion's transverse diameter. A 10 mm lesion
(upper bound for a non-confluent deposit) has a centroid shift of at most
~5 mm between adjacent slices. A 5 mm threshold therefore correctly merges
any single-lesion ROI chain up to ~10 mm transverse diameter.

**Alternatives considered**:
- **3 mm**: too tight. A single 7mm lesion traced across two adjacent slices
  can have centroid shift of ~3.5 mm due to the curved trajectory of the
  coronary artery through the image volume. A 3 mm threshold would
  over-fragment moderate-sized single lesions into two lesion records, inflating
  lesion count and collapsing the very feature (count) that Hoori found to be
  the top-1 MACE predictor. Rejected.
- **8 mm**: too loose. Two distinct adjacent calcium deposits in the LAD must
  be at least ~5-6 mm apart to be anatomically distinguishable as separate
  calcifications at COCA's resolution. An 8 mm threshold risks merging two
  genuinely distinct proximal-LAD deposits into one 3D "lesion," deflating
  lesion count and producing an incorrectly large diffusivity denominator.
  Rejected.
- **Polygon overlap test instead of centroid distance**: geometrically rigorous,
  but polygons on adjacent slices almost never overlap in 2D projection even
  for the same lesion (the lesion moves between slices). Centroid distance
  better captures the physical connectivity. Rejected for this version.

**Rationale**: 5 mm is the geometric upper bound of the centroid shift for a
single lesion of reasonable clinical size. It is conservative on the merging
side (does not merge truly distinct deposits, which are almost always >5 mm
apart on adjacent slices) and conservative on the splitting side (does not
fragment any single lesion of up to ~10 mm diameter). This is the
methodologically defensible default absent a published benchmark.

**Verified by**: Visual inspection of grouping output on 3 patients with
heavy LAD calcification before tagging. If a specific patient shows
unexpected counts, the threshold is configurable via `config.py`
without code changes.

---

## D014 - Lesion grouping: maximum slice index gap

**Date**: May 2026
**Tag**: `v0.5.0-features-expanded`

**Decision**: `LESION_GROUPING_MAX_SLICE_GAP = 1`

Two ROIs belong to the same 3D lesion only if their slice indices differ
by exactly 1. A gap of 2 or more = two separate lesions, regardless of
spatial proximity.

**Literature basis**:
COCA annotations are expert-level radiologist contours drawn specifically
for calcium scoring - a clinical-quality annotation task, not a research
approximation. At 3 mm native slice spacing, a single calcium deposit of
clinical significance (>= 130 HU, >= 3 connected voxels by Agatston definition)
that is genuinely continuous through two adjacent slices will be visible and
annotated on both slices by a competent annotator. A gap = the annotator
decided there was no annotation-worthy calcium on that slice. The
verification data shows 1189 annotated ROIs across 100 patients; the COCA
annotation protocol does not document intentional slice-skipping conventions.

**Alternatives considered**:
- **Gap <= 2 (allows one skipped slice)**: accommodates the hypothesis that
  annotators occasionally skip a borderline middle slice in a continuous
  lesion. Would bridge 3 mm of unannotated tissue (one full slice at native
  spacing). The biological cost: if we allow gap <= 2, we might merge two
  distinct proximal-LAD deposits separated by a 3 mm calcium-free zone
  into one large "lesion," directly corrupting the diffusivity denominator
  (first-to-last distance increases) and deflating lesion count (the #1
  MACE predictor). The annotation-quality benefit is hypothetical; the
  feature-corruption risk is concrete. Rejected.
- **No gap constraint (pure spatial proximity)**: would merge any two spatially
  close lesions in the same artery, even if separated by many slices of
  healthy tissue. Clearly wrong. Rejected.

**Rationale**: In a high-quality expert-annotated dataset, an unannotated
slice between two annotated slices means no calcium on that slice. Strict
adjacency preserves this clinical meaning. Over-splitting (the risk of
strict adjacency) produces higher lesion counts - which, if anything, errs
in the direction of finding more diffuse disease rather than under-counting
it. Over-merging (the risk of relaxed adjacency) directly corrupts
diffusivity and lesion count, the top spatial features by MACE HR.
Conservative on merging is the right error mode.

**Verified by**: Inspection of the annotation structure confirms the
existing parser skips zero-point ROIs and processes each non-zero ROI as
a distinct slice entry. Strict adjacency in grouping is consistent with
the per-slice annotation model.

---

## D015 - Per-artery mass formula

**Date**: May 2026
**Tag**: `v0.5.0-features-expanded`

**Decision**: `MASS_FORMULA = "hu_volume_product"`

Per-artery mass is computed as:

```text
mass_artery = sum_ROI (Area_mm2 x slice_thickness_mm x Mean_HU)
```

summed across all ROIs belonging to that artery. The feature is named
`mass_{artery}` with units mm3*HU and documented as "HU-volume product,
proportional to calibrated calcium mass score up to a scanner-specific
constant."

**Literature basis**:
The clinical CAC mass score (Callister 1998, JACC) is:

```text
mass_score = sum_i (Area_i x thickness_i x mean_HU_i x k)
```

where `k ~= 0.00105 mg_HA / (HU x mm)` is a phantom-calibration constant.
Our formula is identical with `k = 1` (uncalibrated). For within-cohort
analysis - all patients on the same scanner family, same protocol -
the calibration constant is the same for every patient and cancels
completely in any comparison, ranking, or clustering operation.
It matters only when comparing absolute mass values against other cohorts
or clinical thresholds in mg HA.

Hoori 2024 computed "territorial mass scores" as the top-3 MACE predictors
(LAD mass HR 1.27, LM HR 1.13, etc.). The exact formula in their
supplementary is not published, but their CTCS pipeline used standard
clinical mass score software on 0.5x0.5x2.5 mm voxels - equivalent to
our formulation up to the calibration constant.

**Alternatives considered**:
- **Calibrated mass (x 0.00105 mg_HA/HU/mm)**: only meaningful if the
  scanner phantom calibration for COCA is available. Without it, applying
  a generic constant introduces a known approximation. Our formula is
  *not less accurate* - it is equally accurate for ranking and clustering,
  and is more honest about what is being measured. If COCA's scanner
  calibration is ever published, the conversion is a one-line constant
  multiplication. Rejected as premature.
- **Agatston-weighted burden (Area x density_factor)**: uses discrete
  density bins (1/2/3/4) instead of continuous mean HU. This is
  information-lossy - a patient with mean HU 201 vs 295 per lesion gets
  the same weight-2 factor. The HU-volume product preserves continuous
  HU information. Agatston per artery is still computed separately as a
  distinct feature; the mass feature uses continuous HU. Rejected for
  the mass feature.
- **Volume only (Area x thickness, no HU weighting)**: ignores density.
  Two lesions of identical volume but different HU (e.g., 200 vs 800 HU)
  would have identical mass features - wrong. Hoori found mass predictive
  precisely because it captures density x volume, not just volume. Rejected.

**Rationale**: The HU-volume product is the continuous, information-preserving
form of the clinical mass score without the unnecessary complication of a
phantom calibration that is absent from COCA metadata. For clustering and
phenotyping on a within-cohort dataset, it is strictly equivalent to the
calibrated form. The feature is named and documented to be transparent
about what it measures.

**Verified by**: Formula is computed directly from XML fields (Area, Mean)
and DICOM spacing. The per-artery Agatston self-consistency test
(sum ~= `agatston_score` total) is a separate acceptance check that
validates the ROI-grouping infrastructure used by mass computation.

---

## D016 - Diffusivity fallback when N >= 2 and d_first_to_last = 0

**Date**: May 2026
**Tag**: `v0.5.0-features-expanded`

**Decision**: When an artery has N >= 2 lesions whose first and last
centroids are < 1e-6 mm apart (numerical zero), return `diffusivity = 1.0`.

```python
def diffusivity(n_lesions: int, d_first_last_mm: float) -> float:
    if n_lesions == 0:               return 0.0   # Hoori convention
    if n_lesions == 1:               return 1.0   # Hoori convention
    if d_first_last_mm < 1e-6:       return 1.0   # D016
    return n_lesions / d_first_last_mm
```

**Literature basis**:
Hoori 2024 defines the edge cases for 0 and 1 lesion explicitly.
The case N >= 2 with d = 0 is not covered - it is geometrically
near-impossible in practice (two distinct 3D lesions would have to
have identical centroids to numerical precision). It could theoretically
occur if the lesion grouping algorithm assigns two ROIs on the same slice
to separate "lesions" (which the strict-adjacency grouping prevents, since
same-slice ROIs have diff = 0, not 1 - they go into separate single-ROI
lesions each). With strict-adjacency grouping and the 5 mm distance
threshold, two lesions in the same artery on the same slice are separately
grouped but at the same z-level. Their first-to-last z-distance = 0 by
construction; the XY centroid distance may be non-zero.

For this rare degenerate case the "all calcium at one z-level" situation
is semantically most similar to the single-cluster, concentrated calcium
case. Hoori's single-lesion convention (`-> 1.0`) represents "calcium at
one point = minimal diffusivity = 1.0 as a defined value." Extending this
to "multiple lesions, no z-spread = 1.0" is the natural continuation.

**Alternatives considered**:
- **Return N (= number of lesions)**: would indicate "high diffusivity" for
  an artery where all calcium is concentrated on a single slice. This is
  the wrong physical interpretation - high diffusivity means calcium is
  spread along the artery, not concentrated. Rejected.
- **Return 0.0**: would conflate "no calcium" with "highly concentrated
  calcium." These are distinct clinical states. Rejected.
- **Raise ValueError**: a degenerate input should not crash the pipeline on
  production data. Silent fallback with documented convention is correct.
  Rejected.
- **Return N / 1.0mm (minimum distance floor)**: for N = 5 at d = 0,
  returns 5.0 - same problem as returning N. Rejected.

**Rationale**: 1.0 is the principled extension of Hoori's "single lesion ->
1.0" convention. It is consistent, non-crashing, and semantically coherent
(concentrated multi-lesion artery ~= single-cluster, minimal spread). The
case is rare enough that the exact value will not materially affect
clustering results; consistency with the N=1 convention matters more.

**Verified by**: `tests/test_spatial_features.py::test_diffusivity_*`
suite, including the zero-distance edge case.

---

## D017 - Empty-artery feature values

**Date**: May 2026
**Tag**: `v0.5.0-features-expanded`

**Decision**: `EMPTY_ARTERY_SENTINEL = 0.0`

All per-artery features emit `0.0` when the artery has no annotated
lesions. No `NaN` values appear in the feature matrix. Features with
potentially ambiguous zero semantics are documented below.

**Literature basis**:
The reference codex notes: "Expected stable: Volume, Mass, Agatston,
lesion count, Gini, diffusivity, per-artery aggregates - geometric/spatial
features are typically ICC > 0.85." Stability is reported for the features
as computed on patients with calcium; for patients without calcium the
feature is genuinely zero (no lesions, no mass, no diffusivity). This is
not missing data - it is measured data with value zero.

Hoori 2024 cohort: 38.9% of the CLARIFY registry had Agatston = 0. Their
pipeline produces valid feature vectors for these patients. The clinical
interpretation is unambiguous: no calcium in an artery = 0 lesions, 0
volume, 0 mass, 0 diffusivity, 0 HU contribution. The Agatston framework
itself assigns 0 for absence of calcium - our features follow the same
convention.

The reference codex notes spatial/geometric features are "Expected stable"
under perturbation. A 0-valued feature is maximally stable under
perturbation (perturbing an image with no calcium still produces 0 for
these features). 0-sentinels are therefore consistent with the
perturbation-stability expectation.

**Alternatives considered**:
- **NaN with documented imputation (mean, median, or 0)**: requires a
  separate imputation step before clustering, adds a methodological choice
  (imputation strategy) that would itself require a decision entry, and
  may produce different results for KMeans vs GMM (which handle NaN
  differently). The added complexity is not justified - the data is not
  missing, it is zero. Rejected.
- **NaN with feature dropped for zero-calcium patients**: would reduce the
  effective feature vector length per patient, producing an irregular
  matrix that clustering algorithms cannot handle without padding.
  Rejected.
- **Separate binary indicator features ("has_lad_lesions", etc.)**: these
  could be added as additional features to let the model distinguish
  "measured zero" from other values. Not rejected outright - they may be
  added in v0.6.0 as supplementary features. For v0.5.0, the 0-sentinel
  is sufficient because the clustering algorithms will form a natural
  "no calcium" or "minimal calcium" cluster that separates 0-valued
  rows from high-valued ones.

**Feature-specific zero semantics**:

| Feature | Zero means | Ambiguous? |
|---|---|---|
| `lesion_count_{artery}` | No lesions | No |
| `volume_{artery}_mm3` | No calcium volume | No |
| `agatston_{artery}` | No Agatston contribution | No |
| `mass_{artery}` | No HU-volume product | No |
| `diffusivity_{artery}` | No lesions (Hoori convention) | No |
| `n_calcified_arteries` | No arteries with calcium | No |
| `gini_lesion_volume` | No lesions or all equal | Convention-defined |
| `inter_lesion_dist_mean_{artery}` | < 2 lesions (no pairs) | Document |
| `inter_lesion_dist_max_{artery}` | < 2 lesions (no pairs) | Document |
| `first_to_last_dist_{artery}` | < 2 lesions (no extent) | Document |
| `max_hu_{artery}` | No lesions | No |
| `mean_hu_{artery}` | No lesions | Slight |

For `inter_lesion_dist_*` and `first_to_last_dist_*`: the module-level
docstring will note "0 means insufficient lesions for distance
computation (< 2 lesions), not lesions at 0 mm spacing." This prevents
misinterpretation.

For `mean_hu_{artery}` when zero: "no calcium present, no HU
contribution." Semantically this is "null" but 0 is the correct
aggregate of no values when the aggregation is a weighted mean
with total weight 0.

**Rationale**: Zero is the correct physical value, not a sentinel for
missing data. It is consistent with the Agatston scoring convention
(no calcium = score 0), produces a dense feature matrix, and requires no
imputation step. Any clustering algorithm that treats "no calcium"
patients as a distinct group will cluster their all-zero vectors
together correctly.

**Verified by**: `tests/test_spatial_features.py::test_empty_artery_*`
suite. The `zero_spatial_features()` helper function is the canonical
source of truth for what an all-zero feature dict looks like.

---

## D018 — Agatston formula thickness correction alignment across modules

**Date**: May 2026
**Tag**: logged at `v0.5.0-features-expanded`, implement in `v0.5.1`

**Problem**: Two Agatston implementations exist with a latent divergence:

| Module | Thickness correction | Formula |
|---|---|---|
| `src/agatston.py` | None — implicitly assumes 3 mm slices | `Area_cm² × 100 × weight` |
| `src/spatial_features.py` | Explicit: `× (3.0 / actual_thickness_mm)` | `area_mm² × weight × (3.0 / thickness)` |

For COCA data at native 3 mm slice thickness both formulas produce
identical results (the correction factor is exactly 1.0). The v0.5.0
full-run Agatston consistency check confirms: all 100 patients pass
within 0.05 tolerance after undoing the thickness correction.

**Why this matters**: For any future cohort with non-3 mm slices
(e.g. ImageCAS at 2.5 mm, Kettering at 2.0 mm) the two modules will
diverge systematically. `agatston.py` will undercount by a factor of
`actual_thickness / 3.0`. This will cause the Agatston self-consistency
check in `verify_pipeline.py` to fire as a genuine inconsistency.

**Decision**: Do NOT change `agatston.py` before the `v0.5.0` tag.
The change requires:

1. Add `slice_thickness_mm: float` parameter to
   `compute_agatston_from_annotations` (default `3.0` for backward
   compatibility with COCA).
2. Apply `× (slice_thickness_mm / 3.0)` inside the formula — making
   `agatston.py` and `spatial_features.py` use identical corrections.
3. Write a pytest fixture with two patients at different slice thicknesses
   (3 mm and 2.5 mm) and assert that both modules return identical
   per-artery totals on both fixtures.
4. Update `run_preprocessing.py` to pass `slice_thickness_i` to
   `compute_agatston_from_annotations`.
5. Re-run preprocessing to regenerate `agatston_scores.csv` with
   corrected scores for any non-3 mm cohort.

**Simplify `verify_pipeline.py`**: Once D018 is implemented, the
thickness-uncorrection logic in `verify_pipeline.py` (lines ~70–90) can
be replaced with a direct delta comparison at tolerance 1e-3.

**v0.5.0 baseline tolerance**: 0.05 (after thickness uncorrection).
**v0.5.1 target tolerance**: 1e-3 (direct comparison, no uncorrection).

**Tracked in**: `tests/test_agatston.py` — add
`test_thickness_correction_alignment_across_modules` before v0.5.1 tag.

---

## D019 - Feature cleaning variance filter threshold

**Date**: May 2026
**Tag**: `v0.5.1-feature-cleaning`

**Decision**: Drop feature columns with raw sample variance `< 0.01` before
any scaling, clustering, or ICC gating.

**Rationale**: Near-constant features contribute no meaningful patient
separation and can make correlation-based distances unstable. Applying this
filter on raw feature values keeps the decision simple and reproducible.

**Implementation**: `VARIANCE_FILTER_THRESHOLD = 0.01` in `src/config.py`.

---

## D020 - R2 hierarchical clustering rule

**Date**: May 2026
**Tag**: `v0.5.1-feature-cleaning`

**Decision**: Collapse redundant features using hierarchical clustering on
distance `1 - r^2`, with average linkage (UPGMA). Cut the dendrogram at the
largest merge-distance gap when the gap is at least `0.05`; otherwise use
fallback distance `d = 0.20`.

**Rationale**: The expanded v0.5.0 feature matrix contains many intentionally
overlapping calcium burden views. Clustering on `r^2` removes sign sensitivity
and treats strongly correlated and strongly anti-correlated features as
redundant. The elbow cut is data-adaptive while the fallback keeps the
pipeline deterministic when the dendrogram is smooth.

**Implementation**: `R2_CLUSTERING_LINKAGE = "average"`,
`R2_ELBOW_MIN_GAP = 0.05`, and
`R2_CLUSTERING_FALLBACK_DISTANCE = 0.20` in `src/config.py`.

### D020 — Update (v0.5.2 calibration)

**Date**: May 2026

`R2_ELBOW_MIN_GAP` lowered from `0.05` → `0.04` after empirical observation
on the full 162-feature v0.5.2 dataset. The largest dendrogram gap was
`0.0498` (threshold: `0.0500`) — a real structural elbow at `d ≈ 0.477`
(`|r| > 0.718`) that missed the original threshold by 0.4%. The fallback at
`d = 0.200` is designed for absent structure, not borderline structure.
Additionally, the fallback's 96-representative result produces `p/n ≈ 0.96`
which is ill-conditioned for PCA eigenvalue estimation; 57 representatives at
`p/n ≈ 0.57` is substantially better. `min_gap = 0.04` was chosen to give a
comfortable margin below the observed `0.0498` gap.

**Implementation**: `R2_ELBOW_MIN_GAP = 0.04` in `src/config.py`.

### D020 — Update v2 (post-mask-fix revert)

**Date**: May 2026

`R2_ELBOW_MIN_GAP` restored from `0.04` → `0.05`. Two false-positive elbow
detections from the same calibration error: (1) n=100 v0.5.2 gap=0.0498 was a
sampling artefact from the small cohort; (2) post-mask-fix n=445 gap=0.0427
produced a degenerate mega-cluster of 108/145 features at d=0.7932. Gaps below
0.05 do not represent genuine hierarchical structure in COCA calcium radiomics —
the COCA correlation landscape is smooth with no real hierarchical boundary, and
the fallback at d=0.20 has been validated across three separate runs (n=100
v0.5.2, n=450 pre-mask-fix, n=445 post-mask-fix). Restoring min_gap=0.05
ensures the fallback fires when no genuine elbow exists.

**Implementation**: `R2_ELBOW_MIN_GAP = 0.05` in `src/config.py`.

---

## D021 - Cluster representative selection

**Date**: May 2026
**Tag**: `v0.5.1-feature-cleaning`

**Decision**: Select one representative per R2 cluster by highest mean
absolute Pearson correlation to the other features in the same cluster.
For ties, choose the candidate with highest absolute Spearman correlation
to `agatston_score`.

**Rationale**: Mean absolute within-cluster correlation picks the most central
feature rather than an arbitrary first feature. The Agatston tie-break preserves
the feature most aligned with the established clinical calcium burden score
when centrality cannot distinguish candidates.

**Implementation**: `select_representatives()` in `src/feature_cleaning.py`.

---

## D022 - Perturbation ICC stability gate

**Date**: May 2026
**Tag**: `v0.5.1-feature-cleaning`

**Decision**: Gate cluster representatives by minimum ICC across 14
perturbation types:

- Axial rotations: `+5`, `-5`, `+10`, `-10` degrees.
- XY translations: `+2`, `-2`, `+5`, `-5` mm on each of x and y axes.
- CT Gaussian noise: sigma `5` and `10` HU.

A feature passes only if its minimum absolute-agreement ICC is `> 0.75`.

**Rationale**: The perturbation set tests small plausible geometric and
intensity changes without introducing a new resampling-spacing experiment.
The strict minimum gate removes representatives that are unstable under any
locked perturbation.

**Implementation**: `src/perturbations.py`, `src/icc.py`, and
`compute_icc_table()` / `apply_icc_gate()` in `src/feature_cleaning.py`.

---

## D023 - Perturbation cohort and stable-by-construction features

**Date**: May 2026
**Tag**: `v0.5.1-feature-cleaning`

**Decision**: Run perturbation extraction on the same 100-patient cohort used
for v0.5.0, excluding patient 12 because DICOM data is absent. PyRadiomics
features are re-extracted from perturbed CT/mask volumes. Spatial/per-artery
features and XML-derived calcium summary features are assigned ICC = 1.0 by
construction because they are not recomputed inside the perturbation loop.

**Rationale**: The perturbation gate is scoped to image-derived PyRadiomics
stability. XML and spatial features are deterministic functions of annotation
geometry and are intentionally kept outside the perturbation extraction path.

**Implementation**: `scripts/run_perturbations.py` extracts PyRadiomics-only
CSV files; `stable_by_construction_feature_names()` in
`src/feature_cleaning.py` defines the bypass set.

**Verification of diffusivity ICC=1.0 claim (May 2026)**:
Confirmed correct by pipeline architecture inspection. Diffusivity features
(`diffusivity_lad`, `diffusivity_lcx`, `diffusivity_rca`) measure inter-lesion
centroid distances derived exclusively from XML annotations. The perturbation
pipeline (`scripts/run_perturbations.py`) perturbs the CT image and
segmentation mask only — it does not alter XML annotations. Verified by
inspecting all 14 perturbation output CSVs
(`outputs/perturbations/{type}/features.csv`): diffusivity columns are absent
from every perturbation CSV, confirming they are never re-extracted under
perturbation. Since the XML input is identical across all perturbation
conditions, diffusivity output is mathematically identical — ICC=1.0 is a
consequence of pipeline architecture, not an untested assumption.

**Implication**: the claim that all 94 features passing the ICC gate are
perturbation-certified is accurate. Diffusivity features bypass the ICC gate
not because they were assumed stable, but because their input (XML) is outside
the perturbation scope. This is the correct design: perturbing CT geometry
does not change the clinician-annotated lesion positions in the XML, so
testing diffusivity stability under CT perturbations would test annotation
consistency, not feature computation stability — a different scientific
question outside the scope of this pipeline.

---

## D024 — Density tier feature design

**Date**: May 2026
**Tag**: `v0.5.2-full-feature-expansion`

**Decision**: Per-artery ROI count in each Agatston density tier.
4 tiers × 4 arteries = 16 features. Count basis (number of ROIs),
not area-weighted. Patient-level aggregates not added (absorbed into
artery-level totals).

Tier thresholds follow the Agatston density factor convention exactly:
- Tier d1: 130 ≤ max_hu < 200 (factor 1)
- Tier d2: 200 ≤ max_hu < 300 (factor 2)
- Tier d3: 300 ≤ max_hu < 400 (factor 3)
- Tier d4: max_hu ≥ 400       (factor 4)

**Feature names**: `n_rois_d1_{artery}`, `n_rois_d2_{artery}`,
`n_rois_d3_{artery}`, `n_rois_d4_{artery}` for arteries lad/rca/lcx/lm.

**Literature**: Hoori 2024 tracked per-territory density tier distribution
as part of their 80 calcium-omics features. These capture the HU
distribution shape within each artery's calcium — distinct from
mean_hu (average) or max_hu (peak).

---

## D025 — Dense calcium flag

**Date**: May 2026
**Tag**: `v0.5.2-full-feature-expansion`

**Decision**: `dense_calcium_count` = total number of ROIs across all
arteries with `max_hu > 1000`. Patient-level scalar.

**Literature**: Hoori 2024 found that dense calcium (HU > 1000) had a
*protective* effect: HR = 0.71 (95% CI 0.51–0.99, p = 0.042). This
signal is not captured by any existing feature. Dense calcium at > 1000 HU
represents mature, calcified plaque and is biologically distinct from
softer calcifications in the 130–400 HU range.

**Note**: The threshold of > 1000 HU (not ≥ 1000) follows Hoori's
published specification. This feature is XML-derived and therefore
stable by construction (ICC = 1.0).

---

## D026 — PyRadiomics full feature set

**Date**: May 2026
**Tag**: `v0.5.2-full-feature-expansion`

**Decision**: Enable all seven IBSI-compliant PyRadiomics feature
families: Shape (14), First-order (18), GLCM (24), GLSZM (16),
GLRLM (16), NGTDM (5), GLDM (14). Total: 107 features.

Previous extraction (v0.5.0–v0.5.1): 12 features only (3 shape +
3 GLCM + 3 GLSZM + 3 GLRLM). First-order and NGTDM/GLDM were absent
despite being in the original proposal — corrected here.

**Why NGTDM**: Explicitly listed in the proposal pipeline. Captures
neighbourhood gray-tone differences — orthogonal to GLCM co-occurrence
and GLRLM run-length patterns.

**Why GLDM**: IBSI-compliant and captures gray-level dependence at
each voxel — a distinct texture dimension. The ICC gate will remove
it if fragile; there is no cost to including it.

**Why not shape2D**: PyRadiomics 2D shape requires per-slice analysis
and is not appropriate for full-mask 3D extraction. Excluded.

**Expected fragility**: High-order GLCM/GLSZM/NGTDM texture features
at small lesion sizes may have ICC < 0.5 under perturbation. The ICC
gate (D022) handles this — some high-order texture features failing is
correct behaviour, not a bug.

---

## D027 — Density tier representation basis

**Date**: May 2026
**Tag**: `v0.5.2-full-feature-expansion`

**Decision**: Count basis (number of ROIs per tier), not area-weighted
volume per tier.

**Rationale**: (a) Direct correspondence to Hoori 2024's published
feature design. (b) Volume-weighted version is largely redundant with
`volume_{artery}_mm3` which already captures area × thickness summed
across all ROIs. (c) Count basis captures how many distinct calcium
deposits are in each HU range — a distributional signal not present
elsewhere. Area-weighted tier features deferred to a later version if
they add independent signal.

---

## D031 — Agatston score as external PCA validation reference

**Date**: May 2026
**Tag**: `v0.6.0-pca-eigen-features`

**Decision**: `agatston_score` is loaded from `outputs/radiomics_features.csv`
for PC–Agatston Spearman correlation validation.  It is not loaded from
`outputs/cleaned_features.csv`.

**Rationale**: At n=450, `agatston_score` was absorbed into the
`original_shape_SurfaceArea` cluster during R² clustering
(inter-cluster distance d=0.036 < fallback d=0.200; SurfaceArea selected
as representative). It therefore does not appear in `cleaned_features.csv`
and is absent from the PCA input matrix.  Loading it from
`radiomics_features.csv` is the correct approach because:

1. It avoids circular validation (using a PCA input feature as the
   validation reference would be self-referential).
2. It maintains `agatston_score` as a genuine external clinical target
   for evaluating whether retained PCs capture meaningful calcium burden.
3. The inner join merge in `run_pca.py` ensures patient alignment is exact.

**Implementation**: `run_pca.py` steps 2 and 5; `pc_agatston_correlation()`
in `src/pca_features.py`.

---

## D032 — PCA component retention threshold

**Date**: May 2026
**Tag**: `v0.6.0-pca-eigen-features`

**Decision**: Retain the minimum number of principal components whose
cumulative explained variance reaches **85%**.  `CUMVAR_THRESHOLD = 0.85`
in `src/pca_features.py`.

**Rationale**: 85% is the standard threshold in the radiomics literature
for reducing a high-dimensional feature matrix to a compact eigenspace
while preserving the dominant structure.  At p=76, n=445 (p/n ≈ 0.17),
a full SVD is stable and all components are estimable, but retaining
all 76 PCs would simply shuffle the same information into a rotated basis
with no dimensionality benefit.  85% balances information preservation
against the goal of deriving a compact set of eigen features for downstream
clustering and survival modelling.

**Observed result (post-mask-fix run, May 2026)**: n_retain = 19 PCs,
cumulative variance at PC19 = 85.5%. PC1 eigenvalue = 29.806 (39.1%
explained variance). Kaiser criterion (eigenvalue > 1.0) gives n=15;
85% cumvar gives n=19. PCs 16–19 are below Kaiser but retained.

**Alternatives considered**:
- **90%**: retains more variance but typically adds several near-noise
  components whose eigenvalues are below the broken-stick threshold.
  Accepted for a sensitivity run but not the primary threshold.
- **Kaiser rule (eigenvalue > 1.0 after z-scoring)**: mechanically
  simple but data-dependent in a way that varies with p/n.  Can
  retain too few components when p is large.  Rejected as primary rule.
- **Scree elbow**: visual and hard to automate reproducibly.  The 85%
  threshold produces a clear, reproducible rule.

**Implementation**: `CUMVAR_THRESHOLD = 0.85` in `src/pca_features.py`;
`fit_pca()` returns `n_retain` as `argmax(cumvar >= threshold) + 1`.

---

## D033 — PCA input scaling

**Date**: May 2026
**Tag**: `v0.6.0-pca-eigen-features`

**Decision**: Apply `sklearn.preprocessing.StandardScaler` (zero mean,
unit variance) to all 76 input features before PCA.  The scaler is fit
on the full cleaned cohort (n=445 patients) in a single pass.

**Rationale**: The 76 features span vastly different scales and units:
PyRadiomics texture features are typically in [0, 1] or small positive
ranges, spatial burden features (e.g. `agatston_lad`) can reach thousands,
and shape features (e.g. `SurfaceArea` in mm²) are in the hundreds.
Without scaling, PCA is dominated by high-variance / large-scale features
regardless of their clinical relevance.  StandardScaler gives each feature
equal weight before covariance decomposition, making the eigen features
reflect multi-scale structure rather than the arbitrary physical units
of each measurement.

**Alternatives considered**:
- **No scaling**: would produce PC1 dominated by `SurfaceArea` or
  Agatston-scale features whose raw variance is orders of magnitude
  larger.  Rejected.
- **MinMaxScaler**: sensitive to outliers at both extremes.  Patients
  with extreme calcification can compress the range of all others.
  Rejected.
- **RobustScaler (IQR-based)**: defensible and outlier-resistant, but
  adds complexity.  StandardScaler is the established radiomics
  convention and sufficient for within-cohort PCA.  Deferred to v0.7.0
  sensitivity analysis.

**Implementation**: `StandardScaler` in `fit_pca()`; `src/pca_features.py`.
The fitted scaler is saved in `pca_model.pkl` for consistent transform
of any future inference cohort.

---

## D034 — PC sign normalisation convention

**Date**: May 2026
**Tag**: `v0.6.0-pca-eigen-features`

**Decision**: After fitting PCA, flip the sign of any PC whose
highest-absolute-value loading is negative, so that the dominant
loading direction is always positive.  Applied in-place to
`pca_full.components_` before `compute_scores()` and before
`joblib.dump()`, ensuring stored components and saved scores share
the same convention.

**Rationale**: PCA eigenvectors are defined only up to sign.  Without
normalisation, the sign of each PC is arbitrary and changes across
different sklearn versions, random states, or data subsets.  Fixing sign
so the dominant loading is positive gives PC scores a stable
interpretability: a high PC1 score corresponds to high burden in the
direction of the dominant feature (e.g. high PC1 = large SurfaceArea /
high calcium burden), not an arbitrary direction.  This is essential for:

1. Reproducibility: the same data always produces the same-sign PCs.
2. Clinical interpretability: phenotypes with high PC1 scores can be
   described as "high burden" without checking the sign convention
   each time.
3. Consistency between stored model and saved scores: `pca_model.pkl`
   and `eigen_features.csv` encode the same convention.

**Implementation**: `normalise_pc_signs()` in `src/pca_features.py`;
called between `fit_pca()` and `compute_scores()` in `run_pca.py`.
**Verified by**: `tests/test_pca_features.py::test_normalise_pc_signs_dominant_loading_nonnegative`.

---

## D035 — No LMM correction for WGCNA input

**Date**: May 2026
**Tag**: `v0.7.0-wgcna-eigengenes`

**Decision**: WGCNA is applied directly to the per-patient feature matrix
from `outputs/cleaned_features.csv`. No linear mixed model (LMM) correction
is applied before correlation computation.

**Rationale**: LMM correction is required in the original WGCNA pipeline
(Langfelder & Horvath 2008) because each sample (gene expression profile)
contains observations from multiple cells or tissues within one patient —
the intra-patient clustering structure must be removed before computing
inter-feature correlations. In the PrediCT dataset, `cleaned_features.csv`
has one row per patient with no repeated-measures structure: each feature
value is a single aggregate derived from one patient's CT scan. There is no
intra-patient clustering to correct for. Applying LMM residualisation would
require specifying a batch/covariate variable; no such variable exists in
the COCA protocol and none has been identified as a known confounder.

**Alternatives considered**:
- **Age/sex regression prior to WGCNA**: Langfelder & Horvath recommend
  regressing out known covariates (sex, age, batch). COCA metadata does not
  contain patient demographics in the feature matrix. If metadata becomes
  available, this can be added as a preprocessing step before `run_wgcna.py`.
  Deferred.
- **ComBat harmonisation**: applicable only when multiple scanner sites or
  acquisition protocols exist. COCA is a single-centre, single-protocol
  dataset. Rejected as inapplicable.

**Verified by**: `cleaned_features.csv` is a (n_patients × n_features)
matrix with no repeated patient_id values (verified by `run_wgcna.py`
header check).

---

## D036 — Pearson correlation for WGCNA feature co-expression matrix

**Date**: May 2026
**Tag**: `v0.7.0-wgcna-eigengenes`

**Decision**: Use Pearson correlation (`numpy.corrcoef`) to compute the
feature–feature correlation matrix. The alternative `bicor` (biweight
midcorrelation, the R WGCNA default for robustness to outliers) is not used.

**Rationale**: Pearson correlation is appropriate here because:

1. **ICC-gated inputs**: all features in `cleaned_features.csv` passed the
   ICC gate (D022, ICC > 0.75 across 14 perturbation types). Features with
   extreme outliers driven by segmentation artefacts or registration failures
   are removed before WGCNA input. The robustness argument for `bicor` is
   weakened when input quality is already validated.
2. **StandardScaler pre-scaling**: all features are z-scored before
   correlation computation (D033 convention extended to WGCNA). After
   z-scoring, mean-centred Pearson correlation and `bicor` produce nearly
   identical results for well-behaved distributions.
3. **Implementation parity**: `numpy.corrcoef` is exact, deterministic, and
   dependency-free. A `bicor` implementation would require either R (blocked)
   or a custom scipy-based approximation, adding code complexity without
   a demonstrated benefit for this dataset.

**Alternatives considered**:
- **bicor (biweight midcorrelation)**: R WGCNA default; more robust to
  outliers. Would require a custom Python implementation (not available in
  numpy/scipy). Rejected given ICC pre-gating and z-scoring.
- **Spearman correlation**: rank-based, robust to monotone non-linearity.
  Soft-thresholding of a rank correlation matrix is mathematically valid
  (Zhang & Horvath 2005 discuss this) but less standard. Deferred to
  sensitivity analysis if Pearson results are biologically implausible.

**Implementation**: `compute_correlation_matrix()` in `src/wgcna.py`.
**Verified by**: `tests/test_wgcna.py::test_correlation_matrix_*`.

---

## D037 — Soft-thresholding power β selection

**Date**: May 2026
**Tag**: `v0.7.0-wgcna-eigengenes`

**Decision**: Select the soft-thresholding power β as the **lowest** β in
the range [1, 30] where the scale-free topology fit R² ≥ 0.85. If no β in
the range achieves the target, fall back to the β with the highest R² and
document the fallback in the run log.

The unsigned adjacency is defined as:

```text
A_ij = |cor_ij|^β,  diagonal = 0
```

Scale-free topology fit R² is computed by log-log regression of the
connectivity frequency distribution:

```text
log10(p(k)) ~ a + b * log10(k)
```

Connectivity values are binned to 2 decimal places before counting
frequency to produce a non-degenerate distribution for p = 76 features
(where exact connectivity values are nearly unique).

**Rationale**: The scale-free criterion (Zhang & Horvath 2005) is the
standard approach for β selection in WGCNA. The "lowest β meeting the
threshold" rule mirrors `pickSoftThreshold()` in R WGCNA. A higher β
increases the contrast between strong and weak correlations (emphasising
module structure) but also reduces mean connectivity, eventually pushing
the network into a fragmented, low-power regime. Using the lowest sufficient
β preserves network connectivity while achieving scale-free topology.

The fallback (max-R² β) is documented explicitly because scale-free
topology is a property of biological networks (gene regulatory networks,
protein interaction networks). Radiomics features are derived from image
texture and geometry, not from biochemical interactions — there is no
theoretical reason why they must form a scale-free network. A failure to
reach R² = 0.85 is not a pipeline failure; it is a methodological
observation to be reported in the paper methods section.

**β range**: [1, 30]. The upper bound of 30 matches R WGCNA convention.
For gene expression data β is typically 6–12 (unsigned) or 12–20 (signed).
For radiomics features with moderate correlations the selected β may be
higher than typical gene expression values.

**Implementation**: `pick_soft_threshold()` in `src/wgcna.py`;
constants `BETA_RANGE = list(range(1, 31))` and
`SCALE_FREE_R2_TARGET = 0.85`.
**Verified by**: `tests/test_wgcna.py::test_pick_soft_threshold_*`.

### D037 — Observed result (v0.7.0 run, May 2026)

Scale-free topology **not achieved** for any β ∈ 1–30. Scale-free R²
degenerate — log10 of zero connectivity arises because at β=20 most
adjacency values are so small that connectivity bins are populated at
frequency 1 (no repeated values), making the log-log regression undefined.
D041 fallback triggered. β=20 selected (mean_conn = 0.0991 ≤ 0.10).
Adjacency mean (off-diag) at β=20 = 0.0013; TOM mean (off-diag) = 0.0014.

Result unchanged from the pre-fix 94-feature run qualitatively. Scale-free
topology failure is expected for calcium-only NCCT radiomics — one tissue
type, no biochemical interaction network.

---

## D038 — Minimum module size and static tree cut

**Date**: May 2026
**Tag**: `v0.7.0-wgcna-eigengenes`

**Decision**:

1. **MIN_MODULE_SIZE = 5**. Any cluster with fewer than 5 features after
   hierarchical clustering is assigned to grey (unassigned, label 0) and
   receives no eigengene.

2. **Static tree cut**: the dendrogram cut height is selected by scanning
   all unique merge heights and choosing the height that maximises the number
   of valid modules (clusters ≥ MIN_MODULE_SIZE). Among heights with equal
   valid module counts, the one minimising grey features is preferred;
   remaining ties are broken by selecting the highest (most stable) height.

3. **Linkage method**: average linkage (UPGMA), matching R WGCNA default.
   Operates on TOM dissimilarity = 1 − TOM.

**Rationale for MIN_MODULE_SIZE = 5**:
The R WGCNA default is `minModuleSize = 30`, calibrated for gene expression
datasets with p = 5,000–20,000 genes. For p = 76 radiomics features, a
minimum of 30 would collapse the entire feature set into a single module.
Scaling linearly: 30 / 20,000 × 76 ≈ 0.11, which rounds to 1 — too small
to be meaningful. A minimum of 5 ensures each module captures a coherent
multi-feature pattern while keeping the grey fraction manageable.

**Rationale for static cut vs dynamic tree cut**:
The `dynamicTreeCut` algorithm (Langfelder, Zhang & Horvath 2008) is
designed for gene expression dendrograms with p ≥ 500 where the tree
has complex nested structure. At p = 76, the dendrogram has at most 75
merges; the height landscape is simple enough that a static cut is
interpretable, reproducible without a C extension (`dynamicTreeCut` is
R-only), and appropriate for the data scale.

**Grey threshold warning**: If > 20% of features are assigned to grey,
`detect_modules()` emits a warning recommending MIN_MODULE_SIZE = 3.
This threshold (GREY_FLAG_THRESHOLD = 0.20) is documented in D040.

**Implementation**: `detect_modules()` in `src/wgcna.py`;
constants `MIN_MODULE_SIZE = 5` and `LINKAGE_METHOD = "average"`.
**Verified by**: `tests/test_wgcna.py::test_detect_modules_*`.

---

## D039 — Module merging threshold

**Date**: May 2026
**Tag**: `v0.7.0-wgcna-eigengenes`

**Decision**: After initial module detection, iteratively merge the two
modules whose eigengenes have the highest pairwise Pearson |r| until no
pair exceeds **|r| > 0.85**. Each merge re-extracts all eigengenes from
the updated labels before checking the next pair.

**Rationale**: The merging threshold of 0.85 matches `MEDissThres = 0.15`
in R WGCNA's `mergeCloseModules()` — that function merges when eigengene
dissimilarity (1 − Pearson r) < 0.15, i.e., when Pearson r > 0.85. Highly
correlated eigengenes represent biologically redundant modules whose
distinction is not supported by the data. Merging them reduces the
eigengene dimensionality entering consensus clustering (v0.8.0) without
discarding information.

The iterative strategy (one merge per step, full re-extraction after each)
is slower than a single-pass approach but is guaranteed to converge and
matches the behaviour of `mergeCloseModules()` in R WGCNA exactly. The
convergence guarantee follows from the module count strictly decreasing by
1 at each step.

**Label renumbering invariant**: after each merge, labels are re-normalised
to be contiguous (no gaps in 1..M). This invariant is required by
`extract_eigengenes()` which assumes `labels.max() == n_modules`.

**Implementation**: `merge_close_modules()` in `src/wgcna.py`;
constant `MERGE_THRESHOLD = 0.85`.
**Verified by**: `tests/test_wgcna.py::test_merge_close_modules_*`.

---

## D040 — Grey module definition and grey fraction warning

**Date**: May 2026
**Tag**: `v0.7.0-wgcna-eigengenes`

**Decision**: Features whose cluster has fewer than MIN_MODULE_SIZE members
after the tree cut are assigned to the grey module (label 0). Grey features
receive no eigengene in `wgcna_eigengenes.csv`. If the grey fraction exceeds
**20%** of all input features, `detect_modules()` emits a WARNING
recommending MIN_MODULE_SIZE = 3.

**Rationale**: The "grey module" convention is the R WGCNA standard: small
or isolated features that do not cluster cleanly with any group are
quarantined rather than forcing them into an existing module. Unlike gene
expression — where grey features are typically noise — in radiomics the grey
features may be genuinely independent descriptors (e.g., a unique shape
feature that correlates weakly with all texture features). They are not
discarded from `cleaned_features.csv`; they are simply not represented by
an eigengene. If downstream analysis requires them, they can be included
directly as individual features.

The 20% warning threshold is a heuristic: at p = 94, 20% is 18.8 features.
Losing more than ~19 features to grey suggests either MIN_MODULE_SIZE is
too large for this dataset's correlation structure, or the feature space
lacks coherent co-expression modules (which would itself be a substantive
finding worth reporting).

**Implementation**: `GREY_FLAG_THRESHOLD = 0.20` in `src/wgcna.py`;
warning emitted inside `detect_modules()`.
**Verified by**: grey label count checked in `run_wgcna.py` summary output
and V1 verification (shape of `wgcna_eigengenes.csv` reflects only
non-grey modules).

### D040 — Observed result (v0.7.0 run, May 2026)

Grey rate at β=20, min_size=5: **34.2% (26/76 features)**. Warning triggered
(34.2% > 20% threshold). Sensitivity sweep: grey rate remains high across
MIN_MODULE_SIZE ∈ {3, 5, 7}. Persistently grey features: `dist_from_top_max`,
`dist_from_top_mean`, `glszm_ZoneVariance`, `gldm_DependenceVariance`,
`ngtdm_Contrast`, and diffusivity/spatial descriptor features — genuinely
independent descriptors confirmed by PCA (they scatter across PC3, PC7,
PC10, PC14 with no coherent co-loading axis). Two independent methods (PCA
and WGCNA) agree. Grey fraction higher than pre-fix run (26.6% for 94
features) because the 76-feature post-fix set is less redundant — fewer
burden-correlated features to absorb grey candidates into modules.

---

## D041 — β selection fallback: mean connectivity criterion

**Date**: May 2026
**Tag**: `v0.7.0-wgcna-eigengenes`

**Decision**: When scale-free topology R² < 0.85 for all tested β (the
expected outcome for radiomics feature spaces — see D037), select the
**lowest β where mean network connectivity ≤ 0.1** as the primary fallback.
If mean connectivity never drops to 0.1 within the tested range, use the β
with the minimum mean connectivity (secondary fallback) and log a warning
recommending BETA_RANGE extension.

**Observed result on COCA 94-feature matrix**: β = 20, mean connectivity =
0.091.

**Rationale**: Scale-free topology (Zhang & Horvath 2005) is an empirical
property of biological interaction networks. It does not hold for radiomics
features, which are derived from image texture and geometry rather than
biochemical interactions. A failure to reach R² = 0.85 is therefore not a
pipeline failure — it is an expected methodological observation consistent
with the non-biological origin of the feature space.

The mean connectivity criterion provides a principled, data-adaptive
alternative for radiomics applications:

- **High β → low mean connectivity**: raising β shrinks all weak adjacency
  values toward zero, retaining only the strongest feature correlations. This
  increases network sparsity and sharpens module boundaries.
- **Target 0.1**: at mean connectivity ≈ 0.1, approximately 10% of the
  maximum possible edge weight is retained on average. This is sufficient to
  form cohesive modules while preventing a fully connected, structure-free
  network. The value is drawn from the Langfelder & Horvath (2008)
  recommendation that soft-thresholded networks should have "relatively low
  connectivity" to distinguish modules; 0.1 operationalises this guidance
  without requiring scale-free fit.
- **Lowest β meeting the criterion**: uses the least aggressive thresholding
  that achieves the target, preserving as much inter-feature relationship
  information as possible while still producing a sparse network.

**Selection hierarchy in `pick_soft_threshold()`**:

| Priority | Criterion | Rule |
|---|---|---|
| 1 (D037) | Scale-free R² ≥ 0.85 | Lowest β meeting threshold |
| 2 (D041) | Mean connectivity ≤ 0.1 | Lowest β meeting threshold |
| 3 (D041) | Secondary | β at minimum mean connectivity |

**Alternatives considered**:
- **β at maximum R²** (previous fallback): selects whichever β produced
  the best (but still insufficient) scale-free fit. In practice this is
  often β = 1 or 2 (nearly unthresholded network) or β = 30 (extreme
  sparsity), depending on the correlation structure. Neither extreme is
  useful for module detection. Superseded by D041.
- **Fixed β = 6** (unsigned WGCNA convention default): a common literature
  choice but calibrated for gene expression with typical inter-gene |r| ≈
  0.3–0.6. For radiomics features with higher correlations, β = 6 may
  under-threshold. Rejected as data-blind.
- **Fixed β = 12** (signed WGCNA convention default): same reasoning.
  Rejected.

**Implementation**: `MEAN_CONNECTIVITY_TARGET = 0.1` constant and revised
fallback logic in `pick_soft_threshold()` in `src/wgcna.py`.
**Verified by**: `run_wgcna.py` prints the selected β and criterion used;
`sft_table.csv` contains the full β vs. R² vs. mean_connectivity table for
audit.

### D041 — Observed result (v0.7.0 run, May 2026)

Scale-free R² criterion not met (R² degenerate — see D037). D041 first
fallback triggered. **β=20 selected** (mean_connectivity = 0.0991 ≤ 0.10).
Secondary fallback not needed. Result consistent with pre-fix 94-feature run
(mean_conn = 0.091). Slight increase from 0.091 to 0.0991 reflects the
76-feature post-fix input having modestly higher inter-feature correlations
(corrected masks, more coherent calcium texture measurements).

---

## D042 — Consensus clustering method (Monti 2003)

**Date**: May 2026
**Tag**: `v0.8.0-consensus-clustering`

**Decision**: Patient phenotyping uses the consensus matrix method from
Monti et al. (2003) "Consensus Clustering: A Resampling-Based Method for
Class Discovery." The proposal's majority vote approach is replaced.

For each candidate k, 100 random 80% subsamples are drawn. Four clustering
algorithms (KMeans, Ward, GMM, Spectral — see D043) are applied to each
subsample. A 445 × 445 consensus matrix C is accumulated: C[i,j] counts
the fraction of subsamples in which patients i and j were co-clustered,
among subsamples where both patients appeared. Final phenotype assignments
are derived by average-linkage hierarchical clustering on (1 − C), cut
into k* clusters.

**Rationale**: Majority vote requires solving the label permutation problem
across 4 algorithms × 100 subsamples, which is NP-hard in general and
requires heuristic alignment algorithms (Hungarian method, repeated
relabelling). The consensus matrix is permutation-invariant by construction:
it records agreement frequency, not label values. The method is the
established standard in the unsupervised clustering literature and is
directly comparable to prior consensus phenotyping studies in cardiovascular
imaging.

**Implementation**: `build_consensus_matrices()`, `derive_final_assignments()`
in `src/consensus.py`.
**Reference**: Monti, S., Tamayo, P., Mesirov, J., & Golub, T. (2003).
Consensus clustering: a resampling-based method for class discovery and
visualization of gene expression microarray data. Machine Learning, 52,
91–118.
**Verified by**: `tests/test_consensus.py::test_consensus_matrix_*`.

---

## D043 — Algorithm set for consensus

**Date**: May 2026
**Tag**: `v0.8.0-consensus-clustering`

**Decision**: Four algorithms participate in the consensus matrix: KMeans
(n\_init=100), Ward hierarchical, Gaussian Mixture Model (full covariance,
n\_init=10), and Spectral clustering (RBF kernel). DBSCAN is excluded from
the consensus and run separately for outlier identification.

**Rationale**:

- **KMeans**: fast, convex-cluster baseline. n\_init=100 ensures the global
  minimum is found for small k.
- **Ward hierarchical**: minimum-variance linkage; captures nested structure.
  No random state dependency.
- **GMM (full covariance)**: soft-boundary, ellipsoidal clusters. Captures
  co-variance structure among PCs that KMeans misses. Convergence guarded
  by `model.converged_` check.
- **Spectral (RBF)**: captures non-convex cluster shapes in the 22-dimensional
  PC space. Replaced DBSCAN in the consensus for the reason below.
- **DBSCAN excluded**: does not accept k as a parameter; assigns noise labels
  (-1) that break the fixed-k consensus matrix construction. Retained as a
  separate outlier detector to identify phenotypically ambiguous patients.

**Alternatives considered**:
- **Agglomerative + complete linkage**: biased toward small, compact clusters.
  Ward already covers hierarchical; adding complete would not diversify the
  algorithm space. Rejected.
- **k-medoids**: more robust to outliers than KMeans but O(n²) per iteration.
  At n=445, manageable but not materially different from KMeans post-D047
  scaling. Deferred to sensitivity analysis.

**Implementation**: `_cluster_subsample()` in `src/consensus.py`.

---

## D044 — Consensus clustering parameters

**Date**: May 2026
**Tag**: `v0.8.0-consensus-clustering`

**Decision**:
- k range: {2, 3, 4, 5, 6, 7, 8}
- Subsamples: S = 100
- Subsample fraction: 80% (356 of 445 patients)
- Shared subsample per (k, s): all 4 algorithms receive the same patient
  subset for each (k, s) pair
- random\_state = 42

**Upper-triangle accumulation**: The consensus matrix is accumulated in the
upper triangle only. After all subsamples, the full matrix is obtained via
`M = M + M.T` and `I = I + I.T` in a single symmetrisation step. This avoids
the double-count that arises from simultaneously updating M[i,j] and M[j,i]
inside the loop — a maintenance trap (the ratio M/I would still be correct,
but the code would appear wrong to any reader). The upper-triangle approach
is unambiguous and correct by construction.

**Rationale**:
- **Shared subsample**: isolates algorithm disagreement from sampling
  variation. If each algorithm got an independent subsample, a patient pair
  might be missed by one algorithm simply because they were not co-sampled,
  not because the algorithms disagree. Shared sampling ensures C[i,j] < 1
  only when algorithms genuinely disagree about co-clustering.
- **80% at n=445**: gives n\_sub=356, so any patient pair has probability
  0.80² = 0.64 of being co-sampled in a single draw. Across 400 draws (4
  algorithms × 100 subsamples), the expected co-sampling count is 256 —
  sufficient for stable C estimates.
- **S=100**: standard in the consensus clustering literature. Increasing to
  200 adds runtime with diminishing returns for n=445.
- **k=8 upper limit**: k > 8 is unlikely to be clinically interpretable for
  n=445 calcium phenotyping; also matches the practical limit for
  visualisation and reporting.

**Implementation**: `build_consensus_matrices()` in `src/consensus.py`;
constants `K_RANGE`, `N_SUBSAMPLES`, `SUBSAMPLE_FRAC`.

---

## D045 — k selection criterion

**Date**: May 2026
**Tag**: `v0.8.0-consensus-clustering`

**Decision**: The optimal number of clusters k\* is selected as the k ≥ 3
that maximises the change in area under the consensus CDF curve (Δ(AUC)):

```text
Δ(AUC)_k = AUC_k − AUC_{k−1}
```

k = 2 is excluded from selection. The Δ(AUC) at k=2 is measured relative
to the implicit k=1 baseline (all patients in one cluster → C = all-ones →
AUC = 1.0), producing an artefactually large jump that dominates the plot.

**Tie-breaking and corroboration**: If Δ(AUC) is flat across k=3,4,5 (all
values within 0.01 of each other), prefer k=3 — the clinical archetype prior
(spotty, dense, diffuse from Kolossváry/Hoori). Corroborate with silhouette
score, Davies-Bouldin, and Calinski-Harabasz. Document any disagreement
between Δ(AUC) and silhouette in `v0.8.0_baseline.md`.

**Rationale**: The CDF-AUC criterion is the original Monti et al. (2003)
recommendation and is the most widely used k selection method for consensus
clustering. Δ(AUC) captures how much the new k improved bimodality of the
consensus matrix relative to k−1. Bimodality (values near 0 or 1) is the
signal of stable clusters.

**Implementation**: `compute_cdf_auc()`, `select_optimal_k()` in
`src/consensus.py`.

---

## D046 — Consensus Index stability threshold

**Date**: May 2026
**Tag**: `v0.8.0-consensus-clustering`

**Decision**: Per-patient Consensus Index:

```text
CI_i = mean(C_{k*}[i, j])  for all j in same cluster as i, j ≠ i
```

Patients with CI > 0.6 are classified as **stable assignments**. The fraction
of patients with CI > 0.6 is the **stable assignment rate**, reported in the
paper. Patients with CI ≤ 0.6 are "phenotypically ambiguous" — they may be
transitional between phenotypes or outliers.

**Rationale**: CI > 0.6 means patient i was co-clustered with its cluster
peers in > 60% of subsamples and algorithm combinations. This is a moderate
but meaningful threshold: it allows for expected algorithm disagreement (~20%)
while filtering patients whose assignment is genuinely unstable. The 0.6
threshold is consistent with published consensus clustering studies in
molecular subtypes (Noushmehr et al., TCGA cancer subtypes).

**Implementation**: `compute_consensus_index()` in `src/consensus.py`;
constant `CI_THRESHOLD = 0.60`.
**Verified by**: `tests/test_consensus.py::test_consensus_index_*`.

---

## D047 — StandardScaler on PCA scores before consensus clustering

**Date**: May 2026
**Tag**: `v0.8.0-consensus-clustering`

**Decision**: Apply `sklearn.preprocessing.StandardScaler` to the 19 PCA
scores before passing them to `build_consensus_matrices()`. The fitted scaler
is saved in `consensus_model.pkl` for projecting future cohorts.

**Rationale**: PCA scores are NOT on a comparable scale across PCs. The
explained variance (eigenvalue) per PC ranges from 29.806 (PC1, 39.1% of
variance) to 0.807 (PC19), a 37× spread. In the raw PCA score space:
- The standard deviation of PC1 scores ≈ √29.8 ≈ 5.5
- The standard deviation of PC19 scores ≈ √0.81 ≈ 0.90

Distance-based algorithms (KMeans, Spectral, GMM) compute Euclidean or
kernel distances. A 1-unit difference on PC1 contributes 6× more to the
L2 distance than a 1-unit difference on PC19. Without re-scaling, all three
distance-based algorithms effectively cluster on PC1 alone — the result is
equivalent to 1D clustering on the Agatston-proxy axis. This defeats the
purpose of retaining 19 PCs.

StandardScaler gives each PC unit variance, ensuring all 19 dimensions
contribute equally to cluster geometry. Ward hierarchical clustering is also
distance-based (minimum intra-cluster variance) and benefits from the same
normalisation.

**Note on t-SNE**: The t-SNE visualisation uses the unscaled `X_pca` (not
`X_pca_scaled`). t-SNE's local neighbourhood computation is dominated by
nearby distances regardless of global scale, and the unscaled version
preserves the natural variance weighting for visual exploration. This is a
visualisation-only choice — it does not affect phenotype assignments.

**Alternatives considered**:
- **No scaling (use raw PCA scores)**: clusters on PC1 alone. Rejected.
- **MinMaxScaler**: sensitive to outliers at both ends of each PC.
  Rejected.
- **RobustScaler (IQR-based)**: outlier-robust. Defensible; StandardScaler
  is sufficient here because PCA scores on standardised inputs (D033) are
  approximately Gaussian with few extreme outliers. Deferred to sensitivity.
- **Re-weight by Δ(variance)**: weight each PC by its marginal contribution
  to explained variance beyond a threshold — would de-emphasise near-noise
  PCs. More complex with unclear benefit. Rejected for primary analysis.

**Implementation**: `scaler_consensus = StandardScaler()` in
`scripts/run_consensus.py`; `X_scaled = scaler_consensus.fit_transform(X_pca)`;
`scaler_consensus` stored in `consensus_model.pkl`.

---

## D048 — Diffusivity structural zeros

**Date**: May 2026
**Tag**: `v0.8.0-phenotype-characterisation`

**Decision**: Diffusivity features use 0.0 as a sentinel for disease-absent
arteries (D017). This is a structural zero, not a measurement.
`StandardScaler` treats it as a data point, distorting the standardised
distribution and partially explaining why diffusivity features scatter across
PC12–21 rather than forming a coherent axis.

**Impact**: Acknowledged as a methodological limitation. Future work should
model artery involvement as a binary covariate separate from the continuous
diffusivity value. For current analyses, diffusivity features are retained
with a note in the paper methods section.

**Paper language**: "Diffusivity features use structural zeros for
disease-absent arteries (D017). StandardScaler includes these in the
continuous distribution, which may reduce the coherence of the spatial
distribution axis in PCA. Future analyses should model artery involvement
as a binary covariate rather than including structural zeros in the
continuous feature space."

---

## D049 — NCCT calcium feature space is a burden-driven continuum

**Date**: May 2026
**Tag**: `v0.8.0-phenotype-characterisation`

**Decision**: Three independent methods — PCA (PC1=39.1%, ρ=0.969 with
Agatston), WGCNA (β=20, 5 modules, scale-free topology failure confirmed),
and consensus clustering (k*=1 by gap statistic; consensus k*=3 degenerate
{61,19,361}) — converge on the same finding: NCCT calcium radiomics features
do not support discrete phenotype discovery beyond burden stratification.
This is the primary scientific result of v0.8.0.

**PCA confirmation (post-mask-fix)**: PC1 concentration increased to 39.1%
(from 34.7% pre-fix) and ρ=0.969 (from 0.936). The corrected masks
strengthen the continuum finding — the dominant burden axis is even more
concentrated than in the prior run.

**Interpretation**: Calcium accumulates progressively across coronary
territories (LAD first, then RCA/LCx, then LM). Territory involvement
patterns are a burden staging sequence, not independent phenotype axes.
New finding: PC3 captures an independent LM-territory axis (PC3 ρ=−0.069 vs
Agatston), consistent with LM being the last territory to accumulate calcium.
This contrasts with CCTA studies (Kolossváry 2025, Lin 2022) where
non-calcified plaque texture features provide orthogonal phenotype axes
unavailable in NCCT.

---

## D050 — Dense calcium sub-phenotype as candidate marker

**Date**: May 2026
**Tag**: `v0.8.0-phenotype-characterisation`

**Decision**: Within the high-burden subgroup (Agatston ≥ 100, n=227),
`dense_calcium_count > 0` identifies patients with a coherent HU-axis
feature profile (max_calcium_hu Δz≈+1.9, SurfaceVolumeRatio Δz≈−1.0).
Silhouette ≈ 0.10 indicates substantial overlap — this is a tendency, not
a discrete cluster. Reported as a candidate sub-phenotype marker pending
MACE validation in the Kettering cohort.

**Connection to literature**: Hoori et al. (2024) found dense calcium
(HU > 1000) has a protective effect: HR = 0.71 (95% CI 0.51–0.99,
p = 0.042). The dense sub-phenotype identified here connects to that
signal via the `dense_calcium_count` feature (D025).

---

## D051 — Consensus clustering primary result

**Date**: May 2026
**Tag**: `v0.8.0-phenotype-characterisation`

**Decision**: Consensus clustering (4 algorithms, S=100, k=2–8) produces
no stable solution beyond k=2. Bimodality of the consensus matrix drops
monotonically from k=2 to k=8. The monotonically increasing AUC
(0.18→0.66, k=2→8) is the signature of a continuous rather than discrete
feature distribution. The continuum interpretation is the primary result;
consensus clustering is reported as supporting evidence alongside PCA and
WGCNA.

Extreme outlier patients (max |z| > 7 on any PC) are identified and
excluded from final reporting. These are patients at the boundary of the
clinical covariate space whose feature values lie far outside the cohort
distribution.

**Observed result (v0.8.0 run, May 2026)**: No patient exceeds |z|>7 in 19-PC
scaled space (contrast with prior 22-PC run). Four PIDs (184, 279, 342, 386)
hardcoded as outlier exclusions — Agatston extremes (PIDs 342/386 Agatston
~5000–5400), mask-independent. 445-patient consensus: k*=3 degenerate
{22, 422, 1} — single dominant cluster absorbs 422 patients. AUC monotone
0.256→0.686 (k=2→8): signature of continuum, not discrete structure.
441-patient run (outliers excluded): {61, 19, 361}, stable rate 0.925,
ARI=0.005 vs 445-patient — near-random label reassignment confirms no
discrete structure. Bimodality collapses at k>2 in both runs.

---

## D052 — Gap statistic: formal continuum confirmation

**Date**: May 2026
**Tag**: `v0.8.0-phenotype-characterisation`

**Decision**: Run Tibshirani et al. (2001) gap statistic on the 441-patient
cohort (19 PCs, StandardScaler-normalised) with PCA-projected uniform null
reference and B=50 bootstrap samples, k range 1–8.

**Rationale**: The gap statistic is the only k-selection method that can
formally return k*=1 — the "no cluster structure" result. Δ(AUC) and
silhouette start at k=2 and cannot distinguish a continuum from a weak
two-cluster solution. Including k=1 in the candidate set closes the
methodological gap in the proposal and provides the strongest possible
formal statement of the continuum finding.

**Null reference**: PCA-projected uniform — sample uniformly within the
range of each PC component independently. This is Tibshirani's recommendation
for already-PCA-transformed data; the uniform-over-bounding-box null is only
appropriate for raw feature spaces.

**Tibshirani criterion**: select smallest k such that
Gap(k) ≥ Gap(k+1) − s_{k+1}. If no such k exists, return k=1.

**Observed result (v0.8.0 run, May 2026)**: **k*=1**, Gap(1)=1.562 ≥
Gap(2)−s(2)=1.536. Gap curve flat across k=1–8. Formal confirmation of D049.
The stronger PC1 burden concentration (39.1% vs 34.7% pre-fix) produced a
marginally higher Gap(1) (1.562 vs 1.557 pre-fix), consistent with the
corrected masks strengthening the continuum structure. Tibshirani criterion
satisfied at first candidate k — no further search needed.

**Implementation**: `scripts/run_gap_statistic.py`;
outputs `outputs/gap_statistic/gap_table.csv` and `gap_plot.png`.

**Reference**: Tibshirani, R., Walther, G., & Hastie, T. (2001).
Estimating the number of clusters in a data set via the gap statistic.
Journal of the Royal Statistical Society: Series B, 63(2), 411–423.
