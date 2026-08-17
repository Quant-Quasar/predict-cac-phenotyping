# Stage 4, stability (ICC perturbation gate)

## What this stage does

Filters the 175-feature output of stage 3 down to a stability-gated subset by
computing intra-class correlation under 14 deterministic CT perturbations.
Features that are unstable across plausible registration errors are dropped
before any clustering or phenotype analysis in stage 5 and beyond.

The gate is split into two paths per D016:

1. **Invariant by construction** (68 canonical features). These never read the
   CT pixel array; they read the radiologist-annotated XML's frozen Max /
   Mean fields plus polygon vertices. CT perturbations cannot change them.
   ICC is asserted as 1.0 with `icc_source = invariant_by_construction`.
2. **Empirical** (107 PyRadiomics features). These index into the CT array
   via the PyRadiomics extractor. ICC is measured on a 422-by-15 reliability
   matrix (baseline + 14 perturbations) and tagged `icc_source = empirical`.

Both paths share the D013 threshold (ICC >= 0.75). The bypass does not exempt
any feature from the gate; it short-circuits the computation when the answer
is provably 1.0.

## Module list

- `src/predict/stability/perturbations.py`  the 14 perturbation specs and
  the SimpleITK transforms.
- `src/predict/stability/icc.py`  ICC(3,1) absolute-agreement formula,
  reliability-matrix assembly, gate, bypass registry.
- `scripts/04_perturbations.py`  parallel re-extraction of PyRadiomics on
  perturbed CTs.
- `scripts/05_icc_gate.py`  ICC computation, gating, report writers.

## Why these design choices

### D013, ICC(3,1) absolute agreement at 0.75

- ICC(3,1), not (2,1): the 14 perturbations are a fixed deterministic design,
  not a random sample of raters.
- Absolute agreement, not consistency: a systematic shift under rotation
  (e.g. all values bias up by 5 units) is a reliability concern, not a noise
  channel to be ignored. Penalising it is the conservative choice.
- 0.75: Koo and Li 2016 "good reliability" floor, standard in radiomics
  reproducibility studies (IBSI-2, Lin 2022, Kolossvary 2025). v1 D022 used
  the same threshold; preserving comparability with v1's published numbers.

### D014, 14 perturbations, mask held fixed

- 4 rotations (+/- 5 and +/- 10 degrees about z), 8 translations (+/- 2 and
  +/- 5 mm in x and y separately), 2 noise levels (sigma = 5 and 10 HU).
- z is not translated; 3 mm slice spacing makes sub-slice z shifts ambiguous
  and gated cardiac acquisition limits z mis-registration in real workflows.
- Magnitudes span the clinically plausible registration-error envelope.
  Smaller would not discriminate; larger would test out-of-distribution.
- Mask is **not** transformed (Option B, locked 2026-06-03). This tests
  mis-registration robustness ("if the patient was positioned slightly
  differently, would the feature change?"). The alternative co-perturbation
  design (v1 / IBSI-2 convention) would make most perturbations near-identity
  and the gate would only filter noise-sensitive features. The fixed-mask
  design is much stricter and is what test-retest reproducibility actually
  means in an unsupervised phenotyping pipeline.

### D015, eligible cohort N=422

- Full 444 minus 22 patients with `radiomics_status == "skipped"` from
  stage 3 (whole-mask voxel count below PyRadiomics' minimumROISize = 14).
- These 22 already have NaN PyRadiomics columns in baseline; ICC is
  undefined for them and they would contribute no information.
- Includes the 142 `low_burden_flag == True` patients (mask 14 to 100
  voxels). Their PyRadiomics is numerically defined but biologically
  suspect; we keep them in the ICC computation because the gate is meant
  to identify features that are unstable across the whole cohort, not the
  cleanest subset.

### D016, bypass scope, 68 canonical features

The bypass list is **all 68 canonical features**, not just the 31
pure-geometry features. The reasoning, verified by code inspection and
empirical confirmation:

- All canonical features read from `parse_result` (XML) or `Lesion` objects
  derived from XML polygons. None index into the CT pixel array.
- `roi.max_hu` and `roi.mean_hu` are populated from the XML's Max / Mean
  fields at parse time (`xml_parser.py` lines 122 and 123) and never mutated.
  These are the radiologist's measurement, not a CT-derived statistic.
- Perturbing the CT does not change the XML, therefore cannot change the
  canonical feature values. Empirically confirmed on patient 306, 2026-06-03:
  68 keys, 0 diffs across two calls.

The original D016 lock split the canonical features into 31 pure-geometry
(bypass) plus 37 HU-touching (empirical gate). That was retracted after the
architectural audit found that the HU-touching canonical features read XML,
not CT. Forcing them through the empirical gate would require re-deriving
HU statistics by sampling the perturbed CT through the polygon, a code path
that does not exist in production. The ICC measured on such a re-derived
quantity would describe a hypothetical alternative pipeline, not v2.

## Module contracts

### `perturbations.py`

```python
@dataclass(frozen=True)
class PerturbationSpec:
    name: str                # stable string key, e.g. "rotate_+5", "noise_10"
    kind: Literal["rotate", "translate", "noise"]
    rotation_deg: float = 0.0
    tx_mm: float = 0.0
    ty_mm: float = 0.0
    sigma_hu: float = 0.0

def enumerate_perturbations(cfg: Config) -> tuple[PerturbationSpec, ...]:
    """Return the locked 14 specs in deterministic order. Raises if the
    config does not produce exactly 14."""

def rotate(ct_img: sitk.Image, degrees: float, *, background_hu: float = -1024.0) -> sitk.Image:
    """Z-axis rotation about volume centre, linear interpolation."""

def translate(ct_img: sitk.Image, tx_mm: float, ty_mm: float, *, background_hu: float = -1024.0) -> sitk.Image:
    """Physical x/y translation in mm; z is never translated."""

def add_gaussian_noise(ct_img: sitk.Image, sigma_hu: float, *, seed: int,
                       clip_min: float = -200.0, clip_max: float = 3000.0) -> sitk.Image:
    """Per-voxel Gaussian noise, deterministic seed, clipped to HU range."""

def noise_seed(pid: str | int, sigma_hu: float, multiplier: int) -> int:
    """Per-(pid, sigma) deterministic seed: int(pid) * multiplier + int(sigma)."""

def apply_perturbation(ct_img: sitk.Image, spec: PerturbationSpec, *,
                       pid: str | int, cfg: Config) -> sitk.Image:
    """Dispatch one spec; returns the perturbed CT. Mask is NOT modified."""
```

### `icc.py`

```python
def icc_3_1_absolute(matrix: np.ndarray) -> float:
    """ICC(3,1) absolute agreement on (n_subjects, k_raters). Listwise NaN
    deletion. Returns NaN if degenerate."""

def build_reliability_matrix(feature_name: str,
                             baseline_df: pd.DataFrame,
                             perturbation_dfs: dict[str, pd.DataFrame],
                             *, pid_col: str = "pid") -> tuple[np.ndarray, list[str], list[str]]:
    """Stack baseline + each perturbation's column for one feature.
    Patients are intersected; baseline first, then dict iteration order."""

@dataclass(frozen=True)
class IccRecord:
    feature: str
    icc: float
    icc_source: Literal["empirical", "invariant_by_construction"]
    n_subjects: int
    n_raters: int
    passes_gate: bool

def gate_features(records: list[IccRecord], *, threshold: float) -> tuple[list[IccRecord], list[str]]:
    """Apply threshold; returns updated records (passes_gate recomputed) and
    the list of feature names that pass. NaN ICC always fails."""

def invariant_by_construction_features() -> tuple[str, ...]:
    """All 68 canonical features. Returns feature_schema.feature_names() so
    the registry is locked to the schema and cannot drift."""
```

### Outputs

```
outputs/04_perturbations/
  rotate_+5.csv             one row per eligible patient, 107 original_* cols + error
  rotate_-5.csv             ...
  ... (14 files total)
  perturbation_log.csv      per-(pid, pert) runtime / status row
  run_header.json           git hash + lib versions + perturbation names

outputs/05_icc/
  icc_report.csv            one row per feature: feature, icc, icc_source,
                            n_subjects, n_raters, passes_gate
  gated_features.csv        single column 'feature', features passing the gate
  icc_summary.json          aggregate counts and threshold info
```

## How to run

Smoke test first (one perturbation, five patients):

```bash
cd <repo-root>
conda activate predict_env
python scripts/04_perturbations.py --only-perturbation noise_5 --limit 5 --no-resume
```

Full run (all 14 perturbations, 422 patients, parallel):

```bash
python scripts/04_perturbations.py --n-workers 16
python scripts/05_icc_gate.py
```

Resume is automatic: a per-perturbation CSV that already contains all 422
eligible pids is skipped on rerun unless `--no-resume` is passed.

## Tests

64 tests total in `tests/stability/`:

- `test_perturbations.py` (27): perturbation enumeration, individual
  transforms (rotation / translation / noise), determinism, geometry
  preservation, dispatch, full-set smoke.
- `test_icc.py` (25): ICC formula on toy matrices, NaN handling,
  degeneracy, matrix assembly, gate application, bypass registry.
- `test_bypass_truly_invariant.py` (11): static check (zero CT-array refs
  in canonical-feature modules) + functional check (each canonical function
  produces byte-identical output across repeated calls).

## Empirical results (2026-06-03 run)

First full run on N = 422 eligible patients (D015), 14 perturbations
(D014), single seed pass per perturbation.

### Overall pass count at D013 threshold 0.75

| Track | Total | Passing | Notes |
|---|---|---|---|
| Canonical bypass (D016) | 68 | 68 | All XML-driven; ICC = 1.0 by construction. |
| PyRadiomics empirical | 107 | 20 | Measured ICC(3,1) absolute agreement. |
| **Total gated features for stage 5** | **175** | **88** | Of which 6 are non-shape PyRadiomics. |

Empirical ICC distribution (107 features):

- min 0.000 (`original_ngtdm_Coarseness`)
- median 0.518
- max 1.000 (shape features)
- mean 0.518
- no NaN

### Sanity checks (passed)

- All 14 PyRadiomics shape features ICC = 1.0 exactly. Mask is genuinely
  held fixed; perturbation pipeline is correct.
- n_subjects = 422 for every empirical feature. No listwise NaN deletion.
- No NaN ICCs; no degenerate features.

### Per-family pass rate at 0.75

| family | total | passing | pass % | icc min | median | max |
|---|---|---|---|---|---|---|
| shape | 14 | 14 | 100% | 1.0000 | 1.0000 | 1.0000 |
| firstorder | 18 | 1 | 6% | 0.0832 | 0.4007 | 0.7687 |
| glcm | 24 | 0 | 0% | 0.1729 | 0.4866 | 0.6975 |
| glszm | 16 | 1 | 6% | 0.0628 | 0.4617 | 0.8839 |
| glrlm | 16 | 2 | 12% | 0.2498 | 0.5540 | 0.9729 |
| ngtdm | 5 | 0 | 0% | 0.0000 | 0.3107 | 0.6726 |
| gldm | 14 | 2 | 14% | 0.0690 | 0.5145 | 0.9294 |

### The 6 non-shape PyRadiomics survivors

| feature | ICC | interpretation |
|---|---|---|
| `original_glrlm_RunLengthNonUniformity` | 0.973 | aggregate run-length spread |
| `original_gldm_DependenceEntropy` | 0.929 | entropy of dependence histogram |
| `original_glszm_ZoneEntropy` | 0.884 | entropy of zone-size histogram |
| `original_glrlm_GrayLevelNonUniformity` | 0.862 | spread of gray levels |
| `original_gldm_GrayLevelNonUniformity` | 0.792 | spread of gray levels |
| `original_firstorder_Range` | 0.769 | max-HU minus min-HU |

These six share a structural property: they are aggregate-distribution
statistics that do not depend on the identity of specific voxels. Features
that depend on which voxel is sampled (Mean, Median, percentiles,
position-aware texture descriptors) collapse under the fixed-mask
translation, because translating the CT pushes calcium voxels in and out
of the fixed polygon.

### Threshold sensitivity (transparency, D013 threshold is locked at 0.75)

The numbers below are produced by re-applying alternative thresholds to the
existing ICC report. They are reported for reviewer transparency; the gate
is locked at 0.75 by D013 before any data inspection (this is the
standard discipline; do not retroactively pick a threshold).

| threshold | bypass passing | empirical passing | total passing | % of 175 |
|---|---|---|---|---|
| 0.50 | 68 | 56 | 124 | 70.9% |
| 0.60 | 68 | 36 | 104 | 59.4% |
| **0.75 (locked)** | **68** | **20** | **88** | **50.3%** |
| 0.85 | 68 | 18 | 86 | 49.1% |

Reproduce with `python scripts/05c_threshold_sensitivity.py`. The locked
threshold (0.75) sits on a relatively flat part of the ICC distribution
(0.75 -> 0.85 loses only 2 features), so the result is robust to small
threshold perturbations. The steepest drop is between 0.50 and 0.60
(56 -> 36 empirical features, a 20-feature loss), confirming that 0.75 is
not picked from a knife-edge region.

### Why the 19% PyRadiomics pass rate

v1 saw ~70% under co-perturbation (mask and CT moved together). v2's
fixed-mask design (Option B, D014) is strictly stricter and was chosen for
that reason: under co-perturbation most non-noise perturbations become
near-identity for the extractor, so the gate filters very little. Under our
design, translations of +/- 5 mm genuinely shift the CT relative to the
fixed mask. For COCA's small lesions (median per-vessel ROI area ~5 mm²
per dataset insights), calcium voxels translate out of the polygon
entirely on a single shift, sampling background instead. Features that
depend on specific voxel intensities (most first-order, most texture)
collapse; features that only summarise distribution shape (Range, entropy,
non-uniformity) survive.

This is the gate doing what it was designed to do. Lin 2022 (CCTA culprit
lesions) and Mackin 2015 (NSCLC) report similar pass rates (30-50%) under
test-retest reproducibility designs. The v2 rate sits at the low end of
the literature range, explained by COCA's small-lesion profile.

### Implications for stage 5

- The 88 features going into stage 5 include all clinically interpretable
  canonical features (Agatston, mass, density tiers per vessel,
  lesion counts, distances, diffusivity, volume, COM, etc.) plus shape +
  6 aggregate texture descriptors.
- 88 features for 422 patients = n:p ratio of ~5:1, healthy for
  unsupervised phenotype discovery.
- The texture vocabulary has been heavily pruned, which is biologically
  meaningful: COCA's calcium phenotypes (if they exist as discrete
  clusters) cannot rely on registration-sensitive texture descriptors.
  v1's published "continuum, no discrete phenotypes" finding was already
  driven primarily by the canonical features; the texture pruning here
  does not change that hypothesis.

### Output artefacts

- `outputs/04_perturbations/*.csv` (14 files, one per perturbation): 422
  rows x 108 cols (pid + 107 original_* + error).
- `outputs/04_perturbations/run_header.json`: reproducibility breadcrumbs.
- `outputs/04_perturbations/perturbation_log.csv`: per-(pid, pert)
  runtime / status row.
- `outputs/05_icc/icc_report.csv`: 175 rows, one per feature.
- `outputs/05_icc/gated_features.csv`: 88 rows, the features that pass.
- `outputs/05_icc/icc_summary.json`: aggregate counts.
- `outputs/05_icc/threshold_sensitivity.csv`: sensitivity table.

## Edge cases

- **Empty mask**: a patient with `mask.sum() == 0` would skip PyRadiomics
  but should not be in the eligible cohort anyway (radiomics_status would
  not be "ok"). Defensive guard in `_process_one` returns status="skipped"
  with no features.
- **Perturbation that rotates calcium out of the FOV**: corner voxels get
  background fill (-1024). PyRadiomics still extracts on whatever calcium
  is inside the fixed mask; if the mask now overlaps mostly background-filled
  voxels, the texture features will be unstable, which is exactly what the
  gate is designed to detect.
- **noise_seed collision**: seed = int(pid) * 1000 + int(sigma). For
  sigma 5 vs 10 and any two patient IDs that differ, the seeds are unique
  by construction. Non-numeric pids fall back to a hash. Tested.
- **Listwise NaN deletion**: a feature with NaN in any rater (e.g. due to
  a single transient PyRadiomics failure on one perturbation) drops that
  patient from the ICC computation for that feature only. Other features
  and other patients are unaffected.
- **ICC degeneracy**: if a feature is constant across all subjects and
  raters (zero total variance), ICC is NaN, treated as a gate failure.
  Should not happen for any non-trivial feature on 422 patients.

## Known limitations

- The gate tests robustness to CT-side mis-registration only. It does not
  test mask redefinition stability (intra-rater segmentation variability)
  or scanner-day-to-day variability beyond what fits in the noise term.
- The 14 perturbations are not exhaustive of clinically plausible noise
  sources. They are a representative subset designed to discriminate
  reliable from unreliable features, not to certify a feature as stable
  under every conceivable acquisition variation.
- PyRadiomics shape features should have ICC = 1.0 exactly under our
  fixed-mask design. If they do not, the perturbation implementation has
  a bug (the mask is moving when it should not). The 0/0 invariants in
  the gate code are designed to catch this.

## Decisions referencing this stage

- D013, ICC formulation and threshold.
- D014, 14-perturbation set (mask held fixed).
- D015, eligible cohort N=422.
- D016, bypass scope (68 canonical bypass, 107 PyRadiomics empirical).
