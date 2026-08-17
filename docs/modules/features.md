# Stage 3 — `features`

XML-derived scalar features + 3D lesion identification + per-vessel masks + PyRadiomics on the whole-mask. Produces one row per patient with 184 columns: 68 canonical scalar features, 9 metadata + status columns, 107 PyRadiomics features.

## Modules

| Module | Purpose |
|---|---|
| `feature_schema.py` | Canonical 68-key feature name registry. Single source of truth. |
| `agatston.py` | Per-vessel + total Agatston with the **D011 single helper** (`agatston_roi_score`). |
| `lesion_ccl.py` | BFS-based 3D lesion identification (D007) with Z-matched slice indices. |
| `per_vessel_aggregates.py` | Per-vessel volume / mass / mean HU / max HU (XML-derived). |
| `spatial.py` | Lesion counts, diffusivity (D016), Gini, inter-lesion distances, COM. |
| `density_tiers.py` | Per-vessel ROI counts in 4 Agatston HU tiers + dense-calcium count. |
| `per_artery_mask.py` | Filter-then-rasterise per vessel (D008) — reuses `mask_builder`. |
| `radiomics.py` | PyRadiomics wrapper (D009 locked `params.yaml`). |

## What this stage does

Per patient (orchestrated by `scripts/03_features.py`, multi-process):

1. Load native DICOM headers (`load_patient_metadata`) for `slice_positions` and native pixel spacing — header-only, ~10× faster than full image load.
2. Parse XML (`parse_calcium_xml`).
3. Derive `excluded_roi_ids` for this patient from `outputs/02_preprocessed/xml_roundtrip.csv` (non-dirty trips that failed the D002 Max-exact gate).
4. **XML-only paths** (all take `excluded_roi_ids` per D012):
   - Agatston per vessel + total (D011 helper).
   - Per-vessel volume / mass / mean HU / max HU + globals.
   - Density-tier ROI counts + `dense_calcium_count`.
5. **Lesion CCL**: per-vessel BFS with Z-matched slice indices (no more `(num_slices-1)-ImageIndex` inversion bug from v1). Produces 3D `Lesion` dataclasses.
6. **Spatial features** from the `Lesion` list: count, diffusivity, Gini, inter-lesion distances, distance-from-top, center of mass.
7. **PyRadiomics** on the resampled whole-mask using locked config (all 7 IBSI families). Graceful-degrade per **D010** for masks below `minimumROISize=14`: catch the `ValueError`, leave PyRadiomics columns NaN, record `radiomics_status="skipped"`.
8. **Per-artery masks** (built but PyRadiomics on them off by default — `--per-vessel-radiomics` toggles it).
9. Aggregate into one row; write the row + lesion audit + log.

Per-stage outputs in `outputs/03_features/`:

```
features.csv                 # 444 rows × 184 columns (the main artefact)
lesions.csv                  # 3179 rows (one per 3D lesion, audit trail)
feature_extraction_log.csv   # per-patient runtime + counts + radiomics_status
lesion_grouping_probe.csv    # sensitivity probe output (offline)
```

## Why these choices

### Hybrid per-patient feature stack (whole-mask + per-lesion + per-vessel)

- **First-order intensity + texture**: whole-mask PyRadiomics. Texture statistics need many voxels; pooling all calcified voxels per patient gives 100s–1000s of voxels, stable. Per-lesion texture on 15-voxel lesions would be noise.
- **Shape**: whole-mask via PyRadiomics for the aggregatable shape features; per-lesion volume/area via `Lesion` dataclass for spatial aggregation (Gini, lesion count). PyRadiomics shape features on a multi-lesion mask are by convention.
- **Spatial distribution**: pure custom code from XML + lesion CCL. Lesion count, inter-lesion distances, diffusivity (D016), Gini, COM. These are voxel-count-independent and stable at any burden.
- **Per-vessel**: filter-then-rasterise (D008). Same `mask_builder` as stage 2, just with `vessel_filter`. Voxels sum cleanly to whole-mask voxel count.

### Z-matched slice indices everywhere

The `Lesion` dataclass stores CT array slice indices from `slice_matcher.match_roi_to_slice` (D001), not the raw `ImageIndex`. This is what makes the spatial Z values correct in v2 — v1's `(num_slices-1) - ImageIndex` inversion put lesions on wrong slices for the Z computations in `spatial_features._compute_dist_from_top` and the Lesion centroid z.

### Sub-pixel polygon rasterisation everywhere

The per-vessel masks reuse `mask_builder` which uses `cv2.fillPoly` with `shift=4`. Consistent with stage 2's whole-mask. Boundary precision is the same.

### `excluded_roi_ids` is an explicit input to every feature module (D012)

The orchestration script reads `outputs/02_preprocessed/xml_roundtrip.csv` once at start, builds a per-patient exclusion set, and passes it to:

  - `compute_agatston`
  - `group_rois_into_lesions`
  - `compute_per_vessel_aggregates`
  - `compute_density_tiers`
  - `build_per_artery_masks`

Result: every feature value in a row is computed from the same input set. Whole-mask PyRadiomics consumes the stage-2 mask which was already cleaned. Consistency guaranteed.

### `low_burden_flag` instead of pre-gating (D010)

`low_burden_flag = (mask_voxels < 100)` is metadata; no feature is pre-gated. PyRadiomics' own `minimumROISize=14` filters the bottom of the distribution; for those patients the orchestration catches the `ValueError`, NaN's the PyRadiomics columns, and keeps the row's XML-derived features. Downstream filtering on `low_burden_flag` or `radiomics_status` is one column away.

### Single Agatston helper (D011)

`agatston_roi_score(area_cm2, max_hu, slice_thickness_mm)` lives in `agatston.py` and is the only Agatston formula in the codebase. `per_vessel_aggregates.py` does not duplicate it. A regression test asserts both Agatston paths agree at multiple slice thicknesses.

## Module contracts

```python
# feature_schema.py
VESSEL_SUFFIXES: tuple[str, ...]      # ("lad", "rca", "lcx", "lm")
DENSITY_TIERS: tuple[str, ...]         # ("d1", "d2", "d3", "d4")
PER_VESSEL_STEMS: tuple[str, ...]      # 10 stems
GLOBAL_FEATURES: tuple[str, ...]       # 12 keys
def feature_names() -> tuple[str, ...]: ...      # full ordered name list
def zero_features() -> dict[str, float]: ...     # fresh schema-complete zero dict
def n_features() -> int: ...                     # 68

# agatston.py
def density_factor(max_hu: float) -> int: ...
def agatston_roi_score(area_cm2: float, max_hu: float,
                       slice_thickness_mm: float) -> float: ...   # D011 helper
def classify_risk(total: float) -> str: ...                       # "0" / "1-99" / "100-399" / "400+"

@dataclass(frozen=True)
class AgatstonResult:
    total: float
    per_vessel: dict[str, float]
    category: str
    def to_feature_dict(self) -> dict[str, float]: ...

def compute_agatston(parse_result: ParseResult, *,
                     slice_thickness_mm: float,
                     excluded_roi_ids: set | None = None) -> AgatstonResult: ...

# lesion_ccl.py
@dataclass(frozen=True)
class Lesion:
    vessel: str
    roi_keys: tuple[tuple[int, int], ...]
    slice_indices: tuple[int, ...]
    centroid_mm: tuple[float, float, float]
    total_area_mm2: float
    mean_hu_weighted: float
    max_hu: float
    volume_mm3: float
    @property
    def n_rois(self) -> int: ...

def group_rois_into_lesions(parse_result: ParseResult, *,
    slice_positions: Sequence[float],
    pixel_spacing_xy: tuple[float, float],
    slice_thickness_mm: float,
    excluded_roi_ids: set | None = None,
    max_inplane_mm: float = 5.0,
    max_slice_gap: int = 1,
    tolerance_mm: float = 1.5,
) -> dict[str, list[Lesion]]: ...

# per_vessel_aggregates.py
def compute_per_vessel_aggregates(parse_result: ParseResult, *,
                                  slice_thickness_mm: float,
                                  excluded_roi_ids: set | None = None,
                                  ) -> dict[str, float]: ...   # 20 keys

# spatial.py
def gini_coefficient(values: Sequence[float]) -> float: ...
def diffusivity(n_lesions: int, d_first_last_mm: float) -> float: ...   # D016
def compute_spatial_features(lesions_per_vessel: dict[str, list[Lesion]], *,
                             slice_positions: Sequence[float],
                             slice_thickness_mm: float = 3.0,
                             ) -> dict[str, float]: ...   # 26 keys

# density_tiers.py
def density_tier(max_hu: float) -> str | None: ...
def compute_density_tiers(parse_result: ParseResult, *,
                          excluded_roi_ids: set | None = None,
                          ) -> dict[str, float]: ...   # 17 keys

# per_artery_mask.py
def build_per_artery_masks(parse_result: ParseResult,
                           loaded: LoadedPatient, *,
                           excluded_roi_ids: set | None = None,
                           tolerance_mm: float = 1.5,
                           ) -> dict[str, np.ndarray]: ...

# radiomics.py
def create_extractor(params_yaml: Path = DEFAULT_PARAMS_YAML) -> RadiomicsFeatureExtractor: ...
def validate_ct_for_radiomics(ct_array: np.ndarray, pid: str = "") -> None: ...
def extract_pyradiomics(ct_array: np.ndarray, mask_array: np.ndarray,
                        spacing: tuple[float, float, float],
                        extractor, *,
                        label: int = 1, pid: str = "") -> dict[str, float]: ...
```

### Orchestration flow (`scripts/03_features.py` per patient)

```
load_patient_metadata(pid)       ─►  meta (slice_positions, spacing, thickness)
parse_calcium_xml(pid)           ─►  parse_result
excluded_by_pid[pid]             ─►  excluded  (from outputs/02_preprocessed/xml_roundtrip.csv)
np.load(*_ct.npy, *_mask.npy)    ─►  ct, mask

row = zero_features()
row += compute_agatston(...)
row += compute_per_vessel_aggregates(...)
row += compute_density_tiers(...)
lesions = group_rois_into_lesions(...)
row += compute_spatial_features(lesions, ...)

try:
    row += extract_pyradiomics(ct, mask, target_spacing, extractor)
    radiomics_status = "ok"
except ValueError as exc:        # mask below minimumROISize → D010 graceful degrade
    radiomics_status = "skipped"

row + lesions + log → CSV writers
```

## Decisions

- **D007** Lesion grouping rule (BFS, ≤5 mm in-plane, gap=1, skip dirty + excluded).
- **D008** Per-artery masks via filter-then-rasterise.
- **D009** PyRadiomics extractor configuration (locked `params.yaml`).
- **D010** `low_burden_flag = (mask_voxels < 100)` + graceful-degrade for PyRadiomics on tiny masks.
- **D011** Single Agatston thickness-correction helper.
- **D012** `excluded_roi_ids` as explicit input to every feature path.

## Tests (`tests/features/`)

98 unit tests, all passing:

- `test_feature_schema.py` (12) — canonical name list, uniqueness, ordering, zero dict.
- `test_agatston.py` (14) — density factor boundaries, single-helper formula, thickness scaling (D011), classify_risk, per-vessel breakdown, dirty + excluded skipping, empty parse.
- `test_lesion_ccl.py` (17) — BFS edge predicate, same-slice / cross-vessel / gap behaviour, dirty + excluded + too-few-points + unmatched ROIs all skipped, area-weighted centroid, deterministic output keys.
- `test_per_vessel_aggregates.py` (13) — volume/mass formula, area-weighted mean HU, max-of-max, empty-vessel zeros, dirty/excluded skip, thickness scaling.
- `test_spatial.py` (16) — Gini edge cases, diffusivity (D016) full edge-case table, distance metrics, z-sorted first-to-last, dist_from_top clamping, area-weighted COM.
- `test_density_tiers.py` (10) — tier boundary HUs, below-threshold rejection, strict `> 1000` for dense, dirty/excluded skip.
- `test_per_artery_mask.py` (6) — vessel_filter isolation, voxel-count sum, backward-compat with whole-mask, exclusion consistency.
- `test_radiomics.py` (11) — guardrail rejects normalised / low-max / wrong-dtype inputs, geometry wrap, all 7 IBSI families present, feature count in expected range, empty mask returns empty dict.

## Publication-readiness audits

Run after stage 3 via `scripts/03b_cohort_sanity.py`. Outputs
`outputs/03_features/cohort_sanity.json`.

| Audit | Result on 444-patient cohort |
|---|---|
| `row_count_matches_manifest` | OK (444 = 444) |
| `canonical_schema_present` | OK (68 schema keys, none missing) |
| `nan_audit` | OK (NaN only on PyRadiomics columns of `radiomics_status="skipped"` rows) |
| `per_vessel_totals_consistent` | OK (max drift on `agatston_total`, `volume_total_mm3`, `mass_total` = 4.5e-13 / 9.1e-13 / 2.3e-10 — numerical roundoff only) |
| `per_vessel_union_equals_whole` | OK (per-vessel masks OR'd together = whole-mask voxel for voxel, on all 444 patients) |

Additional reported context: 1 patient (PID 116) shows **2 voxels of inter-vessel annotation overlap** (0.08% of their mask). This is the LM-vs-proximal-LAD/LCx annotation overlap phenomenon documented in the dataset analysis report (§ 5.3); the per-vessel masks correctly handle it (their union equals the whole-mask), the small `per_vessel_sum > whole_mask` discrepancy is the overlap arithmetic, not a pipeline bug.

A `run_header.json` is written alongside the features CSV containing the run timestamp, git commit hash, `params.yaml` SHA, Python version, and PyRadiomics / numpy / pandas / SimpleITK / pydicom / OpenCV versions. Anyone reproducing the result has the full dependency baseline.

## Empirical results on the 444-patient cohort

```
Done. ok=444 | low_burden=142 | errors=0 | features.csv columns=184
```

- 444 rows, 184 columns: 68 canonical schema + 9 metadata (`pid`, `kernel`, `scanner_model`, `mask_voxels`, `low_burden_flag`, `roundtrip_quality`, `category`, `radiomics_status`, `radiomics_reason`) + 107 PyRadiomics.
- 422 patients with `radiomics_status="ok"`, 22 with `radiomics_status="skipped"` (mask 6–14 voxels, below PyRadiomics' `minimumROISize=14`).
- 142 patients (32%) have `low_burden_flag=True` (all in Agatston category 1–99, which is consistent).
- 3179 lesions across the cohort (mean 7.5 per patient, max 43).
- No NaN outside the 22 skipped-radiomics rows (and only on PyRadiomics columns there).
- Spearman ρ(`agatston_total`, `max_hu_global`) = 0.88 — coherent cohort-level relationship between burden and density peak.
- Runtime: 24 s for 444 patients at 16 workers (~0.9 patients per worker-second).

### Lesion-grouping sensitivity probe (D007 verification)

10 patients across Agatston quartiles; grouping run at `{3, 5, 8} mm × {gap 1, 2}` (6 settings).

| Region | Behaviour |
|---|---|
| 5 mm / gap=1 vs 5 mm / gap=2 | max ±11.1% — gap parameter has secondary effect |
| 5 mm vs 3 mm | up to +46% lesions (3 mm over-splits) |
| 5 mm vs 8 mm | up to −67% lesions (8 mm over-merges) |
| Low-burden patients (Agatston < 100) | no sensitivity to any threshold (lesion count = 1) |

The default (5 mm, gap=1) sits in the safe middle — conservatively over-split per D007's stated rationale, neighbours-sensitive only for high-burden patients. Accepted as the default for COCA.

## Edge cases handled

- Patients with masks 6–14 voxels (22 cases): PyRadiomics gracefully skipped, XML-derived features intact.
- Empty vessel (no ROIs after exclusion): all per-vessel scalars = 0 (D017).
- Single lesion in a vessel: `diffusivity = 1.0`, distances = 0 (D016, D017).
- Dirty vessel names: never enter any feature path (`vessel=None` upstream).
- Round-trip-failed ROIs (10 ROIs across 7 patients): never enter any feature path (D012).
- ROIs whose `Center` is unmatched in Z (no centre + ImageIndex fallback also fails): not in any lesion, not in any mask.

## Known limitations

- **Texture features on tiny masks**: even where PyRadiomics didn't refuse (mask 15–99 voxels), texture statistics are noisy. `low_burden_flag` flags these. The reduce/analyse stage will run a sensitivity-exclusion analysis (re-run phenotype discovery without `low_burden_flag=True` patients) to confirm whatever structure emerges is robust to this cohort tail.
- **Per-vessel PyRadiomics is off by default**. Enabling it (`--per-vessel-radiomics`) adds 4× extractions per patient (~4 min total). Decision deferred — patient-level whole-mask features are the primary unit.
- **3 mm slice anisotropy**: inherited from preprocess (D005). Affects 3D texture features; consistent across the cohort so phenotyping is not biased, only the absolute interpretation.
- **Binary-vs-partial-volume mask bias**: inherited from preprocess (D002). PyRadiomics consumes the binary mask we built; the systematic boundary-pixel inclusion is a constant across all patients.
- **Lesion grouping is threshold-sensitive at high burden**: the sensitivity probe shows ±50% swings for high-burden patients between 3 mm and 8 mm. The 5 mm/gap=1 default is defensible per D007 but is the single largest parameter to revisit on a new cohort.
