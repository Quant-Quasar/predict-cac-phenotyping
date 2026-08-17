# Stage 2 — `preprocess`

Mask building, resampling, HU clipping, XML correctness gating. Turns the raw DICOM + XML pair into the `(ct.npy, mask.npy)` pair that radiomics consumes.

## Modules

| Module | Purpose |
|---|---|
| `slice_matcher.py` | Map an ROI to its CT slice index by physical Z (D001) |
| `mask_builder.py` | Rasterise 3D binary mask from XML polygons; supports round-trip-driven exclusion |
| `resampling.py` | Resample CT + mask to the target voxel grid (D005) |
| `hu_handling.py` | Clip HU to a safe range and flag metal artefacts (D003) |
| `validate.xml_roundtrip` *(under `validate/`, but invoked here)* | Per-ROI HU sanity check (D002) |

## What this stage does

For each patient in the manifest:

1. **Load** CT + XML via stage 1.
2. **Round-trip first** — for every ROI in the parsed XML, find its slice via D001 Z-matching, rasterise the polygon onto a fresh 2D mask, recompute Mean/Max/Min HU from voxels, compare against XML stats. Produces a list of `ROITrip` records.
3. **Identify failing ROIs** — non-dirty trips whose Max HU does not exactly match the XML's Max HU. Collected as `(image_index, roi_idx_in_slice)` tuples.
4. **Build the 3D mask** — rasterise every non-dirty ROI EXCEPT the failing ones. cv2 sub-pixel polygon fill, integer-grid result.
5. **Integrity check** — assert mask voxel count > 0 (calcium-positive cohort; an empty mask is a bug).
6. **Resample** — CT (B-spline) and mask (nearest-neighbour) to `0.5 × 0.5 × 3.0 mm` (D005).
7. **Clip HU** to `[clip_min, clip_max]` (D003 default `[-200, 3000]`).
8. **Metal flag** — any mask voxel > metal_threshold flags the patient; recorded but not excluded.
9. **Save** `{pid}_ct.npy` (int16 HU) and `{pid}_mask.npy` (uint8 binary).

Per-stage outputs in `outputs/02_preprocessed/`:

```
{pid}_ct.npy           # int16, shape (n_slices_resampled, H, W), HU clipped
{pid}_mask.npy         # uint8 {0,1}, same shape
spacing.json           # target voxel grid
preprocess_report.csv  # per-patient counts, roundtrip rate + quality, metal flag
xml_roundtrip.csv      # per-ROI deltas across the whole cohort (audit trail)
errors.csv             # patients that failed (if any)
```

## Why these choices

### Slice matching by Z, not ImageIndex (D001)
The XML's `Center` field stores the lesion centroid in world coordinates. Matching `Center[2]` against the loaded `slice_positions` is robust to multi-series quirks, duplicate-Z slices, and OsiriX's internal counter. Empirically verified on patient 306: voxel Max HU matches XML Max exactly with direct mapping, fails completely under v1's inverted ImageIndex math. Tolerance is half the slice thickness (1.5 mm at 3 mm native).

`ImageIndex` direct mapping is kept as a fallback for ROIs without a `Center` field. The fallback is unreliable on patient 159 (Case C — duplicate Z) and produces wrong-slice voxels for 3 ROIs; D002 round-trip catches these and they are excluded from the mask (see "Round-trip-driven cleaning" below).

### Sub-pixel polygon rasterisation
XML `Point_px` vertices are floats. We pass them to `cv2.fillPoly` with `shift=4` (1/16 px). For our typical polygon precision this changes nothing vs int-rounded vertices (verified empirically), but the code is correct and forward-compatible if higher-precision XML data ever appears. The mask remains a binary integer-grid result — pixel inclusion is decided by pixel center, which is the standard radiomics convention.

### Round-trip-driven cleaning (D002)
The round-trip is **mandatory** in v2 and runs **before** mask building. Its hard gate is `voxel_max == xml_max` exactly (no tolerance). The Mean is informational (the partial-volume bias against OsiriX produces a consistent 20–60 HU offset that's not a slice-mapping problem).

Failing ROIs are passed as an exclusion set to `build_3d_mask`. The mask is built from passing ROIs only. This means a patient whose XML contains a few bad ROIs still gets a clean mask of their good calcium, instead of a contaminated mask with wrong-slice polygons.

### Target voxel grid `0.5 × 0.5 × 3.0 mm` (D005)
- In-plane 0.5 mm is a round-number common grid for the cohort (native pixel spacing is 0.26–0.71 mm, median ~0.38). About half the cohort is mildly downsampled, half mildly upsampled.
- Z = 3.0 mm preserves native slice thickness. We do NOT resample z to a finer grid; that would synthesise data between 3 mm slabs.
- It is NOT isotropic. The z dimension is 6× the in-plane. The function is named `resample_to_target` (not `to_isotropic`) so this is unambiguous. Texture features (GLCM/GLSZM/etc.) computed in 3D weight in-plane and z voxel pairs differently — a known limitation of NCCT at 3 mm acknowledged in the features-stage notes.

### HU clipping `[-200, 3000]` (D003)
Wide enough to preserve the dense-calcium tail (Hoori's protective-signal HU > 1000) and most metal-artefact pixels (so they can be flagged by `flag_metal_artifact`). No display HU window in pipeline outputs — radiomics consumes raw HU; saving a windowed copy was v1 waste.

### Mask integrity check
A calcium-positive cohort means every patient has at least one annotated lesion. An empty mask post-build or post-resample indicates a bug (most likely all ROIs excluded by the round-trip, which itself signals a deeper problem). The orchestration returns `status=error` and the patient is logged to `errors.csv`.

### Patient quality flag
Each patient is tagged in `preprocess_report.csv`:
- `roundtrip_quality = ok` — all non-dirty ROIs pass.
- `roundtrip_quality = partial` — ≥ 95% pass.
- `roundtrip_quality = poor` — < 95% pass.

This lets downstream stages filter on a single column if needed. Defaults: all qualities accepted.

## Module contracts

```python
# slice_matcher.py
def match_roi_to_slice(
    roi: ROI,
    slice_positions: Sequence[float],
    tolerance_mm: float = 1.5,
) -> int | None:
    """Closest-Z slice index; None if delta > tolerance or roi.center_xyz is None."""

def fallback_image_index_to_slice(image_index: int, n_slices: int) -> int | None:
    """Direct mapping fallback (no inversion). Returns None if out of range."""

# mask_builder.py
@dataclass(frozen=True)
class MaskBuildReport:
    pid: str
    n_rois_total: int
    n_rasterised: int
    n_skipped_dirty: int
    n_skipped_too_few_points: int
    n_skipped_no_match: int
    n_matched_by_fallback: int
    n_excluded_by_roundtrip: int = 0

def build_3d_mask(
    parse_result: ParseResult,
    loaded: LoadedPatient,
    *,
    tolerance_mm: float = 1.5,
    skip_dirty: bool = True,
    excluded_roi_ids: set | None = None,    # {(image_index, roi_idx_in_slice), ...}
) -> tuple[np.ndarray, MaskBuildReport]: ...

def mask_to_sitk(mask: np.ndarray, ct_sitk: sitk.Image) -> sitk.Image: ...

# resampling.py
def resample_to_target(
    ct_sitk: sitk.Image,
    mask_sitk: sitk.Image | None,
    target_spacing: tuple[float, float, float],
    *,
    ct_default_value: float = -1000.0,
    mask_default_value: int = 0,
) -> tuple[sitk.Image, sitk.Image | None]: ...

# hu_handling.py
def clip_hu(ct_array: np.ndarray, clip_min: int, clip_max: int) -> np.ndarray: ...
def flag_metal_artifact(
    ct_array: np.ndarray, mask_array: np.ndarray, threshold: int,
) -> bool: ...

# validate.xml_roundtrip (invoked here, lives in validate/)
@dataclass(frozen=True)
class ROITrip:
    pid: str
    image_index: int
    roi_idx_in_slice: int
    vessel: str | None
    matched_slice_idx: int | None
    matched_via: str        # "center" | "image_index_fallback" | "unmatched" | "dirty"
    n_xml_points: int
    xml_mean: float; xml_max: float; xml_min: float
    voxel_n: int
    voxel_mean: float; voxel_max: float; voxel_min: float
    delta_mean: float; delta_max: float; delta_min: float
    passes: bool                  # max-exact gate (D002)
    reason: str

def xml_roundtrip_check(
    parse_result: ParseResult,
    ct_array: np.ndarray,
    slice_positions: Sequence[float],
    *, tolerance_mm: float = 1.5,
    max_max_delta: float = 0.0,        # HARD gate
    max_mean_delta: float = 200.0,     # informational
    include_dirty: bool = False,
) -> list[ROITrip]: ...

def pass_rate(trips: Sequence[ROITrip]) -> float: ...
def failed_roi_ids(trips: Sequence[ROITrip]) -> set: ...
def trips_to_rows(trips: Sequence[ROITrip]) -> list[dict]: ...
```

### Orchestration flow (`scripts/02_preprocess.py` per patient)

```
load_patient_dicom(pid) ───►  loaded
parse_calcium_xml(pid)  ───►  parse_result
xml_roundtrip_check(parse_result, ct_native, slice_positions)
                         ───►  trips
failed_roi_ids(trips)    ───►  excluded
build_3d_mask(parse_result, loaded, excluded_roi_ids=excluded)
                         ───►  mask, build_report
mask_to_sitk → resample_to_target → clip_hu → flag_metal_artifact
                         ───►  ct_resampled, mask_resampled
np.save / report row / quality flag
```

## Decisions
- D001 — Z-coordinate matching as primary ROI→slice mapping.
- D002 — XML stat round-trip as preprocessing correctness gate (Max-exact, Mean informational).
- D003 — No display HU window in pipeline outputs.
- D005 — Target voxel grid `0.5 × 0.5 × 3.0 mm`.

## Tests
- `tests/preprocess/test_slice_matcher.py` — Z-match closest pick, tolerance reject, no-center handling, fallback direct mapping.
- `tests/preprocess/test_mask_builder.py` — mask lands on stamped slice (regression for v1 D011 flip), dirty-vessel skip, exclusion set behaviour, fallback path, empty parse.
- `tests/preprocess/test_resampling.py` — spacing change, physical extent preservation, NN preserves binary mask.
- `tests/preprocess/test_hu_handling.py` — clip behaviour, metal flag inside/outside mask, empty-mask, shape mismatch.
- `tests/validate/test_xml_roundtrip.py` — synthetic-stamp passes with delta=0, simulated Z-flip fails, dirty skip, fallback path, `roi_idx_in_slice` tracking, `failed_roi_ids` predicate.

31 preprocess + validate tests, all passing.

## Empirical results on the 444-patient cohort

```
Done. ok=444 | perfect=437 partial=7 poor=0 | rois_cleaned=10 | metal_flagged=0 | errors=0
```

- 444 patients preprocessed end-to-end, 0 errors.
- 437 patients (98.4%) have every ROI passing the Max-exact gate.
- 7 patients have one or more ROIs that failed; these are excluded from the saved mask but logged in `xml_roundtrip.csv`.
- 10 ROIs total excluded from masks (0.17% of ~6,000 cohort ROIs).
- 0 metal-artefact flags.
- Output size: 9.7 GB (444 × 2 `.npy` files + reports).

### Breakdown of the 10 excluded ROIs

| Patient | Count | Type | Cause |
|---|---|---|---|
| 159 | 3 | `image_index_fallback` landing on air/soft-tissue slices | XML has no `Center` for these ROIs; patient 159 has a duplicate-Z slice (dataset report Case C) so ImageIndex math is off |
| 126 | 1 | `delta_max=+378` on dense ROI | Binary mask catches a bright pixel just outside OsiriX's float polygon (adjacent dense deposit) |
| 184 | 1 | `delta_max=+207` on dense ROI | Same as 126 |
| 184 | 1 | `mean_delta=-260` on ROI | XML polygon extends partially into air outside body |
| 238 | 1 | `delta_max=+76` on dense ROI | Same partial-volume adjacency as 126 |
| 298 | 1 | `delta_max=+250` on dense ROI | Same |
| 347 | 1 | `delta_max=+40` on ROI | Same |
| 433 | 1 | `delta_max=+52` on dense ROI | Same |

None are systemic. None invalidate slice mapping. All are individually inspectable in `xml_roundtrip.csv`.

## Edge cases handled
- **Patient 388** (1-slice scout + 44-slice scan): the 44-slice scan is loaded by D006; all 22 of 388's ROIs match correctly.
- **Patient 159** (broken DICOM file, duplicate-Z slice): broken file skipped by D006; 3 fallback-matched ROIs excluded from mask via D002.
- **Patient 268** (1.5 mm annotation series replaced with 3.0 mm): excluded by config (D004) — included masks would be spatially inaccurate.
- **GE scanners** (4 patients): excluded by config (D004) — different kernel and slice thickness.
- **Dirty vessel names** (5 entries, 4 patients): `vessel=None`, skipped by `mask_builder`, recorded as `dirty` in the round-trip.
- **Multi-series same-Z** (14 patients including 78): D006 + Z-sort handles correctly without de-duplication; same-Z files are not duplicated in the output because `SimpleITK.ImageSeriesReader` follows the file list order.
- **Polygon vertices with < 3 points**: skipped, counted in `n_skipped_too_few_points`.
- **Patient with no DICOM** (12, 197): caught at discovery (D004).

## Known limitations
- **Binary mask is fatter than OsiriX's anti-aliased polygon at boundaries.** This produces a systematic Mean HU offset (~20–60 HU lower than XML on each ROI). The Max is unaffected. The bias is constant across the cohort; phenotyping is unaffected. PyRadiomics and every standard radiomics tool consume binary masks too, so this is the standard convention.
- **No partial-volume rasterisation in the round-trip checker.** Replicating OsiriX's anti-aliased Mean exactly is possible (super-sample each boundary pixel) but would not change the binary mask we feed forward. Deferred.
- **3 mm slice anisotropy.** The voxel grid is 0.5 × 0.5 × 3.0; texture features computed in 3D will weight axial-pair statistics differently from in-plane pairs. Documented and accepted as a property of NCCT calcium scoring at this slice thickness.
- **No kernel harmonisation here.** The Qr36d (52%) / I30f (46%) mix is a confound, but ComBat or within-kernel z-score operates on extracted features, not on images. It lives in stage 5 (`reduce`).
