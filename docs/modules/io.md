# Stage 1 — `io`

DICOM loading, XML parsing, patient discovery, spacing metadata. The entry point of the pipeline; everything downstream consumes its outputs.

## Modules

| Module | Purpose |
|---|---|
| `dicom_loader.py` | Load patient CT series into SimpleITK with Z-ascending slice order |
| `xml_parser.py` | Parse Apple plist calcium annotation XMLs |
| `patient_discovery.py` | Build the cohort manifest with exclusions applied |
| `spacing.py` | Persist target voxel grid metadata as JSON |

## What this stage does

1. **Discovery** — intersect `data/raw/` (DICOM patient folders) with `data/calcium_xml/` (XML annotation files), drop patients in the configured exclusion list, drop GE scanners, label each remaining patient with scanner/kernel/thickness. Produces `outputs/01_manifest/manifest.csv` and `exclusions.csv`.
2. **Per-patient load** — given a `pid`, return a `LoadedPatient` containing the CT image and the per-slice physical Z positions (the load-bearing primitive for D001 slice matching).
3. **Per-patient XML parse** — return a `ParseResult` with all non-empty ROIs, vessel names normalised, dirty-name set surfaced, and the `Center` field carried through verbatim.

## Why these choices

### Z-ascending slice sort, not InstanceNumber or filename
Empirically: `InstanceNumber` is not always monotonic in Z (patient 78 shows this). Filename order is scanner- and protocol-dependent. `ImagePositionPatient[2]` (IPP[2]) is the only authoritative physical Z per DICOM. Sorting by IPP[2] ascending makes `array[0]` correspond to the most-negative-Z slice, and that ordering is what we hand to `SimpleITK.ImageSeriesReader`. Downstream, the round-trip check (D002) verified this is the order the XML's `Center.z` expects.

### Multi-series rule: largest slice count (D006)
v1 used "lowest SeriesNumber". That broke on two real patients in the cohort:
- **388** — SeriesNumber 2 is a 1-slice scout; SeriesNumber 3 is the 44-slice diagnostic scan. v1 picked the scout.
- **159** — one DICOM has a missing `SeriesInstanceUID`, gets isolated in a 1-file group; v1 selected that group and SimpleITK could not read the file.

The annotated scan is reliably the longest series in the COCA folder structure. D006 sorts by slice count (desc) first, then SeriesNumber, then UID. Patient 78 (two same-length series) still falls through to the SeriesNumber tiebreak — same behaviour as v1.

### Kernel normalisation
Siemens stores `ConvolutionKernel` as a multi-valued tag (`Qr36d\2`). pydicom hands it back as a `MultiValue` whose `str()` is the list repr `"['Qr36d', '2']"` — ugly and unstable for joins/groupbys later. `_normalise_kernel` joins it with `/` (→ `"Qr36d/2"`) so the manifest column and downstream kernel-harmonisation grouping are deterministic.

### Exclusions at discovery (D004)
Every paper-relevant exclusion is applied once, at `discover_patients`. Downstream stages read the manifest and never re-derive eligibility. The audit trail (`exclusions.csv`) records every excluded PID with reason: `no_dicom`, `no_xml`, `config_excluded`, `ge_scanner`.

### `peek_patient_header` for cheap labelling
Discovery only needs scanner/kernel/thickness — it reads one DICOM per patient, not the full series. The full `load_patient_dicom` is invoked only when the actual CT volume is needed (stage 2).

### XML parser exposes `Center`, not just `ImageIndex`
v1's parser only kept `ImageIndex`. v2 keeps `Center[2]` for Z-coordinate matching (D001), plus `Point_px` (2D) and `Point_mm` (3D). The XML's pre-computed HU stats (`Mean`, `Max`, `Min`, `Total`) are carried verbatim so the round-trip checker can compare without re-reading the original CT through OsiriX's renderer.

### Dirty vessel names
Five entries in the cohort have garbage names (`"555614876"`, `"Unnamed"`, `"1"`, …). The parser sets `roi.vessel = None` for any name not in `VESSEL_NAME_MAP` and surfaces them as a set in `ParseResult.dirty_vessel_names`. Mask building skips these by default; round-trip records them with `matched_via="dirty"` and excludes them from pass-rate counts.

## Decisions
- D001 — Z-coordinate matching (this stage provides the `slice_positions` it needs).
- D004 — Cohort exclusions applied at discovery.
- D006 — Multi-series reduction rule (supersedes v1 D012).

## Module contracts

```python
# dicom_loader.py
@dataclass(frozen=True)
class LoadedPatient:
    pid: str
    ct_sitk: sitk.Image
    slice_positions: tuple[float, ...]   # IPP[2] per array slice, ascending
    pixel_spacing: tuple[float, float, float]
    slice_thickness: float
    n_slices: int
    scanner_model: str
    kernel: str                          # normalised, e.g. "Qr36d/2"
    manufacturer: str
    series_uid: str

def load_patient_dicom(pid: str, data_root: Path) -> LoadedPatient: ...
def peek_patient_header(pid: str, data_root: Path) -> dict: ...
def _select_single_series(file_meta: list[dict], pid: str) -> list[dict]: ...
def _normalise_kernel(value) -> str: ...

# xml_parser.py
@dataclass(frozen=True)
class ROI:
    vessel_raw: str
    vessel: str | None                   # LAD/RCA/LCx/LM or None if dirty
    area_cm2: float
    mean_hu: float; max_hu: float; min_hu: float; total_hu: float
    n_points: int
    points_px: tuple[tuple[float, float], ...]
    points_mm: tuple[tuple[float, float, float], ...]
    center_xyz: tuple[float, float, float] | None

@dataclass(frozen=True)
class SliceAnnotation:
    image_index: int
    rois: tuple[ROI, ...]

@dataclass(frozen=True)
class ParseResult:
    pid: str
    slices: tuple[SliceAnnotation, ...]
    dirty_vessel_names: tuple[str, ...]
    n_active_rois: int
    n_dropped_zero_point: int

def parse_calcium_xml(pid: str, xml_dir: Path) -> ParseResult: ...

# patient_discovery.py
@dataclass(frozen=True)
class PatientRecord:
    pid: str; raw_path: Path; xml_path: Path
    manufacturer: str; scanner_model: str; kernel: str
    slice_thickness: float

@dataclass(frozen=True)
class DiscoveryResult:
    included: tuple[PatientRecord, ...]
    excluded_no_dicom: tuple[str, ...]
    excluded_no_xml: tuple[str, ...]
    excluded_by_config: tuple[str, ...]
    excluded_ge: tuple[str, ...]

def discover_patients(
    data_root: Path, *,
    exclude_pids: tuple[str, ...] = (),
    exclude_ge_scanners: bool = True,
) -> DiscoveryResult: ...
def manifest_rows(result: DiscoveryResult) -> list[dict]: ...

# spacing.py
def parse_spacing(value: str) -> tuple[float, float, float]: ...
def save_spacing_metadata(path: Path, target_spacing: tuple[float, float, float]) -> None: ...
def load_spacing_metadata(path: Path) -> tuple[float, float, float]: ...
```

## Tests (`tests/io/`)
- `test_config.py` — config loader, default exclusions, vessel map round-trip.
- `test_dicom_loader.py` — `_select_single_series` rules (scout/scan, equal counts, UID tiebreak, modality filter), kernel normalisation, multi-series warning logging, integration on patients 1 and 78.
- `test_xml_parser.py` — vessel normalisation, dirty-name detection, zero-point ROI filtering, `Center` parsing (valid / missing / malformed), HU stat preservation.
- `test_patient_discovery.py` — intersection logic, exclusion application, GE filter, manifest metadata, audit CSV.
- `test_spacing.py` — parse/save/load round-trip, validation errors.

39 tests total, all passing.

## Empirical results on the 449-patient COCA cohort

```
Discovery: 444 included, 1 config-excluded, 4 GE-excluded, 2 no-DICOM, 0 no-XML
Kernel breakdown:
  Qr36d/2    235 (SOMATOM Force)
  I30f/3     207 (SOMATOM Definition Flash)
  B35f         1
  I36f/3       1
```

The 2 no-DICOM patients (12, 197) match the dataset report. 1 config-exclusion (268) handles the replaced-series patient. 4 GE patients held out from the primary analysis as the dataset analysis recommends.

## Edge cases handled
- Patient 78 (multi-series, same Z): D006 SeriesNumber tiebreak → 34 slices, correct series.
- Patient 388 (1-slice scout + 44-slice scan): D006 slice-count rule → 44-slice series.
- Patient 159 (broken DICOM file with missing UID): empty-UID fallback + slice-count rule → 43-slice series, broken file isolated and skipped.
- Patients with non-CT modality entries: `_select_single_series` filters them out, warns if no CT remains.
- Dirty vessel names (5 entries across 4 patients): `vessel=None`, surfaced in `dirty_vessel_names`.

## Known limitations
- No de-duplication of duplicate-Z slices within a selected series. Patient 159 has 1 such pair; ROIs whose XML references the duplicate-Z position via `ImageIndex` (rather than `Center`) cannot be reliably placed (3 ROIs affected — caught by D002 round-trip and excluded from the mask).
- pydicom emits noisy "Invalid value for VR UI" warnings for many COCA files (non-standard UID format). These are cosmetic and do not affect loading.
