# Cohort and Data

Source: Stanford COCA — gated non-contrast cardiac CT with expert XML calcium annotations.

## Counts
- Patients in `raw/`: 449
- XML annotation files: 451
- Excluded (no DICOM): 12, 197
- Excluded (replaced series, 1.5 mm → 3.0 mm mismatch): 268
- GE scanners held out from primary analysis: 4 patients

**Primary cohort**: 446 patients (Siemens only), 442 after GE exclusion.

## Acquisition
- Slice thickness: 3.0 mm (445 / 449)
- KVP: 120 (446 / 449)
- Image dim: 512 × 512
- Pixel spacing: 0.26 – 0.71 mm (median ≈ 0.38)
- Two dominant kernels: Qr36d (52%), I30f (46%) — confound; harmonised in `preprocess.kernel_harmonise`.

## Annotations
- Format: Apple plist XML (one per patient)
- Vessels: LAD, RCA, LCx, LM (left main)
- Total ROI instances: 6,232
- Dirty vessel names: patients 238, 398, 415, 421 — cleaned in `io.xml_parser`.

## Class distribution
- Zero-calcium patients: **0**. Every patient has at least one annotated calcium deposit. The analytical question is therefore within-burden sub-structure, not burden detection.

## Demographics
- Sex, age, race: not in dataset. No demographic confound checks possible.
