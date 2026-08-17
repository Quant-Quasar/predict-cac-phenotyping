# D005 — Target voxel grid for resampling

**Date**: 2026-06-01
**Stage**: preprocess
**Status**: Active
**Module**: `src/predict/preprocess/resampling.py`

## Decision

Resample CT and mask to a fixed voxel grid: **`0.5 × 0.5 × 3.0 mm`**.

- In-plane (0.5 mm): a common grid for the cohort. Native pixel spacing
  ranges 0.26–0.71 mm with median ≈ 0.38. About half the cohort downsamples,
  half upsamples — no systematic information invention.
- Z (3.0 mm): preserves native slice thickness for 99% of the cohort. The
  remaining 4 GE patients at 2.5 mm are excluded by D004; if reintroduced
  they would upsample slightly in z.

## Anisotropy note

This grid is **not isotropic**. The z spacing is 6× the in-plane. Texture
features (GLCM/GLSZM/GLRLM/NGTDM/GLDM) computed in 3D will therefore weight
in-plane voxel pairs differently from z-pairs. This is acknowledged as a
limitation of NCCT calcium radiomics at 3 mm slice thickness and is documented
in the features-stage notes.

The function is named `resample_to_target` (not `to_isotropic`) to avoid the
misleading naming used in v1.

## Alternatives considered

- **True isotropic (1.0 × 1.0 × 1.0)**: requires synthesising z data between
  3 mm slabs. Texture features would reflect interpolation artefacts more
  than real biology. Rejected.
- **Native (no resample)**: PyRadiomics shape and texture features become
  patient-specific by spacing — not comparable across the cohort. Rejected.
- **0.4 × 0.4 × 3.0** (closer to median native 0.38): defensible. 0.5 is
  preferred for the round-number reproducibility and slightly cheaper
  interpolation; the difference is within the ICC perturbation envelope.

## Verified by

- `tests/preprocess/test_resampling.py`
- `outputs/02_preprocessed/spacing.json` records the active grid.
