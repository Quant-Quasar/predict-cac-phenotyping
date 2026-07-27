"""ROI-to-slice mapping by physical Z coordinate.

The XML's per-ROI ``Center`` field carries the lesion centroid in world
coordinates (x, y, z) in millimetres. Each DICOM slice's
``ImagePositionPatient[2]`` is its world Z. Matching ``Center[2]`` against the
slice Z list (provided by ``predict.io.dicom_loader.LoadedPatient``) gives an
unambiguous 0-based array index into the CT volume.

If the closest slice's Z is more than ``tolerance_mm`` away from the ROI
centre, the match is rejected (returns ``None``). This catches the patient
268 scenario where the DICOM series spacing no longer matches the XML's
original 1.5 mm reconstruction.

Decisions referencing this module:
    D001 — Z-coordinate matching as primary
"""
from __future__ import annotations

from typing import Sequence

from predict.io.xml_parser import ROI


def match_roi_to_slice(
    roi: ROI,
    slice_positions: Sequence[float],
    tolerance_mm: float = 1.5,
) -> int | None:
    """Return the 0-based array index of the slice matching ``roi.center_xyz[2]``.

    Returns ``None`` if the ROI lacks a centre or the closest slice is more
    than ``tolerance_mm`` away in Z.
    """
    if roi.center_xyz is None:
        return None
    z_roi = roi.center_xyz[2]

    best_idx = 0
    best_delta = abs(slice_positions[0] - z_roi)
    for i in range(1, len(slice_positions)):
        d = abs(slice_positions[i] - z_roi)
        if d < best_delta:
            best_delta = d
            best_idx = i

    if best_delta > tolerance_mm:
        return None
    return best_idx


def fallback_image_index_to_slice(image_index: int, n_slices: int) -> int | None:
    """Direct fallback when no ``Center`` is present.

    The convention is ``slice_idx = ImageIndex`` (no inversion). This is the
    correct mapping for the COCA XML format when DICOM files are sorted by
    Z ascending, as ``predict.io.dicom_loader.load_patient_dicom`` does.
    """
    if 0 <= image_index < n_slices:
        return image_index
    return None
