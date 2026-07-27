"""Shared helpers for preprocess and validate tests.

Build a synthetic CT + ROI with known HU values inside the polygon so that
round-trip and slice-matcher tests can assert on exact stats.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from predict.io.dicom_loader import LoadedPatient
from predict.io.xml_parser import ROI, ParseResult, SliceAnnotation


def make_loaded_patient(
    pid: str = "TEST",
    n_slices: int = 10,
    shape_2d: tuple[int, int] = (64, 64),
    z0: float = 0.0,
    dz: float = 3.0,
) -> LoadedPatient:
    """Create a :class:`LoadedPatient` backed by a real SimpleITK image.

    Each slice ``k`` is initialised to HU = ``-200 + 20*k`` so different
    slices have distinguishable means, allowing slice-mismatch errors to be
    detected.
    """
    import SimpleITK as sitk

    arr = np.zeros((n_slices, shape_2d[0], shape_2d[1]), dtype=np.int16)
    for k in range(n_slices):
        arr[k] = -200 + 20 * k
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.5, 0.5, dz))
    img.SetOrigin((0.0, 0.0, z0))
    return LoadedPatient(
        pid=pid,
        ct_sitk=img,
        slice_positions=tuple(z0 + k * dz for k in range(n_slices)),
        pixel_spacing=(0.5, 0.5, dz),
        slice_thickness=dz,
        n_slices=n_slices,
        scanner_model="TEST",
        kernel="Qr36d/2",
        manufacturer="SIEMENS",
        series_uid="test-uid",
    )


def stamp_calcium_square(
    loaded: LoadedPatient,
    slice_idx: int,
    *,
    x0: int = 20,
    y0: int = 20,
    side: int = 10,
    hu: int = 300,
) -> tuple[ROI, np.ndarray]:
    """Burn a square of HU=``hu`` into the CT at ``slice_idx``.

    Returns the matching ROI (with XML-equivalent stats computed from the
    voxels) and the 2D polygon mask used.
    """
    import SimpleITK as sitk

    arr = sitk.GetArrayFromImage(loaded.ct_sitk)
    h, w = arr.shape[1], arr.shape[2]
    poly = np.array(
        [[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side]],
        dtype=np.int32,
    ).reshape((-1, 1, 2))
    mask2d = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask2d, [poly], color=1)

    arr[slice_idx][mask2d == 1] = hu

    img = sitk.GetImageFromArray(arr)
    img.CopyInformation(loaded.ct_sitk)
    object.__setattr__(loaded, "ct_sitk", img)  # frozen dataclass workaround

    voxels = arr[slice_idx][mask2d == 1]
    z = loaded.slice_positions[slice_idx]
    center_xy = (x0 + side / 2, y0 + side / 2, z)

    roi = ROI(
        vessel_raw="Left Anterior Descending Artery",
        vessel="LAD",
        area_cm2=(side * 0.5) * (side * 0.5) / 100.0,
        mean_hu=float(voxels.mean()),
        max_hu=float(voxels.max()),
        min_hu=float(voxels.min()),
        total_hu=float(voxels.sum()),
        n_points=4,
        points_px=tuple((float(p[0][0]), float(p[0][1])) for p in poly),
        points_mm=tuple((p[0][0] * 0.5, p[0][1] * 0.5, z) for p in poly),
        center_xyz=center_xy,
    )
    return roi, mask2d


def make_parse_result(pid: str, image_index: int, roi: ROI) -> ParseResult:
    return ParseResult(
        pid=pid,
        slices=(SliceAnnotation(image_index=image_index, rois=(roi,)),),
        dirty_vessel_names=(),
        n_active_rois=1,
        n_dropped_zero_point=0,
    )
