"""DICOM series loading for the COCA cohort.

Loads a patient's CT series into a SimpleITK image, sorted by physical Z
coordinate (ascending). Multi-series folders are reduced to a single series
by the rule in D012.

The loader returns a :class:`LoadedPatient` carrying everything downstream
stages need:

- the CT image
- per-slice Z positions (so ROIs can be matched by Z; see D001)
- scanner / kernel metadata for kernel harmonisation
- pixel spacing and slice thickness

No HU manipulation, no resampling, no mask handling — those belong to
``predict.preprocess``.

Decisions referencing this module:
    D001 — Z-coordinate matching (provides slice_positions)
    D004 — Cohort exclusions (kernel/manufacturer used by discovery)
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pydicom
import SimpleITK as sitk

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedPatient:
    """Result of loading one patient's CT series."""

    pid: str
    ct_sitk: sitk.Image
    slice_positions: tuple[float, ...]  # IPP[2] per array slice, ascending
    pixel_spacing: tuple[float, float, float]  # (x, y, z) mm
    slice_thickness: float
    n_slices: int
    scanner_model: str
    kernel: str
    manufacturer: str
    series_uid: str


def find_dicom_subfolder(patient_dir: Path) -> Path:
    """Return the folder under ``patient_dir`` that contains the .dcm files."""
    for entry in patient_dir.iterdir():
        if entry.is_dir() and any(f.name.endswith(".dcm") for f in entry.iterdir()):
            return entry
    if any(f.name.endswith(".dcm") for f in patient_dir.iterdir()):
        return patient_dir
    raise FileNotFoundError(f"No DICOM files found under {patient_dir}")


def _select_single_series(
    file_meta: list[dict],
    pid: str,
) -> list[dict]:
    """Reduce a multi-series folder to one series.

    Rule (D012 v2):
      1. Filter to ``Modality == "CT"``.
      2. Pick the series with the **largest slice count** (the annotated
         diagnostic acquisition; scouts/previews are typically 1 slice).
      3. Tiebreak: lowest ``SeriesNumber``.
      4. Final tiebreak: lexicographic ``SeriesInstanceUID``.

    Emits a warning via the module logger if the input was multi-series.
    """
    ct_meta = [m for m in file_meta if str(m["modality"]).upper() == "CT"]
    if not ct_meta:
        _log.warning(
            "Patient %s: no DICOM files with Modality=CT; "
            "falling back to all %d files.",
            pid, len(file_meta),
        )
        ct_meta = file_meta

    groups: dict[str, list[dict]] = defaultdict(list)
    for m in ct_meta:
        groups[m["series_uid"]].append(m)

    if len(groups) == 1:
        return ct_meta

    # Sort: most slices first (desc), then lowest SeriesNumber (asc), then UID (asc).
    summary = sorted(
        ((-len(g), g[0]["series_number"], uid) for uid, g in groups.items()),
    )
    chosen_uid = summary[0][2]

    _log.warning(
        "Patient %s: multi-series folder (%d series). "
        "Selecting series with most slices.",
        pid, len(groups),
    )
    for neg_n, snum, uid in summary:
        marker = "  <-- SELECTED" if uid == chosen_uid else ""
        _log.warning(
            "  Series #%s  uid=%s  (%d slices)%s", snum, uid, -neg_n, marker,
        )

    return groups[chosen_uid]


def _normalise_kernel(value) -> str:
    """Flatten pydicom multi-valued ConvolutionKernel into a stable string.

    Siemens encodes the kernel as a backslash-separated multi-value (e.g.
    ``Qr36d\\2``), which pydicom exposes as a list-like ``MultiValue``. We
    join with ``/`` for a deterministic single-token kernel id used in the
    manifest and kernel-harmonisation grouping.
    """
    if value is None:
        return ""
    try:
        # MultiValue is iterable; a plain string is also iterable but is
        # excluded by the isinstance check below.
        if isinstance(value, str):
            return value.strip()
        parts = [str(v).strip() for v in value]
        return "/".join(p for p in parts if p)
    except TypeError:
        return str(value).strip()


def _read_header(path: str) -> dict:
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    ipp = getattr(ds, "ImagePositionPatient", None)
    z = float(ipp[2]) if ipp is not None else float(getattr(ds, "InstanceNumber", 0))
    return {
        "path": path,
        "instance_number": int(getattr(ds, "InstanceNumber", 0)),
        "z_position": z,
        # Some COCA DICOMs are missing SeriesInstanceUID (patient 159). Use
        # an empty-string fallback so all files in the folder that share the
        # missing tag are grouped as a single series by _select_single_series.
        "series_uid": str(getattr(ds, "SeriesInstanceUID", "")),
        "series_number": int(getattr(ds, "SeriesNumber", 0)),
        "modality": str(getattr(ds, "Modality", "CT")),
        "manufacturer": str(getattr(ds, "Manufacturer", "")).strip(),
        "scanner_model": str(getattr(ds, "ManufacturerModelName", "")).strip(),
        "kernel": _normalise_kernel(getattr(ds, "ConvolutionKernel", "")),
        "slice_thickness": float(getattr(ds, "SliceThickness", 0.0)) or None,
    }


def load_patient_dicom(pid: str, data_root: Path) -> LoadedPatient:
    """Load a patient's CT series.

    Steps:
      1. Find the DICOM subfolder.
      2. Read headers; sort by (Z ascending, InstanceNumber tiebreak).
      3. Reduce to one series (D012) if multi-series.
      4. Build SimpleITK image from the chosen file list.
      5. Return :class:`LoadedPatient` with slice positions + scanner metadata.

    Parameters
    ----------
    pid : str
        Patient ID (directory name under ``data_root / "raw"``).
    data_root : Path
        Root path containing ``raw/`` and ``calcium_xml/`` subdirectories.
    """
    patient_dir = data_root / "raw" / str(pid)
    dcm_dir = find_dicom_subfolder(patient_dir)

    dcm_paths = [str(dcm_dir / f) for f in os.listdir(dcm_dir) if f.endswith(".dcm")]
    if not dcm_paths:
        raise FileNotFoundError(f"No DICOM files in {dcm_dir}")

    headers = [_read_header(p) for p in dcm_paths]
    headers = _select_single_series(headers, pid)
    headers.sort(key=lambda m: (m["z_position"], m["instance_number"]))

    sorted_files = [m["path"] for m in headers]
    slice_positions = tuple(m["z_position"] for m in headers)

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(sorted_files)
    ct_sitk = reader.Execute()

    head = headers[0]
    spacing = ct_sitk.GetSpacing()  # (x, y, z) mm

    return LoadedPatient(
        pid=str(pid),
        ct_sitk=ct_sitk,
        slice_positions=slice_positions,
        pixel_spacing=spacing,
        slice_thickness=head["slice_thickness"] or spacing[2],
        n_slices=len(sorted_files),
        scanner_model=head["scanner_model"],
        kernel=head["kernel"],
        manufacturer=head["manufacturer"],
        series_uid=head["series_uid"],
    )


@dataclass(frozen=True)
class PatientMetadata:
    """Header-only summary of a patient's DICOM series.

    Identical fields to :class:`LoadedPatient` except there's no ``ct_sitk``.
    Used by downstream stages (notably features) that already have the
    preprocessed CT on disk and only need the spatial metadata.
    """
    pid: str
    slice_positions: tuple[float, ...]
    pixel_spacing: tuple[float, float, float]
    slice_thickness: float
    n_slices: int
    scanner_model: str
    kernel: str
    manufacturer: str
    series_uid: str


def load_patient_metadata(pid: str, data_root: Path) -> PatientMetadata:
    """Header-only version of :func:`load_patient_dicom`.

    Reads each DICOM header (``stop_before_pixels=True``), applies the D006
    multi-series reduction, sorts by Z ascending, and returns the spatial
    metadata. ~10× faster than ``load_patient_dicom`` because it skips
    SimpleITK's pixel read.
    """
    patient_dir = data_root / "raw" / str(pid)
    dcm_dir = find_dicom_subfolder(patient_dir)

    dcm_paths = [str(dcm_dir / f) for f in os.listdir(dcm_dir) if f.endswith(".dcm")]
    if not dcm_paths:
        raise FileNotFoundError(f"No DICOM files in {dcm_dir}")

    headers = [_read_header(p) for p in dcm_paths]
    headers = _select_single_series(headers, pid)
    headers.sort(key=lambda m: (m["z_position"], m["instance_number"]))

    head = headers[0]
    # Pixel spacing & thickness derived from the first chosen-series header;
    # consistent with what SimpleITK would have set on the volume.
    import pydicom
    ds = pydicom.dcmread(head["path"], stop_before_pixels=True)
    row_sp = float(ds.PixelSpacing[0]) if hasattr(ds, "PixelSpacing") else 0.0
    col_sp = float(ds.PixelSpacing[1]) if hasattr(ds, "PixelSpacing") else 0.0

    return PatientMetadata(
        pid=str(pid),
        slice_positions=tuple(m["z_position"] for m in headers),
        pixel_spacing=(col_sp, row_sp, head["slice_thickness"] or 0.0),
        slice_thickness=head["slice_thickness"] or 0.0,
        n_slices=len(headers),
        scanner_model=head["scanner_model"],
        kernel=head["kernel"],
        manufacturer=head["manufacturer"],
        series_uid=head["series_uid"],
    )


def peek_patient_header(pid: str, data_root: Path) -> dict:
    """Read a single header from the patient's DICOM folder.

    Used by patient discovery to label scanner / kernel without loading the
    full series. Picks the lexicographically first .dcm in the folder.
    """
    patient_dir = data_root / "raw" / str(pid)
    dcm_dir = find_dicom_subfolder(patient_dir)
    first = sorted(f for f in os.listdir(dcm_dir) if f.endswith(".dcm"))[0]
    return _read_header(str(dcm_dir / first))
