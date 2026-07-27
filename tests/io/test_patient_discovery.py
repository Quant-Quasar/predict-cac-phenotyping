"""Tests for predict.io.patient_discovery.

Builds synthetic ``data/raw`` and ``data/calcium_xml`` trees and verifies
that:
  * intersect of folders and XML files defines the candidate set
  * configured PIDs are excluded
  * GE-manufacturer patients are excluded when the flag is set
  * orphan XMLs (no DICOM) and orphan DICOMs (no XML) are reported
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from predict.io.patient_discovery import discover_patients


def _write_synthetic_dicom(path: Path, *, manufacturer: str = "SIEMENS", model: str = "SOMATOM Force",
                           kernel: str = "Qr36d/2", thickness: float = 3.0,
                           ipp_z: float = 0.0, instance_number: int = 1,
                           series_number: int = 1, series_uid: str | None = None) -> None:
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), Dataset(), file_meta=file_meta, preamble=b"\x00" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = series_uid or generate_uid()
    ds.StudyInstanceUID = generate_uid()
    ds.Modality = "CT"
    ds.Manufacturer = manufacturer
    ds.ManufacturerModelName = model
    ds.ConvolutionKernel = kernel
    ds.SliceThickness = thickness
    ds.InstanceNumber = instance_number
    ds.SeriesNumber = series_number
    ds.ImagePositionPatient = [0.0, 0.0, ipp_z]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.Rows = 16
    ds.Columns = 16
    ds.PixelSpacing = [0.5, 0.5]
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.RescaleSlope = 1
    ds.RescaleIntercept = -1024
    ds.PixelData = np.zeros((16, 16), dtype=np.int16).tobytes()

    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path))


def _make_patient(root: Path, pid: str, *, manufacturer: str = "SIEMENS") -> None:
    series_uid = generate_uid()
    pdir = root / "raw" / pid / "series"
    for i in range(3):
        _write_synthetic_dicom(
            pdir / f"IM-{i:04d}.dcm",
            manufacturer=manufacturer,
            ipp_z=float(i) * 3.0,
            instance_number=i + 1,
            series_uid=series_uid,
        )
    (root / "calcium_xml").mkdir(parents=True, exist_ok=True)
    (root / "calcium_xml" / f"{pid}.xml").write_bytes(b"<plist></plist>")


@pytest.fixture
def synthetic_cohort(tmp_path: Path) -> Path:
    _make_patient(tmp_path, "1")
    _make_patient(tmp_path, "2")
    _make_patient(tmp_path, "12")
    _make_patient(tmp_path, "5", manufacturer="GE MEDICAL SYSTEMS")
    # Orphan XML (no DICOM)
    (tmp_path / "calcium_xml" / "99.xml").write_bytes(b"<plist></plist>")
    # Orphan DICOM (no XML)
    pdir = tmp_path / "raw" / "200" / "series"
    _write_synthetic_dicom(pdir / "IM-0001.dcm", ipp_z=0.0)
    return tmp_path


def test_discovery_intersects_raw_and_xml(synthetic_cohort: Path):
    result = discover_patients(synthetic_cohort, exclude_pids=(), exclude_ge_scanners=False)
    included = {r.pid for r in result.included}
    assert included == {"1", "2", "12", "5"}
    assert result.excluded_no_dicom == ("99",)
    assert result.excluded_no_xml == ("200",)


def test_discovery_applies_config_exclusions(synthetic_cohort: Path):
    result = discover_patients(
        synthetic_cohort, exclude_pids=("12",), exclude_ge_scanners=False
    )
    assert {r.pid for r in result.included} == {"1", "2", "5"}
    assert result.excluded_by_config == ("12",)


def test_discovery_excludes_ge_when_flagged(synthetic_cohort: Path):
    result = discover_patients(
        synthetic_cohort, exclude_pids=(), exclude_ge_scanners=True
    )
    assert {r.pid for r in result.included} == {"1", "2", "12"}
    assert "5" in result.excluded_ge


def test_discovery_records_scanner_metadata(synthetic_cohort: Path):
    result = discover_patients(
        synthetic_cohort, exclude_pids=(), exclude_ge_scanners=False
    )
    rec = next(r for r in result.included if r.pid == "1")
    assert rec.manufacturer == "SIEMENS"
    assert rec.scanner_model == "SOMATOM Force"
    assert rec.kernel == "Qr36d/2"
    assert rec.slice_thickness == 3.0
