"""Patient discovery and cohort manifest construction.

Intersects ``data/raw/`` (DICOM folders) with ``data/calcium_xml/`` (XML
annotations), applies the cohort exclusions from configuration (D004), and
labels each included patient with scanner/kernel metadata for downstream
kernel-harmonisation and stratification.

The output is a manifest (a list of :class:`PatientRecord`), persisted by
the orchestration script to ``outputs/01_manifest/manifest.csv``.

Decisions referencing this module:
    D004 — Cohort exclusions at discovery
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path

from predict.io.dicom_loader import peek_patient_header

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PatientRecord:
    pid: str
    raw_path: Path
    xml_path: Path
    manufacturer: str
    scanner_model: str
    kernel: str
    slice_thickness: float


@dataclass(frozen=True)
class DiscoveryResult:
    included: tuple[PatientRecord, ...]
    excluded_no_dicom: tuple[str, ...]
    excluded_no_xml: tuple[str, ...]
    excluded_by_config: tuple[str, ...]
    excluded_ge: tuple[str, ...]


def _listdir_sorted_int(path: Path, *, dirs: bool = True, suffix: str | None = None) -> list[str]:
    """List entries under ``path``. ``dirs=True`` keeps directories, ``False`` keeps files."""
    out: list[str] = []
    for name in os.listdir(path):
        full = path / name
        if dirs and not full.is_dir():
            continue
        if not dirs and not full.is_file():
            continue
        if suffix is not None and not name.endswith(suffix):
            continue
        out.append(name)
    return sorted(out, key=lambda s: int(s.replace(suffix or "", "")) if (suffix and s.replace(suffix, "").isdigit()) or s.isdigit() else s)


def discover_patients(
    data_root: Path,
    *,
    exclude_pids: tuple[str, ...] = (),
    exclude_ge_scanners: bool = True,
) -> DiscoveryResult:
    """Build the cohort manifest.

    Steps:
      1. Intersect raw DICOM folder names with XML basenames.
      2. Apply ``exclude_pids`` (config-driven, default: 12, 197, 268).
      3. For each remaining patient, read one DICOM header to label scanner +
         kernel + thickness.
      4. If ``exclude_ge_scanners`` is True, drop any patient whose
         ``Manufacturer`` does not start with "SIEMENS" (case-insensitive).
      5. Return :class:`DiscoveryResult` carrying included + excluded sets.

    Patients listed in ``exclude_pids`` are excluded silently (they are
    expected exclusions, not anomalies). Patients failing the GE filter or
    missing DICOM/XML are logged.
    """
    raw_dir = data_root / "raw"
    xml_dir = data_root / "calcium_xml"

    raw_ids = {n for n in os.listdir(raw_dir) if (raw_dir / n).is_dir()}
    xml_ids = {n[:-4] for n in os.listdir(xml_dir) if n.endswith(".xml")}

    no_dicom = sorted(xml_ids - raw_ids, key=_int_key)
    no_xml = sorted(raw_ids - xml_ids, key=_int_key)
    if no_dicom:
        _log.warning("XML present, no DICOM: %s", no_dicom)
    if no_xml:
        _log.warning("DICOM present, no XML: %s", no_xml)

    candidates = sorted(raw_ids & xml_ids, key=_int_key)
    excluded_by_config = tuple(p for p in candidates if p in exclude_pids)
    eligible = [p for p in candidates if p not in exclude_pids]

    included: list[PatientRecord] = []
    excluded_ge: list[str] = []
    for pid in eligible:
        head = peek_patient_header(pid, data_root)
        manuf = head["manufacturer"].upper()
        if exclude_ge_scanners and not manuf.startswith("SIEMENS"):
            excluded_ge.append(pid)
            continue
        included.append(PatientRecord(
            pid=pid,
            raw_path=raw_dir / pid,
            xml_path=xml_dir / f"{pid}.xml",
            manufacturer=head["manufacturer"],
            scanner_model=head["scanner_model"],
            kernel=head["kernel"],
            slice_thickness=head["slice_thickness"] or 0.0,
        ))

    _log.info(
        "Discovery: %d included, %d config-excluded, %d GE-excluded, "
        "%d no-DICOM, %d no-XML",
        len(included), len(excluded_by_config), len(excluded_ge),
        len(no_dicom), len(no_xml),
    )

    return DiscoveryResult(
        included=tuple(included),
        excluded_no_dicom=tuple(no_dicom),
        excluded_no_xml=tuple(no_xml),
        excluded_by_config=excluded_by_config,
        excluded_ge=tuple(excluded_ge),
    )


def _int_key(s: str) -> tuple[int, str]:
    """Sort key that orders numeric strings numerically, others lexicographically."""
    return (0, s) if not s.isdigit() else (-1, f"{int(s):010d}")


def manifest_rows(result: DiscoveryResult) -> list[dict]:
    """Flatten included records into dicts suitable for a CSV."""
    rows = []
    for rec in result.included:
        row = asdict(rec)
        row["raw_path"] = str(rec.raw_path)
        row["xml_path"] = str(rec.xml_path)
        rows.append(row)
    return rows
