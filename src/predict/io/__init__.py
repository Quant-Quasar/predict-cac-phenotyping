"""Stage 1 — io.

DICOM loading, XML parsing, patient discovery, spacing metadata.
"""
from predict.io.dicom_loader import (
    LoadedPatient,
    PatientMetadata,
    load_patient_dicom,
    load_patient_metadata,
    peek_patient_header,
)
from predict.io.patient_discovery import (
    DiscoveryResult,
    PatientRecord,
    discover_patients,
    manifest_rows,
)
from predict.io.spacing import (
    load_spacing_metadata,
    parse_spacing,
    save_spacing_metadata,
)
from predict.io.xml_parser import ROI, ParseResult, SliceAnnotation, parse_calcium_xml

__all__ = [
    "DiscoveryResult",
    "LoadedPatient",
    "ParseResult",
    "PatientMetadata",
    "PatientRecord",
    "ROI",
    "SliceAnnotation",
    "discover_patients",
    "load_patient_dicom",
    "load_patient_metadata",
    "load_spacing_metadata",
    "manifest_rows",
    "parse_calcium_xml",
    "parse_spacing",
    "peek_patient_header",
    "save_spacing_metadata",
]
