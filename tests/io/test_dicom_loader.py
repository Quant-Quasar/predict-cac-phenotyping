"""Tests for predict.io.dicom_loader.

Covers the multi-series selection rule (D012) using fabricated header dicts.
Integration tests against real DICOM data are marked ``integration`` and run
on the remote machine where ``data/raw/`` is present.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from predict.io.dicom_loader import _normalise_kernel, _select_single_series, load_patient_dicom


def _meta(path: str, inst: int, uid: str, snum: int, modality: str = "CT") -> dict:
    return {
        "path": path,
        "instance_number": inst,
        "z_position": float(inst),
        "series_uid": uid,
        "series_number": snum,
        "modality": modality,
        "manufacturer": "SIEMENS",
        "scanner_model": "TEST",
        "kernel": "Qr36d/2",
        "slice_thickness": 3.0,
    }


# -- _select_single_series unit tests --------------------------------------


def test_single_series_returns_unchanged():
    inputs = [_meta(f"f{i}.dcm", i, "uid-A", 1) for i in range(1, 6)]
    assert _select_single_series(inputs, "TEST") == inputs


def test_multi_series_picks_largest_slice_count():
    """Annotated scan vs 1-slice scout: must pick the longer series."""
    scout = [_meta("scout.dcm", 1, "uid-SCOUT", 2)]  # lower SeriesNumber, 1 slice
    scan = [_meta(f"scan{i}.dcm", i, "uid-SCAN", 3) for i in range(1, 45)]  # 44 slices
    out = _select_single_series(scout + scan, "TEST388")
    assert {m["series_uid"] for m in out} == {"uid-SCAN"}
    assert len(out) == 44


def test_multi_series_equal_slice_count_picks_lower_series_number():
    """If both series have the same slice count (Patient 78 scenario),
    fall back to the lowest SeriesNumber."""
    series_a = [_meta(f"a{i}.dcm", i, "uid-A", 3) for i in range(1, 35)]  # 34
    series_b = [_meta(f"b{i}.dcm", i, "uid-B", 4) for i in range(1, 35)]  # 34
    out = _select_single_series(series_a + series_b, "TEST78")
    assert {m["series_uid"] for m in out} == {"uid-A"}


def test_multi_series_final_tiebreak_by_uid_lex():
    """Equal slice count AND equal SeriesNumber → lexicographic UID tiebreak."""
    series_x = [_meta(f"x{i}.dcm", i, "uid-Z", 1) for i in range(1, 4)]
    series_y = [_meta(f"y{i}.dcm", i, "uid-A", 1) for i in range(1, 4)]
    out = _select_single_series(series_x + series_y, "TEST")
    assert {m["series_uid"] for m in out} == {"uid-A"}


def test_multi_series_emits_warning_via_logger(caplog):
    caplog.set_level(logging.WARNING, logger="predict.io.dicom_loader")
    a = [_meta(f"a{i}.dcm", i, "uid-A", 1) for i in range(1, 4)]   # 3 slices
    b = [_meta(f"b{i}.dcm", i, "uid-B", 2) for i in range(1, 6)]   # 5 slices
    out = _select_single_series(a + b, "PID42")
    # New rule: larger slice count wins.
    assert {m["series_uid"] for m in out} == {"uid-B"}
    text = caplog.text
    assert "multi-series" in text.lower()
    assert "PID42" in text
    assert "SELECTED" in text
    assert "uid-A" in text and "uid-B" in text


def test_single_series_no_warning(caplog):
    caplog.set_level(logging.WARNING, logger="predict.io.dicom_loader")
    inputs = [_meta(f"f{i}.dcm", i, "uid-A", 1) for i in range(1, 4)]
    _select_single_series(inputs, "TEST")
    assert "multi-series" not in caplog.text.lower()


def test_modality_filter_drops_non_ct():
    ct = [_meta(f"ct{i}.dcm", i, "uid-CT", 1, modality="CT") for i in range(1, 4)]
    rt = [_meta(f"rt{i}.dcm", i, "uid-RT", 2, modality="RTSTRUCT") for i in range(1, 3)]
    out = _select_single_series(ct + rt, "TEST")
    assert {m["series_uid"] for m in out} == {"uid-CT"}
    assert len(out) == 3


def test_no_ct_modality_falls_back_to_all(caplog):
    caplog.set_level(logging.WARNING, logger="predict.io.dicom_loader")
    inputs = [_meta(f"f{i}.dcm", i, "uid-A", 1, modality="MR") for i in range(1, 4)]
    out = _select_single_series(inputs, "TEST")
    assert len(out) == 3
    assert "no DICOM files with Modality=CT" in caplog.text


def test_normalise_kernel_string_passthrough():
    assert _normalise_kernel("STANDARD") == "STANDARD"
    assert _normalise_kernel("  Qr36d  ") == "Qr36d"


def test_normalise_kernel_multivalue_joined_by_slash():
    assert _normalise_kernel(["Qr36d", "2"]) == "Qr36d/2"
    assert _normalise_kernel(["I30f", "3"]) == "I30f/3"


def test_normalise_kernel_empty_or_none():
    assert _normalise_kernel(None) == ""
    assert _normalise_kernel("") == ""
    assert _normalise_kernel([]) == ""


def test_deterministic_across_input_order():
    a = [_meta(f"a{i}.dcm", i, "uid-A", 1) for i in range(1, 4)]
    b = [_meta(f"b{i}.dcm", i, "uid-B", 2) for i in range(1, 4)]
    out1 = _select_single_series(a + b, "TEST")
    out2 = _select_single_series(b + a, "TEST")
    assert sorted(m["path"] for m in out1) == sorted(m["path"] for m in out2)


# -- integration tests (require real data) ---------------------------------

DATA_ROOT = Path("data")


@pytest.mark.integration
@pytest.mark.skipif(
    not (DATA_ROOT / "raw" / "78").exists(),
    reason="requires data/raw/78",
)
def test_patient_78_multi_series_yields_single_series(caplog):
    caplog.set_level(logging.WARNING, logger="predict.io.dicom_loader")
    p = load_patient_dicom("78", DATA_ROOT)
    assert p.n_slices == 34
    assert "multi-series" in caplog.text.lower()


@pytest.mark.integration
@pytest.mark.skipif(
    not (DATA_ROOT / "raw" / "1").exists(),
    reason="requires data/raw/1",
)
def test_single_series_patient_loads_clean(caplog):
    caplog.set_level(logging.WARNING, logger="predict.io.dicom_loader")
    p = load_patient_dicom("1", DATA_ROOT)
    assert p.n_slices > 0
    assert len(p.slice_positions) == p.n_slices
    # Z positions must be sorted ascending.
    assert list(p.slice_positions) == sorted(p.slice_positions)
    assert "multi-series" not in caplog.text.lower()
