"""Tests for predict.io.xml_parser.

Covers vessel-name normalisation, dirty-name handling, zero-point ROI
filtering, and Center field parsing (the load-bearing primitive for D001).
"""
from __future__ import annotations

import plistlib
from pathlib import Path

import pytest

from predict.io.xml_parser import _parse_center, _parse_point, parse_calcium_xml


def test_parse_point_2d():
    assert _parse_point("(123.5, 47.2)") == (123.5, 47.2)


def test_parse_point_3d():
    assert _parse_point("(1.0, 2.0, 3.0)") == (1.0, 2.0, 3.0)


def test_parse_point_handles_list_input():
    assert _parse_point([1.0, 2.0, 3.0]) == (1.0, 2.0, 3.0)


def test_parse_center_valid():
    assert _parse_center("(10.0, 20.0, -30.5)") == (10.0, 20.0, -30.5)


def test_parse_center_none_when_absent():
    assert _parse_center(None) is None


def test_parse_center_none_when_2d_only():
    assert _parse_center("(10.0, 20.0)") is None


def test_parse_center_none_when_malformed():
    assert _parse_center("not a center") is None


# -- end-to-end parser tests against synthetic plist files -----------------


def _write_plist(path: Path, payload: dict) -> None:
    with open(path, "wb") as f:
        plistlib.dump(payload, f)


def _roi(name: str, n_points: int = 4, mean: float = 200.0, max_hu: float = 250.0) -> dict:
    return {
        "Name": name,
        "NumberOfPoints": n_points,
        "Area": 0.05,
        "Mean": mean,
        "Max": max_hu,
        "Min": 130.0,
        "Total": 800.0,
        "Point_px": ["(10.5, 10.0)", "(20.0, 10.0)", "(20.0, 20.0)", "(10.5, 20.0)"],
        "Point_mm": [
            "(1.0, 1.0, -50.0)",
            "(2.0, 1.0, -50.0)",
            "(2.0, 2.0, -50.0)",
            "(1.0, 2.0, -50.0)",
        ],
        "Center": "(15.0, 15.0, -50.0)",
    }


def test_parser_normalises_known_vessel_names(tmp_path):
    _write_plist(tmp_path / "P1.xml", {
        "Images": [
            {"ImageIndex": 5, "ROIs": [_roi("Left Anterior Descending Artery")]},
            {"ImageIndex": 6, "ROIs": [_roi("Right Coronary Artery")]},
            {"ImageIndex": 7, "ROIs": [_roi("Left Circumflex Artery")]},
            {"ImageIndex": 8, "ROIs": [_roi("Left Coronary Artery")]},
        ],
    })
    result = parse_calcium_xml("P1", tmp_path)
    assert [s.rois[0].vessel for s in result.slices] == ["LAD", "RCA", "LCx", "LM"]
    assert result.dirty_vessel_names == ()


def test_parser_marks_dirty_vessel_names(tmp_path):
    _write_plist(tmp_path / "P2.xml", {
        "Images": [
            {"ImageIndex": 1, "ROIs": [_roi("LAD"), _roi("555614876")]},
            {"ImageIndex": 2, "ROIs": [_roi("1"), _roi("Unnamed")]},
        ],
    })
    result = parse_calcium_xml("P2", tmp_path)
    dirty = result.dirty_vessel_names
    assert "555614876" in dirty
    assert "1" in dirty
    assert "Unnamed" in dirty
    assert "LAD" in dirty  # not in VESSEL_NAME_MAP (only full names accepted)
    # Vessels for dirty names are None.
    assert all(
        (roi.vessel is None) == (roi.vessel_raw in dirty)
        for s in result.slices for roi in s.rois
    )


def test_parser_drops_zero_point_rois(tmp_path):
    payload = {
        "Images": [{
            "ImageIndex": 0,
            "ROIs": [
                _roi("Left Anterior Descending Artery"),
                {"Name": "RCA placeholder", "NumberOfPoints": 0},
                {"Name": "LCx placeholder", "NumberOfPoints": 0},
            ],
        }],
    }
    _write_plist(tmp_path / "P3.xml", payload)
    result = parse_calcium_xml("P3", tmp_path)
    assert result.n_active_rois == 1
    assert result.n_dropped_zero_point == 2
    assert len(result.slices[0].rois) == 1


def test_parser_extracts_center(tmp_path):
    _write_plist(tmp_path / "P4.xml", {
        "Images": [{"ImageIndex": 0, "ROIs": [_roi("Right Coronary Artery")]}],
    })
    result = parse_calcium_xml("P4", tmp_path)
    roi = result.slices[0].rois[0]
    assert roi.center_xyz == (15.0, 15.0, -50.0)


def test_parser_handles_missing_center(tmp_path):
    roi = _roi("Right Coronary Artery")
    del roi["Center"]
    _write_plist(tmp_path / "P5.xml", {"Images": [{"ImageIndex": 0, "ROIs": [roi]}]})
    result = parse_calcium_xml("P5", tmp_path)
    assert result.slices[0].rois[0].center_xyz is None


def test_parser_preserves_hu_stats(tmp_path):
    roi = _roi("Left Anterior Descending Artery", mean=178.0, max_hu=210.0)
    roi["Min"] = 132.0
    roi["Total"] = 4500.0
    roi["Area"] = 0.123
    _write_plist(tmp_path / "P6.xml", {"Images": [{"ImageIndex": 0, "ROIs": [roi]}]})
    result = parse_calcium_xml("P6", tmp_path)
    out = result.slices[0].rois[0]
    assert out.mean_hu == 178.0
    assert out.max_hu == 210.0
    assert out.min_hu == 132.0
    assert out.total_hu == 4500.0
    assert out.area_cm2 == 0.123
