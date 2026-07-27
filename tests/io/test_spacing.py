"""Tests for predict.io.spacing."""
from __future__ import annotations

from pathlib import Path

import pytest

from predict.io.spacing import load_spacing_metadata, parse_spacing, save_spacing_metadata


def test_parse_spacing_three_values():
    assert parse_spacing("0.5,0.5,3.0") == (0.5, 0.5, 3.0)


def test_parse_spacing_rejects_wrong_count():
    with pytest.raises(ValueError, match="exactly three"):
        parse_spacing("0.5,0.5")


def test_parse_spacing_rejects_non_positive():
    with pytest.raises(ValueError, match="positive"):
        parse_spacing("0.5,0,-3")


def test_parse_spacing_rejects_non_numeric():
    with pytest.raises(ValueError, match="three comma-separated"):
        parse_spacing("a,b,c")


def test_spacing_metadata_round_trip(tmp_path: Path):
    path = tmp_path / "spacing.json"
    save_spacing_metadata(path, (0.5, 0.5, 3.0))
    assert load_spacing_metadata(path) == (0.5, 0.5, 3.0)


def test_load_spacing_rejects_missing_key(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text('{"foo": [0.5, 0.5, 3.0]}', encoding="utf-8")
    with pytest.raises(ValueError, match="target_spacing"):
        load_spacing_metadata(path)
