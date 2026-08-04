"""Tests for predict.validate.cross_cohort_ari (D031 re-exporter)."""
from __future__ import annotations

import filecmp
from pathlib import Path

import pandas as pd
import pytest

from predict.validate.cross_cohort_ari import (
    ARI_PASS_THRESHOLD,
    SOURCE_REQUIRED_COLS,
    consolidate,
    write,
)


@pytest.fixture
def stage7_csv(tmp_path: Path) -> Path:
    """Synthetic stage 7 cross_cohort_ari.csv mimicking the real schema."""
    df = pd.DataFrame({
        "partition": ["spatial_k2", "spatial_k2",
                      "burden_k3", "burden_k3"],
        "stratum": ["Qr36d_2", "I30f_3", "Qr36d_2", "I30f_3"],
        "n_shared_pids": [220, 200, 220, 200],
        "ari": [0.93, 0.81, 0.74, 0.70],
        "passes": [True, True, False, False],
    })
    p = tmp_path / "cross_cohort_ari.csv"
    df.to_csv(p, index=False)
    return p


def test_consolidate_columns(stage7_csv: Path):
    out = consolidate(stage7_csv)
    expected = ["partition", "stratum", "n_shared_pids", "ari",
                "pass_threshold", "pass_verdict"]
    assert list(out.columns) == expected


def test_consolidate_row_count_preserved(stage7_csv: Path):
    src = pd.read_csv(stage7_csv)
    out = consolidate(stage7_csv)
    assert len(out) == len(src)


def test_consolidate_pass_threshold_constant(stage7_csv: Path):
    out = consolidate(stage7_csv)
    assert (out["pass_threshold"] == ARI_PASS_THRESHOLD).all()
    assert ARI_PASS_THRESHOLD == 0.80


def test_consolidate_pass_verdict_matches_ari_threshold(stage7_csv: Path):
    out = consolidate(stage7_csv)
    # 0.93 >= 0.80 -> True; 0.81 >= 0.80 -> True; 0.74 < 0.80 -> False; 0.70 -> False
    assert out["pass_verdict"].tolist() == [True, True, False, False]


def test_consolidate_pass_verdict_overridable(stage7_csv: Path):
    out = consolidate(stage7_csv, pass_threshold=0.95)
    # Only 0.95 threshold: nothing passes
    assert not out["pass_verdict"].any()
    assert (out["pass_threshold"] == 0.95).all()


def test_consolidate_raises_on_missing_file(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError, match="stage 7"):
        consolidate(missing)


def test_consolidate_raises_on_missing_columns(tmp_path: Path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"partition": ["x"], "stratum": ["y"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing expected stage-7 columns"):
        consolidate(bad)


def test_write_produces_byte_identical_output(stage7_csv: Path, tmp_path: Path):
    out1 = tmp_path / "out1.csv"
    out2 = tmp_path / "out2.csv"
    write(stage7_csv, out1)
    write(stage7_csv, out2)
    assert filecmp.cmp(out1, out2, shallow=False)


def test_source_required_cols_are_canonical():
    # Regression: tests + the module agree on the exact source schema.
    assert SOURCE_REQUIRED_COLS == (
        "partition", "stratum", "n_shared_pids", "ari", "passes",
    )
