"""Tests for scripts/09_validate.py orchestrator (lightweight, no real data)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def _load_orchestrator():
    """Load scripts/09_validate.py as a module."""
    root = Path(__file__).resolve().parents[2]
    p = root / "scripts" / "09_validate.py"
    spec = importlib.util.spec_from_file_location("scripts_09_validate", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────── helpers + flags ───────────────────────


def test_orchestrator_loads():
    mod = _load_orchestrator()
    assert hasattr(mod, "main")
    assert hasattr(mod, "run_d029_holdout")
    assert hasattr(mod, "run_d030_loo")
    assert hasattr(mod, "run_d031_consolidation")


def test_discover_holdout_pids_from_exclusions_csv(tmp_path: Path,
                                                     monkeypatch):
    mod = _load_orchestrator()
    # Build a minimal fake Config object with an outputs dir.
    excl_dir = tmp_path / "01_manifest"
    excl_dir.mkdir()
    pd.DataFrame({
        "pid": ["19", "28", "76", "77", "12"],
        "reason": ["ge_scanner"] * 4 + ["no_dicom"],
    }).to_csv(excl_dir / "exclusions.csv", index=False)

    class FakePaths:
        outputs = tmp_path

    class FakeCfg:
        paths = FakePaths()

    pids = mod._discover_holdout_pids(FakeCfg())
    assert pids == ["19", "28", "76", "77"]


def test_discover_holdout_pids_falls_back_when_csv_missing(tmp_path: Path):
    mod = _load_orchestrator()

    class FakePaths:
        outputs = tmp_path  # no exclusions.csv

    class FakeCfg:
        paths = FakePaths()

    pids = mod._discover_holdout_pids(FakeCfg())
    assert pids == ["19", "28", "76", "77"]


def test_xml_pass_by_pid_aggregation():
    mod = _load_orchestrator()
    trips = pd.DataFrame({
        "pid": ["19", "19", "19", "28", "28", "76"],
        "passes": [True, True, False, True, True, True],
        "matched_via": ["z", "z", "z", "z", "z", "dirty"],
        "image_index": [0, 1, 2, 0, 1, 0],
        "roi_idx_in_slice": [0, 0, 0, 0, 0, 0],
    })
    out = mod._xml_pass_by_pid_from_trips(trips)
    assert out["19"] is False           # one non-dirty fail
    assert out["28"] is True
    # pid 76 has only one ROI which is matched_via=dirty -> all-dropped -> True
    # (empty after filter; .all() on empty returns True by NumPy convention).
    assert out["76"] is True


def test_d031_consolidation_smoke(tmp_path: Path):
    mod = _load_orchestrator()
    # Synthesise a stage-7 cross_cohort_ari.csv.
    stage7 = tmp_path / "07_analyse"
    stage7.mkdir(parents=True)
    pd.DataFrame({
        "partition": ["spatial_k2", "burden_k3"],
        "stratum": ["Qr36d_2", "Qr36d_2"],
        "n_shared_pids": [220, 220],
        "ari": [0.91, 0.55],
        "passes": [True, False],
    }).to_csv(stage7 / "cross_cohort_ari.csv", index=False)

    class FakePaths:
        outputs = tmp_path

    class FakeCfg:
        paths = FakePaths()

    out = tmp_path / "08_validate"
    out.mkdir()
    import logging
    summary = mod.run_d031_consolidation(FakeCfg(), out, logging.getLogger("t"))
    assert summary["verdict"] == "ok"
    df = pd.read_csv(out / "cross_cohort_ari_consolidated.csv")
    assert list(df.columns) == ["partition", "stratum", "n_shared_pids",
                                 "ari", "pass_threshold", "pass_verdict"]
    assert summary["n_passing"] == 1


def test_d031_consolidation_skips_when_stage7_missing(tmp_path: Path):
    mod = _load_orchestrator()

    class FakePaths:
        outputs = tmp_path

    class FakeCfg:
        paths = FakePaths()

    import logging
    out = tmp_path / "08_validate"
    out.mkdir()
    summary = mod.run_d031_consolidation(FakeCfg(), out, logging.getLogger("t"))
    assert summary["verdict"] == "skipped"
    assert summary["reason"] == "stage7_missing"
