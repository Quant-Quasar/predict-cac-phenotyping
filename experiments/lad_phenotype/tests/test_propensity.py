"""Tests for the propensity matching infrastructure."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from propensity import (  # noqa: E402
    MatchResult,
    caliper_match,
    match_yield,
    standardised_mean_difference,
)


def test_smd_zero_on_identical_groups():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    assert standardised_mean_difference(a, b) == pytest.approx(0.0)


def test_smd_sign_matches_mean_difference():
    a = np.array([10.0, 11.0, 12.0])
    b = np.array([1.0, 2.0, 3.0])
    smd = standardised_mean_difference(a, b)
    assert smd > 0  # a > b


def test_smd_handles_constant_vector():
    a = np.array([5.0, 5.0, 5.0])
    b = np.array([5.0, 5.0, 5.0])
    # both vectors are constant + equal -> 0
    assert standardised_mean_difference(a, b) == 0.0
    # constant unequal -> nan (pooled sd is 0)
    a = np.array([5.0, 5.0, 5.0])
    b = np.array([7.0, 7.0, 7.0])
    assert np.isnan(standardised_mean_difference(a, b))


def test_caliper_match_basic_balanced_case():
    """Cases and controls drawn from the same distribution should match
    well, with post-match SMD near zero."""
    rng = np.random.default_rng(0)
    cases = pd.Series(rng.normal(0, 1, 50), index=[f"c{i}" for i in range(50)])
    controls = pd.Series(rng.normal(0, 1, 200),
                          index=[f"x{i}" for i in range(200)])
    result = caliper_match(cases, controls, caliper_sd=0.2, k=3)
    assert isinstance(result, MatchResult)
    assert result.n_cases_matched >= 40  # most cases find controls
    assert abs(result.smd_post) < 0.1


def test_caliper_match_without_replacement():
    """Each control should be used at most once."""
    rng = np.random.default_rng(1)
    cases = pd.Series(rng.normal(0, 1, 30), index=[f"c{i}" for i in range(30)])
    controls = pd.Series(rng.normal(0, 1, 90),
                          index=[f"x{i}" for i in range(90)])
    result = caliper_match(cases, controls, k=3)
    used = result.pairs["control_pid"].tolist()
    assert len(used) == len(set(used))


def test_caliper_match_caliper_excludes_far_controls():
    """A tight caliper should leave some cases unmatched and some
    controls unused."""
    rng = np.random.default_rng(2)
    cases = pd.Series(rng.normal(5, 0.5, 20), index=[f"c{i}" for i in range(20)])
    controls = pd.Series(rng.normal(-5, 0.5, 60),
                          index=[f"x{i}" for i in range(60)])
    result = caliper_match(cases, controls, caliper_sd=0.05, k=3)
    assert result.n_cases_matched == 0
    assert result.pairs.empty


def test_caliper_match_overlap_raises():
    cases = pd.Series([1.0, 2.0], index=["a", "b"])
    controls = pd.Series([1.5, 2.5], index=["b", "c"])  # "b" in both
    with pytest.raises(ValueError, match="overlap"):
        caliper_match(cases, controls)


def test_caliper_match_zero_variance_raises():
    cases = pd.Series([1.0, 1.0, 1.0], index=["a", "b", "c"])
    controls = pd.Series([1.0, 1.0, 1.0], index=["x", "y", "z"])
    with pytest.raises(ValueError, match="zero variance"):
        caliper_match(cases, controls)


def test_caliper_match_seed_reproducibility():
    rng = np.random.default_rng(3)
    cases = pd.Series(rng.normal(0, 1, 30), index=[f"c{i}" for i in range(30)])
    controls = pd.Series(rng.normal(0, 1, 90),
                          index=[f"x{i}" for i in range(90)])
    a = caliper_match(cases, controls, random_state=42)
    b = caliper_match(cases, controls, random_state=42)
    pd.testing.assert_frame_equal(a.pairs, b.pairs)


def test_caliper_match_respects_k():
    """No case should have more than k matched controls."""
    rng = np.random.default_rng(4)
    cases = pd.Series(rng.normal(0, 1, 10), index=[f"c{i}" for i in range(10)])
    controls = pd.Series(rng.normal(0, 1, 200),
                          index=[f"x{i}" for i in range(200)])
    k = 2
    result = caliper_match(cases, controls, caliper_sd=0.5, k=k)
    counts = result.pairs.groupby("case_pid").size()
    assert (counts <= k).all()


def test_match_yield_basic():
    rng = np.random.default_rng(5)
    cases = pd.Series(rng.normal(0, 1, 100), index=[f"c{i}" for i in range(100)])
    controls = pd.Series(rng.normal(0, 1, 300),
                          index=[f"x{i}" for i in range(300)])
    result = caliper_match(cases, controls, caliper_sd=0.2, k=3)
    y = match_yield(result)
    assert 0.0 <= y <= 1.0
    assert y == pytest.approx(result.n_cases_matched / result.n_cases_in)
