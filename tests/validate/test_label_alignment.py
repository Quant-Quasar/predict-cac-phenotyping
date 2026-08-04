"""Tests for predict.validate.label_alignment (stage 8 shared helper)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predict.validate.label_alignment import (
    DIFFUSE_LABEL_INT,
    DIFFUSE_LABEL_STR,
    FOCAL_LABEL_INT,
    FOCAL_LABEL_STR,
    apply_mapping,
    canonical_numeric_labels,
    canonical_string_labels,
    determine_mapping,
)


@pytest.fixture
def synthetic_two_cluster():
    """Cluster 0 = high n_calc (diffuse); cluster 1 = low (focal)."""
    pids = [f"p{i:03d}" for i in range(20)]
    labels = pd.Series([0] * 10 + [1] * 10, index=pids, name="label")
    raw = pd.DataFrame(
        {
            # cluster 0 median = 3, cluster 1 median = 1
            "n_calcified_arteries": [3, 3, 3, 4, 3, 3, 2, 3, 4, 3,
                                      1, 1, 1, 1, 2, 1, 1, 0, 1, 1],
            "agatston_total": [100.0] * 20,
        },
        index=pids,
    )
    return raw, labels


def test_determine_mapping_focal_is_lower_median(synthetic_two_cluster):
    raw, labels = synthetic_two_cluster
    mapping = determine_mapping(raw, labels)
    assert mapping[1] == FOCAL_LABEL_STR  # lower median = focal
    assert mapping[0] == DIFFUSE_LABEL_STR


def test_canonical_string_labels_relabels_correctly(synthetic_two_cluster):
    raw, labels = synthetic_two_cluster
    out = canonical_string_labels(raw, labels)
    # Cluster 1 (low n_calc) should be "focal"
    assert (out[labels == 1] == FOCAL_LABEL_STR).all()
    assert (out[labels == 0] == DIFFUSE_LABEL_STR).all()


def test_canonical_numeric_labels_focal_is_zero(synthetic_two_cluster):
    raw, labels = synthetic_two_cluster
    out = canonical_numeric_labels(raw, labels)
    assert (out[labels == 1] == FOCAL_LABEL_INT).all()
    assert (out[labels == 0] == DIFFUSE_LABEL_INT).all()
    assert out.dtype == int


def test_apply_mapping_is_idempotent(synthetic_two_cluster):
    raw, labels = synthetic_two_cluster
    mapping = determine_mapping(raw, labels)
    once = apply_mapping(labels, mapping)
    twice = apply_mapping(once, mapping)
    pd.testing.assert_series_equal(once, twice)


def test_tie_breaker_uses_mean_when_medians_equal():
    """Median 1 vs 1 in both, but cluster A has mean 1.2 and B has mean 0.8."""
    pids = [f"p{i:03d}" for i in range(10)]
    labels = pd.Series([0] * 5 + [1] * 5, index=pids)
    raw = pd.DataFrame(
        {
            # both medians = 1, but mean differs: 0->1.2, 1->0.8
            "n_calcified_arteries": [0, 1, 1, 1, 3,
                                      0, 0, 1, 1, 2],
        },
        index=pids,
    )
    # Sanity check on the construction:
    assert np.median(raw.loc[labels == 0, "n_calcified_arteries"]) == \
           np.median(raw.loc[labels == 1, "n_calcified_arteries"])
    mapping = determine_mapping(raw, labels, tie_break="mean")
    # cluster 1 has lower mean -> focal
    assert mapping[1] == FOCAL_LABEL_STR


def test_tie_breaker_strict_raises_on_median_tie():
    pids = [f"p{i:03d}" for i in range(10)]
    labels = pd.Series([0] * 5 + [1] * 5, index=pids)
    raw = pd.DataFrame(
        {"n_calcified_arteries": [1] * 10},  # both medians AND means equal
        index=pids,
    )
    with pytest.raises(ValueError, match="identical median"):
        determine_mapping(raw, labels, tie_break="raise")


def test_mapping_raises_when_not_two_clusters():
    pids = [f"p{i:03d}" for i in range(9)]
    labels = pd.Series([0, 1, 2] * 3, index=pids)
    raw = pd.DataFrame({"n_calcified_arteries": [1] * 9}, index=pids)
    with pytest.raises(ValueError, match="exactly 2"):
        determine_mapping(raw, labels)
