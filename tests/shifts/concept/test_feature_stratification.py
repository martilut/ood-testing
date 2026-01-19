import numpy as np
import pandas as pd
import pytest

from oodt.shifts.concept.feature_stratification import FeatureStratificationShift


def test_feature_stratification_numeric_no_empty_partitions():
    # --- synthetic dataset ---
    X = pd.DataFrame(
        {
            "feature": np.linspace(0, 100, 200),
            "other": np.random.randn(200),
        }
    )
    y = pd.Series(np.zeros(len(X)), index=X.index)

    # --- define shift ---
    shift = FeatureStratificationShift(
        feature="feature",
        n_partitions=4,
        random_state=42,
    )

    # --- labels ---
    labels = shift.get_partition_labels(X, y)

    # --- basic invariants ---
    assert isinstance(labels, pd.Series)
    assert labels.index.equals(X.index)
    assert labels.dtype.kind in {"i", "u"}

    # --- partition ids ---
    assert set(labels.unique()) == {0, 1, 2, 3}

    # --- base class validation ---
    partitions = shift.get_partition_indices(X, y)

    assert len(partitions) == 4
    assert all(len(idx) > 0 for idx in partitions.values())


def test_feature_stratification_missing_feature_raises():
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([0, 1, 0])

    shift = FeatureStratificationShift(feature="missing")

    with pytest.raises(ValueError, match="Feature 'missing'"):
        shift.get_partition_labels(X, y)


def test_feature_stratification_categorical_too_few_values_raises():
    X = pd.DataFrame({"cat": ["a", "b", "a", "b"]})
    y = pd.Series([0, 0, 1, 1])

    shift = FeatureStratificationShift(
        feature="cat",
        n_partitions=3,
    )

    with pytest.raises(ValueError, match="cannot form 3 non-empty partitions"):
        shift.get_partition_indices(X, y)
