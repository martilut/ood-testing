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
        features=["feature"],
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

    # --- partition indices ---
    partitions = shift.get_partition_indices(X, y)

    assert all(isinstance(idx, pd.Index) for idx in partitions.values())
    assert all(len(idx) > 0 for idx in partitions.values())


def test_feature_stratification_categorical():
    X = pd.DataFrame({"cat": ["a", "b", "a", "c", "b", "c"]})
    y = pd.Series([0, 0, 1, 1, 0, 1])

    shift = FeatureStratificationShift(features=["cat"])

    labels = shift.get_partition_labels(X, y)

    # Each class should become a partition
    unique_classes = set(X["cat"].unique())
    partition_classes = set(labels.unique())
    assert len(partition_classes) == len(unique_classes)
    assert set(labels) == set(range(len(unique_classes)))

    partitions = shift.get_partition_indices(X, y)
    assert all(len(idx) > 0 for idx in partitions.values())


def test_feature_stratification_multiple_features():
    X = pd.DataFrame({
        "num": [1, 2, 3, 4, 5, 6],
        "cat": ["A", "A", "B", "B", "C", "C"]
    })
    y = pd.Series([0] * len(X))

    shift = FeatureStratificationShift(features=["num", "cat"], n_partitions=3)

    labels = shift.get_partition_labels(X, y)

    # Maximum possible partitions = 3 (num bins) * 3 (cat classes) = 9
    assert labels.max() <= 8
    assert labels.min() == 0

    partitions = shift.get_partition_indices(X, y)
    # All partitions returned should correspond to non-empty combinations
    for idx in partitions.values():
        assert len(idx) > 0


def test_missing_feature_raises():
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([0, 1, 0])

    shift = FeatureStratificationShift(features=["missing"])

    with pytest.raises(ValueError, match="Feature 'missing'"):
        shift.get_partition_labels(X, y)

if __name__ == "__main__":
    pytest.main()
