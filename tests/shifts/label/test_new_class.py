import pandas as pd
import pytest

from oodt.shifts.label.new_class import NewClassShift


def test_new_class_shift_basic():
    # --- synthetic dataset ---
    y = pd.Series([0, 1, 2, 0, 1, 2], name="target")
    X = pd.DataFrame({"feat": range(len(y))})

    shift = NewClassShift(n_classes=3)

    # --- basic invariants ---
    labels = shift.get_partition_labels(X, y)
    assert isinstance(labels, pd.Series)
    assert labels.index.equals(y.index)
    assert labels.dtype.kind in {"i", "u"}
    assert set(labels.unique()) <= set(range(shift.n_partitions))
    assert shift.n_partitions == 3


def test_new_class_shift_labels_match_y():
    # The labels should match y exactly
    y = pd.Series([0, 1, 2, 0, 1, 2], name="target")
    X = pd.DataFrame({"feat": range(len(y))})

    shift = NewClassShift(n_classes=3)
    labels = shift.get_partition_labels(X, y)

    # Check that the labels equal y
    pd.testing.assert_series_equal(labels, y.astype(int))


def test_new_class_shift_partition_indices_consistency():
    # Labels and partition indices should match
    y = pd.Series([0, 1, 2, 0, 1, 2], name="target")
    X = pd.DataFrame({"feat": range(len(y))})

    shift = NewClassShift(n_classes=3)
    labels = shift.get_partition_labels(X, y)
    partitions = shift.get_partition_indices(X, y)

    # Keys match partition IDs
    assert set(partitions.keys()) == set(range(shift.n_partitions))

    # Partition indices correspond to labels
    for pid, idx in partitions.items():
        assert (labels.loc[idx] == pid).all()


def test_new_class_shift_invalid_y_type():
    # Non-integer y should fail with astype(int)
    X = pd.DataFrame({"feat": [1, 2, 3]})
    y = pd.Series(["a", "b", "c"])

    shift = NewClassShift(n_classes=3)

    with pytest.raises(ValueError):
        shift.get_partition_labels(X, y)

if __name__ == "__main__":
    pytest.main()
