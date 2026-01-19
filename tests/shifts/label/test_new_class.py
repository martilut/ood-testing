from pathlib import Path

import pandas as pd

from oodt.data.loaders import CSVDataset
from oodt.shifts.label.new_class import NewClassShift
from oodt.utils.utils import get_project_path


def test_new_class_shift_on_electricity():
    # --- load dataset ---
    dataset_dir = get_project_path() / Path("datasets/partitions/electricity_small")
    paths = {
        "source": dataset_dir / "source.csv",
        "target": dataset_dir / "target.csv",
    }

    dataset = CSVDataset(
        path=paths,
        target_col="class",
        name="electricity_small",
    )
    dataset.load()

    X, y = dataset.data, dataset.target

    assert len(X) == len(y)
    assert y.name == "class"

    # --- define shift ---
    held_out_classes = sorted(y.unique())[:1]
    shift = NewClassShift(held_out_classes)

    # --- basic invariants ---
    assert shift.n_partitions == 1 + len(held_out_classes)

    labels = shift.get_partition_labels(X, y)

    # --- label validity ---
    assert isinstance(labels, pd.Series)
    assert labels.index.equals(y.index)
    assert set(labels.unique()).issubset(set(range(shift.n_partitions)))

    # --- semantic correctness ---
    # ID partition must NOT contain held-out classes
    id_indices = labels[labels == 0].index
    assert not y.loc[id_indices].isin(held_out_classes).any()

    # Each held-out class must map to its own partition
    for i, cls in enumerate(held_out_classes, start=1):
        cls_indices = labels[labels == i].index
        assert len(cls_indices) > 0
        assert y.loc[cls_indices].eq(cls).all()

    # --- partition indices consistency ---
    partitions = shift.get_partition_indices(X, y)
    assert set(partitions.keys()) == set(range(shift.n_partitions))

    for pid, idx in partitions.items():
        assert (labels.loc[idx] == pid).all()