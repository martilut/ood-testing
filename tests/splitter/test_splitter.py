import numpy as np
import pytest
import pandas as pd
from oodt.splitter.splitter import TrainTestSplitter

# --- synthetic dataset helper ---
def create_synthetic_dataset():
    # 12 samples, 2 classes
    X = pd.DataFrame({
        "feat1": range(12),
        "feat2": np.arange(12) * 10
    })
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1], name="target")

    # define 3 partitions
    partitions = {
        0: pd.Index([0, 1, 2, 3]),
        1: pd.Index([4, 5, 6, 7]),
        2: pd.Index([8, 9, 10, 11])
    }
    return X, y, partitions


# --- mode 0: train=id only, test=id only ---
def test_mode_0_id_id():
    X, y, partitions = create_synthetic_dataset()
    splitter = TrainTestSplitter(
        partitions,
        mode=0,
        train_ratio=0.5,
        test_ratio=0.5,
        stratify=True,
        id_partitions=[0,1,2]
    )

    X_train, y_train, X_test, y_test, meta = splitter.split(X, y)

    # --- all samples from ID partitions ---
    id_indices = partitions[0].append(partitions[1]).append(partitions[2])
    assert set(X_train.index).issubset(set(id_indices))
    assert set(X_test.index).issubset(set(id_indices))

    # --- disjoint ---
    assert X_train.index.intersection(X_test.index).empty


# --- mode 1: train=id only, test=ood only ---
def test_mode_1_id_ood():
    X, y, partitions = create_synthetic_dataset()
    splitter = TrainTestSplitter(
        partitions,
        mode=1,
        train_ratio=0.5,
        test_ratio=0.5,
        stratify=True,
        id_partitions=[0,1],
        ood_partitions=[2]
    )

    X_train, y_train, X_test, y_test, meta = splitter.split(X, y)

    # --- train only from ID partitions ---
    id_indices = partitions[0].append(partitions[1])
    assert set(X_train.index).issubset(set(id_indices))

    # --- test only from OOD partitions ---
    ood_indices = partitions[2]
    assert set(X_test.index).issubset(set(ood_indices))

    # --- disjoint ---
    assert X_train.index.intersection(X_test.index).empty


# --- mode 2: train=id only, test=id+ood ---
def test_mode_2_id_id_plus_ood():
    X, y, partitions = create_synthetic_dataset()
    splitter = TrainTestSplitter(
        partitions,
        mode=2,
        train_ratio=0.5,
        test_ratio=0.5,
        stratify=True,
        id_partitions=[0,1],
        ood_partitions=[2]
    )

    X_train, y_train, X_test, y_test, meta = splitter.split(X, y)

    # --- train only ID partitions ---
    id_indices = partitions[0].append(partitions[1])
    assert set(X_train.index).issubset(set(id_indices))

    # --- test can include ID and OOD ---
    combined_indices = id_indices.append(partitions[2])
    assert set(X_test.index).issubset(set(combined_indices))

    # --- disjoint ---
    assert X_train.index.intersection(X_test.index).empty


# --- mode 3: train=id+ood, test=id+ood ---
def test_mode_3_id_plus_ood_id_plus_ood():
    X, y, partitions = create_synthetic_dataset()
    splitter = TrainTestSplitter(
        partitions,
        mode=3,
        train_ratio=0.5,
        test_ratio=0.5,
        stratify=True,
        id_partitions=[0,1],
        ood_partitions=[2]
    )

    X_train, y_train, X_test, y_test, meta = splitter.split(X, y)

    # --- train + test cover all partitions ---
    all_indices = partitions[0].append(partitions[1]).append(partitions[2])
    assert set(X_train.index).union(set(X_test.index)) == set(all_indices)

    # --- disjoint ---
    assert X_train.index.intersection(X_test.index).empty

    # --- meta info ---
    assert set(meta["train_id_indices"]).issubset(set(partitions[0].append(partitions[1])))
    assert set(meta["test_ood_indices"]).issubset(set(partitions[2]))


# --- stratification preserves class ratios ---
def test_stratification_preserves_classes():
    X, y, partitions = create_synthetic_dataset()
    splitter = TrainTestSplitter(
        partitions,
        mode=0,
        train_ratio=0.5,
        test_ratio=0.5,
        stratify=True,
        id_partitions=[0,1,2]
    )

    X_train, y_train, X_test, y_test, _ = splitter.split(X, y)

    # check that both classes appear in train/test
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}
