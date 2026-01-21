import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Tuple


class TrainTestSplitter:
    """
    Create train/test splits based on partitions returned by a shift strategy.
    Supports different modes, stratification, and preserves ID/OOD metadata.

    Modes (use `mode` argument as integer):
        0: train = ID only, test = ID only
        1: train = ID only, test = OOD only
        2: train = ID only, test = ID + OOD
        3: train = ID + OOD, test = ID + OOD
    """

    def __init__(
        self,
        partitions: Dict[int, pd.Index],
        mode: int = 0,
        train_ratio: float = 0.7,
        test_ratio: float = 0.3,
        id_partitions: Optional[List[int]] = None,
        ood_partitions: Optional[List[int]] = None,
        stratify: bool = True,
        random_state: Optional[int] = None,
    ):
        """
        partitions: dict of {partition_id: sample indices} returned by strategy
        mode: integer, see class docstring
        id_partitions: list of partition IDs considered ID
        ood_partitions: list of partition IDs considered OOD
        train_ratio: fraction of selected data to use for training
        test_ratio: fraction of selected data to use for testing
        stratify: stratified sampling by labels if True
        random_state: seed for reproducibility
        """
        self.partitions = partitions
        self.mode = mode
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.id_partitions = id_partitions or [pid for pid in partitions.keys()]
        self.ood_partitions = ood_partitions or []
        self.stratify = stratify
        self.random_state = random_state

    def split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Dict[str, pd.Index]]:
        # --- collect indices ---
        id_indices = pd.Index([])
        for pid in self.id_partitions:
            if pid in self.partitions:
                id_indices = id_indices.append(self.partitions[pid])

        ood_indices = pd.Index([])
        for pid in self.ood_partitions:
            if pid in self.partitions:
                ood_indices = ood_indices.append(self.partitions[pid])

        # --- sample helper ---
        def sample_indices(indices, ratio):
            if len(indices) == 0:
                return pd.Index([])
            if self.stratify:
                return train_test_split(
                    indices,
                    train_size=ratio,
                    stratify=y.loc[indices],
                    random_state=self.random_state,
                )[0]
            else:
                return indices.to_series().sample(frac=ratio, random_state=self.random_state).index

        # --- prepare train/test based on mode ---
        if self.mode == 0:  # train = ID, test = ID
            train_idx = sample_indices(id_indices, self.train_ratio)
            test_idx = sample_indices(id_indices.difference(train_idx), self.test_ratio)
        elif self.mode == 1:  # train = ID, test = OOD
            train_idx = sample_indices(id_indices, self.train_ratio)
            test_idx = sample_indices(ood_indices, self.test_ratio)
        elif self.mode == 2:  # train = ID, test = ID + OOD
            train_idx = sample_indices(id_indices, self.train_ratio)
            test_id_idx = sample_indices(id_indices.difference(train_idx), self.test_ratio)
            test_ood_idx = sample_indices(ood_indices, self.test_ratio)
            test_idx = test_id_idx.append(test_ood_idx)
        elif self.mode == 3:  # train = ID + OOD, test = ID + OOD
            all_indices = id_indices.append(ood_indices)
            train_idx = sample_indices(all_indices, self.train_ratio)
            test_idx = all_indices.difference(train_idx)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # --- return datasets ---
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]

        # --- metadata ---
        meta = {
            "train_id_indices": train_idx.intersection(id_indices),
            "train_ood_indices": train_idx.intersection(ood_indices),
            "test_id_indices": test_idx.intersection(id_indices),
            "test_ood_indices": test_idx.intersection(ood_indices),
        }

        return X_train, y_train, X_test, y_test, meta
