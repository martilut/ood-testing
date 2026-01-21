from typing import Optional, List
import pandas as pd
import numpy as np
from oodt.shifts.base import BaseShiftStrategy


class FeatureStratificationShift(BaseShiftStrategy):
    """
    Concept shift: split samples based on one or more features.
    Each partition corresponds to a combination of feature bins/classes.
    Empty partitions are ignored.
    """

    def __init__(
        self,
        features: List[str],
        n_partitions: int = 2,
        random_state: Optional[int] = None,
    ):
        """
        Args:
            features: list of feature names to stratify on
            n_partitions: number of partitions per numeric feature
            random_state: optional seed
        """
        if not features:
            raise ValueError("At least one feature must be specified")
        super().__init__(
            name="FeatureStratificationShift",
            n_partitions=n_partitions,  # used per numeric feature
            random_state=random_state,
        )
        self.features = features

    def get_partition_labels(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> pd.Series:
        for f in self.features:
            if f not in X.columns:
                raise ValueError(f"Feature '{f}' not in DataFrame columns")

        # Create a DataFrame to store feature-level labels
        feature_labels = pd.DataFrame(index=X.index)

        for f in self.features:
            col = X[f]

            if np.issubdtype(col.dtype, np.number):
                # numeric → n_partitions equal-width bins
                labels = pd.cut(
                    col, bins=self.n_partitions, labels=False, include_lowest=True
                )
            else:
                # categorical → each class is a partition
                unique_vals = pd.Index(col.unique())
                mapping = {v: i for i, v in enumerate(unique_vals)}
                labels = col.map(mapping)

            feature_labels[f] = labels.astype(int)

        # Combine feature labels into tuples (one tuple per sample)
        partition_tuples = feature_labels.apply(lambda row: tuple(row), axis=1)

        # Map tuples to integer IDs (only for non-empty partitions)
        unique_partitions = {v: i for i, v in enumerate(partition_tuples.unique())}
        partition_ids = partition_tuples.map(unique_partitions).astype(int)

        return partition_ids
