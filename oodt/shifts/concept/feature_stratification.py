from typing import Dict, Optional, Union
import pandas as pd
import numpy as np
from oodt.shifts.base import BaseShiftStrategy


class FeatureStratificationShift(BaseShiftStrategy):
    """
    Concept shift: split samples based on the value of a selected feature.
    Each partition corresponds to a range or category of the feature,
    enabling OOD evaluation on feature subgroups.
    """

    def __init__(
        self,
        feature: str,
        n_partitions: int = 2,
        bins: Optional[Union[int, list]] = None,
        shuffle: bool = True,
        random_state: Optional[int] = None
    ):
        """
        Args:
            feature: feature column name to stratify on
            n_partitions: number of partitions to create
            bins: optional bin edges (if numeric); if None and numeric, will use equal-width bins
            shuffle: shuffle samples within each partition
            random_state: optional seed
        """
        super().__init__()
        self.feature = feature
        self.n_partitions = n_partitions
        self.bins = bins
        self.shuffle = shuffle
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    def get_partition_labels(self, X: pd.DataFrame, y: pd.Series = None) -> pd.Series:
        """
        Assign each sample to a partition based on the value of the specified feature.

        Returns:
            pd.Series: partition label (int) for each sample
        """
        if self.feature not in X.columns:
            raise ValueError(f"Feature '{self.feature}' not in DataFrame columns")

        col = X[self.feature]

        # Numeric feature
        if np.issubdtype(col.dtype, np.number):
            if self.bins is not None:
                # Use user-provided bins
                labels = pd.cut(col, bins=self.bins, labels=False, include_lowest=True)
            else:
                # Equal-width bins
                labels = pd.cut(col, bins=self.n_partitions, labels=False, include_lowest=True)
        else:
            # Categorical feature
            unique_vals = col.unique()
            if len(unique_vals) <= self.n_partitions:
                # Each unique value is a partition
                mapping = {v: i for i, v in enumerate(unique_vals)}
                labels = col.map(mapping)
            else:
                # Group categories into n_partitions roughly equal in size
                labels = pd.qcut(col.rank(method="first"), q=self.n_partitions, labels=False)

        labels = labels.astype(int)

        if self.shuffle:
            labels = labels.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        # Map back to original index
        labels.index = X.index

        return labels

    def get_partition_indices(self, X: pd.DataFrame, y: pd.Series = None) -> Dict[int, pd.Index]:
        """
        Return dict mapping partition_id -> sample indices
        """
        labels = self.get_partition_labels(X, y)
        partitions = {pid: labels.index[labels == pid] for pid in labels.unique()}
        return partitions
