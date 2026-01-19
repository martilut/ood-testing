from typing import Optional, Union
import pandas as pd
import numpy as np
from oodt.shifts.base import BaseShiftStrategy


class FeatureStratificationShift(BaseShiftStrategy):
    """
    Concept shift: split samples based on the value of a selected feature.
    Each partition corresponds to a range or category of the feature.
    """

    def __init__(
        self,
        feature: str,
        n_partitions: int = 2,
        bins: Optional[Union[int, list]] = None,
        random_state: Optional[int] = None,
    ):
        """
        Args:
            feature: feature column name to stratify on
            n_partitions: number of partitions (>=2)
            bins: optional bin edges (numeric features only)
            random_state: optional seed
        """
        super().__init__(
            name="FeatureStratificationShift",
            n_partitions=n_partitions,
            random_state=random_state,
        )
        self.feature = feature
        self.bins = bins

    def get_partition_labels(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> pd.Series:
        if self.feature not in X.columns:
            raise ValueError(f"Feature '{self.feature}' not in DataFrame columns")

        col = X[self.feature]

        # --- numeric feature ---
        if np.issubdtype(col.dtype, np.number):
            if self.bins is not None:
                labels = pd.cut(
                    col,
                    bins=self.bins,
                    labels=False,
                    include_lowest=True,
                )
            else:
                labels = pd.cut(
                    col,
                    bins=self.n_partitions,
                    labels=False,
                    include_lowest=True,
                )

        # --- categorical feature ---
        else:
            unique_vals = pd.Index(col.unique())

            if len(unique_vals) < self.n_partitions:
                raise ValueError(
                    f"{self.name}: feature '{self.feature}' has only "
                    f"{len(unique_vals)} unique values, "
                    f"cannot form {self.n_partitions} non-empty partitions."
                )

            mapping = {
                v: i % self.n_partitions
                for i, v in enumerate(unique_vals)
            }
            labels = col.map(mapping)

        labels = labels.astype(int)
        labels.index = X.index

        # base class will enforce non-empty partitions
        return labels
