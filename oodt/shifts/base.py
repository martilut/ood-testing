from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
import numpy as np


class BaseShiftStrategy(ABC):
    """
    Abstract base class for OOD / data shift strategies.

    A shift strategy assigns each sample to a partition.
    Partition IDs can be used to simulate multiple OOD regions or levels.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        random_state: Optional[int] = None,
        n_partitions: int = 2,
    ):
        """
        Args:
            name: Optional name of the shift strategy
            random_state: Optional seed for reproducibility
            n_partitions: Number of partitions (>=2)
        """
        self.name = name or self.__class__.__name__
        self.random_state = random_state
        self.n_partitions = max(2, n_partitions)
        if self.random_state is not None:
            np.random.seed(self.random_state)

    @abstractmethod
    def get_partition_labels(
        self, X: pd.DataFrame, y: pd.Series
    ) -> pd.Series:
        """
        Assign each sample to a partition.

        Args:
            X: Feature DataFrame (N x F)
            y: Target Series (N,)

        Returns:
            pd.Series of integers [0, n_partitions-1], indexed by X.index
        """
        pass

    def get_partition_indices(self, X: pd.DataFrame, y: pd.Series) -> dict[int, pd.Index]:
        labels = self.get_partition_labels(X, y)
        self._validate_labels(labels)

        partitions = {
            pid: labels.index[labels == pid]
            for pid in range(self.n_partitions)
        }

        empty = [pid for pid, idx in partitions.items() if len(idx) == 0]
        if empty:
            raise ValueError(
                f"{self.name}: empty partitions detected: {empty}. "
                "This shift strategy requires all partitions to be non-empty."
            )

        return partitions

    def _validate_labels(self, labels: pd.Series):
        used = set(labels.unique())
        expected = set(range(self.n_partitions))

        missing = expected - used
        if missing:
            raise ValueError(
                f"{self.name}: partitions {sorted(missing)} are empty."
            )

    def __repr__(self):
        return f"<{self.name} shift strategy, n_partitions={self.n_partitions}>"
