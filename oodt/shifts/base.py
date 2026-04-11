from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
import numpy as np


class BaseShiftStrategy(ABC):
    """
    Abstract base class for OOD / data shift strategies.

    A shift strategy partitions the dataset into multiple subsets, which can be
    interpreted as different distributions (ID/OOD regions or multiple OOD regimes).

    The core idea:
    - Each sample is assigned a partition ID.
    - Partitions are later used by the pipeline and splitter to simulate or define OOD.

    Attributes
    ----------
    name : str
        Name of the shift strategy (defaults to class name).

    random_state : int, optional
        Random seed for reproducibility. If provided, sets NumPy global seed.

    n_partitions : int
        Number of partitions to generate.
        Must be >= 2. Enforced automatically.

    Notes
    -----
    - Partition IDs must be integers in range [0, n_partitions - 1].
    - All partitions must be non-empty (strict requirement).
    - Output must be deterministic if `random_state` is set.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        random_state: Optional[int] = None,
        n_partitions: int = 2,
    ):
        """
        Initialize shift strategy.

        Parameters
        ----------
        name : str, optional
            Custom name of the strategy.

        random_state : int, optional
            Random seed for reproducibility.

        n_partitions : int, default=2
            Number of partitions to create (minimum = 2).
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

        Parameters
        ----------
        X : pd.DataFrame, shape (N, F)
            Feature matrix.

        y : pd.Series, shape (N,)
            Target values.

        Returns
        -------
        pd.Series
            Partition labels for each sample.

            Requirements:
            - Index must match X.index
            - Values must be integers in [0, n_partitions - 1]
            - All partitions must be present (no missing IDs)

        Example
        -------
        [0, 0, 1, 1, 2, 2, ...]
        """
        pass

    def get_partition_indices(self, X: pd.DataFrame, y: pd.Series) -> dict[int, pd.Index]:
        """
        Convert partition labels into index-based partitions.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        y : pd.Series
            Target values.

        Returns
        -------
        dict[int, pd.Index]
            Mapping from partition ID → indices of samples.

            Example:
            {
                0: Index([...]),
                1: Index([...]),
                2: Index([...])
            }

        Raises
        ------
        ValueError
            If:
            - Some partitions are missing
            - Any partition is empty

        Notes
        -----
        This is the main interface used by the pipeline and splitter.
        """
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
        """
        Validate partition labels.

        Parameters
        ----------
        labels : pd.Series
            Output of `get_partition_labels`.

        Raises
        ------
        ValueError
            If any partition ID in [0, n_partitions-1] is missing.

        Notes
        -----
        This ensures full coverage of partitions.
        """
        used = set(labels.unique())
        expected = set(range(self.n_partitions))

        missing = expected - used
        if missing:
            raise ValueError(
                f"{self.name}: partitions {sorted(missing)} are empty."
            )

    def __repr__(self):
        """
        String representation of the shift strategy.
        """
        return f"<{self.name} shift strategy, n_partitions={self.n_partitions}>"
