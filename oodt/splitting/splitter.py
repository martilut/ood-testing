import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Tuple


class TrainTestSplitter:
    """
    Train/test splitter with explicit support for ID/OOD partitions.

    This class creates splits based on partition assignments produced by a shift strategy.
    It supports multiple evaluation modes and preserves ID/OOD membership metadata.

    Modes (controlled via `mode`):

    0: train = ID only, test = ID only
    1: train = ID only, test = OOD only
    2: train = ID only, test = ID + OOD (balanced)
    3: train = ID + OOD, test = ID + OOD

    Attributes
    ----------
    partitions : Dict[int, pd.Index]
        Mapping from partition ID → sample indices.

        Example:
        {
            0: Index([...]),
            1: Index([...]),
            2: Index([...])
        }

    mode : int
        Splitting mode (0–3), controlling how ID and OOD samples are used.

    train_ratio : float
        Fraction of selected data used for training.

    test_ratio : float
        Fraction of selected data used for testing (only used in some modes).

    id_partitions : List[int]
        Partition IDs considered in-distribution (ID).

    ood_partitions : List[int]
        Partition IDs considered out-of-distribution (OOD).

    stratify : bool
        Whether to use label stratification during sampling.

    random_state : int, optional
        Random seed for reproducibility.

    Notes
    -----
    - Partitions must be provided before calling `split()`.
    - ID/OOD semantics are defined explicitly via `id_partitions` and `ood_partitions`.
    - Metadata returned by `split()` is REQUIRED by downstream pipeline components.
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
        Initialize splitter.

        Parameters
        ----------
        partitions : Dict[int, pd.Index]
            Partition mapping from shift strategy.

        mode : int, default=0
            Splitting mode (see class docstring).

        train_ratio : float, default=0.7
            Fraction of data used for training.

        test_ratio : float, default=0.3
            Fraction of data used for testing.

        id_partitions : List[int], optional
            Partition IDs treated as ID.
            Defaults to all partitions if not provided.

        ood_partitions : List[int], optional
            Partition IDs treated as OOD.
            Defaults to empty list.

        stratify : bool, default=True
            Whether to stratify sampling by labels.

        random_state : int, optional
            Random seed.
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
    ) -> Tuple[
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
        pd.Series,
        Dict[str, pd.Index]
    ]:
        """
        Perform train/test split based on configured mode and partitions.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        y : pd.Series
            Target vector.

        Returns
        -------
        X_train : pd.DataFrame
        y_train : pd.Series
        X_test : pd.DataFrame
        y_test : pd.Series
        meta : Dict[str, pd.Index]

            Metadata describing ID/OOD membership:

            - "train_id_indices"
            - "train_ood_indices"
            - "test_id_indices"
            - "test_ood_indices"

            These indices are aligned with X and are REQUIRED for OOD evaluation.

        Notes
        -----
        CRITICAL CONTRACT:
        - `meta["test_ood_indices"]` must correctly identify OOD samples in X_test.
        - Downstream pipeline constructs OOD mask via:
              X_test.index.isin(meta["test_ood_indices"])
        """

        # ============================================================
        # Collect ID / OOD indices
        # ============================================================

        id_indices = pd.Index([])
        for pid in self.id_partitions:
            if pid in self.partitions:
                id_indices = id_indices.append(self.partitions[pid])

        ood_indices = pd.Index([])
        for pid in self.ood_partitions:
            if pid in self.partitions:
                ood_indices = ood_indices.append(self.partitions[pid])

        # ============================================================
        # Sampling helper
        # ============================================================

        def sample_indices(indices, ratio):
            """
            Sample a subset of indices.

            Uses stratified sampling if enabled.
            """
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
                return indices.to_series().sample(
                    frac=ratio,
                    random_state=self.random_state
                ).index

        # ============================================================
        # Mode logic
        # ============================================================

        if self.mode == 0:
            # Train: ID, Test: ID
            train_idx = sample_indices(id_indices, self.train_ratio)
            test_idx = sample_indices(
                id_indices.difference(train_idx),
                self.test_ratio
            )

        elif self.mode == 1:
            # Train: ID, Test: OOD
            train_idx = sample_indices(id_indices, self.train_ratio)
            test_idx = sample_indices(ood_indices, self.test_ratio)

        elif self.mode == 2:
            # Train: ID, Test: balanced ID + OOD

            train_idx = sample_indices(id_indices, self.train_ratio)
            remaining_id = id_indices.difference(train_idx)

            n_test_id = len(remaining_id)
            n_test_ood = len(ood_indices)
            n_equal = min(n_test_id, n_test_ood) // 2

            if n_equal > 0:
                test_id_idx = remaining_id.to_series().sample(
                    n=n_equal,
                    random_state=self.random_state
                ).index

                test_ood_idx = ood_indices.to_series().sample(
                    n=n_equal,
                    random_state=self.random_state
                ).index

                test_idx = test_id_idx.append(test_ood_idx)
            else:
                test_idx = pd.Index([])

        elif self.mode == 3:
            # Train: ID + OOD, Test: ID + OOD

            all_indices = id_indices.append(ood_indices)
            train_idx = sample_indices(all_indices, self.train_ratio)
            test_idx = all_indices.difference(train_idx)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # ============================================================
        # Build datasets
        # ============================================================

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
