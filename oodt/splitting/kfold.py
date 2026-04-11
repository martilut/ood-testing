import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from typing import Dict, Generator, List, Literal, Optional, Tuple, Union


class KFoldSplitter:
    """
    K-fold cross-validation splitter with explicit support for ID/OOD partitions.

    Supports two top-level split strategies, controlled by `split_strategy`:

    "cv"            — standard k-fold: yields k (train, val, meta) folds.
    "train_val_test" — first carves out a held-out test set, then runs k-fold
                       CV on the remainder, yielding k (train, val, test, meta)
                       folds.  The test set is fixed across all folds.

    Modes (controlled via `mode`) define how ID/OOD pools map to split roles.
    They apply identically under both strategies:

    0: train = ID,        val/test = ID
    1: train = ID,        val/test = OOD
    2: train = ID,        val/test = balanced ID + OOD
    3: train = ID + OOD,  val/test = ID + OOD

    Under "train_val_test", the mode governs both the val fold (rotated across
    k folds) and the fixed test set with the same composition rule.

    Attributes
    ----------
    partitions : Dict[int, pd.Index]
        Mapping from partition ID → sample indices.
    n_splits : int
        Number of CV folds (k).
    mode : int
        Splitting mode (0–3).
    split_strategy : {"cv", "train_val_test"}
        Top-level strategy.
    test_ratio : float
        Fraction of the relevant pool reserved as the fixed test set.
        Only used when split_strategy="train_val_test".
    id_partitions : List[int]
        Partition IDs treated as in-distribution.
    ood_partitions : List[int]
        Partition IDs treated as out-of-distribution.
    stratify : bool
        Whether to use label stratification.
    random_state : int, optional
        Random seed.

    Notes
    -----
    - `split()` is always a generator; consume with a for-loop or list().
    - meta keys are a superset of TrainTestSplitter keys for drop-in compat:
        "train_id_indices", "train_ood_indices"
        "val_id_indices",   "val_ood_indices"
        "test_id_indices",  "test_ood_indices"   ← alias of val_* under "cv"
                                                    real held-out set under "train_val_test"
    - CRITICAL CONTRACT: meta["val_ood_indices"] / meta["test_ood_indices"]
      correctly identify OOD samples in X_val / X_test respectively.
      Downstream OOD mask: X_val.index.isin(meta["val_ood_indices"])
    """

    def __init__(
        self,
        partitions: Dict[int, pd.Index],
        n_splits: int = 5,
        mode: int = 0,
        split_strategy: Literal["cv", "train_val_test"] = "cv",
        test_ratio: float = 0.2,
        id_partitions: Optional[List[int]] = None,
        ood_partitions: Optional[List[int]] = None,
        stratify: bool = True,
        random_state: Optional[int] = None,
    ):
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if not 0.0 < test_ratio < 1.0:
            raise ValueError(f"test_ratio must be in (0, 1), got {test_ratio}")
        if split_strategy not in ("cv", "train_val_test"):
            raise ValueError(f"Unknown split_strategy: {split_strategy!r}")

        self.partitions      = partitions
        self.n_splits        = n_splits
        self.mode            = mode
        self.split_strategy  = split_strategy
        self.test_ratio      = test_ratio
        self.id_partitions   = id_partitions or list(partitions.keys())
        self.ood_partitions  = ood_partitions or []
        self.stratify        = stratify
        self.random_state    = random_state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect(self, partition_ids: List[int]) -> pd.Index:
        """Concatenate indices belonging to the requested partition IDs."""
        idx = pd.Index([])
        for pid in partition_ids:
            if pid in self.partitions:
                idx = idx.append(self.partitions[pid])
        return idx

    def _make_cv(self) -> Union[StratifiedKFold, KFold]:
        """Return a configured (Stratified)KFold instance."""
        if self.stratify:
            return StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
        return KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

    def _cv_splits(
        self, indices: pd.Index, y: pd.Series
    ) -> List[Tuple[pd.Index, pd.Index]]:
        """
        Run k-fold on `indices`, return list of (train_idx, val_idx) tuples
        using pandas Index objects aligned with the original X/y.
        """
        cv     = self._make_cv()
        arr    = indices.to_numpy()
        labels = y.loc[indices].to_numpy() if self.stratify else None

        return [
            (pd.Index(arr[tr]), pd.Index(arr[va]))
            for tr, va in cv.split(arr, labels)
        ]

    def _carve_test(
        self, indices: pd.Index, y: pd.Series
    ) -> Tuple[pd.Index, pd.Index]:
        """
        Split `indices` into (remainder, test) using test_ratio.

        Returns (remainder_idx, test_idx).  CV will run on remainder_idx.
        """
        if self.stratify:
            remainder, test = train_test_split(
                indices,
                test_size=self.test_ratio,
                stratify=y.loc[indices],
                random_state=self.random_state,
            )
        else:
            test = (
                indices.to_series()
                .sample(frac=self.test_ratio, random_state=self.random_state)
                .index
            )
            remainder = indices.difference(test)
        return pd.Index(remainder), pd.Index(test)

    @staticmethod
    def _balanced_sample(
        idx_a: pd.Index,
        idx_b: pd.Index,
        random_state: Optional[int],
    ) -> pd.Index:
        """
        Draw equal-sized samples from two index pools and concatenate.
        Used by mode 2 to balance ID and OOD in val/test.
        """
        n = min(len(idx_a), len(idx_b))
        if n == 0:
            return pd.Index([])
        a = idx_a.to_series().sample(n=n, random_state=random_state).index
        b = idx_b.to_series().sample(n=n, random_state=random_state).index
        return a.append(b)

    def _build_meta(
        self,
        train_idx:   pd.Index,
        val_idx:     pd.Index,
        id_indices:  pd.Index,
        ood_indices: pd.Index,
        test_idx:    Optional[pd.Index] = None,
    ) -> Dict[str, pd.Index]:
        """
        Construct metadata dict from train / val / (optional) test index sets.

        Under "cv":             test_* keys are aliases of val_* keys.
        Under "train_val_test": test_* keys reflect the real held-out test set.
        """
        val_id  = val_idx.intersection(id_indices)
        val_ood = val_idx.intersection(ood_indices)

        if test_idx is not None:
            test_id  = test_idx.intersection(id_indices)
            test_ood = test_idx.intersection(ood_indices)
        else:
            # "cv" mode: alias val as test for backward compat
            test_id  = val_id
            test_ood = val_ood

        return {
            "train_id_indices":  train_idx.intersection(id_indices),
            "train_ood_indices": train_idx.intersection(ood_indices),
            "val_id_indices":    val_id,
            "val_ood_indices":   val_ood,
            "test_id_indices":   test_id,
            "test_ood_indices":  test_ood,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Generator:
        """
        Yield k fold tuples according to split_strategy.

        "cv" strategy yields:
            (X_train, y_train, X_val, y_val, meta)

        "train_val_test" strategy yields:
            (X_train, y_train, X_val, y_val, X_test, y_test, meta)

        The test set is identical across all k folds.

        Parameters
        ----------
        X : pd.DataFrame
        y : pd.Series

        Yields
        ------
        See strategy description above.  meta keys:
            "train_id_indices", "train_ood_indices"
            "val_id_indices",   "val_ood_indices"
            "test_id_indices",  "test_ood_indices"
        """
        id_indices  = self._collect(self.id_partitions)
        ood_indices = self._collect(self.ood_partitions)

        if self.split_strategy == "cv":
            yield from self._run_cv(X, y, id_indices, ood_indices,
                                    test_idx=None)
        else:
            # Carve out the fixed test set first, then CV on the remainder.
            id_remainder,  id_test  = self._carve_test(id_indices,  y)
            ood_remainder, ood_test = (
                self._carve_test(ood_indices, y)
                if len(ood_indices) > 0
                else (pd.Index([]), pd.Index([]))
            )
            test_idx = self._compose_test(
                id_test, ood_test, id_indices, ood_indices, y
            )
            yield from self._run_cv(
                X, y,
                id_remainder, ood_remainder,
                test_idx=test_idx,
                id_indices_full=id_indices,
                ood_indices_full=ood_indices,
            )

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def _compose_test(
        self,
        id_test:     pd.Index,
        ood_test:    pd.Index,
        id_indices:  pd.Index,
        ood_indices: pd.Index,
        y:           pd.Series,
    ) -> pd.Index:
        """
        Build the fixed test set index according to the active mode.

        Mirrors the same composition rules used for val folds so that
        test composition is consistent with what the model was validated on.
        """
        if self.mode == 0:
            return id_test
        elif self.mode == 1:
            return ood_test
        elif self.mode == 2:
            return self._balanced_sample(id_test, ood_test, self.random_state)
        elif self.mode == 3:
            return id_test.append(ood_test)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _run_cv(
        self,
        X:               pd.DataFrame,
        y:               pd.Series,
        id_indices:      pd.Index,
        ood_indices:     pd.Index,
        test_idx:        Optional[pd.Index],
        id_indices_full:  Optional[pd.Index] = None,
        ood_indices_full: Optional[pd.Index] = None,
    ) -> Generator:
        """
        Dispatch to the per-mode CV generator.

        `id_indices` / `ood_indices` are the pools available for CV
        (i.e. with test samples already removed under train_val_test).
        `id_indices_full` / `ood_indices_full` are the original full pools,
        used only for meta intersection so test indices resolve correctly.
        """
        # For meta intersections always use the full (pre-carve) pools.
        id_full  = id_indices_full  if id_indices_full  is not None else id_indices
        ood_full = ood_indices_full if ood_indices_full is not None else ood_indices

        args = (X, y, id_indices, ood_indices, test_idx, id_full, ood_full)
        if self.mode == 0:
            yield from self._mode_0(*args)
        elif self.mode == 1:
            yield from self._mode_1(*args)
        elif self.mode == 2:
            yield from self._mode_2(*args)
        elif self.mode == 3:
            yield from self._mode_3(*args)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    # ------------------------------------------------------------------
    # Fold emission
    # ------------------------------------------------------------------

    def _emit(
        self,
        X:           pd.DataFrame,
        y:           pd.Series,
        train_idx:   pd.Index,
        val_idx:     pd.Index,
        id_full:     pd.Index,
        ood_full:    pd.Index,
        test_idx:    Optional[pd.Index],
    ) -> tuple:
        """
        Build and yield one fold tuple.

        Emits 5-tuple under "cv", 7-tuple under "train_val_test".
        """
        meta = self._build_meta(train_idx, val_idx, id_full, ood_full, test_idx)
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_val,   y_val   = X.loc[val_idx],   y.loc[val_idx]

        if test_idx is not None:
            X_test, y_test = X.loc[test_idx], y.loc[test_idx]
            return X_train, y_train, X_val, y_val, X_test, y_test, meta

        return X_train, y_train, X_val, y_val, meta

    # ------------------------------------------------------------------
    # Per-mode generators
    # ------------------------------------------------------------------

    def _mode_0(self, X, y, id_indices, ood_indices, test_idx, id_full, ood_full):
        """train = ID folds, val = ID fold."""
        for train_idx, val_idx in self._cv_splits(id_indices, y):
            yield self._emit(X, y, train_idx, val_idx, id_full, ood_full, test_idx)

    def _mode_1(self, X, y, id_indices, ood_indices, test_idx, id_full, ood_full):
        """train = ID folds (rotated), val = OOD fold (rotated)."""
        id_splits  = self._cv_splits(id_indices,  y)
        ood_splits = self._cv_splits(ood_indices, y)
        for (train_idx, _), (_, val_idx) in zip(id_splits, ood_splits):
            yield self._emit(X, y, train_idx, val_idx, id_full, ood_full, test_idx)

    def _mode_2(self, X, y, id_indices, ood_indices, test_idx, id_full, ood_full):
        """train = ID folds, val = balanced ID + OOD."""
        id_splits  = self._cv_splits(id_indices,  y)
        ood_splits = self._cv_splits(ood_indices, y)
        for (train_idx, val_id_idx), (_, val_ood_idx) in zip(id_splits, ood_splits):
            val_idx = self._balanced_sample(val_id_idx, val_ood_idx, self.random_state)
            yield self._emit(X, y, train_idx, val_idx, id_full, ood_full, test_idx)

    def _mode_3(self, X, y, id_indices, ood_indices, test_idx, id_full, ood_full):
        """train = ID + OOD folds, val = ID + OOD fold."""
        all_indices = id_indices.append(ood_indices)
        for train_idx, val_idx in self._cv_splits(all_indices, y):
            yield self._emit(X, y, train_idx, val_idx, id_full, ood_full, test_idx)
