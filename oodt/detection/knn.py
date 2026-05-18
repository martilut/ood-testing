"""
K-Nearest Neighbour OOD detector.

Scores a test sample by its average distance to the k nearest training
neighbours.  High distance → likely OOD.

Two variants are supported via `aggregation`:
  "mean"   — average distance to all k neighbours (default)
  "max"    — distance to the k-th (farthest) neighbour only

Reference
---------
Hendrycks & Gimpel (2017); Sun et al., "Out-of-Distribution Detection with
Deep Nearest Neighbors", ICML 2022.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from oodt.detection.base import BaseOODDetector


class KNNOODDetector(BaseOODDetector):
    """
    OOD detector based on k-nearest-neighbour distance in feature space.

    Parameters
    ----------
    k : int
        Number of neighbours to consider.
    aggregation : {"mean", "max"}
        How to aggregate the k distances into a single score.
        "mean" gives a smoother signal; "max" is more sensitive to
        isolated high-distance dimensions.
    metric : str
        Distance metric accepted by sklearn NearestNeighbors
        (e.g. "euclidean", "cosine", "manhattan").
    normalize : bool
        If True, fit a StandardScaler on training data before computing
        distances.  Recommended when features have different scales.
    n_jobs : int
        Parallelism for NearestNeighbors.  -1 = all cores.
    """

    def __init__(
        self,
        k: int = 5,
        aggregation: Literal["mean", "max"] = "mean",
        metric: str = "euclidean",
        normalize: bool = True,
        n_jobs: int = -1,
    ) -> None:
        super().__init__()
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if aggregation not in ("mean", "max"):
            raise ValueError(f"aggregation must be 'mean' or 'max', got {aggregation!r}")

        self.k = k
        self.aggregation = aggregation
        self.metric = metric
        self.normalize = normalize
        self.n_jobs = n_jobs

        self._nn: Optional[NearestNeighbors] = None
        self._scaler: Optional[StandardScaler] = None

    # ------------------------------------------------------------------
    # BaseOODDetector interface
    # ------------------------------------------------------------------

    def fit(self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None) -> "KNNOODDetector":
        """
        Index the training set.

        Parameters
        ----------
        X_train : pd.DataFrame
            ID training features.
        y_train : ignored
            Kept for API compatibility.
        """
        X = X_train.to_numpy(dtype=float)

        if self.normalize:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)

        self._nn = NearestNeighbors(
            n_neighbors=self.k,
            metric=self.metric,
            n_jobs=self.n_jobs,
        )
        self._nn.fit(X)
        self.is_fitted_ = True
        return self

    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute OOD scores for each sample.

        Higher score = more likely OOD.

        Parameters
        ----------
        X : pd.DataFrame
            Samples to score.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
        """
        self._check_is_fitted()
        Xn = X.to_numpy(dtype=float)

        if self.normalize and self._scaler is not None:
            Xn = self._scaler.transform(Xn)

        distances, _ = self._nn.kneighbors(Xn)  # shape (n, k)

        if self.aggregation == "mean":
            return distances.mean(axis=1)
        else:  # "max"
            return distances[:, -1]

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """
        Binary OOD prediction (1 = OOD, 0 = ID).

        Parameters
        ----------
        X : pd.DataFrame
        threshold : float, optional
            Score threshold.  If None, uses ``self.threshold_`` set during
            ``fit_threshold()``.
        """
        scores = self.score_samples(X)
        thr = threshold if threshold is not None else self._get_threshold()
        return (scores >= thr).astype(int)

    def get_params(self) -> dict:
        return {
            "k": self.k,
            "aggregation": self.aggregation,
            "metric": self.metric,
            "normalize": self.normalize,
        }
