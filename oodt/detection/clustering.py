"""
Clustering-based OOD detector (k-means).

A sample's OOD score is its distance to the nearest cluster centroid.
Intuitively, ID samples cluster tightly around centroids learned from
training data; OOD samples fall far from all centroids.

Two score variants:
  "min_dist"  — distance to the nearest centroid (default)
  "soft"      — negative log of the normalised assignment probability
                (a Gaussian kernel applied to centroid distances, similar
                to a Gaussian Mixture likelihood)

Reference
---------
Lee et al., "A Simple Unified Framework for Detecting Out-of-Distribution
Samples and Adversarial Attacks", NeurIPS 2018  (Mahalanobis variant;
this uses Euclidean k-means as a lighter alternative).
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from oodt.detection.base import BaseOODDetector


class ClusteringOODDetector(BaseOODDetector):
    """
    OOD detector based on distance to k-means cluster centroids.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.  A good starting point is sqrt(n_train / 2).
    score_type : {"min_dist", "soft"}
        "min_dist" — raw distance to nearest centroid.
        "soft"     — negative log normalised Gaussian assignment probability.
                     More sensitive to samples that are equidistant from all
                     centroids (i.e. genuinely outside the cluster structure).
    bandwidth : float, optional
        Bandwidth for the Gaussian kernel used in "soft" scoring.
        Defaults to the median intra-cluster distance on training data.
    normalize : bool
        Fit a StandardScaler on training data before clustering.
    mini_batch : bool
        Use MiniBatchKMeans for large datasets (> ~50k samples).
    random_state : int, optional
    """

    def __init__(
        self,
        n_clusters: int = 8,
        score_type: Literal["min_dist", "soft"] = "min_dist",
        bandwidth: Optional[float] = None,
        normalize: bool = True,
        mini_batch: bool = False,
        random_state: Optional[int] = 42,
    ) -> None:
        super().__init__()
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")

        self.n_clusters = n_clusters
        self.score_type = score_type
        self.bandwidth = bandwidth
        self.normalize = normalize
        self.mini_batch = mini_batch
        self.random_state = random_state

        self._km: Optional[KMeans | MiniBatchKMeans] = None
        self._scaler: Optional[StandardScaler] = None
        self._bandwidth_: Optional[float] = None

    # ------------------------------------------------------------------
    # BaseOODDetector interface
    # ------------------------------------------------------------------

    def fit(self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None) -> "ClusteringOODDetector":
        """
        Fit k-means on training data and optionally calibrate bandwidth.

        Parameters
        ----------
        X_train : pd.DataFrame
        y_train : ignored
        """
        X = X_train.to_numpy(dtype=float)

        if self.normalize:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)

        KMClass = MiniBatchKMeans if self.mini_batch else KMeans
        self._km = KMClass(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init="auto" if not self.mini_batch else 3,
        )
        self._km.fit(X)

        # Calibrate bandwidth from training data
        if self.score_type == "soft":
            if self.bandwidth is not None:
                self._bandwidth_ = self.bandwidth
            else:
                dists = self._centroid_distances(X)            # (n, k)
                min_dists = dists.min(axis=1)                  # (n,)
                self._bandwidth_ = float(np.median(min_dists)) or 1.0

        self.is_fitted_ = True
        return self

    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute OOD scores.  Higher = more likely OOD.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
        """
        self._check_is_fitted()
        Xn = X.to_numpy(dtype=float)

        if self.normalize and self._scaler is not None:
            Xn = self._scaler.transform(Xn)

        dists = self._centroid_distances(Xn)  # (n, k)

        if self.score_type == "min_dist":
            return dists.min(axis=1)

        # "soft": OOD score = -log p(x)  where p ∝ sum_k exp(-d²/2h²)
        h = self._bandwidth_
        log_probs = np.log(np.exp(-dists**2 / (2 * h**2)).sum(axis=1) + 1e-12)
        return -log_probs

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Binary OOD prediction (1 = OOD, 0 = ID)."""
        scores = self.score_samples(X)
        thr = threshold if threshold is not None else self._get_threshold()
        return (scores >= thr).astype(int)

    def get_params(self) -> dict:
        return {
            "n_clusters": self.n_clusters,
            "score_type": self.score_type,
            "bandwidth": self.bandwidth,
            "normalize": self.normalize,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _centroid_distances(self, X: np.ndarray) -> np.ndarray:
        """Return (n_samples, n_clusters) Euclidean distance matrix."""
        centroids = self._km.cluster_centers_       # (k, d)
        # squared distances — broadcasting: (n,1,d) - (1,k,d)
        diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        return np.sqrt((diff**2).sum(axis=2))       # (n, k)

    @property
    def centroids_(self) -> np.ndarray:
        """Cluster centroids in (optionally scaled) feature space."""
        self._check_is_fitted()
        return self._km.cluster_centers_

    @property
    def labels_(self) -> np.ndarray:
        """Cluster assignment for each training sample."""
        self._check_is_fitted()
        return self._km.labels_
