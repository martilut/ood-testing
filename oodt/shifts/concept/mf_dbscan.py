"""
MF-DBSCAN shift strategy.

Builds the same metafeature (MF) space as MFKMeansShift, but clusters it
with DBSCAN instead of KMeans. Centroids are computed as cluster means;
noise points (DBSCAN label -1) are reassigned to their nearest cluster.

NOTE: DBSCAN's actual cluster count depends on `eps` and `min_samples`
and is NOT enforced to equal `n_partitions`. After fitting, `n_partitions`
is updated to the actual number of clusters discovered. This means
MFDBSCANShift is **not safe** to plug into pipelines that require an
exact partition count (e.g. OODPipeline's splitter) — it is intended for
use in OOD detectors (via MFKMeansOODDetector) where only `centroids`,
`project_samples`, `get_partition_labels` and `get_meta_info` are needed.
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from oodt.shifts.concept.mf_kmeans import MFKMeansShift


class MFDBSCANShift(MFKMeansShift):
    """DBSCAN clustering in metafeature space."""

    def __init__(
        self,
        mf_name: Union[str, List[str]],
        n_partitions: int = 2,
        n_bins: int = 5,
        percent: float = 0.1,
        summary: Optional[List[str]] = None,
        random_state: Optional[int] = None,
        verbose: bool = True,
        mode: str = "matrix",
        metric: str = "frobenius",
        include_sample_features: bool = True,
        # DBSCAN-specific
        eps: Optional[float] = None,        # None → auto via kNN heuristic
        min_samples: int = 5,
    ):
        super().__init__(
            mf_name=mf_name,
            n_partitions=n_partitions,
            n_bins=n_bins,
            percent=percent,
            summary=summary,
            random_state=random_state,
            verbose=verbose,
            mode=mode,
            metric=metric,
            include_sample_features=include_sample_features,
        )
        self.eps = eps
        self.min_samples = min_samples

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    @staticmethod
    def _flatten_mf_space(mf_space) -> np.ndarray:
        return np.array([np.asarray(m, dtype=float).reshape(-1) for m in mf_space])

    def _auto_eps(self, X_vec: np.ndarray) -> float:
        """Heuristic: median of k-th nearest-neighbour distances."""
        k = max(1, min(self.min_samples, len(X_vec) - 1))
        if k < 1 or len(X_vec) < 2:
            return 0.5
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X_vec)
        distances, _ = nn.kneighbors(X_vec)
        return float(np.median(distances[:, -1]))

    # --------------------------------------------------
    # Clustering
    # --------------------------------------------------

    def get_partition_labels(self, X: pd.DataFrame, y: pd.Series):
        mf_space = self._build_mf_space(X, y)
        X_vec = self._flatten_mf_space(mf_space)

        eps = self.eps if self.eps is not None else self._auto_eps(X_vec)
        if self.verbose:
            print(f"  [MFDBSCANShift] eps={eps:.4g}, min_samples={self.min_samples}")

        db = DBSCAN(eps=eps, min_samples=self.min_samples).fit(X_vec)
        raw_labels = db.labels_

        # find real clusters (exclude noise)
        unique_clusters = sorted([c for c in set(raw_labels) if c != -1])

        if not unique_clusters:
            # everything was noise — fall back to a single cluster centred at
            # the mean of the MF space
            if self.verbose:
                print("  [MFDBSCANShift] all points are noise; using a single cluster")
            raw_labels = np.zeros_like(raw_labels)
            unique_clusters = [0]

        cluster_map = {c: i for i, c in enumerate(unique_clusters)}

        # compute centroids in the original MF shape (matrix or vector)
        if self.mode == "matrix":
            centroids = []
            for c in unique_clusters:
                members = [mf_space[i] for i in range(len(mf_space)) if raw_labels[i] == c]
                centroids.append(np.mean(members, axis=0))
        else:
            mf_arr = np.asarray(mf_space)
            centroids = np.vstack([
                np.mean(mf_arr[raw_labels == c], axis=0) for c in unique_clusters
            ])

        # reassign noise to nearest cluster centroid (in flattened space)
        centroid_vecs = np.array([np.asarray(c, dtype=float).reshape(-1) for c in centroids])
        labels = np.empty(len(X_vec), dtype=int)
        for i, lbl in enumerate(raw_labels):
            if lbl == -1:
                dists = np.linalg.norm(centroid_vecs - X_vec[i], axis=1)
                labels[i] = int(np.argmin(dists))
            else:
                labels[i] = cluster_map[lbl]

        self.centroids = centroids
        # actual cluster count may differ from requested n_partitions
        self.n_partitions = len(unique_clusters)

        self.meta_["final_labels"] = labels
        self.meta_["final_centroids"] = self.centroids
        self.meta_["dbscan_eps"] = float(eps)
        self.meta_["dbscan_min_samples"] = self.min_samples
        self.meta_["dbscan_n_noise"] = int(np.sum(raw_labels == -1))
        self.meta_["dbscan_n_clusters_found"] = len(unique_clusters)

        return pd.Series(labels, index=X.index)
