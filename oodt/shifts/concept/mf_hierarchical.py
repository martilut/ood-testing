"""
MF-Hierarchical shift strategy.

Builds the same metafeature (MF) space as MFKMeansShift, but clusters it
with sklearn's AgglomerativeClustering. Produces exactly `n_partitions`
clusters; centroids are computed as cluster means.
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from oodt.shifts.concept.mf_kmeans import MFKMeansShift


class MFHierarchicalShift(MFKMeansShift):
    """Agglomerative (hierarchical) clustering in metafeature space."""

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
        # Hierarchical-specific
        linkage: str = "ward",          # "ward", "average", "complete", "single"
        agg_metric: str = "euclidean",  # only honoured when linkage != "ward"
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
        self.linkage = linkage
        self.agg_metric = agg_metric

    # --------------------------------------------------
    # Clustering
    # --------------------------------------------------

    def get_partition_labels(self, X: pd.DataFrame, y: pd.Series):
        mf_space = self._build_mf_space(X, y)
        X_vec = np.array([np.asarray(m, dtype=float).reshape(-1) for m in mf_space])

        # safety: number of samples must exceed n_clusters
        n_clusters = min(self.n_partitions, max(1, len(X_vec) - 1))

        kwargs = dict(n_clusters=n_clusters, linkage=self.linkage)
        if self.linkage != "ward":
            kwargs["metric"] = self.agg_metric

        if self.verbose:
            print(f"  [MFHierarchicalShift] linkage={self.linkage}, k={n_clusters}")

        agg = AgglomerativeClustering(**kwargs).fit(X_vec)
        labels = agg.labels_

        # compute centroids in original shape
        if self.mode == "matrix":
            centroids = []
            for c in range(n_clusters):
                members = [mf_space[i] for i in range(len(mf_space)) if labels[i] == c]
                if members:
                    centroids.append(np.mean(members, axis=0))
                else:
                    centroids.append(mf_space[0])
        else:
            mf_arr = np.asarray(mf_space)
            centroids = []
            for c in range(n_clusters):
                mask = labels == c
                if mask.any():
                    centroids.append(np.mean(mf_arr[mask], axis=0))
                else:
                    centroids.append(mf_arr[0])
            centroids = np.vstack(centroids)

        self.centroids = centroids
        self.n_partitions = n_clusters

        self.meta_["final_labels"] = labels
        self.meta_["final_centroids"] = self.centroids
        self.meta_["hierarchical_linkage"] = self.linkage
        self.meta_["hierarchical_metric"] = self.agg_metric

        return pd.Series(labels, index=X.index)
