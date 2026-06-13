"""
Plain KMeans on source features (no metafeatures).

Serves as the "no-MF" baseline in the MF-KMeans comparison. Exposes the
same interface as the MF shifts (`centroids`, `project_samples`,
`get_meta_info`) so it plugs into MFKMeansOODDetector and the existing
OODPipeline without changes.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from oodt.shifts.base import BaseShiftStrategy


class RawKMeansShift(BaseShiftStrategy):
    """KMeans clustering directly on raw source features."""

    def __init__(
        self,
        n_partitions: int = 2,
        random_state: Optional[int] = None,
        n_init: int = 10,
        max_iter: int = 300,
    ):
        super().__init__(
            name="RawKMeansShift",
            n_partitions=n_partitions,
            random_state=random_state,
        )
        self.n_init = n_init
        self.max_iter = max_iter

        self.centroids = None
        self.meta_: dict = {}

    # ----------------------------------------------------
    # API shared with MF-shifts
    # ----------------------------------------------------

    def get_meta_info(self) -> dict:
        return self.meta_

    def get_partition_labels(self, X: pd.DataFrame, y: pd.Series = None) -> pd.Series:
        X_np = X.to_numpy()

        km = KMeans(
            n_clusters=self.n_partitions,
            random_state=self.random_state,
            n_init=self.n_init,
            max_iter=self.max_iter,
        )
        labels = km.fit_predict(X_np)
        self.centroids = km.cluster_centers_

        # populated for downstream plotting / pipeline metadata
        self.meta_ = {
            "mode": "raw",
            "mf_name": None,
            "mf_space_train": X_np,        # raw features stand in for the MF space
            "final_labels": labels,
            "final_centroids": self.centroids,
        }

        return pd.Series(labels, index=X.index)

    def project_samples(self, X_new: pd.DataFrame):
        """Identity projection — raw features as-is."""
        if hasattr(X_new, "to_numpy"):
            return X_new.to_numpy()
        return np.asarray(X_new)
