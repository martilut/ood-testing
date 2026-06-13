import numpy as np
import pandas as pd

from oodt.detection.base import BaseOODDetector
from oodt.shifts.concept.mf_kmeans import MFKMeansShift


class MFKMeansOODDetector(BaseOODDetector):
    """
    OOD detector based on MF-KMeans clustering in metafeature space.

    Score = distance to nearest centroid in MF space.
    """

    def __init__(self, shift: MFKMeansShift):
        super().__init__()
        self.shift = shift

        self._centroids = None

    # --------------------------------------------------
    # FIT
    # --------------------------------------------------

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series = None):
        labels = self.shift.get_partition_labels(X_train, y_train)

        # build MF space from shift (IMPORTANT: already computed in shift)
        mf_space = self._get_mf_space()

        self._centroids = self.shift.centroids

        self.is_fitted_ = True
        return self

    # --------------------------------------------------
    # SCORE
    # --------------------------------------------------

    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()

        mf_space = self._project(X)

        centroids = self._centroids

        # convert centroids + mf_space to 2D vectors safely
        X_vec = np.array([self._flatten(x) for x in mf_space])
        C_vec = np.array([self._flatten(c) for c in centroids])

        # distance to nearest centroid
        scores = []
        for x in X_vec:
            dists = np.linalg.norm(C_vec - x, axis=1)
            scores.append(np.min(dists))

        return np.array(scores)

    # --------------------------------------------------
    # PREDICT
    # --------------------------------------------------

    def predict(self, X: pd.DataFrame, threshold=None):
        scores = self.score_samples(X)
        thr = threshold if threshold is not None else self._get_threshold()
        return (scores >= thr).astype(int)

    # --------------------------------------------------
    # PARAMS
    # --------------------------------------------------

    def get_params(self):
        return {
            "shift": self.shift.get_meta_info()
        }

    # --------------------------------------------------
    # INTERNAL HELPERS
    # --------------------------------------------------

    def _project(self, X):
        return self.shift.project_samples(X)

    def _get_mf_space(self):
        meta = self.shift.get_meta_info()
        return meta.get("mf_space_train")

    def _flatten(self, x):
        """
        Robust flattening for:
        - vector MF: (D,)
        - matrix MF: (P, B, B)
        """
        x = np.asarray(x)
        return x.reshape(-1)
