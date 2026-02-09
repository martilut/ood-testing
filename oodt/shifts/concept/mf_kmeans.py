from typing import Optional, List, Union

import numpy as np
import pandas as pd

from oodt.shifts.base import BaseShiftStrategy
from oodt.utils.kmeans import build_metafeature_space, assign_clusters_matrix, update_centroids_matrix, assign_clusters, \
    update_centroids, stratify_features_with_edges, build_metafeature_space_matrix


class MFKMeansShift(BaseShiftStrategy):
    """
    Metafeature KMeans Shift Strategy

    Supports:
      - vector mode (local neighborhoods)
      - matrix mode (feature-bin stratification)

    Also supports:
      - projection of unseen test samples into MF space
        via project_samples()
    """

    def __init__(
        self,
        mf_name: Union[str, List[str]],
        n_partitions: int = 2,
        n_bins: int = 5,
        percent: float = 0.1,
        summary: Optional[List[str]] = None,
        random_state: Optional[int] = None,
        max_iter: int = 50,
        patience: int = 5,
        verbose: bool = True,
        mode: str = "vector",      # "vector" or "matrix"
        metric: str = "frobenius", # matrix distance metric
    ):
        super().__init__(
            name="MFKMeansShift",
            n_partitions=n_partitions,
            random_state=random_state,
        )

        self.mf_name = mf_name
        self.n_partitions = n_partitions
        self.n_bins = n_bins
        self.percent = percent
        self.summary = summary

        self.mode = mode
        self.metric = metric

        self.max_iter = max_iter
        self.patience = patience
        self.verbose = verbose

        # learned centroids
        self.centroids = None
        self.expected_len = None

        # training reference data (needed for projection)
        self.X_ref_ = None
        self.y_ref_ = None

        # matrix-mode storage
        self.partition_mf_ = None
        self.bin_edges_ = None

        # metadata storage
        self.meta_: dict = {}

    # ============================================================
    # Metadata accessor
    # ============================================================

    def get_meta_info(self) -> dict:
        return self.meta_

    # ============================================================
    # Fit clustering + return partition labels
    # ============================================================

    def get_partition_labels(self, X: pd.DataFrame, y: pd.Series):

        X_np = X.to_numpy()
        y_np = y.to_numpy()

        # store training reference for later projection
        self.X_ref_ = X_np
        self.y_ref_ = y_np

        rng = np.random.default_rng(self.random_state)

        # store config
        self.meta_ = {
            "mode": self.mode,
            "mf_name": self.mf_name,
            "summary": self.summary,
            "percent": self.percent,
            "n_bins": self.n_bins,
            "metric": self.metric,
        }

        # =====================================================
        # Build MF space (train)
        # =====================================================

        if self.mode == "vector":

            mf_space = build_metafeature_space(
                X_np, y_np,
                mf_name=self.mf_name,
                percent=self.percent,
                summary=self.summary,
            )

            init_ids = rng.choice(len(mf_space), self.n_partitions, replace=False)
            self.centroids = mf_space[init_ids]

            self.meta_["mf_space_train"] = mf_space

        elif self.mode == "matrix":

            mf_space, bins, partitions, bin_edges, expected_len = build_metafeature_space_matrix(
                X_np, y_np,
                mf_name=self.mf_name,
                n_bins=self.n_bins,
                summary=self.summary,
            )
            self.expected_len = expected_len

            init_ids = rng.choice(len(mf_space), self.n_partitions, replace=False)
            self.centroids = [mf_space[i] for i in init_ids]

            # store bin structure for test projection
            self.partition_mf_ = partitions
            self.bin_edges_ = bin_edges

            self.meta_["mf_space_train"] = mf_space
            self.meta_["bins_train"] = bins

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # =====================================================
        # Run KMeans iterations
        # =====================================================

        labels = np.zeros(len(X_np), dtype=int)

        prev_labels = None
        stable_iters = 0

        for it in range(self.max_iter):

            if self.mode == "matrix":
                labels = assign_clusters_matrix(mf_space, self.centroids, self.metric)
                new_centroids = update_centroids_matrix(mf_space, labels, self.n_partitions)

            else:
                labels = assign_clusters(mf_space, self.centroids)
                new_centroids = update_centroids(mf_space, labels, self.n_partitions)

            # convergence check
            if prev_labels is not None and np.all(prev_labels == labels):
                stable_iters += 1
                if stable_iters >= self.patience:
                    if self.verbose:
                        print(f"Converged after {it+1} iterations")
                    break
            else:
                stable_iters = 0

            prev_labels = labels.copy()
            self.centroids = new_centroids

        # store results
        self.meta_["final_labels"] = labels
        self.meta_["final_centroids"] = self.centroids

        return pd.Series(labels, index=X.index)

    # ============================================================
    # Project unseen samples into MF space
    # ============================================================

    def project_samples(self, X_new: pd.DataFrame) -> list[np.ndarray]:
        if self.mode == "vector":
            # vector mode stays the same
            return build_metafeature_space(
                X_new.to_numpy(),
                np.zeros(len(X_new)),  # dummy target
                self.mf_name,
                percent=self.percent,
                summary=self.summary,
            )

        elif self.mode == "matrix":
            projected = []

            # compute bin indices for new samples
            bins_array = stratify_features_with_edges(X_new.to_numpy(), self.n_bins)[0]

            for i, sample in enumerate(X_new.to_numpy()):
                sample_matrix = []

                for f_idx, bin_id in enumerate(bins_array[i]):
                    partition_mf_ = self.partition_mf_

                    # safe lookup: use nearest available bin if bin_id missing
                    if bin_id not in partition_mf_[f_idx]:
                        available_bins = np.array(list(partition_mf_[f_idx].keys()))
                        if len(available_bins) > 0:
                            bin_id = available_bins[np.argmin(np.abs(available_bins - bin_id))]
                        else:
                            # fallback: no bins at all, use zeros
                            mf_vec = np.zeros(self.expected_len)
                            sample_matrix.append(mf_vec)
                            continue

                    mf_vec = partition_mf_[f_idx][bin_id]

                    # ensure correct length
                    if len(mf_vec) != self.expected_len:
                        padded = np.zeros(self.expected_len)
                        padded[:len(mf_vec)] = mf_vec
                        mf_vec = padded

                    sample_matrix.append(mf_vec)

                projected.append(np.vstack(sample_matrix))

            return projected

        else:
            raise ValueError(f"Unknown mode: {self.mode}")
