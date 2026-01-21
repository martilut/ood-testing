from __future__ import annotations
from typing import Optional, List, Union
import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm
from sklearn.preprocessing import normalize
from oodt.shifts.base import BaseShiftStrategy
from pymfe.mfe import MFE


class MFKMeansShift(BaseShiftStrategy):
    """
    Shift strategy using local metafeatures and k-means clustering to create partitions.
    """

    def __init__(
        self,
        mf_name: Union[str, List[str]],
        n_partitions: int = 2,
        percent: float = 0.1,
        summary: Optional[List[str]] = None,
        random_state: Optional[int] = None,
        max_iter: int = 50,
        patience: int = 5,
        verbose: bool = True,
    ):
        super().__init__(
            name="MFKMeansShift",
            n_partitions=n_partitions,
            random_state=random_state
        )
        self.mf_name = mf_name
        self.percent = percent
        self.summary = summary
        self.max_iter = max_iter
        self.patience = patience
        self.verbose = verbose
        self.centroids: Optional[np.ndarray] = None

    def get_partition_labels(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        X_np = X.to_numpy()
        y_np = y.to_numpy()

        # --- build MF space ---
        mf_space = build_metafeature_space(X_np, y_np, self.mf_name, percent=self.percent, summary=self.summary)

        # --- initialize centroids ---
        rng = np.random.default_rng(self.random_state)
        self.centroids = mf_space[rng.choice(len(mf_space), self.n_partitions, replace=False)]

        # --- k-means clustering ---
        labels = np.zeros(len(X), dtype=int)
        prev_labels = None
        stable_iters = 0

        for it in range(self.max_iter):
            labels = assign_clusters(mf_space, self.centroids)
            new_centroids = update_centroids(mf_space, labels, self.n_partitions)
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

        return pd.Series(labels, index=X.index)


def get_neighborhood_indices(x: np.ndarray, point: np.ndarray, percent: float, exclude_self: bool = True) -> np.ndarray:
    n = len(x)
    k_neighbours = max(1, int(n * percent))
    distances = np.linalg.norm(x - point, axis=1)
    if exclude_self:
        self_idx = np.argmin(distances)
        distances[self_idx] = np.inf
    sorted_indices = np.argsort(distances)
    return sorted_indices[:k_neighbours]

def _flatten_feature_values(ft_values):
    parts = []
    for v in ft_values:
        if v is None:
            parts.append(np.array([0.0]))
        else:
            if np.isscalar(v):
                parts.append(np.array([v]))
            else:
                arr = np.asarray(v, dtype=float)
                arr = np.where(np.isnan(arr), 0.0, arr)
                parts.append(arr)
    if not parts:
        return None
    flat = np.concatenate(parts, axis=0)
    flat = np.where(np.isfinite(flat), flat, 0.0)
    return flat

def compute_metafeature(x_sub: np.ndarray, y_sub: np.ndarray, mf_name: Union[str, List[str]], summary: Optional[List[str]] = None) -> np.ndarray:
    if isinstance(mf_name, str):
        features = [mf_name]
    else:
        features = list(mf_name)

    mfe = MFE(features=features, summary=summary)
    try:
        mfe.fit(x_sub, y_sub)
        _, ft_values = mfe.extract()
    except Exception:
        ft_values = None

    if not ft_values:
        return None
    return _flatten_feature_values(ft_values)

def build_metafeature_space(x: np.ndarray, y: np.ndarray, mf_name: Union[str, List[str]], percent: float = 0.1, summary: Optional[List[str]] = None) -> np.ndarray:
    mf_vectors = []
    placeholder_indices = []

    for idx in tqdm(range(len(x)), desc="Building MF space"):
        neighbors = get_neighborhood_indices(x, x[idx], percent)
        mf_vec = compute_metafeature(x[neighbors], y[neighbors], mf_name, summary)
        if mf_vec is None:
            mf_vectors.append(None)
            placeholder_indices.append(idx)
        else:
            mf_vectors.append(mf_vec)

    if all(v is None for v in mf_vectors):
        raise RuntimeError("All metafeature computations failed.")

    first_nonnull = next(v for v in mf_vectors if v is not None)
    target_dim = first_nonnull.shape[0]

    processed = []
    for v in mf_vectors:
        if v is None:
            processed.append(np.zeros(target_dim))
        else:
            if v.shape[0] < target_dim:
                padded = np.zeros(target_dim)
                padded[:v.shape[0]] = v
                processed.append(padded)
            elif v.shape[0] > target_dim:
                processed.append(v[:target_dim])
            else:
                processed.append(v)
    return np.vstack(processed)

def assign_clusters(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    labels = []
    for vec in x:
        dists = [np.linalg.norm(vec - c) for c in centroids]
        labels.append(np.argmin(dists))
    return np.array(labels)

def update_centroids(x: np.ndarray, labels: np.ndarray, k_clusters: int) -> np.ndarray:
    centroids = []
    for i in range(k_clusters):
        if np.any(labels == i):
            centroids.append(np.mean(x[labels == i], axis=0))
        else:
            centroids.append(x[np.random.randint(0, len(x))])
    return np.vstack(centroids)
