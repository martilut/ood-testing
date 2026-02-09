import warnings
from typing import Optional, List, Union

import numpy as np
from pymfe.mfe import MFE
from sklearn.preprocessing import normalize
from tqdm import tqdm

warnings.filterwarnings("ignore")


def compute_metafeature(
    x_sub: np.ndarray,
    y_sub: np.ndarray,
    mf_name: Union[str, List[str]],
    summary: Optional[List[str]] = None,
) -> Optional[np.ndarray]:

    mfe = MFE(features=mf_name, summary=summary)

    try:
        mfe.fit(x_sub, y_sub)
        _, ft_values = mfe.extract()
    except Exception:
        return None

    if not ft_values or ft_values[0] is None:
        return None

    values = ft_values[0]

    if np.isscalar(values):
        values = [values]

    cleaned = [
        0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else v
        for v in values
    ]

    return np.array(cleaned)


def get_neighborhood_indices(x, point, percent, exclude_self=True):

    n = len(x)
    k = max(1, int(n * percent))

    distances = np.linalg.norm(x - point, axis=1)

    if exclude_self:
        distances[np.argmin(distances)] = np.inf

    return np.argsort(distances)[:k]


def build_metafeature_space(x, y, mf_name, percent, summary):

    vectors = []

    for idx in tqdm(range(len(x)), desc="Building MF space (vector)"):

        neigh_idx = get_neighborhood_indices(x, x[idx], percent)

        mf_vec = compute_metafeature(
            x[neigh_idx],
            y[neigh_idx],
            mf_name,
            summary
        )

        if mf_vec is None:
            mf_vec = np.zeros(1)

        vectors.append(mf_vec)

    return np.vstack(vectors)


def assign_clusters(x, centroids):

    labels = []
    for vec in x:
        dists = [np.linalg.norm(vec - c) for c in centroids]
        labels.append(np.argmin(dists))

    return np.array(labels)


def update_centroids(x, labels, k):

    centroids = []
    for i in range(k):

        if np.any(labels == i):
            centroids.append(np.mean(x[labels == i], axis=0))
        else:
            centroids.append(x[np.random.randint(0, len(x))])

    return np.vstack(centroids)


def stratify_features_with_edges(x: np.ndarray, n_bins: int):

    bins_matrix = []
    partitions = {}
    bin_edges = {}

    for f_idx in range(x.shape[1]):

        col = x[:, f_idx]

        edges = np.histogram_bin_edges(col, bins=n_bins)
        bin_edges[f_idx] = edges

        labels = np.digitize(col, edges) - 1
        labels = np.clip(labels, 0, n_bins - 1)

        bins_matrix.append(labels)

        partitions[f_idx] = {
            b: np.where(labels == b)[0]
            for b in range(n_bins)
            if len(np.where(labels == b)[0]) > 0
        }

    return np.vstack(bins_matrix).T, partitions, bin_edges


def build_metafeature_space_matrix(x, y, mf_name, n_bins, summary):
    """
    Build metafeature matrices for training samples.
    Ensures all MF vectors have consistent length and handles empty partitions.

    Returns:
        sample_matrices: list of np.ndarray (per sample)
        bins: bin indices per feature
        partitions: dict of partition MF vectors
        bin_edges: edges used for digitization
        expected_len: length of each MF vector
    """

    # Step 1: stratify features into bins
    bins, partitions, bin_edges = stratify_features_with_edges(x, n_bins)

    # Step 2: compute MF vector per feature-bin partition
    for f_idx in partitions:
        for b in partitions[f_idx]:
            idxs = partitions[f_idx][b]
            mf_vec = compute_metafeature(x[idxs], y[idxs], mf_name, summary)

            if mf_vec is None or len(mf_vec) == 0:
                mf_vec = np.zeros(1)  # placeholder; will fix length later
            partitions[f_idx][b] = mf_vec

    # Step 3: determine expected MF length
    all_vecs = [v for f in partitions for v in partitions[f].values()]
    expected_len = max(len(v) for v in all_vecs)

    # Step 4: pad all partition vectors to expected_len
    for f_idx in partitions:
        for b in partitions[f_idx]:
            vec = partitions[f_idx][b]
            if len(vec) != expected_len:
                padded = np.zeros(expected_len)
                padded[:len(vec)] = vec
                partitions[f_idx][b] = padded

    # Step 5: build MF matrix per sample
    sample_matrices = []

    for i in tqdm(range(x.shape[0]), desc="Building MF space (matrix)"):
        sample_bins = bins[i]
        matrix = []

        for f_idx, bin_id in enumerate(sample_bins):
            mf_vec = partitions[f_idx].get(bin_id, np.zeros(expected_len))
            matrix.append(mf_vec)

        sample_matrices.append(np.vstack(matrix))

    return sample_matrices, bins, partitions, bin_edges, expected_len


def compute_matrix_distance(a, b, method):

    if method == "frobenius":
        return np.linalg.norm(a - b, ord="fro")

    elif method == "cosine":
        a_flat = a.flatten().reshape(1, -1)
        b_flat = b.flatten().reshape(1, -1)
        return 1 - np.dot(normalize(a_flat), normalize(b_flat).T).item()

    else:
        raise ValueError(f"Unknown metric: {method}")


def assign_clusters_matrix(x, centroids, method):

    labels = []
    for mat in x:
        dists = [compute_matrix_distance(mat, c, method) for c in centroids]
        labels.append(np.argmin(dists))

    return np.array(labels)


def update_centroids_matrix(x, labels, k):

    centroids = []

    for i in range(k):

        cluster = [x[j] for j in range(len(x)) if labels[j] == i]

        if cluster:
            centroids.append(np.mean(cluster, axis=0))
        else:
            centroids.append(x[np.random.randint(0, len(x))])

    return centroids
