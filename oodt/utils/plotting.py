from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import IncrementalPCA


# ============================================================
# Internal helpers
# ============================================================

def _incremental_pca_transform(matrices, pca: IncrementalPCA, batch_size: int) -> np.ndarray:
    """Transform a list of matrices through a fitted IncrementalPCA in batches."""
    n = len(matrices)
    result_2d = np.empty((n, 2), dtype=np.float64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = np.array([m.flatten() for m in matrices[start:end]], dtype=np.float64)
        result_2d[start:end] = pca.transform(batch)

    return result_2d


def _save_blank_plot(title: str, path):
    """Save an empty placeholder plot when PCA cannot be fitted."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax in axes:
        ax.text(
            0.5, 0.5,
            "MF space degenerate\n(all matrices identical)",
            ha="center", va="center", transform=ax.transAxes, fontsize=12,
        )
        ax.set_axis_off()
    plt.suptitle(title)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()
    plt.close(fig)


# ============================================================
# Plot MF Space (Train clusters + Test ID/OOD)
# ============================================================

def plot_mf_space(
    result,
    reduction: str = "pca",
    title: str = "Metafeature Space",
    path: str = None,
    batch_size: int = 512,
    jitter_scale: float = 1e-10,
):
    """
    Plot the metafeature space using IncrementalPCA to keep memory usage low.

    Parameters
    ----------
    result        : pipeline result object
    reduction     : ignored (always PCA), kept for API compatibility
    title         : plot title
    path          : save path; if None the plot is shown interactively
    batch_size    : samples flattened at once — lower = less peak RAM, more CPU.
                    Must be >= 2 (n_components).
    jitter_scale  : std of Gaussian noise added before SVD to break degeneracy.
                    Negligible effect on PCA geometry.
    """
    meta = result.metadata

    mf_train = meta["shift_meta"]["mf_space_train"]
    mf_test  = meta["test_projection"]

    n_train = len(mf_train)

    # IncrementalPCA requires batch_size >= n_components (2)
    effective_batch = max(batch_size, 2)

    rng = np.random.default_rng(42)

    # ----------------------------
    # Fit IncrementalPCA on train
    # ----------------------------
    pca = IncrementalPCA(n_components=2)
    batches_fitted = 0

    for start in range(0, n_train, effective_batch):
        end   = min(start + effective_batch, n_train)
        batch = np.array([m.flatten() for m in mf_train[start:end]], dtype=np.float64)

        # Skip batches where every row is identical — SVD diverges on them
        if np.allclose(batch, batch[0]):
            continue

        # Tiny jitter breaks numerical near-degeneracy without shifting the geometry
        batch += rng.normal(0.0, jitter_scale, batch.shape)

        try:
            pca.partial_fit(batch)
            batches_fitted += 1
        except np.linalg.LinAlgError as e:
            print(f"  [WARNING] SVD did not converge for batch [{start}:{end}], skipping. ({e})")

    if batches_fitted == 0:
        # Every batch was degenerate — nothing to plot
        print("  [WARNING] PCA fit failed entirely — MF space is degenerate. Saving blank plot.")
        _save_blank_plot(title, path)
        return

    # ----------------------------
    # Transform train and test
    # ----------------------------
    train_2d = _incremental_pca_transform(mf_train, pca, effective_batch)
    test_2d  = _incremental_pca_transform(mf_test,  pca, effective_batch)

    # ----------------------------
    # Partition labels / cluster ids
    # ----------------------------
    train_labels = meta.get("train_partitions", meta.get("partitions", {}))

    train_cluster_ids = np.full(n_train, -1)
    train_index = result.X_train.index.to_numpy()

    for pid, idx_list in train_labels.items():
        idx_list        = np.array(idx_list)
        valid_global    = idx_list[np.isin(idx_list, train_index)]
        local_positions = np.where(np.isin(train_index, valid_global))[0]
        train_cluster_ids[local_positions] = pid

    # ----------------------------
    # OOD mask for test samples
    # ----------------------------
    ood_mask = result.X_test.index.isin(meta.get("test_ood_indices", []))

    # ----------------------------
    # Plot two horizontal subplots
    # ----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Left: train clusters colored ----
    scatter = axes[0].scatter(
        train_2d[:, 0],
        train_2d[:, 1],
        c=train_cluster_ids,
        cmap="tab20",
        alpha=0.7,
        s=10,
    )
    axes[0].set_title("Train clusters")
    axes[0].grid(True)
    legend1 = axes[0].legend(*scatter.legend_elements(), title="Cluster")
    axes[0].add_artist(legend1)

    # ---- Right: train gray, test ID orange, test OOD blue ----
    axes[1].scatter(train_2d[:, 0], train_2d[:, 1],
                    color="lightgray", alpha=0.6, s=10, label="Train")
    axes[1].scatter(test_2d[~ood_mask, 0], test_2d[~ood_mask, 1],
                    color="orange", marker="x", s=20, label="Test ID")
    axes[1].scatter(test_2d[ood_mask, 0], test_2d[ood_mask, 1],
                    color="blue", marker="x", s=20, label="Test OOD")
    axes[1].set_title("Train gray + Test ID/OOD")
    axes[1].grid(True)
    axes[1].legend()

    plt.suptitle(title)
    plt.tight_layout()

    if path is not None:
        plt.savefig(path)
    else:
        plt.show()

    plt.close(fig)  # release figure memory immediately


# ============================================================
# Distance Stats for ID vs OOD Test Samples
# ============================================================

def compute_distance_stats(result) -> Dict[str, Any]:
    """
    Compute distance statistics from test points to nearest centroid.

    Requires:
        metadata["test_centroid_distance"]
        metadata["test_ood_indices"]

    Returns
    -------
    Dict[str, Any]
        Summary statistics for ID vs OOD test samples
    """
    meta = result.metadata

    if "test_centroid_distance" not in meta:
        raise ValueError("No test_centroid_distance in metadata")

    dist = meta["test_centroid_distance"]
    ood_mask = result.X_test.index.isin(meta["test_ood_indices"])

    dist_id  = dist[~ood_mask]
    dist_ood = dist[ood_mask]

    def summarize(x):
        return {
            "count":  len(x),
            "mean":   float(np.mean(x)),
            "std":    float(np.std(x)),
            "min":    float(np.min(x)),
            "max":    float(np.max(x)),
            "median": float(np.median(x)),
        }

    return {
        "ID_test":  summarize(dist_id),
        "OOD_test": summarize(dist_ood),
    }


# ============================================================
# Pretty print stats
# ============================================================

def print_distance_stats(stats: Dict[str, Any]):
    for group, values in stats.items():
        print("=" * 50)
        print(group)
        print("=" * 50)
        for k, v in values.items():
            print(f"{k:>8}: {v}")
        print()