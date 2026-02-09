from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


# ============================================================
# Plot MF Space (Train clusters + Test ID/OOD)
# ============================================================

def plot_mf_space(
    result,
    reduction: str = "pca",
    title: str = "Metafeature Space",
    path: str = None
):
    meta = result.metadata

    train_mf = np.array([m.flatten() for m in meta["shift_meta"]["mf_space_train"]])
    test_mf = np.array([m.flatten() for m in meta["test_projection"]])

    # Partition labels dict
    train_labels = meta.get("train_partitions", meta.get("partitions", {}))

    # ----------------------------
    # Fix index mismatch properly
    # ----------------------------
    train_cluster_ids = np.full(len(result.X_train), -1)
    train_index = result.X_train.index.to_numpy()

    for pid, idx_list in train_labels.items():
        idx_list = np.array(idx_list)
        valid_global = idx_list[np.isin(idx_list, train_index)]
        local_positions = np.where(np.isin(train_index, valid_global))[0]
        train_cluster_ids[local_positions] = pid

    # ----------------------------
    # OOD mask for test samples
    # ----------------------------
    ood_mask = result.X_test.index.isin(meta.get("test_ood_indices", []))

    pca = PCA(n_components=2)
    pca.fit(train_mf)
    train_2d = pca.transform(train_mf)
    test_2d = pca.transform(test_mf)

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
    axes[1].scatter(
        train_2d[:, 0],
        train_2d[:, 1],
        color="lightgray",
        alpha=0.6,
        s=10,
        label="Train"
    )
    axes[1].scatter(
        test_2d[~ood_mask, 0],
        test_2d[~ood_mask, 1],
        color="orange",
        marker="x",
        s=20,
        label="Test ID"
    )
    axes[1].scatter(
        test_2d[ood_mask, 0],
        test_2d[ood_mask, 1],
        color="blue",
        marker="x",
        s=20,
        label="Test OOD"
    )
    axes[1].set_title("Train gray + Test ID/OOD")
    axes[1].grid(True)
    axes[1].legend()

    plt.suptitle(title)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()



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

    # OOD mask
    ood_mask = result.X_test.index.isin(meta["test_ood_indices"])

    dist_id = dist[~ood_mask]
    dist_ood = dist[ood_mask]

    def summarize(x):
        return {
            "count": len(x),
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "median": float(np.median(x)),
        }

    return {
        "ID_test": summarize(dist_id),
        "OOD_test": summarize(dist_ood),
    }


# ============================================================
# Pretty print stats
# ============================================================

def print_distance_stats(stats: Dict[str, Any]):
    """
    Print distance stats nicely.
    """

    for group, values in stats.items():
        print("=" * 50)
        print(group)
        print("=" * 50)

        for k, v in values.items():
            print(f"{k:>8}: {v}")
        print()
