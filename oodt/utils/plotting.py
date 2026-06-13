from typing import Dict, Any, Optional

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


def _fit_incremental_pca(matrices, batch_size: int, jitter_scale: float = 1e-10):
    """Fit an IncrementalPCA(n_components=2) on a list of matrices. Returns (pca, n_batches_fitted)."""
    effective_batch = max(batch_size, 2)
    rng = np.random.default_rng(42)

    pca = IncrementalPCA(n_components=2)
    batches_fitted = 0

    n = len(matrices)
    for start in range(0, n, effective_batch):
        end = min(start + effective_batch, n)
        batch = np.array([m.flatten() for m in matrices[start:end]], dtype=np.float64)

        if np.allclose(batch, batch[0]):
            continue

        batch += rng.normal(0.0, jitter_scale, batch.shape)

        try:
            pca.partial_fit(batch)
            batches_fitted += 1
        except np.linalg.LinAlgError as e:
            print(f"  [WARNING] SVD did not converge for batch [{start}:{end}], skipping. ({e})")

    return pca, batches_fitted, effective_batch


def _save_blank_plot(title: str, path, n_axes: int = 2):
    """Save an empty placeholder plot when PCA cannot be fitted."""
    fig, axes = plt.subplots(1, n_axes, figsize=(7 * n_axes, 6))
    if n_axes == 1:
        axes = [axes]
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
    """
    meta = result.metadata

    mf_train = meta["shift_meta"]["mf_space_train"]
    mf_test  = meta["test_projection"]

    n_train = len(mf_train)

    pca, batches_fitted, effective_batch = _fit_incremental_pca(mf_train, batch_size, jitter_scale)

    if batches_fitted == 0:
        print("  [WARNING] PCA fit failed entirely — MF space is degenerate. Saving blank plot.")
        _save_blank_plot(title, path)
        return

    train_2d = _incremental_pca_transform(mf_train, pca, effective_batch)
    test_2d  = _incremental_pca_transform(mf_test,  pca, effective_batch)

    train_labels = meta.get("train_partitions", meta.get("partitions", {}))

    train_cluster_ids = np.full(n_train, -1)
    train_index = result.X_train.index.to_numpy()

    for pid, idx_list in train_labels.items():
        idx_list        = np.array(idx_list)
        valid_global    = idx_list[np.isin(idx_list, train_index)]
        local_positions = np.where(np.isin(train_index, valid_global))[0]
        train_cluster_ids[local_positions] = pid

    ood_mask = result.X_test.index.isin(meta.get("test_ood_indices", []))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    scatter = axes[0].scatter(
        train_2d[:, 0], train_2d[:, 1],
        c=train_cluster_ids, cmap="tab20", alpha=0.7, s=10,
    )
    axes[0].set_title("Train clusters")
    axes[0].grid(True)
    legend1 = axes[0].legend(*scatter.legend_elements(), title="Cluster")
    axes[0].add_artist(legend1)

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

    plt.close(fig)


# ============================================================
# Plot OOD classification correctness in MF space
# ============================================================

def plot_ood_classification(
    mf_train,
    mf_test,
    ood_true: np.ndarray,
    ood_pred: np.ndarray,
    scores: Optional[np.ndarray] = None,
    title: str = "OOD classification in MF space",
    path: Optional[str] = None,
    batch_size: int = 512,
    jitter_scale: float = 1e-10,
):
    """
    Project train + test MF representations into 2D via IncrementalPCA fitted on train,
    then plot test points colored by classification correctness.

    Categories:
        - TN: ID samples correctly labelled ID  (green dot)
        - TP: OOD samples correctly labelled OOD (blue dot)
        - FP: ID samples wrongly labelled OOD   (red x)
        - FN: OOD samples wrongly labelled ID   (orange x)

    Three subplots:
        1. Train MF space + true ID/OOD test overlay (ground truth view)
        2. Train MF space + predicted ID/OOD test overlay (prediction view)
        3. Correctness map (TP/TN/FP/FN)

    Parameters
    ----------
    mf_train : list of np.ndarray
        Training MF representations (each can be vector or 2D matrix).
    mf_test : list of np.ndarray
        Test MF representations, same format as mf_train.
    ood_true : np.ndarray
        Boolean / 0-1 array, True/1 if the test sample is from the target (OOD).
    ood_pred : np.ndarray
        Boolean / 0-1 array, True/1 if the detector predicted OOD.
    scores : np.ndarray, optional
        Continuous OOD scores. If provided, an additional 4th subplot shows them.
    """
    ood_true = np.asarray(ood_true).astype(bool)
    ood_pred = np.asarray(ood_pred).astype(bool)

    if len(ood_true) != len(mf_test) or len(ood_pred) != len(mf_test):
        raise ValueError(
            f"Length mismatch: mf_test={len(mf_test)}, ood_true={len(ood_true)}, "
            f"ood_pred={len(ood_pred)}"
        )

    n_axes = 3 if scores is None else 4

    pca, batches_fitted, effective_batch = _fit_incremental_pca(mf_train, batch_size, jitter_scale)

    if batches_fitted == 0:
        print("  [WARNING] PCA fit failed entirely — MF space is degenerate. Saving blank plot.")
        _save_blank_plot(title, path, n_axes=n_axes)
        return

    train_2d = _incremental_pca_transform(mf_train, pca, effective_batch)
    test_2d = _incremental_pca_transform(mf_test, pca, effective_batch)

    # classification categories
    tp = ood_true & ood_pred           # OOD predicted OOD
    tn = (~ood_true) & (~ood_pred)     # ID predicted ID
    fp = (~ood_true) & ood_pred        # ID predicted OOD
    fn = ood_true & (~ood_pred)        # OOD predicted ID

    fig, axes = plt.subplots(1, n_axes, figsize=(7 * n_axes, 6))

    # ---- 1. Ground truth view ----
    ax = axes[0]
    ax.scatter(train_2d[:, 0], train_2d[:, 1],
               color="lightgray", alpha=0.5, s=10, label="Train")
    ax.scatter(test_2d[~ood_true, 0], test_2d[~ood_true, 1],
               color="tab:orange", marker="o", s=25, alpha=0.85,
               edgecolors="black", linewidths=0.3, label="Test ID (source)")
    ax.scatter(test_2d[ood_true, 0], test_2d[ood_true, 1],
               color="tab:blue", marker="o", s=25, alpha=0.85,
               edgecolors="black", linewidths=0.3, label="Test OOD (target)")
    ax.set_title("Ground truth (source vs target)")
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)

    # ---- 2. Prediction view ----
    ax = axes[1]
    ax.scatter(train_2d[:, 0], train_2d[:, 1],
               color="lightgray", alpha=0.5, s=10, label="Train")
    ax.scatter(test_2d[~ood_pred, 0], test_2d[~ood_pred, 1],
               color="tab:orange", marker="o", s=25, alpha=0.85,
               edgecolors="black", linewidths=0.3, label="Predicted ID")
    ax.scatter(test_2d[ood_pred, 0], test_2d[ood_pred, 1],
               color="tab:blue", marker="o", s=25, alpha=0.85,
               edgecolors="black", linewidths=0.3, label="Predicted OOD")
    ax.set_title("Predictions")
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)

    # ---- 3. Correctness map ----
    ax = axes[2]
    ax.scatter(train_2d[:, 0], train_2d[:, 1],
               color="lightgray", alpha=0.4, s=10, label="Train")
    if tn.any():
        ax.scatter(test_2d[tn, 0], test_2d[tn, 1],
                   color="tab:green", marker="o", s=28, alpha=0.85,
                   edgecolors="black", linewidths=0.3,
                   label=f"TN: ID→ID ({int(tn.sum())})")
    if tp.any():
        ax.scatter(test_2d[tp, 0], test_2d[tp, 1],
                   color="tab:blue", marker="o", s=28, alpha=0.85,
                   edgecolors="black", linewidths=0.3,
                   label=f"TP: OOD→OOD ({int(tp.sum())})")
    if fp.any():
        ax.scatter(test_2d[fp, 0], test_2d[fp, 1],
                   color="tab:red", marker="x", s=45, linewidths=1.5,
                   label=f"FP: ID→OOD ({int(fp.sum())})")
    if fn.any():
        ax.scatter(test_2d[fn, 0], test_2d[fn, 1],
                   color="tab:orange", marker="x", s=45, linewidths=1.5,
                   label=f"FN: OOD→ID ({int(fn.sum())})")

    acc = float((tp.sum() + tn.sum()) / max(1, len(ood_true)))
    ax.set_title(f"Correctness (acc={acc:.3f})")
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)

    # ---- 4. Score map (optional) ----
    if scores is not None:
        ax = axes[3]
        ax.scatter(train_2d[:, 0], train_2d[:, 1],
                   color="lightgray", alpha=0.4, s=10)
        sc = ax.scatter(test_2d[:, 0], test_2d[:, 1],
                        c=np.asarray(scores), cmap="viridis",
                        s=25, alpha=0.9, edgecolors="black", linewidths=0.3)
        plt.colorbar(sc, ax=ax, label="OOD score")
        ax.set_title("OOD scores")
        ax.grid(True)

    plt.suptitle(title)
    plt.tight_layout()

    if path is not None:
        plt.savefig(path, dpi=110)
    else:
        plt.show()

    plt.close(fig)


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
