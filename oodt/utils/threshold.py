"""
OOD Threshold Finder
====================
Given centroid distances for test samples (already in PipelineResult),
finds the optimal distance threshold via ROC curve that best separates
ID (0) from OOD (1) samples.

Handles both:
  - direction: OOD can be farther OR closer to centroids than ID
  - label polarity: picks whichever assignment (dist >= t or dist <= t) gives better macro F1
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, f1_score
from dataclasses import dataclass


@dataclass
class ThresholdResult:
    threshold: float
    auc_roc: float
    y_true: np.ndarray
    y_pred: np.ndarray
    y_score: np.ndarray  # raw distances (not negated)
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray


def find_ood_threshold(result) -> ThresholdResult:
    """
    Find optimal distance threshold to separate ID vs OOD test samples.

    Steps:
      1. Auto-detect score direction (negate if AUC < 0.5).
      2. Pick best threshold via Youden's J on the ROC curve.
      3. Try both label assignments (dist >= t and dist <= t),
         keep whichever gives higher macro F1.
    """
    meta = result.metadata

    if "test_centroid_distance" not in meta:
        raise ValueError("No test_centroid_distance in metadata. Run pipeline first.")

    distances = meta["test_centroid_distance"]
    ood_mask = result.X_test.index.isin(meta["test_ood_indices"])
    y_true = ood_mask.astype(int)

    if len(np.unique(y_true)) < 2:
        raise ValueError("Need both ID and OOD samples in test set.")

    # Step 1: auto-detect direction
    auc_raw = roc_auc_score(y_true, distances)
    inverted = auc_raw < 0.5
    scores = -distances if inverted else distances

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)  # always >= 0.5

    # Step 2: best threshold via Youden's J
    best_idx = np.argmax(tpr - fpr)
    best_score_threshold = thresholds[best_idx]
    best_threshold = -best_score_threshold if inverted else best_score_threshold

    # Step 3: try both label assignments, pick best macro F1
    y_pred_geq = (distances >= best_threshold).astype(int)
    y_pred_leq = (distances <= best_threshold).astype(int)

    f1_geq = f1_score(y_true, y_pred_geq, average="macro", zero_division=0)
    f1_leq = f1_score(y_true, y_pred_leq, average="macro", zero_division=0)

    y_pred = y_pred_geq if f1_geq >= f1_leq else y_pred_leq

    return ThresholdResult(
        threshold=best_threshold,
        auc_roc=auc,
        y_true=y_true,
        y_pred=y_pred,
        y_score=distances,
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
    )


def plot_ood_threshold(tr: ThresholdResult, path: str = None):
    """
    Two-panel plot:
      Left  - ROC curve with optimal threshold marked
      Right - Distance distributions for ID vs OOD with threshold line
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ---- Left: ROC curve ----
    ax = axes[0]
    ax.plot(tr.fpr, tr.tpr, color="steelblue", lw=2,
            label=f"AUC = {tr.auc_roc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)

    best_idx = np.argmax(tr.tpr - tr.fpr)
    ax.scatter(tr.fpr[best_idx], tr.tpr[best_idx],
               color="crimson", zorder=5, s=80,
               label=f"Threshold = {tr.threshold:.4f}")

    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Right: Distance distributions ----
    ax = axes[1]
    id_dist  = tr.y_score[tr.y_true == 0]
    ood_dist = tr.y_score[tr.y_true == 1]

    bins = np.linspace(tr.y_score.min(), tr.y_score.max(), 50)
    ax.hist(id_dist,  bins=bins, alpha=0.6, color="orange",    label="ID",  density=True)
    ax.hist(ood_dist, bins=bins, alpha=0.6, color="steelblue", label="OOD", density=True)
    ax.axvline(tr.threshold, color="crimson", lw=2, linestyle="--",
               label=f"Threshold = {tr.threshold:.4f}")

    ax.set_xlabel("Distance to nearest centroid")
    ax.set_ylabel("Density")
    ax.set_title("Distance Distributions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    report = classification_report(tr.y_true, tr.y_pred,
                                   target_names=["ID", "OOD"], output_dict=True)
    f1_id  = report["ID"]["f1-score"]
    f1_ood = report["OOD"]["f1-score"]
    plt.suptitle(
        f"AUC-ROC: {tr.auc_roc:.3f}  |  F1 ID: {f1_id:.3f}  |  F1 OOD: {f1_ood:.3f}",
        fontsize=12,
    )

    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150)
        plt.close()
    else:
        plt.show()


def print_ood_threshold_report(tr: ThresholdResult):
    print("=" * 50)
    print(f"  AUC-ROC   : {tr.auc_roc:.4f}")
    print(f"  Threshold : {tr.threshold:.4f}")
    print("=" * 50)
    print(classification_report(tr.y_true, tr.y_pred, target_names=["ID", "OOD"]))
    id_dist  = tr.y_score[tr.y_true == 0]
    ood_dist = tr.y_score[tr.y_true == 1]
    print(f"  ID  dist  — mean: {id_dist.mean():.4f}  std: {id_dist.std():.4f}")
    print(f"  OOD dist  — mean: {ood_dist.mean():.4f}  std: {ood_dist.std():.4f}")
    print("=" * 50)