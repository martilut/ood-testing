from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    roc_auc_score,
    roc_curve,
    f1_score, precision_score, recall_score
)
from sklearn.preprocessing import normalize


@dataclass
class MetricsResult:
    """
    Container for evaluation results.

    Attributes
    ----------
    id_metrics : Dict[str, float]
        Predictive metrics computed on in-distribution (ID) samples only.

    ood_metrics : Dict[str, float]
        Predictive metrics computed on out-of-distribution (OOD) samples only.

    global_metrics : Dict[str, float]
        Metrics that require both ID and OOD samples, typically OOD detection metrics.
        Example:
        {
            "auroc": float,
            "fpr@95tpr": float
        }
    """
    id_metrics: Dict[str, float]
    ood_metrics: Dict[str, float]
    global_metrics: Dict[str, float]


class MetricsEvaluator:
    """
    Evaluator for predictive performance and OOD detection.

    This class separates evaluation into:
    - ID performance (model behavior on in-distribution data),
    - OOD performance (generalization to shifted data),
    - global OOD detection metrics (quality of OOD scoring).

    Notes
    -----
    - Predictive metrics are computed independently for ID and OOD subsets.
    - OOD detection metrics assume availability of `ood_scores`.
    - Convention: higher OOD score => more OOD-like sample.
    """

    def __init__(self, task: str = "classification"):
        """
        Initialize evaluator.

        Parameters
        ----------
        task : {"classification", "regression"}, default="classification"
            Type of prediction task.

        Raises
        ------
        ValueError
            If task is not supported.
        """
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        self.task = task

    def evaluate(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            ood_mask: np.ndarray,
            ood_scores: Optional[np.ndarray] = None,
    ) -> MetricsResult:
        id_mask = ~ood_mask

        # Per-class probabilities for roc_auc — use raw ood_scores if 1-D
        # (already 1 - max_prob from pipeline), otherwise pass None and skip.
        id_scores = ood_scores[id_mask] if ood_scores is not None else None
        ood_scores_ = ood_scores[ood_mask] if ood_scores is not None else None

        id_metrics = self._predictive_metrics(y_true[id_mask], y_pred[id_mask], id_scores)
        ood_metrics = self._predictive_metrics(y_true[ood_mask], y_pred[ood_mask], ood_scores_)

        global_metrics: Dict[str, float] = {}
        if ood_scores is not None:
            global_metrics["auroc"] = auroc(
                id_scores=ood_scores[id_mask],
                ood_scores=ood_scores[ood_mask],
            )
            global_metrics["fpr@95tpr"] = fpr_at_tpr(
                id_scores=ood_scores[id_mask],
                ood_scores=ood_scores[ood_mask],
                tpr_level=0.95,
            )

        return MetricsResult(
            id_metrics=id_metrics,
            ood_metrics=ood_metrics,
            global_metrics=global_metrics,
        )

    def _predictive_metrics(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            y_scores: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        if y_true.size == 0:
            return {}

        if self.task == "classification":
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
                "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            }
            if y_scores is not None:
                try:
                    metrics["roc_auc"] = roc_auc_score(
                        y_true, y_scores,
                        multi_class="ovr",
                        average="weighted",
                    )
                except ValueError:
                    pass  # single class in subset → skip
            return metrics
        else:
            return {
                "rmse": mean_squared_error(y_true, y_pred, squared=False),
                "mae": mean_absolute_error(y_true, y_pred),
            }


# =========================
# Predictive metrics
# =========================

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average='weighted'))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred, squared=False))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


# =========================
# OOD metrics
# =========================

def auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    y_true = np.concatenate(
        [np.zeros(len(id_scores)), np.ones(len(ood_scores))]
    )
    scores = np.concatenate([id_scores, ood_scores])
    return float(roc_auc_score(y_true, scores))


def fpr_at_tpr(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    tpr_level: float = 0.95,
) -> float:
    y_true = np.concatenate(
        [np.zeros(len(id_scores)), np.ones(len(ood_scores))]
    )
    scores = np.concatenate([id_scores, ood_scores])

    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.searchsorted(tpr, tpr_level, side="left")
    return float(fpr[min(idx, len(fpr) - 1)])


# =========================
# Distribution divergence
# =========================

def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = normalize((p + eps).reshape(1, -1), norm="l1")[0]
    q = normalize((q + eps).reshape(1, -1), norm="l1")[0]
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def histogram_divergence(
    id_values: np.ndarray,
    ood_values: np.ndarray,
    bins: int = 20,
    method: str = "js",
) -> float:
    hist_id, bin_edges = np.histogram(
        id_values, bins=bins, density=True
    )
    hist_ood, _ = np.histogram(
        ood_values, bins=bin_edges, density=True
    )

    if method == "kl":
        return kl_divergence(hist_id, hist_ood)
    elif method == "js":
        return js_divergence(hist_id, hist_ood)
    else:
        raise ValueError(f"Unknown divergence method: {method}")
