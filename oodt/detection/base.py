"""
Abstract base class for all OOD detectors in the oodt framework.

Every detector exposes:
    fit(X_train, y_train=None) → self
    score_samples(X)           → np.ndarray   (higher = more OOD)
    predict(X, threshold=None) → np.ndarray   (1 = OOD, 0 = ID)
    fit_threshold(X_val, fpr)  → float        (calibrate threshold)
    get_params()               → dict

Design notes
------------
- ``score_samples`` is the primary output: a continuous, higher-is-worse
  anomaly score.  Callers that need a binary label should call ``predict``.
- ``fit_threshold`` calibrates a decision boundary on held-out ID data so
  that at most ``target_fpr`` fraction of ID samples are flagged as OOD.
  This is optional; if not called, ``predict`` raises unless a threshold
  is supplied explicitly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


class BaseOODDetector(ABC):
    """
    Abstract base class for OOD detectors.

    Subclasses must implement:
        fit(X_train, y_train=None)
        score_samples(X)
        predict(X, threshold=None)
        get_params()
    """

    def __init__(self) -> None:
        self.is_fitted_: bool = False
        self.threshold_: Optional[float] = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None) -> "BaseOODDetector":
        """
        Fit the detector on in-distribution training data.

        Parameters
        ----------
        X_train : pd.DataFrame
        y_train : pd.Series, optional
            Required by supervised detectors; ignored by unsupervised ones.

        Returns
        -------
        self
        """

    @abstractmethod
    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute a scalar OOD score for each sample.

        Convention: **higher score = more likely OOD**.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """
        Return binary OOD labels: 1 = OOD, 0 = ID.

        Parameters
        ----------
        X : pd.DataFrame
        threshold : float, optional
            Decision threshold on the score.  If None, uses the threshold
            stored by ``fit_threshold``.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,) with values in {0, 1}
        """

    @abstractmethod
    def get_params(self) -> dict:
        """Return a dict of detector hyper-parameters."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def fit_threshold(
        self,
        X_id_val: pd.DataFrame,
        target_fpr: float = 0.05,
    ) -> float:
        """
        Calibrate a decision threshold on held-out ID validation data.

        Sets ``self.threshold_`` so that at most ``target_fpr`` fraction
        of ID samples are predicted as OOD (false positives).

        Parameters
        ----------
        X_id_val : pd.DataFrame
            Held-out **ID** samples (no OOD contamination).
        target_fpr : float
            Desired false-positive rate on ID data (default 0.05 = 5%).

        Returns
        -------
        threshold : float
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before fit_threshold().")
        if not 0.0 < target_fpr < 1.0:
            raise ValueError(f"target_fpr must be in (0, 1), got {target_fpr}")

        scores = self.score_samples(X_id_val)
        self.threshold_ = float(np.quantile(scores, 1.0 - target_fpr))
        return self.threshold_

    def evaluate(
        self,
        X: pd.DataFrame,
        ood_mask: np.ndarray,
        threshold: Optional[float] = None,
    ) -> dict:
        """
        Evaluate detector on a labelled test set.

        Parameters
        ----------
        X : pd.DataFrame
        ood_mask : np.ndarray of bool, shape (n_samples,)
            True where a sample is OOD.
        threshold : float, optional

        Returns
        -------
        metrics : dict with keys:
            "auroc"    — area under the ROC curve
            "aupr"     — area under precision-recall (OOD as positive)
            "fpr95"    — FPR at 95% TPR (ID as positive; lower is better)
            "det_acc"  — detection accuracy at given threshold (if provided)
        """
        from sklearn.metrics import roc_auc_score, average_precision_score

        scores = self.score_samples(X)
        y_true = ood_mask.astype(int)

        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            raise ValueError("ood_mask must contain both ID and OOD samples.")

        auroc = float(roc_auc_score(y_true, scores))
        aupr  = float(average_precision_score(y_true, scores))
        fpr95 = self._fpr_at_tpr(scores, y_true, tpr_target=0.95)

        result = {"auroc": auroc, "aupr": aupr, "fpr95": fpr95}

        if threshold is not None or self.threshold_ is not None:
            thr = threshold if threshold is not None else self.threshold_
            preds = (scores >= thr).astype(int)
            result["det_acc"] = float((preds == y_true).mean())

        return result

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                f"{type(self).__name__} is not fitted. Call fit() first."
            )

    def _get_threshold(self) -> float:
        if self.threshold_ is None:
            raise RuntimeError(
                "No threshold set.  Call fit_threshold() or pass threshold "
                "explicitly to predict()."
            )
        return self.threshold_

    @staticmethod
    def _fpr_at_tpr(
        scores: np.ndarray,
        y_true: np.ndarray,
        tpr_target: float = 0.95,
    ) -> float:
        """FPR at a given TPR level (for OOD detection benchmarking)."""
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, scores)
        # Find the smallest FPR where TPR >= tpr_target
        idx = np.searchsorted(tpr, tpr_target)
        if idx >= len(fpr):
            return float(fpr[-1])
        return float(fpr[idx])
