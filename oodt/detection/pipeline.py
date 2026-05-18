"""
OODDetectionPipeline — train, compare, and rank multiple OOD detectors.

Usage
-----
::

    from oodt.detection.pipeline import OODDetectionPipeline
    from oodt.detection.knn import KNNOODDetector
    from oodt.detection.clustering import ClusteringOODDetector
    from oodt.detection.energy import EnergyOODDetector

    pipeline = OODDetectionPipeline(
        detectors={
            "knn_5":      KNNOODDetector(k=5),
            "kmeans":     ClusteringOODDetector(n_clusters=8),
            "energy_gmm": EnergyOODDetector(mode="density"),
        },
        target_fpr=0.05,
    )

    results = pipeline.run(
        X_train=X_train,  # ID training data (no OOD)
        y_train=y_train,
        X_val=X_val,      # Held-out ID data for threshold calibration
        X_test=X_test,    # Mixed ID + OOD evaluation set
        ood_mask=mask,    # Boolean array: True where sample is OOD
    )

    pipeline.print_report(results)
    df = pipeline.to_dataframe(results)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from oodt.detection.base import BaseOODDetector


# -----------------------------------------------------------------------
# Result container
# -----------------------------------------------------------------------

@dataclass
class DetectorResult:
    """
    Evaluation result for a single OOD detector.

    Attributes
    ----------
    name : str
        Detector identifier.
    auroc : float
        Area under the ROC curve.  1.0 is perfect; 0.5 is random.
    aupr : float
        Area under the precision-recall curve (OOD as positive class).
    fpr95 : float
        False-positive rate at 95% true-positive rate.  Lower is better.
    det_acc : float or None
        Detection accuracy at the calibrated threshold.
    threshold : float or None
        Calibrated decision threshold.
    fit_time_s : float
        Wall-clock seconds for fit().
    score_time_s : float
        Wall-clock seconds for score_samples() on the test set.
    scores : np.ndarray
        Raw OOD scores on the test set (higher = more OOD).
    params : dict
        Detector hyper-parameters.
    """
    name: str
    auroc: float
    aupr: float
    fpr95: float
    det_acc: Optional[float]
    threshold: Optional[float]
    fit_time_s: float
    score_time_s: float
    scores: np.ndarray
    params: dict = field(default_factory=dict)


# -----------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------

class OODDetectionPipeline:
    """
    End-to-end pipeline for training and comparing OOD detectors.

    Parameters
    ----------
    detectors : dict[str, BaseOODDetector]
        Named detector instances.  They are trained in the order given.
    target_fpr : float
        Desired false-positive rate for threshold calibration (default 0.05).
    verbose : bool
        Print progress during run().
    """

    def __init__(
        self,
        detectors: Dict[str, BaseOODDetector],
        target_fpr: float = 0.05,
        verbose: bool = True,
    ) -> None:
        if not detectors:
            raise ValueError("At least one detector must be provided.")
        if not 0.0 < target_fpr < 1.0:
            raise ValueError(f"target_fpr must be in (0, 1), got {target_fpr}")

        self.detectors = detectors
        self.target_fpr = target_fpr
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        ood_mask: np.ndarray,
        y_train: Optional[pd.Series] = None,
        X_val: Optional[pd.DataFrame] = None,
    ) -> Dict[str, DetectorResult]:
        """
        Fit all detectors, calibrate thresholds, and evaluate.

        Parameters
        ----------
        X_train : pd.DataFrame
            In-distribution training features.
        X_test : pd.DataFrame
            Test features (mix of ID and OOD samples).
        ood_mask : np.ndarray of bool
            True where a test sample is OOD.
        y_train : pd.Series, optional
            Training labels (required for supervised detectors such as
            EnergyOODDetector in "classifier" mode).
        X_val : pd.DataFrame, optional
            Held-out ID validation data for threshold calibration.
            If None, ``X_train`` is used (less reliable; prefer providing
            a separate split).

        Returns
        -------
        results : dict[name, DetectorResult]
        """
        X_cal = X_val if X_val is not None else X_train
        results: Dict[str, DetectorResult] = {}

        for name, detector in self.detectors.items():
            if self.verbose:
                print(f"  [{name}] fitting ...", end="", flush=True)

            # Fit
            t0 = time.perf_counter()
            detector.fit(X_train, y_train)
            fit_time = time.perf_counter() - t0

            # Threshold calibration
            detector.fit_threshold(X_cal, target_fpr=self.target_fpr)

            # Score test set
            t0 = time.perf_counter()
            scores = detector.score_samples(X_test)
            score_time = time.perf_counter() - t0

            # Evaluate
            metrics = detector.evaluate(X_test, ood_mask, threshold=detector.threshold_)

            results[name] = DetectorResult(
                name=name,
                auroc=metrics["auroc"],
                aupr=metrics["aupr"],
                fpr95=metrics["fpr95"],
                det_acc=metrics.get("det_acc"),
                threshold=detector.threshold_,
                fit_time_s=fit_time,
                score_time_s=score_time,
                scores=scores,
                params=detector.get_params(),
            )

            if self.verbose:
                r = results[name]
                print(
                    f" done  "
                    f"AUROC={r.auroc:.3f}  "
                    f"AUPR={r.aupr:.3f}  "
                    f"FPR95={r.fpr95:.3f}  "
                    f"({fit_time:.2f}s fit)"
                )

        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def to_dataframe(self, results: Dict[str, DetectorResult]) -> pd.DataFrame:
        """
        Convert results to a tidy DataFrame sorted by AUROC (descending).

        Returns
        -------
        pd.DataFrame with columns:
            detector, auroc, aupr, fpr95, det_acc,
            threshold, fit_time_s, score_time_s
        """
        rows = []
        for r in results.values():
            rows.append({
                "detector":     r.name,
                "auroc":        round(r.auroc, 4),
                "aupr":         round(r.aupr, 4),
                "fpr95":        round(r.fpr95, 4),
                "det_acc":      round(r.det_acc, 4) if r.det_acc is not None else None,
                "threshold":    round(r.threshold, 6) if r.threshold is not None else None,
                "fit_time_s":   round(r.fit_time_s, 3),
                "score_time_s": round(r.score_time_s, 4),
            })
        df = pd.DataFrame(rows).sort_values("auroc", ascending=False)
        df = df.reset_index(drop=True)
        df.index += 1   # 1-based rank
        df.index.name = "rank"
        return df

    def print_report(self, results: Dict[str, DetectorResult]) -> None:
        """Pretty-print a comparison table."""
        df = self.to_dataframe(results)
        print("\n=== OOD Detection Results ===")
        print(df.to_string())
        print()

    @staticmethod
    def best(
        results: Dict[str, DetectorResult],
        metric: str = "auroc",
    ) -> DetectorResult:
        """
        Return the detector with the best score on a given metric.

        Parameters
        ----------
        metric : str
            One of "auroc", "aupr", "det_acc" (higher is better) or
            "fpr95" (lower is better).
        """
        if metric == "fpr95":
            return min(results.values(), key=lambda r: getattr(r, metric))
        return max(results.values(), key=lambda r: (getattr(r, metric) or 0.0))
