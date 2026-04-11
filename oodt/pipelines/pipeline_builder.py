from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal

import numpy as np
import pandas as pd

from oodt.data.base import BaseTabularDataset
from oodt.data.preprocessing import Preprocessor
from oodt.metrics.metrics import MetricsEvaluator, MetricsResult
from oodt.shifts.base import BaseShiftStrategy
from oodt.utils.utils import get_partition_indices


# ============================================================
# Result containers
# ============================================================

@dataclass
class PipelineResult:
    """Single-split result (TrainTestSplitter). Unchanged."""
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    metadata: Dict[str, Any]
    metrics: MetricsResult


@dataclass
class FoldResult:
    """
    Result for one CV fold.

    Attributes
    ----------
    fold : int
        Zero-based fold index.
    X_train, y_train : train split.
    X_val, y_val     : validation split.
    X_test, y_test   : held-out test split (only under "train_val_test" strategy;
                       equals val split under plain "cv").
    metadata         : same keys as PipelineResult.metadata.
    metrics          : MetricsResult evaluated on the val split.
    test_metrics     : MetricsResult evaluated on the fixed test split,
                       or None under plain "cv".
    """
    fold: int
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    metadata: Dict[str, Any]
    metrics: MetricsResult
    test_metrics: Optional[MetricsResult] = None


@dataclass
class CVPipelineResult:
    """
    Aggregated result over all k folds.

    Attributes
    ----------
    fold_results : List[FoldResult]
        One entry per fold.
    metrics : MetricsResult
        Mean metrics across folds (val splits).
    test_metrics : MetricsResult or None
        Mean metrics across folds on the fixed test set,
        or None under plain "cv" strategy.
    std : Dict[str, Dict[str, float]]
        Per-subset, per-metric standard deviation across folds.
        Keys: "id_metrics", "ood_metrics", "global_metrics".
    """
    fold_results: List[FoldResult]
    metrics: MetricsResult
    test_metrics: Optional[MetricsResult]
    std: Dict[str, Dict[str, float]]


# ============================================================
# Aggregation helper
# ============================================================

def _aggregate_metrics(
    fold_metrics: List[MetricsResult],
) -> tuple[MetricsResult, Dict[str, Dict[str, float]]]:
    """
    Average MetricsResult objects across folds.

    Returns
    -------
    mean_result : MetricsResult
    std_dict    : {subset: {metric: std}}
    """
    subsets = ("id_metrics", "ood_metrics", "global_metrics")
    means, stds = {}, {}

    for subset in subsets:
        all_dicts = [getattr(r, subset) for r in fold_metrics]
        keys = all_dicts[0].keys()
        means[subset] = {
            k: float(np.mean([d[k] for d in all_dicts]))
            for k in keys
        }
        stds[subset] = {
            k: float(np.std([d[k] for d in all_dicts], ddof=1))
            for k in keys
        }

    mean_result = MetricsResult(
        id_metrics=means["id_metrics"],
        ood_metrics=means["ood_metrics"],
        global_metrics=means["global_metrics"],
    )
    return mean_result, stds


# ============================================================
# Main Pipeline
# ============================================================

class OODPipeline:
    """
    End-to-end pipeline for OOD-aware evaluation on tabular data.

    Supports both TrainTestSplitter (returns PipelineResult) and
    KFoldSplitter (returns CVPipelineResult).  All other behaviour
    is unchanged.
    """

    def __init__(
        self,
        model: Any,
        shift_strategy: BaseShiftStrategy,
        splitter,
        metrics: MetricsEvaluator,
        mode: Literal["unknown_ood", "known_ood"] = "unknown_ood",
        preprocessor: Preprocessor | None = None,
    ):
        self.model = model
        self.shift_strategy = shift_strategy
        self.splitter = splitter
        self.metrics = metrics
        self.mode = mode
        self.preprocessor = preprocessor

    # ----------------------------------------------------------
    # Public entry point
    # ----------------------------------------------------------

    def run(self, dataset: BaseTabularDataset):
        """
        Execute the pipeline.

        Returns PipelineResult   when splitter is TrainTestSplitter.
        Returns CVPipelineResult when splitter is KFoldSplitter.
        """
        from oodt.splitting.kfold import KFoldSplitter

        if isinstance(self.splitter, KFoldSplitter):
            return self._run_cv(dataset)
        return self._run_single(dataset)

    # ----------------------------------------------------------
    # Single split (original behaviour, untouched)
    # ----------------------------------------------------------

    def _run_single(self, dataset: BaseTabularDataset) -> PipelineResult:
        meta: Dict[str, Any] = {}

        if self.mode == "unknown_ood":
            X, y = dataset.data, dataset.target
            partitions = self.shift_strategy.get_partition_indices(X, y)
            self.splitter.partitions = partitions
            X_train, y_train, X_test, y_test, meta = self.splitter.split(X, y)
            meta["partitions"] = partitions

        elif self.mode == "known_ood":
            if dataset.ood_split is None or dataset.ood_target is None:
                raise ValueError(
                    "Mode 'known_ood' requires dataset with predefined OOD splits"
                )
            self.splitter.partitions = get_partition_indices(dataset.ood_target)
            X, y = dataset.data, dataset.target
            X_train, y_train, X_test, y_test, meta = self.splitter.split(X, y)
            partitions = self.shift_strategy.get_partition_indices(X_train, y_train)
            meta["train_partitions"] = partitions
        else:
            raise ValueError(f"Unknown pipeline mode: {self.mode}")

        if self.preprocessor is not None:
            X_train = self.preprocessor.fit_transform(X_train)
            X_test = self.preprocessor.transform(X_test)

        self._attach_shift_meta(meta, X_test)
        self.model.fit(X_train, y_train)
        metrics_result = self._evaluate(X_test, y_test, meta)

        return PipelineResult(
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
            metadata=meta,   metrics=metrics_result,
        )

    # ----------------------------------------------------------
    # CV loop
    # ----------------------------------------------------------

    def _run_cv(self, dataset: BaseTabularDataset) -> CVPipelineResult:
        """Partition once, then iterate folds."""
        if self.mode == "unknown_ood":
            X, y = dataset.data, dataset.target
            partitions = self.shift_strategy.get_partition_indices(X, y)
            self.splitter.partitions = partitions

        elif self.mode == "known_ood":
            if dataset.ood_split is None or dataset.ood_target is None:
                raise ValueError(
                    "Mode 'known_ood' requires dataset with predefined OOD splits"
                )
            self.splitter.partitions = get_partition_indices(dataset.ood_target)
            X, y = dataset.data, dataset.target
            partitions = None
        else:
            raise ValueError(f"Unknown pipeline mode: {self.mode}")

        fold_results: List[FoldResult] = []
        is_tvt = self.splitter.split_strategy == "train_val_test"

        for fold_idx, fold in enumerate(self.splitter.split(X, y)):

            # Unpack — 7-tuple under train_val_test, 5-tuple under cv
            if is_tvt:
                X_train, y_train, X_val, y_val, X_test, y_test, meta = fold
            else:
                X_train, y_train, X_val, y_val, meta = fold
                X_test, y_test = X_val, y_val

            # Attach partition info to meta
            if self.mode == "unknown_ood":
                meta["partitions"] = partitions
            elif self.mode == "known_ood":
                train_parts = self.shift_strategy.get_partition_indices(
                    X_train, y_train
                )
                meta["train_partitions"] = train_parts

            self._attach_shift_meta(meta, X_val)

            # Fresh model per fold
            import copy
            fold_model = copy.deepcopy(self.model)

            if self.preprocessor is not None:
                fold_pre = copy.deepcopy(self.preprocessor)
                X_train = fold_pre.fit_transform(X_train)
                X_val = fold_pre.transform(X_val)
                if is_tvt:
                    X_test = fold_pre.transform(X_test)
                meta["preprocessor"] = fold_pre

            fold_model.fit(X_train, y_train)

            # Evaluate on val
            val_metrics = self._evaluate(
                X_val, y_val, meta,
                model=fold_model,
                ood_key="val_ood_indices",
            )

            # Evaluate on fixed test set (train_val_test only)
            test_metrics = None
            if is_tvt:
                test_meta = {
                    "test_ood_indices": meta["test_ood_indices"],
                }
                test_metrics = self._evaluate(
                    X_test, y_test, test_meta,
                    model=fold_model,
                    ood_key="test_ood_indices",
                )

            fold_results.append(FoldResult(
                fold=fold_idx,
                X_train=X_train, y_train=y_train,
                X_val=X_val,     y_val=y_val,
                X_test=X_test,   y_test=y_test,
                metadata=meta,
                metrics=val_metrics,
                test_metrics=test_metrics,
            ))

        # Aggregate val metrics
        mean_metrics, std_dict = _aggregate_metrics(
            [fr.metrics for fr in fold_results]
        )

        # Aggregate test metrics (if available)
        mean_test_metrics = None
        if is_tvt:
            mean_test_metrics, _ = _aggregate_metrics(
                [fr.test_metrics for fr in fold_results]
            )

        return CVPipelineResult(
            fold_results=fold_results,
            metrics=mean_metrics,
            test_metrics=mean_test_metrics,
            std=std_dict,
        )

    # ----------------------------------------------------------
    # Shared helpers
    # ----------------------------------------------------------

    def _attach_shift_meta(
        self, meta: Dict[str, Any], X_eval: pd.DataFrame
    ) -> None:
        """Attach shift strategy metadata and projections to meta dict."""
        if hasattr(self.shift_strategy, "get_meta_info"):
            meta["shift_meta"] = self.shift_strategy.get_meta_info()

        if hasattr(self.shift_strategy, "project_samples"):
            projection = self.shift_strategy.project_samples(X_eval)
            meta["test_projection"] = projection

            if hasattr(self.shift_strategy, "centroids"):
                centroids = self.shift_strategy.centroids
                if isinstance(projection, np.ndarray):
                    meta["test_centroid_distance"] = np.array([
                        np.min([np.linalg.norm(v - c) for c in centroids])
                        for v in projection
                    ])
                elif isinstance(projection, list):
                    meta["test_centroid_distance"] = np.array([
                        np.min([np.linalg.norm(mat - c, ord="fro") for c in centroids])
                        for mat in projection
                    ])

    def _evaluate(
        self,
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
        meta: Dict[str, Any],
        model=None,
        ood_key: str = "test_ood_indices",
    ) -> MetricsResult:
        """Predict, build OOD mask, and run metrics evaluator."""
        if model is None:
            model = self.model

        ood_mask = X_eval.index.isin(meta[ood_key])
        y_pred = model.predict(X_eval)

        ood_scores: Optional[np.ndarray] = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_eval)
            ood_scores = 1.0 - np.max(probs, axis=1)
        elif hasattr(model, "decision_function"):
            ood_scores = model.decision_function(X_eval).astype(float)

        return self.metrics.evaluate(
            y_true=y_eval.to_numpy(),
            y_pred=y_pred,
            ood_mask=ood_mask,
            ood_scores=ood_scores,
        )
