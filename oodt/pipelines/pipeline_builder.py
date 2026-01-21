from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from oodt.shifts.base import BaseShiftStrategy
from oodt.splitting.splitter import TrainTestSplitter
from oodt.metrics.metrics import MetricsEvaluator, MetricsResult


@dataclass
class PipelineResult:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    metadata: Dict[str, Any]
    metrics: MetricsResult


class OODPipeline:
    """
    End-to-end pipeline:
      user_train_data
        → shift strategy (partitions)
        → splitter (modes)
        → model training
        → metrics (ID / OOD separated)
    """

    def __init__(
        self,
        model: Any,
        shift_strategy: BaseShiftStrategy,
        splitter: TrainTestSplitter,
        metrics: MetricsEvaluator,
    ):
        self.model = model
        self.shift_strategy = shift_strategy
        self.splitter = splitter
        self.metrics = metrics

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> PipelineResult:
        # --- apply shift strategy ---
        partitions = self.shift_strategy.get_partition_indices(X, y)

        # --- inject partitions into splitter ---
        self.splitter.partitions = partitions

        # --- create train/test ---
        X_train, y_train, X_test, y_test, meta = self.splitter.split(X, y)

        # --- train model ---
        self.model.fit(X_train, y_train)

        # --- predictions ---
        y_pred = self.model.predict(X_test)

        # --- OOD mask for test ---
        ood_mask = X_test.index.isin(meta["test_ood_indices"])

        # --- optional OOD scores ---
        ood_scores: Optional[np.ndarray] = None
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_test)
            ood_scores = 1.0 - np.max(probs, axis=1)
        elif hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X_test)
            ood_scores = scores.astype(float)

        # --- evaluate ---
        metrics_result = self.metrics.evaluate(
            y_true=y_test.to_numpy(),
            y_pred=y_pred,
            ood_mask=ood_mask,
            ood_scores=ood_scores,
        )

        return PipelineResult(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            metadata=meta,
            metrics=metrics_result,
        )
