from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal

import numpy as np
import pandas as pd

from oodt.data.base import BaseTabularDataset
from oodt.shifts.base import BaseShiftStrategy
from oodt.splitting.splitter import TrainTestSplitter
from oodt.metrics.metrics import MetricsEvaluator, MetricsResult
from oodt.utils.utils import get_partition_indices


# ============================================================
# Result container
# ============================================================

@dataclass
class PipelineResult:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    metadata: Dict[str, Any]
    metrics: MetricsResult


# ============================================================
# Main Pipeline
# ============================================================

class OODPipeline:
    """
    End-to-end pipeline with two modes:

    Mode 1 (unknown OOD):
        partition → split

    Mode 2 (known OOD):
        split → partition (only train)

    Now supports:
        - test projection into MF space
        - storing projection inside metadata
    """

    def __init__(
        self,
        model: Any,
        shift_strategy: BaseShiftStrategy,
        splitter: TrainTestSplitter,
        metrics: MetricsEvaluator,
        mode: Literal["unknown_ood", "known_ood"] = "unknown_ood",
    ):
        self.model = model
        self.shift_strategy = shift_strategy
        self.splitter = splitter
        self.metrics = metrics
        self.mode = mode

    # ============================================================
    # Run pipeline
    # ============================================================

    def run(self, dataset: BaseTabularDataset) -> PipelineResult:

        meta: Dict[str, Any] = {}

        # ============================================================
        # MODE 1: Unknown OOD → partition first, then split
        # ============================================================

        if self.mode == "unknown_ood":

            X = dataset.data
            y = dataset.target

            # --- Step 1: apply shift strategy on full dataset ---
            partitions = self.shift_strategy.get_partition_indices(X, y)

            # --- Step 2: inject partitions into splitter ---
            self.splitter.partitions = partitions

            # --- Step 3: create train/test ---
            X_train, y_train, X_test, y_test, meta = self.splitter.split(X, y)

            # store partitions for debugging
            meta["partitions"] = partitions

        # ============================================================
        # MODE 2: Known OOD → split first, then partition train only
        # ============================================================

        elif self.mode == "known_ood":

            if dataset.ood_split is None or dataset.ood_target is None:
                raise ValueError(
                    "Mode 'known_ood' requires dataset with predefined OOD splits"
                )

            # predefined partitions from dataset
            self.splitter.partitions = get_partition_indices(dataset.ood_target)

            X = dataset.data
            y = dataset.target

            # --- Step 1: split train/test FIRST ---
            X_train, y_train, X_test, y_test, meta = self.splitter.split(X, y)

            # --- Step 2: partition ONLY training data ---
            partitions = self.shift_strategy.get_partition_indices(X_train, y_train)

            # store partitions for reference/debugging
            meta["train_partitions"] = partitions

        else:
            raise ValueError(f"Unknown pipeline mode: {self.mode}")

        # ============================================================
        # Store shift metadata
        # ============================================================

        if hasattr(self.shift_strategy, "get_meta_info"):
            meta["shift_meta"] = self.shift_strategy.get_meta_info()

        # ============================================================
        # NEW: Project test samples into MF space
        # ============================================================

        if hasattr(self.shift_strategy, "project_samples"):
            test_projection = self.shift_strategy.project_samples(X_test)

            meta["test_projection"] = test_projection

            if hasattr(self.shift_strategy, "centroids"):

                centroids = self.shift_strategy.centroids

                if isinstance(test_projection, np.ndarray):

                    dist_to_centroid = []

                    for v in test_projection:
                        d = np.min([
                            np.linalg.norm(v - c)
                            for c in centroids
                        ])
                        dist_to_centroid.append(d)

                    meta["test_centroid_distance"] = np.array(dist_to_centroid)

                elif isinstance(test_projection, list):

                    dist_to_centroid = []

                    for mat in test_projection:
                        d = np.min([
                            np.linalg.norm(mat - c, ord="fro")
                            for c in centroids
                        ])
                        dist_to_centroid.append(d)

                    meta["test_centroid_distance"] = np.array(dist_to_centroid)

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        ood_mask = X_test.index.isin(meta["test_ood_indices"])

        ood_scores: Optional[np.ndarray] = None

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_test)
            ood_scores = 1.0 - np.max(probs, axis=1)

        elif hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X_test)
            ood_scores = scores.astype(float)

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
