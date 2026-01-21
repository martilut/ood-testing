from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from oodt.data.loaders import CSVDataset
from oodt.pipelines.pipeline_builder import OODPipeline
from oodt.metrics.metrics import MetricsEvaluator
from oodt.shifts.concept.feature_stratification import FeatureStratificationShift
from oodt.shifts.concept.mf_kmeans import MFKMeansShift
from oodt.splitting.splitter import TrainTestSplitter
from oodt.shifts.base import BaseShiftStrategy
from oodt.utils.utils import get_project_path


# =========================
# Example shift strategy
# =========================

class RandomPartitionShift(BaseShiftStrategy):
    def get_partition_labels(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        rng = np.random.RandomState(self.random_state)
        labels = rng.randint(0, self.n_partitions, size=len(X))
        return pd.Series(labels, index=X.index)


# =========================
# Load dataset
# =========================

dataset_dir = get_project_path() / Path("datasets/partitions/synt")
paths = {
    "source": dataset_dir / "source.csv",
    "target": dataset_dir / "target.csv",
}

dataset = CSVDataset(
    path=paths,
    target_col="y",
    name="electricity_small",
)
dataset.load()

X = dataset.data
y = dataset.target


# =========================
# Initialize components
# =========================

shift_strategy = MFKMeansShift(
    mf_name=["var"],
    n_partitions=2,
    random_state=42,
)

splitter = TrainTestSplitter(
    partitions={},          # injected by pipeline
    mode=2,                 # train = ID, test = ID + OOD
    train_ratio=0.7,
    test_ratio=0.3,
    id_partitions=[0],
    ood_partitions=[1],
    stratify=True,
    random_state=42,
)

model = LogisticRegression(
    max_iter=1000,
    n_jobs=-1,
)

metrics = MetricsEvaluator(task="classification")


# =========================
# Run experiment
# =========================

pipeline = OODPipeline(
    model=model,
    shift_strategy=shift_strategy,
    splitter=splitter,
    metrics=metrics,
)

result = pipeline.run(X, y)


# =========================
# Results
# =========================

print("ID metrics:")
print(result.metrics.id_metrics)

print("\nOOD metrics:")
print(result.metrics.ood_metrics)

print("\nGlobal OOD metrics:")
print(result.metrics.global_metrics)

print("\nMetadata:")
for k, v in result.metadata.items():
    print(k, len(v))
