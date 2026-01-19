from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class BaseTabularDataset(ABC):
    def __init__(self, name: str):
        self.name = name
        self.data: pd.DataFrame = None
        self.target: pd.Series = None
        self.target_col: Optional[str] = None
        self.feature_types: dict = {}
        self.metadata: dict = {}
        self.ood_split: Optional[dict] = None

    @abstractmethod
    def load(self):
        """Load the dataset (single or multiple files)"""
        pass

    def _load_splits(self, paths: dict, reader_func):
        """Helper to load multiple splits"""
        self.ood_split = {}
        for split_name, split_path in paths.items():
            df = reader_func(split_path)
            if getattr(self, "target_col", None) not in df.columns:
                raise ValueError(f"Target column {self.target_col} not found in {split_name} split")
            y = df.pop(self.target_col)
            self.ood_split[split_name] = {"data": df, "target": y}

        self.data = pd.concat([v["data"] for v in self.ood_split.values()], ignore_index=True)
        self.target = pd.concat([v["target"] for v in self.ood_split.values()], ignore_index=True)
        self.feature_types = {col: str(self.data[col].dtype) for col in self.data.columns}

    def summary(self):
        """Print a summary of the dataset"""
        print(f"Dataset: {self.name}")
        print(f"Number of samples: {len(self.data)}")
        print(f"Number of features: {self.data.shape[1]}")
        print(f"Feature types: {self.feature_types}")
        if self.ood_split:
            print(f"OOD Splits: {list(self.ood_split.keys())}")
