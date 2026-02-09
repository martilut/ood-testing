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
        self.ood_target: Optional[pd.Series] = None
        self.ood_partitions: Optional[dict[str, pd.Index]] = None

    @abstractmethod
    def load(self):
        """Load the dataset (single or multiple files)"""
        pass

    def _load_splits(self, paths: dict, reader_func: callable, reader_kwargs: dict = None):
        """Helper to load multiple splits"""
        self.ood_split = {}
        for split_name, split_path in paths.items():
            df = reader_func(split_path, **(reader_kwargs or {}))
            df.reset_index(drop=True, inplace=True)
            if getattr(self, "target_col", None) not in df.columns:
                raise ValueError(f"Target column {self.target_col} not found in {split_name} split")
            y = df.pop(self.target_col)
            self.ood_split[split_name] = {"data": df, "target": y}

        self.data = pd.concat([v["data"] for v in self.ood_split.values()], ignore_index=True)
        self.ood_target = pd.concat(
            [pd.Series([k] * len(v["target"]), index=v["target"].index) for k, v in enumerate(self.ood_split.values())],
            ignore_index=True
        )

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

    def combine_id(self, ood_split_name: str = "target") -> tuple[pd.DataFrame, pd.Series]:
        x_list = []
        y_list = []
        for split_name, split_data in self.ood_split.items():
            if split_name != ood_split_name:
                x_list.append(split_data["data"])
                y_list.append(split_data["target"])
        X_combined = pd.concat(x_list, ignore_index=True)
        y_combined = pd.concat(y_list, ignore_index=True)
        return X_combined, y_combined