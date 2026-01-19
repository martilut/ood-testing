from pathlib import Path
from typing import Optional

import pandas as pd

from .base import BaseTabularDataset


class CSVDataset(BaseTabularDataset):
    def __init__(self, path: str, target_col: str, name: str = "CSV Dataset"):
        super().__init__(name)
        self.path = Path(path)
        self.target_col = target_col

    def load(self):
        self.data = pd.read_csv(self.path)
        if self.target_col not in self.data.columns:
            raise ValueError(f"Target column {self.target_col} not found in dataset")
        self.target = self.data.pop(self.target_col)
        self.feature_types = {col: str(self.data[col].dtype) for col in self.data.columns}


class ParquetDataset(CSVDataset):
    def load(self):
        self.data = pd.read_parquet(self.path)
        if self.target_col not in self.data.columns:
            raise ValueError(f"Target column {self.target_col} not found in dataset")
        self.target = self.data.pop(self.target_col)
        self.feature_types = {col: str(self.data[col].dtype) for col in self.data.columns}


class SklearnDataset(BaseTabularDataset):
    def __init__(self, loader_func, name: str = "Sklearn Dataset", target_name: Optional[str] = None):
        super().__init__(name)
        self.loader_func = loader_func
        self.target_name = target_name

    def load(self):
        dataset = self.loader_func()
        self.data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        self.target = pd.Series(dataset.target, name=self.target_name or "target")
        self.feature_types = {col: "numerical" for col in self.data.columns}
