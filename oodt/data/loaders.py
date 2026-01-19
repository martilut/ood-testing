from pathlib import Path
from typing import Optional, Union

import pandas as pd

from .base import BaseTabularDataset


class CSVDataset(BaseTabularDataset):
    def __init__(self, path, target_col: str, name="CSV Dataset"):
        super().__init__(name)
        self.path = path
        self.target_col = target_col

    def load(self):
        if isinstance(self.path, str):
            df = pd.read_csv(self.path)
            if self.target_col not in df.columns:
                raise ValueError(f"Target column {self.target_col} not found in dataset")
            self.target = df.pop(self.target_col)
            self.data = df
            self.feature_types = {col: str(df[col].dtype) for col in df.columns}
        elif isinstance(self.path, dict):
            paths = {k: Path(v) for k, v in self.path.items()}
            self._load_splits(paths, pd.read_csv)
        else:
            raise ValueError("path must be str or dict of split_name -> path")

class ParquetDataset(CSVDataset):
    def load(self):
        if isinstance(self.path, str):
            df = pd.read_parquet(self.path)
            if self.target_col not in df.columns:
                raise ValueError(f"Target column {self.target_col} not found in dataset")
            self.target = df.pop(self.target_col)
            self.data = df
            self.feature_types = {col: str(df[col].dtype) for col in df.columns}
        elif isinstance(self.path, dict):
            paths = {k: Path(v) for k, v in self.path.items()}
            self._load_splits(paths, pd.read_parquet)
        else:
            raise ValueError("path must be str or dict of split_name -> path")

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
