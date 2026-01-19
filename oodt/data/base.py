from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd


class BaseTabularDataset(ABC):
    """
    Base class for tabular datasets.
    Supports ID/OOD splitting, train/test splits, and metadata.
    """

    def __init__(self, name: str):
        self.name = name
        self.data: pd.DataFrame = None
        self.target: pd.Series = None
        self.feature_types: dict = {}  # {"feature_name": "categorical"/"numerical"}
        self.metadata: dict = {}  # additional info, e.g., n_classes

    @abstractmethod
    def load(self):
        """
        Load data into self.data and self.target
        """
        pass

    @abstractmethod
    def get_splits(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Return train/test splits: X_train, y_train, X_test, y_test
        """
        pass

    def summary(self):
        print(f"Dataset: {self.name}")
        print(f"Shape: {self.data.shape}")
        print(f"Features: {self.feature_types}")
        if self.target is not None:
            print(f"Target distribution:\n{self.target.value_counts(normalize=True)}")
