from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, Tuple

import pandas as pd


class BaseTabularDataset(ABC):
    """
    Abstract base class for tabular datasets with optional OOD (out-of-distribution) support.

    This class standardizes dataset handling for the OOD framework, including:
    - feature/target storage,
    - predefined dataset splits,
    - subset-level OOD labeling.

    Attributes
    ----------
    name : str
        Dataset identifier.

    data : pd.DataFrame
        Feature matrix containing all samples. If multiple splits are used,
        this is a concatenation of all split feature matrices.

    target : pd.Series
        Target values aligned with `data`.

    target_col : str, optional
        Name of the target column in raw data files. Must be set before loading.

    feature_types : dict
        Mapping from column name to dtype (as string).
        Example: {"age": "int64", "income": "float64"}.

    metadata : dict
        Additional dataset-specific metadata (e.g., source, preprocessing info).

    ood_split : dict, optional
        Dictionary containing per-split data:
        {
            "split_name": {
                "data": pd.DataFrame,
                "target": pd.Series
            }
        }

        Example:
        {
            "train": {...},
            "test": {...},
            "ood": {...}
        }

    ood_target : pd.Series, optional
        Integer-encoded labels indicating which split each sample belongs to.
        Length matches `self.data`.

        Example:
        [0, 0, 0, ..., 1, 1, ..., 2, 2, ...]

    ood_partitions : dict[str, pd.Index], optional
        Mapping from partition name to indices in `self.data`.

        Example:
        {
            "id": Index([...]),
            "ood": Index([...])
        }

        Not automatically populated — can be used by downstream components.
    """

    def __init__(self, name: str):
        """
        Initialize dataset.

        Parameters
        ----------
        name : str
            Dataset name.
        """
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
        """
        Load the dataset.

        Must be implemented in subclasses.

        Expected behavior:
        - Load raw data (single file or multiple splits).
        - Set `self.data` and `self.target`, OR
        - Use `_load_splits()` for multi-split datasets.
        """
        pass

    def _load_splits(
        self,
        paths: Dict[str, str],
        reader_func: Callable,
        reader_kwargs: dict = None
    ):
        """
        Load dataset from multiple predefined splits.

        Parameters
        ----------
        paths : dict
            Mapping from split name to file path.

        reader_func : callable
            Function used to read each split (e.g., `pd.read_csv`).

        reader_kwargs : dict, optional
            Additional arguments for reader_func.

        Raises
        ------
        ValueError
            If `target_col` is missing in any split.
        """
        self.ood_split = {}

        for split_name, split_path in paths.items():
            df = reader_func(split_path, **(reader_kwargs or {}))
            df.reset_index(drop=True, inplace=True)

            if getattr(self, "target_col", None) not in df.columns:
                raise ValueError(
                    f"Target column {self.target_col} not found in {split_name} split"
                )

            y = df.pop(self.target_col)

            self.ood_split[split_name] = {
                "data": df,
                "target": y
            }

        # Concatenate splits
        self.data = pd.concat(
            [v["data"] for v in self.ood_split.values()],
            ignore_index=True
        )

        self.target = pd.concat(
            [v["target"] for v in self.ood_split.values()],
            ignore_index=True
        )

        # Create OOD labels (split index per sample)
        self.ood_target = pd.concat(
            [
                pd.Series([k] * len(v["target"]), index=v["target"].index)
                for k, v in enumerate(self.ood_split.values())
            ],
            ignore_index=True
        )

        self.feature_types = {
            col: str(self.data[col].dtype) for col in self.data.columns
        }

    def summary(self):
        """
        Print dataset summary.
        """
        print(f"Dataset: {self.name}")
        print(f"Number of samples: {len(self.data)}")
        print(f"Number of features: {self.data.shape[1]}")
        print(f"Feature types: {self.feature_types}")

        if self.ood_split:
            print(f"OOD Splits: {list(self.ood_split.keys())}")

    def combine_id(
        self,
        ood_split_name: str = "target"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Combine all splits except the specified OOD split.

        Parameters
        ----------
        ood_split_name : str
            Split to exclude (treated as OOD).

        Returns
        -------
        X_combined : pd.DataFrame
            Features from ID splits.

        y_combined : pd.Series
            Targets from ID splits.
        """
        if self.ood_split is None:
            raise ValueError("OOD splits are not initialized. Call load() first.")

        x_list = []
        y_list = []

        for split_name, split_data in self.ood_split.items():
            if split_name != ood_split_name:
                x_list.append(split_data["data"])
                y_list.append(split_data["target"])

        X_combined = pd.concat(x_list, ignore_index=True)
        y_combined = pd.concat(y_list, ignore_index=True)

        return X_combined, y_combined
