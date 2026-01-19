import pytest
from importlib import import_module
from hydra.utils import instantiate
import pandas as pd


# Helper to convert string to callable
def load_callable(path: str):
    module_path, func_name = path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, func_name)


@pytest.mark.parametrize(
    "loader_path, expected_rows, expected_cols",
    [
        ("sklearn.datasets.load_iris", 150, 4),
        ("sklearn.datasets.load_wine", 178, 13)
    ]
)
def test_sklearn_dataset_loading(loader_path, expected_rows, expected_cols):
    """
    Test that SklearnDataset loads datasets correctly.
    """
    dataset_cfg = {
        "_target_": "oodt.data.loaders.SklearnDataset",
        "loader_func": load_callable(loader_path),
        "name": "Test Dataset",
        "target_name": "target"
    }

    dataset = instantiate(dataset_cfg)
    dataset.load()

    # Basic checks
    assert dataset.data is not None
    assert dataset.target is not None
    assert dataset.data.shape == (expected_rows, expected_cols)
    assert len(dataset.target) == expected_rows

    # Ensure summary runs without errors
    dataset.summary()


def test_sklearn_dataset_preprocessing():
    """
    Test that the Preprocessor works with a small dataset.
    """
    from oodt.data.preprocessing import Preprocessor
    from sklearn.datasets import load_iris

    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    preprocessor = Preprocessor(
        scaling="standard",
        encoding="onehot",
        impute_strategy="mean"
    )

    X_processed = preprocessor.fit_transform(X)

    # Preprocessor should return a DataFrame
    assert isinstance(X_processed, pd.DataFrame)
    # Number of rows unchanged
    assert X_processed.shape[0] == X.shape[0]


if __name__ == "__main__":
    pytest.main()
