from pathlib import Path

import pandas as pd

from oodt.data.base import BaseTabularDataset


def get_project_path() -> str:
    return str(Path(__file__).parent.parent.parent)


def get_partition_indices(labels: pd.Series) -> dict[int, pd.Index]:
    partitions = {
        pid: labels.index[labels == pid]
        for pid in range(labels.nunique())
    }

    empty = [pid for pid, idx in partitions.items() if len(idx) == 0]
    if empty:
        raise ValueError(
            f"Empty partitions detected: {empty}. "
        )

    return partitions