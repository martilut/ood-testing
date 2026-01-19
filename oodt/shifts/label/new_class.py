from typing import List, Dict
import pandas as pd
from oodt.shifts.base import BaseShiftStrategy


class NewClassShift(BaseShiftStrategy):
    """
    Label shift strategy where one or more classes are held out
    and treated as OOD. Useful for single-class OOD evaluation
    in multi-class/multi-label classification.
    """

    def __init__(self, held_out_classes: List[int]):
        """
        Args:
            held_out_classes: list of class labels to hold out as OOD
        """
        super().__init__()
        self.held_out_classes = held_out_classes

    def get_partition_labels(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """
        Assign each sample to a partition:
            - partition 0: ID (all classes except held-out)
            - partition 1..N: each held-out class as separate OOD partition
        """
        labels = pd.Series(index=y.index, dtype=int)

        # ID partition
        labels[~y.isin(self.held_out_classes)] = 0

        # OOD partitions
        for i, cls in enumerate(self.held_out_classes, start=1):
            labels[y == cls] = i

        return labels

    def get_partition_indices(self, X: pd.DataFrame, y: pd.Series) -> Dict[int, pd.Index]:
        """
        Convenience method: dict mapping partition_id -> sample indices
        """
        labels = self.get_partition_labels(X, y)
        partitions = {pid: labels.index[labels == pid] for pid in range(len(labels.unique()))}
        return partitions
