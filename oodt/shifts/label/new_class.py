from typing import List, Optional

import pandas as pd

from oodt.shifts.base import BaseShiftStrategy


class NewClassShift(BaseShiftStrategy):
    def __init__(
        self,
        held_out_classes: List[int],
        name: Optional[str] = None,
        random_state: Optional[int] = None,
    ):
        self.held_out_classes = held_out_classes

        super().__init__(
            name=name or "NewClassShift",
            random_state=random_state,
            n_partitions=1 + len(self.held_out_classes),
        )

    def get_partition_labels(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        labels = pd.Series(index=y.index, dtype=int)

        labels[~y.isin(self.held_out_classes)] = 0

        for i, cls in enumerate(self.held_out_classes, start=1):
            labels[y == cls] = i

        return labels
