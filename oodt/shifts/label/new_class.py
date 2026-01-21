from typing import Optional

import pandas as pd

from oodt.shifts.base import BaseShiftStrategy


class NewClassShift(BaseShiftStrategy):
    def __init__(
        self,
        n_classes: int,
        name: Optional[str] = None,
        random_state: Optional[int] = None,
    ):
        self.n_classes = n_classes

        super().__init__(
            name=name or "NewClassShift",
            random_state=random_state,
            n_partitions=self.n_classes,
        )

    def get_partition_labels(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        return y.astype(int)
