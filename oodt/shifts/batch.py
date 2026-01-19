import pandas as pd
import numpy as np
from typing import Dict, Optional


class BatchSampler:
    """
    Construct batches containing samples from multiple partitions (ID/OOD).
    Supports different training/testing modes for OOD experiments and
    keeps track of partition IDs for per-partition metrics computation.
    """

    def __init__(
        self,
        partitions: Dict[int, pd.Index],
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ):
        """
        Args:
            partitions: dict mapping partition_id -> sample indices
                        partition 0 is assumed to be ID
            batch_size: number of samples per batch
            shuffle: shuffle samples before batching
            random_state: optional seed for reproducibility
        """
        self.partitions = partitions
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

        # Flatten all indices for iteration
        self.all_indices = pd.Index(np.concatenate(list(partitions.values())))
        self._prepare_indices()

    def _prepare_indices(self):
        """Prepare the order of indices for simple iteration"""
        self.current_pointer = 0
        self.indices_order = self.all_indices.copy()
        if self.shuffle:
            self.indices_order = np.random.permutation(self.indices_order)

    def __iter__(self):
        self._prepare_indices()
        return self

    def __next__(self) -> pd.Index:
        if self.current_pointer >= len(self.indices_order):
            raise StopIteration

        end = min(self.current_pointer + self.batch_size, len(self.indices_order))
        batch_idx = self.indices_order[self.current_pointer:end]
        self.current_pointer = end
        return pd.Index(batch_idx)

    # ------------------------
    # Mode-based batch creation
    # ------------------------
    def get_mode_batch(
        self,
        mode: str = "train_id_test_id",
        train_fraction: float = 1.0,
        test_fraction: float = 1.0,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Dict[str, pd.Series]]:
        """
        Generate a batch according to the mode.

        Modes:
            "train_id_test_id"        -> train only ID, test only ID
            "train_id_test_ood"       -> train only ID, test only OOD
            "train_id_ood_test_all"   -> train ID+OOD, test ID+OOD
            "train_id_test_all"       -> train ID, test ID+OOD

        Args:
            mode: mode name
            train_fraction: fraction of train samples to include
            test_fraction: fraction of test samples to include
            batch_size: optional batch size (overrides self.batch_size)

        Returns:
            dict with keys "train" and "test", each containing:
                - "indices": pd.Index of samples
                - "partitions": pd.Series mapping index -> partition_id
        """
        if batch_size is None:
            batch_size = self.batch_size

        # Partition 0 = ID, others = OOD
        id_indices = self.partitions.get(0, pd.Index([]))
        ood_indices = pd.Index(np.concatenate([v for k,v in self.partitions.items() if k != 0]))

        # Shuffle
        if self.shuffle:
            id_indices = np.random.permutation(id_indices)
            ood_indices = np.random.permutation(ood_indices)

        # Apply fractions
        id_train = pd.Index(id_indices[: int(len(id_indices) * train_fraction)])
        id_test = pd.Index(id_indices[: int(len(id_indices) * test_fraction)])
        ood_train = pd.Index(ood_indices[: int(len(ood_indices) * train_fraction)])
        ood_test = pd.Index(ood_indices[: int(len(ood_indices) * test_fraction)])

        # Build batches based on mode
        mode = mode.lower()
        if mode == "train_id_test_id":
            train_idx = id_train
            test_idx = id_test
        elif mode == "train_id_test_ood":
            train_idx = id_train
            test_idx = ood_test
        elif mode == "train_id_ood_test_all":
            train_idx = pd.Index(np.concatenate([id_train, ood_train]))
            test_idx = pd.Index(np.concatenate([id_test, ood_test]))
        elif mode == "train_id_test_all":
            train_idx = id_train
            test_idx = pd.Index(np.concatenate([id_test, ood_test]))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Apply batch size limit
        train_idx = train_idx[:batch_size]
        test_idx = test_idx[:batch_size]

        return {
            "train": {
                "indices": train_idx,
                "partitions": self._get_partition_ids(train_idx)
            },
            "test": {
                "indices": test_idx,
                "partitions": self._get_partition_ids(test_idx)
            }
        }

    # ------------------------
    # Helper: map indices -> partition IDs
    # ------------------------
    def _get_partition_ids(self, indices: pd.Index) -> pd.Series:
        """
        Return partition ID for each sample in the given indices
        """
        partition_labels = pd.Series(index=indices, dtype=int)
        for pid, idx in self.partitions.items():
            mask = np.isin(indices, idx)
            partition_labels[indices[mask]] = pid
        return partition_labels

