import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from collections import defaultdict
from copy import deepcopy
from typing import Iterator, Tuple


# ==================================================
# SPLITTER
# ==================================================

class OODCVSplitter:
    """
    Cross-validation splitter with explicit ID/OOD control.
    """

    def __init__(
        self,
        n_splits: int = 5,
        mode: int = 2,
        id_ratio: float = 0.5,
        random_state: int = 42,
    ):
        assert mode in [0, 1, 2, 3]
        self.n_splits = n_splits
        self.mode = mode
        self.id_ratio = id_ratio
        self.random_state = random_state

    def split(self, X, y, ood_mask) -> Iterator[Tuple[np.ndarray, np.ndarray]]:

        rng = np.random.RandomState(self.random_state)

        id_idx = np.where(~ood_mask)[0]
        ood_idx = np.where(ood_mask)[0]

        kf = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )

        for train_id_idx, test_id_idx in kf.split(id_idx):

            train_idx = id_idx[train_id_idx]
            test_id_part = id_idx[test_id_idx]

            # -------------------------
            # TEST SET
            # -------------------------
            if self.mode == 0:
                test_idx = test_id_part

            elif self.mode == 1:
                test_idx = rng.choice(ood_idx, size=len(test_id_part), replace=True)

            elif self.mode == 2:
                n = len(test_id_part)
                n_id = int(n * self.id_ratio)
                n_ood = n - n_id

                id_part = rng.choice(test_id_part, size=n_id, replace=True)
                ood_part = rng.choice(ood_idx, size=n_ood, replace=True)

                test_idx = np.concatenate([id_part, ood_part])

            elif self.mode == 3:
                n = len(test_id_part)

                id_part = test_id_part
                ood_part = rng.choice(ood_idx, size=n, replace=True)

                test_idx = np.concatenate([id_part, ood_part])

                # add OOD to train
                n_ood_train = int(len(train_idx) * (1 - self.id_ratio))
                ood_train = rng.choice(ood_idx, size=n_ood_train, replace=True)
                train_idx = np.concatenate([train_idx, ood_train])

            rng.shuffle(test_idx)

            yield train_idx, test_idx


# ==================================================
# PIPELINE
# ==================================================

class OODCVPipeline:

    def __init__(
        self,
        detectors: dict,
        n_splits: int = 5,
        mode: int = 2,
        id_ratio: float = 0.5,
        target_fpr: float = 0.05,
        preprocessor=None,
        verbose: bool = True,
    ):
        self.detectors = detectors
        self.n_splits = n_splits
        self.mode = mode
        self.id_ratio = id_ratio
        self.target_fpr = target_fpr
        self.preprocessor = preprocessor
        self.verbose = verbose

    # ==================================================
    # RUN
    # ==================================================

    def run(self, X, y, ood_mask):

        splitter = OODCVSplitter(
            n_splits=self.n_splits,
            mode=self.mode,
            id_ratio=self.id_ratio,
        )

        all_results = defaultdict(list)

        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, ood_mask)):

            if self.verbose:
                print(f"\n=== Fold {fold + 1}/{self.n_splits} ===")

            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            ood_test = ood_mask[test_idx]

            # -------------------------
            # preprocess
            # -------------------------
            if self.preprocessor:
                prep = deepcopy(self.preprocessor)
                X_train = prep.fit_transform(X_train)
                X_test = prep.transform(X_test)

            # -------------------------
            # VALIDATION SPLIT (FIXED)
            # -------------------------
            train_id_idx = train_idx[~ood_mask[train_idx]]

            rng = np.random.RandomState(42 + fold)

            if len(train_id_idx) < 2:
                raise ValueError("Not enough ID samples for validation split")

            val_size = max(1, int(0.2 * len(train_id_idx)))

            val_id_idx = rng.choice(train_id_idx, size=val_size, replace=False)
            train_id_idx = np.setdiff1d(train_id_idx, val_id_idx)

            X_train = X.iloc[train_id_idx]
            y_train = y.iloc[train_id_idx]
            X_val = X.iloc[val_id_idx]

            # -------------------------
            # PIPELINE
            # -------------------------
            from oodt.detection.pipeline import OODDetectionPipeline

            pipeline = OODDetectionPipeline(
                detectors={k: deepcopy(v) for k, v in self.detectors.items()},
                target_fpr=self.target_fpr,
                verbose=self.verbose,
            )

            results = pipeline.run(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                X_test=X_test,
                ood_mask=ood_test,
            )

            # -------------------------
            # STORE RAW METRICS (NOT OBJECTS)
            # -------------------------
            for name, res in results.items():
                all_results[name].append({
                    "auroc": float(res.auroc),
                    "aupr": float(res.aupr),
                    "fpr95": float(res.fpr95),
                    "det_acc": float(res.det_acc) if res.det_acc is not None else None
                })

        return self._aggregate(all_results)

    # ==================================================
    # FIXED AGGREGATION (REAL STD GUARANTEED)
    # ==================================================

    def _aggregate(self, all_results):

        def mean_std(vals):
            vals = np.array(vals, dtype=float)
            vals = vals[~np.isnan(vals)]

            if len(vals) == 0:
                return None, None
            if len(vals) == 1:
                return float(vals[0]), 0.0

            return float(vals.mean()), float(vals.std(ddof=1))

        rows = []

        for name, res_list in all_results.items():

            aurocs = [r["auroc"] for r in res_list]
            auprs = [r["aupr"] for r in res_list]
            fpr95s = [r["fpr95"] for r in res_list]

            accs = [r["det_acc"] for r in res_list if r["det_acc"] is not None]

            auroc_mean, auroc_std = mean_std(aurocs)
            aupr_mean, aupr_std = mean_std(auprs)
            fpr95_mean, fpr95_std = mean_std(fpr95s)
            acc_mean, acc_std = mean_std(accs)

            rows.append({
                "detector": name,

                "auroc_mean": auroc_mean,
                "auroc_std": auroc_std,

                "aupr_mean": aupr_mean,
                "aupr_std": aupr_std,

                "fpr95_mean": fpr95_mean,
                "fpr95_std": fpr95_std,

                "det_acc_mean": acc_mean,
                "det_acc_std": acc_std,
            })

        df = pd.DataFrame(rows)

        return df.sort_values("auroc_mean", ascending=False).reset_index(drop=True)