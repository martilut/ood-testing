from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from copy import deepcopy

from oodt.data.loaders import CSVDataset
from oodt.detection.cv_pipeline import OODCVPipeline
from oodt.data.preprocessing import Preprocessor

from oodt.detection.knn import KNNOODDetector
from oodt.detection.clustering import ClusteringOODDetector
from oodt.detection.energy import EnergyOODDetector

from oodt.detection.mf_kmeans_detector import MFKMeansOODDetector
from oodt.shifts.concept.mf_kmeans import MFKMeansShift

from oodt.utils.utils import get_project_path
from oodt.utils.plotting import plot_ood_classification


# ==================================================
# CONFIG
# ==================================================

PROJECT_ROOT = Path(get_project_path())
DATASETS_ROOT = PROJECT_ROOT / "datasets" / "partitions"
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "results" / "self"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ==================================================
# BASE DETECTORS
# ==================================================

base_detectors = {
    "knn_5": KNNOODDetector(k=5),
    "kmeans": ClusteringOODDetector(n_clusters=8),
    "energy": EnergyOODDetector(mode="density"),
}


# ==================================================
# MF-KMEANS GRID
# ==================================================

mf_names = ["mean", "sd", "var", "eigenvalues", "mut_inf", "attr_ent"]
clusters_list = [3]
modes = ["matrix"]

mf_detectors = {}

for mf, k, mode in product(mf_names, clusters_list, modes):

    shift = MFKMeansShift(
        mf_name=[mf],
        n_partitions=k,
        mode=mode,
        random_state=RANDOM_STATE,
        include_sample_features=True,  # matrix mode: append sample's own features
    )

    name = f"mf_kmeans_{mf}_k{k}_{mode}"

    mf_detectors[name] = MFKMeansOODDetector(shift)


# ==================================================
# ALL DETECTORS
# ==================================================

detectors = {
    **base_detectors,
    **mf_detectors
}


# ==================================================
# PIPELINE
# ==================================================

preprocessor = Preprocessor(
    scaling="standard",
    encoding="onehot",
)

cv_pipeline = OODCVPipeline(
    detectors=detectors,
    n_splits=5,
    mode=2,
    id_ratio=0.5,
    target_fpr=0.05,
    preprocessor=preprocessor,
    verbose=True,
)


# ==================================================
# HELPERS
# ==================================================

def load_dataset_from_folder(folder: Path) -> CSVDataset:
    csv_files = sorted(folder.glob("*.csv"))

    if len(csv_files) < 2:
        raise ValueError(f"{folder} must contain at least 2 csv files")

    sample_df = pd.read_csv(csv_files[0])
    target_col = sample_df.columns[-1]

    paths = {f.stem: f for f in csv_files}

    dataset = CSVDataset(
        path=paths,
        target_col=target_col,
        name=folder.name,
    )
    dataset.load()

    return dataset


def build_ood_mask(dataset: CSVDataset) -> np.ndarray:
    return dataset.ood_target != 0


# ==================================================
# PLOTTING PER DATASET (metrics)
# ==================================================

def plot_dataset(df, dataset_name, save_dir):
    metrics = [
        "auroc_mean",
        "aupr_mean",
        "fpr95_mean",
    ]

    for metric in metrics:
        plt.figure(figsize=(10, 4))

        df_sorted = df.sort_values(
            metric,
            ascending=(metric == "fpr95_mean")
        )

        x = np.arange(len(df_sorted))

        plt.bar(
            x,
            df_sorted[metric]
        )

        plt.xticks(x, df_sorted["detector"], rotation=90)
        plt.title(f"{dataset_name} — {metric}")
        plt.tight_layout()

        plt.savefig(save_dir / f"{dataset_name}_{metric}.png")
        plt.close()


# ==================================================
# PLOTTING GLOBAL (metrics)
# ==================================================

def plot_global(summary_df, save_dir):
    metrics = [
        "auroc_mean",
        "aupr_mean",
        "fpr95_mean",
    ]

    for metric in metrics:
        plt.figure(figsize=(10, 4))

        df_sorted = summary_df.sort_values(
            metric,
            ascending=(metric == "fpr95_mean")
        )

        x = np.arange(len(df_sorted))

        plt.bar(
            x,
            df_sorted[metric]
        )

        plt.xticks(x, df_sorted["detector"], rotation=90)
        plt.title(f"GLOBAL — {metric}")
        plt.tight_layout()

        plt.savefig(save_dir / f"GLOBAL_{metric}.png")
        plt.close()


# ==================================================
# MF-KMEANS DIAGNOSTIC PLOTS (per dataset)
# ==================================================

def _single_split(X, y, ood_mask, id_ratio=0.5, train_frac=0.7, seed=RANDOM_STATE):
    """Build one train/test split mirroring the CV pipeline (mode=2)."""
    rng = np.random.RandomState(seed)

    id_idx = np.where(~ood_mask)[0]
    ood_idx = np.where(ood_mask)[0]

    rng.shuffle(id_idx)
    n_train = int(len(id_idx) * train_frac)
    train_idx = id_idx[:n_train]
    test_id_part = id_idx[n_train:]

    n = len(test_id_part)
    n_id = int(n * id_ratio)
    n_ood = max(1, n - n_id)

    id_part = rng.choice(test_id_part, size=n_id, replace=True) if n_id > 0 else np.array([], dtype=int)
    ood_part = rng.choice(ood_idx, size=n_ood, replace=True) if len(ood_idx) > 0 else np.array([], dtype=int)

    test_idx = np.concatenate([id_part, ood_part])
    rng.shuffle(test_idx)

    return train_idx, test_idx


def run_diagnostic_plots(X, y, ood_mask, dataset_name, save_dir):
    """
    Train each MF-KMeans detector once on a single split and plot the OOD
    classification correctness in MF space.
    """
    diag_dir = save_dir / "mf_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    train_idx, test_idx = _single_split(X, y, ood_mask)

    X_train_raw = X.iloc[train_idx]
    y_train_raw = y.iloc[train_idx]
    X_test_raw = X.iloc[test_idx]
    ood_test_true = ood_mask[test_idx]

    # preprocess (same as CV pipeline)
    prep = deepcopy(preprocessor)
    X_train = prep.fit_transform(X_train_raw)
    X_test = prep.transform(X_test_raw)

    # use only ID samples for fitting (matches OODDetectionPipeline behaviour)
    train_ood_mask = ood_mask[train_idx]
    id_train_mask = ~train_ood_mask
    X_train_id = X_train.iloc[id_train_mask] if hasattr(X_train, "iloc") else X_train[id_train_mask]
    y_train_id = y_train_raw.iloc[id_train_mask] if hasattr(y_train_raw, "iloc") else y_train_raw[id_train_mask]

    if len(X_train_id) < 5 or len(X_test) < 2:
        print(f"  [diagnostic] skipping {dataset_name}: not enough samples")
        return

    for det_name, detector_template in mf_detectors.items():
        print(f"  [diagnostic] {det_name}")
        try:
            detector = deepcopy(detector_template)
            detector.fit(X_train_id, y_train_id)

            scores = detector.score_samples(X_test)

            # threshold = quantile of scores on a held-out ID portion
            # (simple approximation: target_fpr quantile of training scores)
            train_scores = detector.score_samples(X_train_id)
            thr = np.quantile(train_scores, 1.0 - 0.05)  # target_fpr=0.05
            ood_pred = (scores >= thr).astype(bool)

            mf_train = detector.shift.get_meta_info()["mf_space_train"]
            mf_test = detector.shift.project_samples(X_test)

            out_path = diag_dir / f"{det_name}_ood_classification.png"
            plot_ood_classification(
                mf_train=mf_train,
                mf_test=mf_test,
                ood_true=ood_test_true,
                ood_pred=ood_pred,
                scores=scores,
                title=f"{dataset_name} — {det_name}",
                path=out_path,
            )
        except Exception as e:
            print(f"  [diagnostic ERROR] {det_name}: {e}")


# ==================================================
# MAIN LOOP
# ==================================================

all_results = []

for dataset_folder in sorted(DATASETS_ROOT.iterdir()):
    if not dataset_folder.is_dir():
        continue

    name = dataset_folder.name
    print(f"\n\n========== DATASET: {name} ==========")

    out_dir = OUTPUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        dataset = load_dataset_from_folder(dataset_folder)

        X = dataset.data
        y = dataset.target
        ood_mask = build_ood_mask(dataset)

        df = cv_pipeline.run(X, y, ood_mask)
        df["dataset"] = name

        print(df)

        df.to_csv(out_dir / "results.csv", index=False)

        plot_dataset(df, name, out_dir)

        # ----- MF diagnostic plots (single-split, per dataset) -----
        print(f"\n--- MF diagnostic plots for {name} ---")
        run_diagnostic_plots(X, y, ood_mask, name, out_dir)

        all_results.append(df)

    except Exception as e:
        print(f"[ERROR] {name}: {e}")


# ==================================================
# FINAL CHECK
# ==================================================

if len(all_results) == 0:
    raise RuntimeError("No datasets were successfully processed.")

final_df = pd.concat(all_results, ignore_index=True)

print("\n\n================ FINAL RESULTS ================")
print(final_df)

final_df.to_csv(OUTPUT_ROOT / "all_results.csv", index=False)


# ==================================================
# GLOBAL SUMMARY
# ==================================================

summary = (
    final_df
    .groupby("detector")
    .agg(
        auroc_mean=("auroc_mean", "mean"),
        auroc_std=("auroc_mean", "std"),

        aupr_mean=("aupr_mean", "mean"),
        aupr_std=("aupr_mean", "std"),

        fpr95_mean=("fpr95_mean", "mean"),
        fpr95_std=("fpr95_mean", "std"),
    )
    .reset_index()
)

print("\n\n=========== SUMMARY ===========")
print(summary)

summary.to_csv(OUTPUT_ROOT / "summary.csv", index=False)


# ==================================================
# GLOBAL PLOTS
# ==================================================

plot_global(summary, OUTPUT_ROOT)
