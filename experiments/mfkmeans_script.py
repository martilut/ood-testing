"""
mfkmeans_script.py — compares MF-KMeans variants on partition datasets.

Variants compared (NEW):
    1. raw_kmeans       — plain KMeans on raw source features (no MF)
    2. mf_single_<mf>   — MF-KMeans with a SINGLE metafeature + source features
                          (original behaviour; one entry per name in `mf_names`)
    3. mf_multi_all     — MF-KMeans with the FULL LIST of metafeatures
                          + source features

Layout for results:
    results_dir / dataset_name / <variant_name> / clusters_<k> / <mode>

Notes
-----
- `raw_kmeans` is independent of `mode` (no MF space is built); it is run
  once per cluster count and saved under mode="none".
- All variants share the same splitter and metrics so the resulting
  `summary_metrics.csv` is directly comparable across variants.
- MF-DBSCAN / MF-Hierarchical are NOT included here because their cluster
  count is data-dependent and cannot guarantee the `n_partitions` required
  by `OODPipeline`'s splitter. They are only used in `detection.py`.
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from oodt.data.loaders import CSVDataset
from oodt.metrics.metrics import MetricsEvaluator
from oodt.pipelines.pipeline_builder import OODPipeline
from oodt.shifts.concept.mf_kmeans import MFKMeansShift
from oodt.shifts.concept.raw_kmeans import RawKMeansShift
from oodt.splitting.splitter import TrainTestSplitter
from oodt.utils.plotting import plot_mf_space, compute_distance_stats
from oodt.utils.utils import get_project_path

# =========================
# Experiment Settings
# =========================
datasets_dir = Path(get_project_path()) / Path("datasets/partitions")
mf_names = ["mean", "sd", "eigenvalues", "mut_inf", "attr_ent"]
clusters_list = [3]
modes = ["vector", "matrix"]

results_dir = Path(get_project_path()) / "experiment_multiple"
results_dir.mkdir(exist_ok=True, parents=True)

summary_records = []


# =========================
# Build variant grid
# =========================

def build_variants(mf_names, clusters_list, modes):
    """
    Yield (variant_name, mf_list_or_None, n_clusters, mode) tuples.

    - variant_name == "raw_kmeans" with mode == "none" -> RawKMeansShift
    - variant_name == "mf_single_<mf>" -> MFKMeansShift(mf_name=[<mf>])
    - variant_name == "mf_multi_all"   -> MFKMeansShift(mf_name=mf_names)
    """
    for n_clusters in clusters_list:
        # # 1) Plain KMeans on raw features (no MF, no mode)
        # yield ("raw_kmeans", None, n_clusters, "none")
        #
        # # 2) MF-KMeans with a single metafeature  (original behaviour)
        # for mf in mf_names:
        #     for mode in modes:
        #         yield (f"mf_single_{mf}", [mf], n_clusters, mode)

        # 3) MF-KMeans with the full list of metafeatures
        for mode in modes:
            yield ("mf_multi_all", list(mf_names), n_clusters, mode)


def build_shift(variant_name, mf_list, n_clusters, mode):
    """Instantiate the shift strategy for a given variant tuple."""
    if variant_name == "raw_kmeans":
        return RawKMeansShift(
            n_partitions=n_clusters,
            random_state=42,
        )

    if mode == "vector":
        return MFKMeansShift(
            mf_name=mf_list,
            n_partitions=n_clusters,
            random_state=42,
            mode="vector",
        )

    # matrix
    return MFKMeansShift(
        mf_name=mf_list,
        n_partitions=n_clusters,
        random_state=42,
        mode="matrix",
        n_bins=100,
    )


# =========================
# Main loop
# =========================

for dataset_folder in datasets_dir.iterdir():
    if not dataset_folder.is_dir():
        continue

    dataset_name = dataset_folder.name
    print(f"\n=== Running experiments for dataset: {dataset_name} ===")

    # Detect target column (last column in source CSV)
    source_csv = dataset_folder / "source.csv"
    df_source = pd.read_csv(source_csv)
    target_col = df_source.columns[-1]

    # Load dataset
    paths = {
        "source": dataset_folder / "source.csv",
        "target": dataset_folder / "target.csv",
    }
    dataset = CSVDataset(
        path=paths,
        target_col=target_col,
        name=dataset_name,
        has_index=True,
    )
    dataset.load()

    # Loop through configurations
    for variant_name, mf_list, n_clusters, mode in build_variants(
        mf_names, clusters_list, modes
    ):
        save_dir = (
            results_dir
            / dataset_name
            / variant_name
            / f"clusters_{n_clusters}"
            / mode
        )
        plot_path = save_dir / "plot.png"
        stats_path = save_dir / "stats.csv"
        metrics_path = save_dir / "metrics.csv"

        # Skip if results already exist
        if plot_path.exists() and stats_path.exists() and metrics_path.exists():
            print(
                f"\n-- Skipping existing experiment: variant={variant_name}, "
                f"clusters={n_clusters}, mode={mode}"
            )
            continue

        print(
            f"\n-- Running experiment: variant={variant_name}, "
            f"clusters={n_clusters}, mode={mode} --"
        )

        # Initialize shift strategy
        shift_strategy = build_shift(variant_name, mf_list, n_clusters, mode)

        # Initialize splitter
        splitter = TrainTestSplitter(
            partitions={},
            mode=2,
            train_ratio=0.7,
            test_ratio=0.3,
            id_partitions=[0],
            ood_partitions=[1],
            stratify=True,
            random_state=42,
        )

        # Initialize model
        model = LogisticRegression(max_iter=1000, n_jobs=-1)
        metrics = MetricsEvaluator(task="classification")

        # Initialize pipeline
        pipeline = OODPipeline(
            model=model,
            shift_strategy=shift_strategy,
            splitter=splitter,
            metrics=metrics,
            mode="known_ood",
        )

        # Run pipeline
        result = pipeline.run(dataset=dataset)

        # =========================
        # Prepare folder structure
        # =========================
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save plots
        plot_mf_space(result, path=plot_path)

        # Save stats
        stats = compute_distance_stats(result)
        pd.DataFrame(stats).to_csv(stats_path, index=False)

        # =========================
        # Save metrics
        # =========================
        metrics_dict = {}
        if hasattr(result, "metrics") and result.metrics is not None:
            for k, v in result.metrics.global_metrics.items():
                metrics_dict[f"global_{k}"] = v
            for k, v in result.metrics.id_metrics.items():
                metrics_dict[f"id_{k}"] = v
            for k, v in result.metrics.ood_metrics.items():
                metrics_dict[f"ood_{k}"] = v

        # Flatten stats and merge into metrics_dict
        for k, v in stats.items():
            metrics_dict[f"stat_{k}"] = v

        # Save metrics CSV
        metrics_df = pd.DataFrame([metrics_dict])
        metrics_df.to_csv(metrics_path, index=False)

        print(f"Saved plot: {plot_path}")
        print(f"Saved stats: {stats_path}")
        print(f"Saved metrics: {metrics_path}")

        # =========================
        # Add to summary
        # =========================
        metrics_dict.update({
            "dataset": dataset_name,
            "variant": variant_name,
            "mf_list": ",".join(mf_list) if mf_list else "",
            "n_clusters": n_clusters,
            "mode": mode,
        })
        summary_records.append(metrics_dict)

# =========================
# Save summary CSV
# =========================
summary_df = pd.DataFrame(summary_records)
summary_csv_path = results_dir / "summary_metrics.csv"
summary_df.to_csv(summary_csv_path, index=False)
print(f"\nAll summary metrics (including stats) saved to: {summary_csv_path}")