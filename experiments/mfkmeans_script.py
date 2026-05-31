import itertools
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from oodt.data.loaders import CSVDataset
from oodt.metrics.metrics import MetricsEvaluator
from oodt.pipelines.pipeline_builder import OODPipeline
from oodt.shifts.concept.mf_kmeans import MFKMeansShift
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

results_dir = Path(get_project_path()) / "experiment_plotting"
results_dir.mkdir(exist_ok=True, parents=True)

summary_records = []

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
    for mf_name, n_clusters, mode in itertools.product(mf_names, clusters_list, modes):
        save_dir = results_dir / dataset_name / mf_name / f"clusters_{n_clusters}" / mode
        plot_path = save_dir / "plot.png"
        stats_path = save_dir / "stats.csv"
        metrics_path = save_dir / "metrics.csv"

        # Skip if results already exist
        if plot_path.exists() and stats_path.exists() and metrics_path.exists():
            print(f"\n-- Skipping existing experiment: MF={mf_name}, clusters={n_clusters}, mode={mode}")
            continue

        print(f"\n-- Running experiment: MF={mf_name}, clusters={n_clusters}, mode={mode} --")

        # Initialize shift strategy
        if mode == "vector":
            shift_strategy = MFKMeansShift(
                mf_name=[mf_name],
                n_partitions=n_clusters,
                random_state=42,
                mode="vector",
            )
        else:
            shift_strategy = MFKMeansShift(
                mf_name=[mf_name],
                n_partitions=n_clusters,
                random_state=42,
                mode="matrix",
                n_bins=100,
            )

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
            mode="known_ood"
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
            "mf_name": mf_name,
            "n_clusters": n_clusters,
            "mode": mode
        })
        summary_records.append(metrics_dict)

# =========================
# Save summary CSV
# =========================
summary_df = pd.DataFrame(summary_records)
summary_csv_path = results_dir / "summary_metrics.csv"
summary_df.to_csv(summary_csv_path, index=False)
print(f"\nAll summary metrics (including stats) saved to: {summary_csv_path}")