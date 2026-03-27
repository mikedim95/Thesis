from __future__ import annotations

import contextlib
import importlib.util
import io
import math
import re
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor


MODULE_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = MODULE_DIR.parent
DATASETS_ROOT = PYTHON_ROOT / "datasets"
VIRGIN_DATASET_DIR = DATASETS_ROOT / "virgin"
REFORMED_DATASET_DIR = DATASETS_ROOT / "reformed"
RESULT_FILES = {
    "IForest": MODULE_DIR / "ISOLATION_FOREST_method.txt",
    "LOF": MODULE_DIR / "LOCAL_OUTLIER_FACTOR_method.txt",
    "SAND": MODULE_DIR / "SUBSEQUENCE_ANOMALY_DETECTION_method.txt",
}
FIGURE_OUTPUT_DIR = MODULE_DIR / "presentation_figures"

DISPLAY_NAME_MAP = {
    "IForest": "Isolation Forest",
    "LOF": "Local Outlier Factor",
    "SAND": "Subsequence Anomaly Detection",
}

RESULT_LINE_PATTERN = re.compile(
    r"^fileName:(?P<dataset_name>[^,]+), "
    r"AUC:(?P<auc>[^,]+), R_AUC:(?P<range_auc>[^,]+), "
    r"Precision:(?P<precision>[^,]+), Recall:(?P<recall>[^,]+), F:(?P<f1>[^,]+), "
    r"ExistenceReward:(?P<existence_reward>[^,]+), OverlapReward:(?P<overlap_reward>[^,]+), "
    r"AP:(?P<ap>[^,]+), R_AP:(?P<range_ap>[^,]+), Precisionk:(?P<precision_at_k>[^,]+), "
    r"R_precision:(?P<range_precision>[^,]+), R_recall:(?P<range_recall>[^,]+), R_f:(?P<range_f1>[^,]+), "
    r"tn_count:(?P<tn_count>\d+), fn_count:(?P<fn_count>\d+), fp_count:(?P<fp_count>\d+), tp_count:(?P<tp_count>\d+), "
    r"elapsed_time:(?P<elapsed_time_sec>[^ ]+) seconds, datapoint_count:(?P<datapoint_count>\d+)$"
)

METRIC_LABELS = {
    "auc": "AUC",
    "range_auc": "Range AUC",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "ap": "Average Precision",
    "elapsed_time_per_1000": "Seconds / 1000 points",
}

DEMO_DATASETS = [
    "031_UCR_Anomaly_DISTORTEDInternalBleeding20_2700_5759_5919",
    "250_UCR_Anomaly_weallwalk_2951_7290_7296",
]

ALGORITHM_SOURCE_MAP = {
    "IForest": PYTHON_ROOT / "IFORESTAnomalyDetection" / "Utils" / "iforest.py",
    "LOF": PYTHON_ROOT / "LOFAnomalyDetection" / "Utils" / "lof.py",
    "SAND": PYTHON_ROOT / "newSANDEffort" / "Utils" / "sand.py",
}

PIPELINE_FILE_MAP = {
    "Virgin datasets": VIRGIN_DATASET_DIR,
    "Reformed datasets": REFORMED_DATASET_DIR,
    "Reform script": DATASETS_ROOT / "reform.py",
    "Isolation Forest results": RESULT_FILES["IForest"],
    "LOF results": RESULT_FILES["LOF"],
    "SAND results": RESULT_FILES["SAND"],
    "Notebook builder": MODULE_DIR / "build_final_presentation_notebook.py",
    "Presentation notebook": MODULE_DIR / "final_anomaly_detection_presentation.ipynb",
}

NORMALIZATION_METHODS = ("none", "zscore", "minmax", "robust")


def set_presentation_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (14, 8),
            "figure.facecolor": "#fffaf3",
            "axes.facecolor": "#fffdf8",
            "axes.edgecolor": "#d8d0c6",
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "--",
            "font.size": 11,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.facecolor": "#fffaf3",
        }
    )


def _save_figure(fig: plt.Figure, save_path: Path | None) -> None:
    if save_path is None:
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220)


def parse_results_file(path: Path, algorithm: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        match = RESULT_LINE_PATTERN.match(line.strip())
        if not match:
            continue

        row = match.groupdict()
        numeric_fields = {
            "auc",
            "range_auc",
            "precision",
            "recall",
            "f1",
            "existence_reward",
            "overlap_reward",
            "ap",
            "range_ap",
            "precision_at_k",
            "range_precision",
            "range_recall",
            "range_f1",
            "elapsed_time_sec",
        }
        int_fields = {
            "tn_count",
            "fn_count",
            "fp_count",
            "tp_count",
            "datapoint_count",
        }
        for field in numeric_fields:
            row[field] = float(row[field])
        for field in int_fields:
            row[field] = int(row[field])

        row["algorithm"] = algorithm
        row["algorithm_display"] = DISPLAY_NAME_MAP[algorithm]
        row["elapsed_time_per_1000"] = (
            row["elapsed_time_sec"] * 1000.0 / row["datapoint_count"]
        )
        row.update(parse_dataset_name(row["dataset_name"]))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.sort_values(["dataset_sequence", "algorithm"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_all_results() -> pd.DataFrame:
    frames = [parse_results_file(path, algorithm) for algorithm, path in RESULT_FILES.items()]
    return pd.concat(frames, ignore_index=True)


def parse_dataset_name(dataset_name: str) -> dict[str, Any]:
    parts = dataset_name.split("_")
    if len(parts) < 7:
        raise ValueError(f"Unexpected dataset name format: {dataset_name}")

    sequence = int(parts[0])
    core_name = "_".join(parts[3:-3])
    train_until = int(parts[-3])
    anomaly_start = int(parts[-2])
    anomaly_end = int(parts[-1])

    variant = "RAW"
    clean_name = core_name
    if core_name.startswith("DISTORTED"):
        variant = "DISTORTED"
        clean_name = core_name[len("DISTORTED") :]
    elif core_name.startswith("NOISE"):
        variant = "NOISE"
        clean_name = core_name[len("NOISE") :]

    family = re.sub(r"\d+$", "", clean_name) or clean_name
    anomaly_length = max(0, anomaly_end - anomaly_start + 1)

    return {
        "dataset_sequence": sequence,
        "dataset_core_name": core_name,
        "dataset_clean_name": clean_name,
        "dataset_family": family,
        "dataset_variant": variant,
        "train_until": train_until,
        "anomaly_start": anomaly_start,
        "anomaly_end": anomaly_end,
        "anomaly_length": anomaly_length,
    }


def build_dataset_catalog(results: pd.DataFrame) -> pd.DataFrame:
    catalog = (
        results.sort_values(["dataset_sequence", "algorithm"])
        .drop_duplicates("dataset_name")
        .copy()
    )
    catalog["anomaly_ratio"] = catalog["anomaly_length"] / catalog["datapoint_count"]
    return catalog


def build_algorithm_summary(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby(["algorithm", "algorithm_display"], as_index=False)
        .agg(
            dataset_count=("dataset_name", "nunique"),
            median_auc=("auc", "median"),
            median_range_auc=("range_auc", "median"),
            median_precision=("precision", "median"),
            median_recall=("recall", "median"),
            median_f1=("f1", "median"),
            median_ap=("ap", "median"),
            median_time_per_1000=("elapsed_time_per_1000", "median"),
            mean_time_per_1000=("elapsed_time_per_1000", "mean"),
        )
        .sort_values("median_f1", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def repo_layout_frame() -> pd.DataFrame:
    rows = []
    for label, path in PIPELINE_FILE_MAP.items():
        rows.append(
            {
                "Component": label,
                "Path": str(path),
                "Exists": path.exists(),
            }
        )
    for algorithm, path in ALGORITHM_SOURCE_MAP.items():
        rows.append(
            {
                "Component": f"{DISPLAY_NAME_MAP[algorithm]} source",
                "Path": str(path),
                "Exists": path.exists(),
            }
        )
    return pd.DataFrame(rows)


def default_algorithm_parameters() -> dict[str, dict[str, Any]]:
    return {
        "IForest": {
            "n_estimators": 100,
            "max_samples": "auto",
            "contamination": 0.1,
            "max_features": 1.0,
            "bootstrap": False,
            "random_state": 42,
        },
        "LOF": {
            "n_neighbors": 20,
            "contamination": 0.1,
            "algorithm": "auto",
            "leaf_size": 30,
            "metric": "minkowski",
            "p": 2,
        },
        "SAND": {
            "k": None,
            "alpha": 0.5,
            "init_length": 5000,
            "batch_size": 2000,
            "subsequence_multiplier": 4,
            "overlapping_rate": None,
            "prefer_new_implementation": True,
        },
    }


def algorithm_parameter_frame(parameters: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for algorithm in ["IForest", "LOF", "SAND"]:
        for name, value in parameters.get(algorithm, {}).items():
            rows.append(
                {
                    "Algorithm": DISPLAY_NAME_MAP[algorithm],
                    "Parameter": name,
                    "Value": value,
                }
            )
    return pd.DataFrame(rows)


def algorithm_overview_frame(summary: pd.DataFrame) -> pd.DataFrame:
    summary_map = {
        row["algorithm"]: row for _, row in summary.iterrows()
    }
    rows = [
        {
            "Algorithm": "Isolation Forest",
            "Core Idea": "Randomly isolate window embeddings with tree partitions.",
            "Temporal Awareness": "Indirect, via sliding windows.",
            "Edge Profile": "Fast and lightweight.",
            "Median F1": summary_map["IForest"]["median_f1"],
            "Median AUC": summary_map["IForest"]["median_auc"],
            "Median s / 1000 pts": summary_map["IForest"]["median_time_per_1000"],
        },
        {
            "Algorithm": "Local Outlier Factor",
            "Core Idea": "Compare local density of each window with its neighbours.",
            "Temporal Awareness": "Indirect, via sliding windows.",
            "Edge Profile": "Accurate but more distance-heavy.",
            "Median F1": summary_map["LOF"]["median_f1"],
            "Median AUC": summary_map["LOF"]["median_auc"],
            "Median s / 1000 pts": summary_map["LOF"]["median_time_per_1000"],
        },
        {
            "Algorithm": "Subsequence AD / SAND",
            "Core Idea": "Cluster and compare subsequences to a streaming normal model.",
            "Temporal Awareness": "Direct, subsequence-native.",
            "Edge Profile": "Best temporal context, highest runtime risk.",
            "Median F1": summary_map["SAND"]["median_f1"],
            "Median AUC": summary_map["SAND"]["median_auc"],
            "Median s / 1000 pts": summary_map["SAND"]["median_time_per_1000"],
        },
    ]
    return pd.DataFrame(rows)


def plot_algorithm_cards(summary: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Algorithm Portraits", y=1.02, fontsize=20, fontweight="bold")

    color_map = {
        "IForest": "#385170",
        "LOF": "#7d5a50",
        "SAND": "#d97b29",
    }
    descriptions = {
        "IForest": (
            "Isolation Forest",
            "Principle\nRandom partitions isolate anomalous windows in fewer steps.",
            "Temporal handling\nUses sliding-window embeddings rather than raw points.",
            "Practical read\nGood baseline when speed matters more than rich sequence context.",
        ),
        "LOF": (
            "Local Outlier Factor",
            "Principle\nFlags windows whose local density is much lower than nearby windows.",
            "Temporal handling\nAlso relies on sliding-window embeddings.",
            "Practical read\nOften stronger on local deviations, but distance computations scale less kindly.",
        ),
        "SAND": (
            "Subsequence AD / SAND",
            "Principle\nMatches streaming subsequences against a weighted normal-pattern model.",
            "Temporal handling\nNative subsequence reasoning, not just embedded points.",
            "Practical read\nMost expressive for collective anomalies, but the heaviest method to deploy.",
        ),
    }

    summary_map = {row["algorithm"]: row for _, row in summary.iterrows()}
    for axis, algorithm in zip(axes, ["IForest", "LOF", "SAND"]):
        axis.set_axis_off()
        card = FancyBboxPatch(
            (0.02, 0.03),
            0.96,
            0.94,
            boxstyle="round,pad=0.02,rounding_size=18",
            linewidth=1.5,
            facecolor="#fff7ed",
            edgecolor=color_map[algorithm],
            transform=axis.transAxes,
        )
        axis.add_patch(card)

        title, principle, temporal, practical = descriptions[algorithm]
        stats = summary_map[algorithm]
        y = 0.88
        axis.text(
            0.08,
            y,
            title,
            transform=axis.transAxes,
            fontsize=16,
            fontweight="bold",
            color=color_map[algorithm],
        )
        y -= 0.14
        for block in (principle, temporal, practical):
            axis.text(0.08, y, block, transform=axis.transAxes, va="top", fontsize=10.5)
            y -= 0.22
        axis.text(
            0.08,
            0.18,
            (
                f"Median AUC: {stats['median_auc']:.2f}\n"
                f"Median F1: {stats['median_f1']:.2f}\n"
                f"Median s / 1000 pts: {stats['median_time_per_1000']:.2f}"
            ),
            transform=axis.transAxes,
            fontsize=10.5,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "#ffffff", "edgecolor": "#eadbc8"},
        )

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_dataset_landscape(catalog: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Dataset Landscape", fontsize=20, fontweight="bold", y=1.01)

    lengths = catalog["datapoint_count"].to_numpy()
    ratios = catalog["anomaly_ratio"].to_numpy()

    axes[0, 0].hist(lengths, bins=24, color="#4c78a8", edgecolor="white")
    axes[0, 0].axvline(
        np.median(lengths),
        color="#f58518",
        linestyle="--",
        linewidth=2,
        label=f"Median = {np.median(lengths):,.0f}",
    )
    axes[0, 0].set_title("Dataset Length Distribution")
    axes[0, 0].set_xlabel("Datapoints")
    axes[0, 0].set_ylabel("Dataset count")
    axes[0, 0].legend()

    positive_ratios = np.where(ratios > 0, ratios, np.nan)
    epsilon = np.nanmin(positive_ratios) / 2.0
    ratio_plot = np.where(ratios > 0, ratios, epsilon)
    axes[0, 1].hist(
        ratio_plot,
        bins=np.logspace(np.log10(epsilon), np.log10(ratio_plot.max()), 28),
        color="#72b7b2",
        edgecolor="white",
    )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_title("Anomaly Ratio Distribution")
    axes[0, 1].set_xlabel("Anomaly ratio (log scale)")
    axes[0, 1].set_ylabel("Dataset count")

    variant_colors = {"RAW": "#6b7280", "DISTORTED": "#d97706", "NOISE": "#2563eb"}
    for variant, subset in catalog.groupby("dataset_variant"):
        axes[1, 0].scatter(
            subset["datapoint_count"],
            subset["anomaly_ratio"],
            s=45,
            alpha=0.8,
            color=variant_colors.get(variant, "#6b7280"),
            label=variant,
            edgecolors="white",
            linewidths=0.5,
        )
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_title("Length vs Anomaly Ratio")
    axes[1, 0].set_xlabel("Datapoints (log scale)")
    axes[1, 0].set_ylabel("Anomaly ratio")
    axes[1, 0].legend()

    variant_counts = (
        catalog["dataset_variant"]
        .value_counts()
        .reindex(["RAW", "DISTORTED", "NOISE"])
        .fillna(0)
    )
    axes[1, 1].bar(
        variant_counts.index,
        variant_counts.values,
        color=[variant_colors.get(name, "#6b7280") for name in variant_counts.index],
    )
    axes[1, 1].set_title("Benchmark Composition by Variant")
    axes[1, 1].set_xlabel("Variant")
    axes[1, 1].set_ylabel("Dataset count")

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_top_families(catalog: pd.DataFrame, top_n: int = 12, save_path: Path | None = None) -> plt.Figure:
    family_counts = catalog["dataset_family"].value_counts().head(top_n).sort_values()
    fig, axis = plt.subplots(figsize=(12, 7))
    axis.barh(family_counts.index, family_counts.values, color="#8c6d62")
    axis.set_title(f"Top {top_n} Dataset Families in the Benchmark")
    axis.set_xlabel("Dataset count")
    axis.set_ylabel("Family")
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_median_metric_heatmap(summary: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    metrics = [
        ("median_auc", "AUC"),
        ("median_range_auc", "Range AUC"),
        ("median_precision", "Precision"),
        ("median_recall", "Recall"),
        ("median_f1", "F1"),
        ("median_ap", "AP"),
    ]
    heat = summary.set_index("algorithm_display")[[column for column, _ in metrics]]
    labels = [label for _, label in metrics]

    fig, axis = plt.subplots(figsize=(11, 4.8))
    cmap = LinearSegmentedColormap.from_list("paper", ["#fff7ed", "#f7b267", "#d35400"])
    image = axis.imshow(heat.to_numpy(), cmap=cmap, aspect="auto")
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels)
    axis.set_yticks(range(len(heat.index)))
    axis.set_yticklabels(heat.index)
    axis.set_title("Median Benchmark Metrics by Algorithm")

    for row_idx in range(heat.shape[0]):
        for col_idx in range(heat.shape[1]):
            axis.text(
                col_idx,
                row_idx,
                f"{heat.iloc[row_idx, col_idx]:.2f}",
                ha="center",
                va="center",
                color="#1f2937",
                fontsize=10,
            )

    fig.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_metric_boxplots(results: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    metrics = ["auc", "range_auc", "precision", "recall", "f1", "ap"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    palette = ["#4c78a8", "#f58518", "#54a24b"]

    for axis, metric in zip(axes.ravel(), metrics):
        values = [
            results.loc[results["algorithm"] == algorithm, metric].to_numpy()
            for algorithm in ["IForest", "LOF", "SAND"]
        ]
        box = axis.boxplot(
            values,
            patch_artist=True,
            labels=[DISPLAY_NAME_MAP[key] for key in ["IForest", "LOF", "SAND"]],
        )
        for patch, color in zip(box["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        axis.set_title(f"{METRIC_LABELS[metric]} distribution")
        axis.tick_params(axis="x", rotation=10)

    fig.suptitle(
        "Metric Distributions Across the Full Benchmark",
        fontsize=19,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_runtime_analysis(results: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    colors = {"IForest": "#4c78a8", "LOF": "#f58518", "SAND": "#54a24b"}

    runtime_data = [
        results.loc[results["algorithm"] == algorithm, "elapsed_time_per_1000"].to_numpy()
        for algorithm in ["IForest", "LOF", "SAND"]
    ]
    box = axes[0].boxplot(
        runtime_data,
        patch_artist=True,
        labels=[DISPLAY_NAME_MAP[key] for key in ["IForest", "LOF", "SAND"]],
    )
    for patch, algorithm in zip(box["boxes"], ["IForest", "LOF", "SAND"]):
        patch.set_facecolor(colors[algorithm])
        patch.set_alpha(0.65)
    axes[0].set_yscale("log")
    axes[0].set_title("Runtime per 1000 Datapoints")
    axes[0].set_ylabel("Seconds (log scale)")

    for algorithm in ["IForest", "LOF", "SAND"]:
        subset = results.loc[results["algorithm"] == algorithm]
        axes[1].scatter(
            subset["datapoint_count"],
            subset["elapsed_time_sec"],
            label=DISPLAY_NAME_MAP[algorithm],
            s=24,
            alpha=0.6,
            color=colors[algorithm],
        )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("Runtime Scaling Against Dataset Size")
    axes[1].set_xlabel("Datapoints (log scale)")
    axes[1].set_ylabel("Elapsed time in seconds (log scale)")
    axes[1].legend()

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def compute_win_table(results: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "auc": True,
        "range_auc": True,
        "precision": True,
        "recall": True,
        "f1": True,
        "ap": True,
        "elapsed_time_per_1000": False,
    }
    win_counts: dict[tuple[str, str], float] = {}

    for _, dataset_frame in results.groupby("dataset_name"):
        for metric, higher_is_better in metrics.items():
            values = dataset_frame[["algorithm", metric]].copy()
            best_value = values[metric].max() if higher_is_better else values[metric].min()
            winners = values.loc[np.isclose(values[metric], best_value, atol=1e-12), "algorithm"]
            contribution = 1.0 / len(winners)
            for algorithm in winners:
                win_counts[(algorithm, metric)] = win_counts.get((algorithm, metric), 0.0) + contribution

    rows = []
    for algorithm in ["IForest", "LOF", "SAND"]:
        row = {"algorithm": algorithm, "algorithm_display": DISPLAY_NAME_MAP[algorithm]}
        for metric in metrics:
            row[metric] = win_counts.get((algorithm, metric), 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_win_heatmap(results: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    win_table = compute_win_table(results).set_index("algorithm_display")
    metric_order = ["auc", "range_auc", "precision", "recall", "f1", "ap", "elapsed_time_per_1000"]
    labels = [METRIC_LABELS[key] for key in metric_order]
    values = win_table[metric_order]

    fig, axis = plt.subplots(figsize=(12, 4.5))
    cmap = LinearSegmentedColormap.from_list("wins", ["#fff7ed", "#fb923c", "#9a3412"])
    image = axis.imshow(values.to_numpy(), cmap=cmap, aspect="auto")
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels)
    axis.set_yticks(range(len(values.index)))
    axis.set_yticklabels(values.index)
    axis.set_title("Win Counts Across Datasets and Metrics")
    for row_idx in range(values.shape[0]):
        for col_idx in range(len(metric_order)):
            axis.text(
                col_idx,
                row_idx,
                f"{values.iloc[row_idx, col_idx]:.1f}",
                ha="center",
                va="center",
                fontsize=10,
            )
    fig.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_family_heatmap(
    results: pd.DataFrame,
    catalog: pd.DataFrame,
    metric: str = "f1",
    top_n_families: int = 12,
    save_path: Path | None = None,
) -> plt.Figure:
    family_order = catalog["dataset_family"].value_counts().head(top_n_families).index
    subset = results.loc[results["dataset_family"].isin(family_order)]
    pivot = (
        subset.pivot_table(
            index="dataset_family",
            columns="algorithm_display",
            values=metric,
            aggfunc="median",
        )
        .reindex(family_order)
        .fillna(0.0)
    )

    fig, axis = plt.subplots(figsize=(11, 8))
    cmap = LinearSegmentedColormap.from_list("families", ["#fff7ed", "#f59e0b", "#166534"])
    image = axis.imshow(pivot.to_numpy(), cmap=cmap, aspect="auto")
    axis.set_xticks(range(len(pivot.columns)))
    axis.set_xticklabels(pivot.columns, rotation=15)
    axis.set_yticks(range(len(pivot.index)))
    axis.set_yticklabels(pivot.index)
    axis.set_title(f"Family-Level Median {METRIC_LABELS.get(metric, metric)}")
    for row_idx in range(pivot.shape[0]):
        for col_idx in range(pivot.shape[1]):
            axis.text(
                col_idx,
                row_idx,
                f"{pivot.iloc[row_idx, col_idx]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )
    fig.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def ensure_reformed_dataset(dataset_name: str) -> Path:
    REFORMED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REFORMED_DATASET_DIR / f"{dataset_name}.txt"
    if output_path.exists():
        return output_path

    virgin_path = VIRGIN_DATASET_DIR / f"{dataset_name}.txt"
    if not virgin_path.exists():
        raise FileNotFoundError(f"Could not find virgin dataset: {virgin_path}")

    details = parse_dataset_name(dataset_name)
    anomaly_start = details["anomaly_start"]
    anomaly_end = details["anomaly_end"]

    lines_out: list[str] = []
    for line_number, raw_line in enumerate(
        virgin_path.read_text(encoding="utf-8", errors="ignore").splitlines(),
        start=1,
    ):
        values = raw_line.strip().split()
        for value in values:
            label = 1 if anomaly_start <= line_number <= anomaly_end else 0
            lines_out.append(f"{value},{label}")

    output_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return output_path


def load_reformed_dataset(dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    dataset_path = ensure_reformed_dataset(dataset_name)
    data = np.loadtxt(dataset_path, delimiter=",")
    return data[:, 0].astype(float), data[:, 1].astype(int)


def apply_normalization(
    values: np.ndarray,
    method: str = "zscore",
    clip_quantile: float | None = None,
) -> np.ndarray:
    method = method.lower()
    if method not in NORMALIZATION_METHODS:
        raise ValueError(f"Unsupported normalization method: {method}")

    transformed = np.asarray(values, dtype=float).copy()
    if clip_quantile is not None:
        if not (0.0 < clip_quantile < 0.5):
            raise ValueError("clip_quantile must be between 0 and 0.5.")
        lower, upper = np.quantile(transformed, [clip_quantile, 1.0 - clip_quantile])
        transformed = np.clip(transformed, lower, upper)

    if method == "none":
        return transformed
    if method == "minmax":
        return minmax_scale(transformed)
    if method == "zscore":
        mean = float(np.mean(transformed))
        std = float(np.std(transformed))
        return np.zeros_like(transformed) if np.isclose(std, 0.0) else (transformed - mean) / std
    if method == "robust":
        median = float(np.median(transformed))
        q1, q3 = np.quantile(transformed, [0.25, 0.75])
        iqr = float(q3 - q1)
        return np.zeros_like(transformed) if np.isclose(iqr, 0.0) else (transformed - median) / iqr
    raise AssertionError("Unhandled normalization branch.")


def dataset_statistics_frame(values: np.ndarray, normalized_values: np.ndarray) -> pd.DataFrame:
    rows = []
    for label, series in [("Raw", values), ("Prepared", normalized_values)]:
        rows.append(
            {
                "Series": label,
                "Mean": float(np.mean(series)),
                "Std": float(np.std(series)),
                "Min": float(np.min(series)),
                "Median": float(np.median(series)),
                "Max": float(np.max(series)),
            }
        )
    return pd.DataFrame(rows)


def _segment_slices(metadata: dict[str, Any], series_length: int) -> dict[str, slice]:
    train_stop = max(0, min(int(metadata["train_until"]), series_length))
    anomaly_start = max(0, min(int(metadata["anomaly_start"]), series_length))
    anomaly_stop = max(anomaly_start, min(int(metadata["anomaly_end"]) + 1, series_length))
    return {
        "Train": slice(0, train_stop),
        "Anomaly": slice(anomaly_start, anomaly_stop),
        "Post-anomaly": slice(anomaly_stop, series_length),
    }


def _safe_segment_summary(values: np.ndarray) -> dict[str, float]:
    series = np.asarray(values, dtype=float)
    if series.size == 0:
        return {
            "Mean": float("nan"),
            "Std": float("nan"),
            "Min": float("nan"),
            "Median": float("nan"),
            "Max": float("nan"),
            "Mean |delta|": float("nan"),
        }
    deltas = np.abs(np.diff(series))
    return {
        "Mean": float(np.mean(series)),
        "Std": float(np.std(series)),
        "Min": float(np.min(series)),
        "Median": float(np.median(series)),
        "Max": float(np.max(series)),
        "Mean |delta|": float(np.mean(deltas)) if deltas.size else 0.0,
    }


def build_dataset_deep_dive_frame(bundle: dict[str, Any]) -> pd.DataFrame:
    metadata = bundle["metadata"]
    point_count = len(bundle["raw_values"])
    anomaly_ratio = (metadata["anomaly_length"] / point_count) if point_count else float("nan")
    rows = [
        {"Metric": "Dataset", "Value": bundle["dataset_name"]},
        {"Metric": "Family", "Value": metadata["dataset_family"]},
        {"Metric": "Variant", "Value": metadata["dataset_variant"]},
        {"Metric": "Datapoints", "Value": point_count},
        {"Metric": "Training cutoff", "Value": metadata["train_until"]},
        {"Metric": "Anomaly start", "Value": metadata["anomaly_start"]},
        {"Metric": "Anomaly end", "Value": metadata["anomaly_end"]},
        {"Metric": "Anomaly length", "Value": metadata["anomaly_length"]},
        {"Metric": "Anomaly ratio", "Value": round(anomaly_ratio * 100.0, 3)},
        {"Metric": "Estimated window", "Value": bundle["window"]},
        {"Metric": "Normalization", "Value": bundle["normalization_method"]},
        {"Metric": "Clip quantile", "Value": bundle["clip_quantile"] if bundle["clip_quantile"] is not None else "None"},
    ]
    return pd.DataFrame(rows)


def build_dataset_segment_frame(bundle: dict[str, Any]) -> pd.DataFrame:
    metadata = bundle["metadata"]
    raw_values = np.asarray(bundle["raw_values"], dtype=float)
    prepared_values = np.asarray(bundle["normalized_values"], dtype=float)
    segment_map = _segment_slices(metadata, len(raw_values))

    rows = []
    for series_label, series in [("Raw", raw_values), ("Prepared", prepared_values)]:
        rows.append(
            {
                "Series": series_label,
                "Segment": "Whole series",
                "Points": len(series),
                "Share %": 100.0,
                **_safe_segment_summary(series),
            }
        )
        for segment_label, segment_slice in segment_map.items():
            segment_values = series[segment_slice]
            rows.append(
                {
                    "Series": series_label,
                    "Segment": segment_label,
                    "Points": int(segment_values.size),
                    "Share %": (float(segment_values.size) * 100.0 / len(series)) if len(series) else float("nan"),
                    **_safe_segment_summary(segment_values),
                }
            )
    return pd.DataFrame(rows)


def prepare_dataset_bundle(
    dataset_name: str,
    normalization_method: str = "zscore",
    clip_quantile: float | None = None,
    window_override: int | None = None,
) -> dict[str, Any]:
    raw_values, labels = load_reformed_dataset(dataset_name)
    normalized_values = apply_normalization(
        raw_values,
        method=normalization_method,
        clip_quantile=clip_quantile,
    )
    metadata = parse_dataset_name(dataset_name)
    window = window_override or estimate_window_length(normalized_values)
    return {
        "dataset_name": dataset_name,
        "metadata": metadata,
        "raw_values": raw_values,
        "normalized_values": normalized_values,
        "labels": labels,
        "window": window,
        "normalization_method": normalization_method,
        "clip_quantile": clip_quantile,
        "statistics": dataset_statistics_frame(raw_values, normalized_values),
    }


def plot_preprocessing_story(
    bundle: dict[str, Any],
    save_path: Path | None = None,
    context_points: int = 1000,
) -> plt.Figure:
    metadata = bundle["metadata"]
    raw_values = bundle["raw_values"]
    normalized_values = bundle["normalized_values"]
    start = max(0, metadata["anomaly_start"] - context_points)
    end = min(len(raw_values), metadata["anomaly_end"] + context_points)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Beginning: Dataset Preparation and Normalization", fontsize=19, fontweight="bold", y=1.01)

    axes[0, 0].plot(raw_values[start:end], color="#475569", linewidth=1.2)
    axes[0, 0].axvspan(
        metadata["anomaly_start"] - start,
        metadata["anomaly_end"] - start,
        color="#f59e0b",
        alpha=0.18,
    )
    axes[0, 0].set_title("Raw sensor stream")
    axes[0, 0].set_ylabel("Raw value")

    axes[0, 1].plot(normalized_values[start:end], color="#0f766e", linewidth=1.2)
    axes[0, 1].axvspan(
        metadata["anomaly_start"] - start,
        metadata["anomaly_end"] - start,
        color="#f59e0b",
        alpha=0.18,
    )
    axes[0, 1].set_title(f"Prepared stream ({bundle['normalization_method']})")
    axes[0, 1].set_ylabel("Prepared value")

    axes[1, 0].hist(raw_values, bins=40, color="#94a3b8", edgecolor="white")
    axes[1, 0].set_title("Raw distribution")
    axes[1, 0].set_xlabel("Raw value")
    axes[1, 0].set_ylabel("Count")

    axes[1, 1].hist(normalized_values, bins=40, color="#14b8a6", edgecolor="white")
    axes[1, 1].set_title("Prepared distribution")
    axes[1, 1].set_xlabel("Prepared value")
    axes[1, 1].set_ylabel("Count")

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_dataset_deep_dive(
    bundle: dict[str, Any],
    save_path: Path | None = None,
    context_points: int = 1000,
) -> plt.Figure:
    metadata = bundle["metadata"]
    raw_values = np.asarray(bundle["raw_values"], dtype=float)
    prepared_values = np.asarray(bundle["normalized_values"], dtype=float)
    segment_frame = build_dataset_segment_frame(bundle)
    segment_labels = ["Train", "Anomaly", "Post-anomaly"]
    start = max(0, metadata["anomaly_start"] - context_points)
    end = min(len(raw_values), metadata["anomaly_end"] + context_points + 1)

    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    fig.suptitle(
        f"Deep Dive Dataset Analysis: {bundle['dataset_name']}",
        fontsize=19,
        fontweight="bold",
        y=1.01,
    )

    axes[0, 0].plot(raw_values, color="#475569", linewidth=1.0)
    axes[0, 0].axvline(
        metadata["train_until"],
        color="#2563eb",
        linestyle="--",
        linewidth=1.3,
        label="Training cutoff",
    )
    axes[0, 0].axvspan(
        metadata["anomaly_start"],
        metadata["anomaly_end"],
        color="#f59e0b",
        alpha=0.18,
        label="Annotated anomaly",
    )
    axes[0, 0].set_title("Full raw series with benchmark annotations")
    axes[0, 0].set_xlabel("Datapoint index")
    axes[0, 0].set_ylabel("Raw value")
    axes[0, 0].legend(loc="upper right")

    axes[0, 1].plot(
        np.arange(start, end),
        raw_values[start:end],
        color="#334155",
        linewidth=1.2,
        label="Raw signal",
    )
    axes[0, 1].axvspan(
        metadata["anomaly_start"],
        metadata["anomaly_end"],
        color="#f59e0b",
        alpha=0.18,
    )
    axes[0, 1].set_title("Zoomed raw context around the anomaly")
    axes[0, 1].set_xlabel("Datapoint index")
    axes[0, 1].set_ylabel("Raw value")

    axis_prepared = axes[0, 1].twinx()
    axis_prepared.plot(
        np.arange(start, end),
        prepared_values[start:end],
        color="#0f766e",
        linewidth=1.0,
        alpha=0.85,
        label=f"Prepared ({bundle['normalization_method']})",
    )
    axis_prepared.set_ylabel("Prepared value")

    left_handles, left_labels = axes[0, 1].get_legend_handles_labels()
    right_handles, right_labels = axis_prepared.get_legend_handles_labels()
    axes[0, 1].legend(left_handles + right_handles, left_labels + right_labels, loc="upper right")

    prepared_segments = (
        segment_frame.loc[
            (segment_frame["Series"] == "Prepared") & (segment_frame["Segment"].isin(segment_labels)),
            ["Segment", "Mean", "Std", "Mean |delta|"],
        ]
        .set_index("Segment")
        .reindex(segment_labels)
    )
    x = np.arange(len(segment_labels))
    axes[1, 0].bar(x - 0.18, prepared_segments["Mean"], width=0.36, color="#0f766e", label="Mean")
    axes[1, 0].bar(x + 0.18, prepared_segments["Std"], width=0.36, color="#f97316", label="Std")
    axes[1, 0].plot(
        x,
        prepared_segments["Mean |delta|"],
        color="#7c3aed",
        marker="o",
        linewidth=2.0,
        label="Mean |delta|",
    )
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(segment_labels)
    axes[1, 0].set_title("Prepared segment profile")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].legend(loc="upper right")

    train_values = prepared_values[_segment_slices(metadata, len(prepared_values))["Train"]]
    anomaly_values = prepared_values[_segment_slices(metadata, len(prepared_values))["Anomaly"]]
    post_values = prepared_values[_segment_slices(metadata, len(prepared_values))["Post-anomaly"]]
    if train_values.size:
        axes[1, 1].hist(train_values, bins=36, alpha=0.6, color="#2563eb", label="Train")
    if anomaly_values.size:
        axes[1, 1].hist(anomaly_values, bins=min(20, max(5, anomaly_values.size)), alpha=0.8, color="#f59e0b", label="Anomaly")
    if post_values.size:
        axes[1, 1].hist(post_values, bins=36, alpha=0.45, color="#6b7280", label="Post-anomaly")
    axes[1, 1].set_title("Prepared value distribution by segment")
    axes[1, 1].set_xlabel("Prepared value")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].legend(loc="upper right")

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def estimate_window_length(values: np.ndarray, default: int = 125) -> int:
    series = np.asarray(values, dtype=float).ravel()
    if series.size < 10:
        return default
    series = series[: min(20000, series.size)]
    series = series - np.mean(series)
    denom = np.dot(series, series)
    if not np.isfinite(denom) or denom <= 0:
        return default

    max_lag = min(400, series.size - 2)
    autocorr = np.correlate(series, series, mode="full")[series.size - 1 : series.size + max_lag] / denom
    candidates: list[tuple[float, int]] = []
    for lag in range(3, min(301, len(autocorr) - 1)):
        if autocorr[lag] > autocorr[lag - 1] and autocorr[lag] > autocorr[lag + 1]:
            candidates.append((float(autocorr[lag]), lag))
    if not candidates:
        return default
    return max(candidates, key=lambda item: item[0])[1]


def rolling_window_matrix(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(values, dtype=float).reshape(-1, 1)
    if window >= len(values):
        raise ValueError("Window length must be smaller than the time-series length.")
    return np.lib.stride_tricks.sliding_window_view(np.asarray(values, dtype=float), window)


def minmax_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    min_value = np.min(values)
    max_value = np.max(values)
    if np.isclose(max_value, min_value):
        return np.zeros_like(values, dtype=float)
    return (values - min_value) / (max_value - min_value)


def to_pointwise_scores(window_scores: np.ndarray, window: int) -> np.ndarray:
    window_scores = np.asarray(window_scores, dtype=float).ravel()
    if window_scores.size == 0:
        return window_scores
    prefix = np.repeat(window_scores[0], math.ceil((window - 1) / 2))
    suffix = np.repeat(window_scores[-1], (window - 1) // 2)
    return np.concatenate([prefix, window_scores, suffix])


def threshold_scores(scores: np.ndarray, threshold_std_multiplier: float = 3.0) -> tuple[np.ndarray, float]:
    threshold = float(np.mean(scores) + threshold_std_multiplier * np.std(scores))
    predictions = (scores > threshold).astype(int)
    return predictions, threshold


def compute_point_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold_std_multiplier: float = 3.0,
) -> dict[str, Any]:
    predictions, threshold = threshold_scores(scores, threshold_std_multiplier=threshold_std_multiplier)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = float("nan")
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    return {
        "auc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": threshold,
        "threshold_std_multiplier": threshold_std_multiplier,
        "predictions": predictions,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def run_iforest_demo(
    values: np.ndarray,
    labels: np.ndarray,
    window: int | None = None,
    params: dict[str, Any] | None = None,
    threshold_std_multiplier: float = 3.0,
) -> dict[str, Any]:
    window = window or estimate_window_length(values)
    matrix = rolling_window_matrix(values, window)
    params = params or {}
    start = time.perf_counter()
    model = IsolationForest(
        n_estimators=params.get("n_estimators", 100),
        max_samples=params.get("max_samples", "auto"),
        contamination=params.get("contamination", 0.1),
        max_features=params.get("max_features", 1.0),
        bootstrap=params.get("bootstrap", False),
        n_jobs=1,
        random_state=params.get("random_state", 42),
    )
    model.fit(matrix)
    window_scores = -model.score_samples(matrix)
    runtime = time.perf_counter() - start
    point_scores = minmax_scale(to_pointwise_scores(window_scores, window))
    metrics = compute_point_metrics(
        labels[: len(point_scores)],
        point_scores,
        threshold_std_multiplier=threshold_std_multiplier,
    )
    return {
        "algorithm": "IForest",
        "algorithm_display": DISPLAY_NAME_MAP["IForest"],
        "window": window,
        "scores": point_scores,
        "runtime_sec": runtime,
        "params": params,
        **metrics,
    }


def run_lof_demo(
    values: np.ndarray,
    labels: np.ndarray,
    window: int | None = None,
    params: dict[str, Any] | None = None,
    threshold_std_multiplier: float = 3.0,
) -> dict[str, Any]:
    window = window or estimate_window_length(values)
    matrix = rolling_window_matrix(values, window)
    params = params or {}
    start = time.perf_counter()
    model = LocalOutlierFactor(
        n_neighbors=params.get("n_neighbors", 20),
        contamination=params.get("contamination", 0.1),
        algorithm=params.get("algorithm", "auto"),
        leaf_size=params.get("leaf_size", 30),
        metric=params.get("metric", "minkowski"),
        p=params.get("p", 2),
        n_jobs=1,
    )
    model.fit(matrix)
    window_scores = -model.negative_outlier_factor_
    runtime = time.perf_counter() - start
    point_scores = minmax_scale(to_pointwise_scores(window_scores, window))
    metrics = compute_point_metrics(
        labels[: len(point_scores)],
        point_scores,
        threshold_std_multiplier=threshold_std_multiplier,
    )
    return {
        "algorithm": "LOF",
        "algorithm_display": DISPLAY_NAME_MAP["LOF"],
        "window": window,
        "scores": point_scores,
        "runtime_sec": runtime,
        "params": params,
        **metrics,
    }


def _load_sand_class(prefer_new: bool = True):
    if not hasattr(np, "Inf"):
        np.Inf = np.inf  # type: ignore[attr-defined]
    if not hasattr(np, "NaN"):
        np.NaN = np.nan  # type: ignore[attr-defined]

    candidates = [
        PYTHON_ROOT / "newSANDEffort" / "Utils" / "sand.py",
        PYTHON_ROOT / "SANDAnomalyDetection" / "Utils" / "sand.py",
    ]
    if not prefer_new:
        candidates.reverse()

    last_error: Exception | None = None
    for path in candidates:
        if not path.exists():
            continue
        try:
            module_name = f"_sand_runtime_{path.parent.parent.name}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="h5py not installed.*")
                spec.loader.exec_module(module)
            return module.SAND
        except Exception as exc:  # pragma: no cover
            last_error = exc
    if last_error is not None:
        raise last_error
    raise FileNotFoundError("Could not locate a SAND implementation in the repo.")


def run_sand_demo(
    values: np.ndarray,
    labels: np.ndarray,
    window: int | None = None,
    prefer_new: bool = True,
    params: dict[str, Any] | None = None,
    threshold_std_multiplier: float = 3.0,
) -> dict[str, Any]:
    window = window or estimate_window_length(values)
    params = params or {}
    subsequence_multiplier = int(params.get("subsequence_multiplier", 4))
    subsequence_length = max(window * subsequence_multiplier, window + 1)
    if subsequence_length >= len(values):
        raise ValueError("Series is too short for the SAND subsequence configuration.")

    init_length = min(int(params.get("init_length", 5000)), len(values) - subsequence_length - 1)
    init_length = max(init_length, subsequence_length + 1)
    batch_size = min(int(params.get("batch_size", 2000)), len(values) - init_length)
    batch_size = max(batch_size, subsequence_length + 1)

    def subsequence_count(span: int) -> int:
        upper = min(span, len(values) - subsequence_length)
        return max(0, len(range(0, upper, subsequence_length)))

    k = params.get("k")
    if k is None:
        k = max(2, min(6, subsequence_count(init_length), subsequence_count(batch_size)))
    k = int(k)
    overlapping_rate = params.get("overlapping_rate")
    if overlapping_rate is None:
        overlapping_rate = max(1, subsequence_length)
    overlapping_rate = int(overlapping_rate)

    SAND = _load_sand_class(prefer_new=prefer_new)
    model = SAND(pattern_length=window, subsequence_length=subsequence_length, k=k)
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(
            values,
            online=True,
            alpha=float(params.get("alpha", 0.5)),
            init_length=init_length,
            batch_size=batch_size,
            overlaping_rate=overlapping_rate,
            verbose=False,
        )
    runtime = time.perf_counter() - start
    point_scores = minmax_scale(np.asarray(model.decision_scores_, dtype=float))
    metrics = compute_point_metrics(
        labels[: len(point_scores)],
        point_scores,
        threshold_std_multiplier=threshold_std_multiplier,
    )
    return {
        "algorithm": "SAND",
        "algorithm_display": DISPLAY_NAME_MAP["SAND"],
        "window": window,
        "scores": point_scores,
        "runtime_sec": runtime,
        "params": params,
        "subsequence_length": subsequence_length,
        **metrics,
    }


def archived_dataset_metrics(results: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    columns = ["algorithm_display", "auc", "precision", "recall", "f1", "elapsed_time_sec"]
    subset = results.loc[results["dataset_name"] == dataset_name, columns].copy()
    subset.rename(columns={"elapsed_time_sec": "archived_runtime_sec"}, inplace=True)
    subset.sort_values("algorithm_display", inplace=True)
    subset.reset_index(drop=True, inplace=True)
    return subset


def run_single_algorithm(
    algorithm: str,
    values: np.ndarray,
    labels: np.ndarray,
    window: int,
    params: dict[str, Any] | None = None,
    threshold_std_multiplier: float = 3.0,
    prefer_new_sand: bool = True,
) -> dict[str, Any]:
    if algorithm == "IForest":
        return run_iforest_demo(
            values,
            labels,
            window=window,
            params=params,
            threshold_std_multiplier=threshold_std_multiplier,
        )
    if algorithm == "LOF":
        return run_lof_demo(
            values,
            labels,
            window=window,
            params=params,
            threshold_std_multiplier=threshold_std_multiplier,
        )
    if algorithm == "SAND":
        sand_params = dict(params or {})
        prefer_flag = bool(sand_params.pop("prefer_new_implementation", prefer_new_sand))
        return run_sand_demo(
            values,
            labels,
            window=window,
            prefer_new=prefer_flag,
            params=sand_params,
            threshold_std_multiplier=threshold_std_multiplier,
        )
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def run_demo_suite(
    dataset_name: str,
    results: pd.DataFrame | None = None,
    include_sand: bool = True,
    prefer_new_sand: bool = True,
    normalization_method: str = "zscore",
    clip_quantile: float | None = None,
    window_override: int | None = None,
    algorithm_params: dict[str, dict[str, Any]] | None = None,
    threshold_std_multiplier: float = 3.0,
) -> dict[str, Any]:
    bundle = prepare_dataset_bundle(
        dataset_name,
        normalization_method=normalization_method,
        clip_quantile=clip_quantile,
        window_override=window_override,
    )
    values = bundle["normalized_values"]
    labels = bundle["labels"]
    window = bundle["window"]
    algorithm_params = algorithm_params or default_algorithm_parameters()

    algorithm_runs = [
        run_single_algorithm(
            "IForest",
            values,
            labels,
            window=window,
            params=algorithm_params.get("IForest"),
            threshold_std_multiplier=threshold_std_multiplier,
            prefer_new_sand=prefer_new_sand,
        ),
        run_single_algorithm(
            "LOF",
            values,
            labels,
            window=window,
            params=algorithm_params.get("LOF"),
            threshold_std_multiplier=threshold_std_multiplier,
            prefer_new_sand=prefer_new_sand,
        ),
    ]
    sand_error = None
    if include_sand:
        try:
            algorithm_runs.append(
                run_single_algorithm(
                    "SAND",
                    values,
                    labels,
                    window=window,
                    params=algorithm_params.get("SAND"),
                    threshold_std_multiplier=threshold_std_multiplier,
                    prefer_new_sand=prefer_new_sand,
                )
            )
        except Exception as exc:  # pragma: no cover
            sand_error = str(exc)

    live_metrics = pd.DataFrame(
        [
            {
                "algorithm": run["algorithm"],
                "algorithm_display": run["algorithm_display"],
                "live_auc": run["auc"],
                "live_precision": run["precision"],
                "live_recall": run["recall"],
                "live_f1": run["f1"],
                "live_runtime_sec": run["runtime_sec"],
                "window": run["window"],
                "tp": run["tp"],
                "fp": run["fp"],
                "fn": run["fn"],
            }
            for run in algorithm_runs
        ]
    ).sort_values("algorithm_display")

    archive = archived_dataset_metrics(results, dataset_name) if results is not None else None
    comparison = live_metrics.merge(archive, on="algorithm_display", how="left") if archive is not None else live_metrics

    return {
        "dataset_name": dataset_name,
        "metadata": parse_dataset_name(dataset_name),
        "raw_values": bundle["raw_values"],
        "values": values,
        "normalized_values": values,
        "labels": labels,
        "window": window,
        "runs": {run["algorithm"]: run for run in algorithm_runs},
        "comparison": comparison.reset_index(drop=True),
        "sand_error": sand_error,
        "normalization_method": normalization_method,
        "clip_quantile": clip_quantile,
        "statistics": bundle["statistics"],
        "algorithm_params": algorithm_params,
        "threshold_std_multiplier": threshold_std_multiplier,
    }


def plot_live_demo_suite(
    demo: dict[str, Any],
    save_path: Path | None = None,
    context_points: int = 1200,
) -> plt.Figure:
    values = demo.get("raw_values", demo["values"])
    metadata = demo["metadata"]
    start = max(0, metadata["anomaly_start"] - context_points)
    end = min(len(values), metadata["anomaly_end"] + context_points)

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(
        f"Live Walkthrough: {demo['dataset_name']}",
        fontsize=18,
        fontweight="bold",
        y=1.01,
    )

    axes[0].plot(values[start:end], color="#374151", linewidth=1.2)
    axes[0].axvspan(
        metadata["anomaly_start"] - start,
        metadata["anomaly_end"] - start,
        color="#f59e0b",
        alpha=0.18,
        label="Ground-truth anomaly",
    )
    axes[0].set_title("Raw signal around the anomaly window")
    axes[0].set_ylabel("Value")
    axes[0].legend(loc="upper right")

    colors = {"IForest": "#4c78a8", "LOF": "#f58518", "SAND": "#54a24b"}
    for axis, algorithm in zip(axes[1:], ["IForest", "LOF", "SAND"]):
        axis.axvspan(
            metadata["anomaly_start"] - start,
            metadata["anomaly_end"] - start,
            color="#f59e0b",
            alpha=0.18,
        )
        run = demo["runs"].get(algorithm)
        if run is None:
            message = demo["sand_error"] if algorithm == "SAND" else "Run unavailable."
            axis.text(0.02, 0.5, message, transform=axis.transAxes, va="center", fontsize=11)
            axis.set_title(DISPLAY_NAME_MAP[algorithm])
            continue

        scores = run["scores"][start:end]
        axis.plot(scores, color=colors[algorithm], linewidth=1.4)
        axis.axhline(run["threshold"], color="#b91c1c", linestyle="--", linewidth=1.1)
        axis.set_ylim(-0.02, max(1.02, float(scores.max()) + 0.05))
        axis.set_title(
            f"{DISPLAY_NAME_MAP[algorithm]} | AUC={run['auc']:.2f}, F1={run['f1']:.2f}, runtime={run['runtime_sec']:.2f}s"
        )
        axis.set_ylabel("Score")

    axes[-1].set_xlabel("Datapoint index (zoomed window)")
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def sweep_parameter(
    dataset_name: str,
    algorithm: str,
    parameter_name: str,
    parameter_values: list[Any],
    normalization_method: str = "zscore",
    clip_quantile: float | None = None,
    window_override: int | None = None,
    algorithm_params: dict[str, dict[str, Any]] | None = None,
    threshold_std_multiplier: float = 3.0,
) -> pd.DataFrame:
    base_params = default_algorithm_parameters()
    if algorithm_params is not None:
        for key, value in algorithm_params.items():
            base_params[key].update(value)

    bundle = prepare_dataset_bundle(
        dataset_name,
        normalization_method=normalization_method,
        clip_quantile=clip_quantile,
        window_override=window_override,
    )

    rows = []
    for value in parameter_values:
        params_for_run = {key: dict(val) for key, val in base_params.items()}
        params_for_run[algorithm][parameter_name] = value
        run = run_single_algorithm(
            algorithm,
            bundle["normalized_values"],
            bundle["labels"],
            window=bundle["window"],
            params=params_for_run[algorithm],
            threshold_std_multiplier=threshold_std_multiplier,
            prefer_new_sand=bool(params_for_run["SAND"].get("prefer_new_implementation", True)),
        )
        rows.append(
            {
                "algorithm": algorithm,
                "algorithm_display": DISPLAY_NAME_MAP[algorithm],
                "parameter_name": parameter_name,
                "parameter_value": value,
                "auc": run["auc"],
                "precision": run["precision"],
                "recall": run["recall"],
                "f1": run["f1"],
                "runtime_sec": run["runtime_sec"],
                "window": run["window"],
            }
        )
    return pd.DataFrame(rows)


def plot_parameter_sweep(
    sweep_frame: pd.DataFrame,
    save_path: Path | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    x = range(len(sweep_frame))
    labels = [str(value) for value in sweep_frame["parameter_value"].tolist()]

    axes[0].plot(x, sweep_frame["auc"], marker="o", label="AUC", color="#2563eb")
    axes[0].plot(x, sweep_frame["f1"], marker="o", label="F1", color="#ea580c")
    axes[0].plot(x, sweep_frame["recall"], marker="o", label="Recall", color="#16a34a")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, rotation=20)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title(
        f"{sweep_frame['algorithm_display'].iloc[0]}: metric response to {sweep_frame['parameter_name'].iloc[0]}"
    )
    axes[0].set_xlabel("Parameter value")
    axes[0].set_ylabel("Metric value")
    axes[0].legend()

    axes[1].bar(x, sweep_frame["runtime_sec"], color="#8b5cf6", alpha=0.8)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, rotation=20)
    axes[1].set_title("Runtime response")
    axes[1].set_xlabel("Parameter value")
    axes[1].set_ylabel("Runtime (seconds)")

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def build_key_findings(summary: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    best_f1 = summary.loc[summary["median_f1"].idxmax(), "algorithm_display"]
    fastest = summary.loc[summary["median_time_per_1000"].idxmin(), "algorithm_display"]
    longest = catalog.loc[catalog["datapoint_count"].idxmax()]
    sparsest = catalog.loc[catalog["anomaly_ratio"].idxmin()]
    densest = catalog.loc[catalog["anomaly_ratio"].idxmax()]

    return pd.DataFrame(
        [
            {"Finding": "Benchmark size", "Value": f"{catalog['dataset_name'].nunique()} datasets"},
            {"Finding": "Most accurate median F1", "Value": best_f1},
            {"Finding": "Fastest median runtime", "Value": fastest},
            {"Finding": "Longest series", "Value": f"{longest['dataset_name']} ({longest['datapoint_count']:,} points)"},
            {"Finding": "Sparsest anomaly ratio", "Value": f"{sparsest['dataset_name']} ({sparsest['anomaly_ratio']:.5f})"},
            {"Finding": "Densest anomaly ratio", "Value": f"{densest['dataset_name']} ({densest['anomaly_ratio']:.5f})"},
        ]
    )
