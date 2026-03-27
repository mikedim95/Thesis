from __future__ import annotations

import html
import importlib
import json
import math
import re
import shutil
import sys
import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import HTML, display

warnings.filterwarnings(
    "ignore", message=".*h5py not installed.*", category=UserWarning)


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent.parent
DATASET_ROOT = PROJECT_ROOT / "datasets"
RAW_DATASET_DIR = DATASET_ROOT / "raw"
NORMALIZED_DATASET_ROOT = DATASET_ROOT / "normalized"
LEGACY_VIRGIN_DIR = PROJECT_ROOT.parent / "datasets" / "virgin"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULT_TABLES_DIR = RESULTS_DIR / "tables"
RESULT_PER_ALGORITHM_TABLES_DIR = RESULT_TABLES_DIR / "per_algorithm"
RESULT_FIGURES_DIR = RESULTS_DIR / "figures"
RESULT_ALGORITHM_PANEL_DIR = RESULT_FIGURES_DIR / "algorithm_panels"
RESULT_DEEP_DIVE_DIR = RESULT_FIGURES_DIR / "deep_dives"
RESULT_SCORES_DIR = RESULTS_DIR / "scores"

RAW_DATASET_DIR.mkdir(parents=True, exist_ok=True)
NORMALIZED_DATASET_ROOT.mkdir(parents=True, exist_ok=True)


def _move_result_file(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    shutil.move(str(source), str(destination))


def ensure_results_layout() -> None:
    for directory in (
        RESULTS_DIR,
        RESULT_TABLES_DIR,
        RESULT_PER_ALGORITHM_TABLES_DIR,
        RESULT_FIGURES_DIR,
        RESULT_ALGORITHM_PANEL_DIR,
        RESULT_DEEP_DIVE_DIR,
        RESULT_SCORES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    for filename in (
        "run_configuration.csv",
        "selected_run_parameters.csv",
        "dataset_preparation_summary.csv",
        "benchmark_results.csv",
        "dataset_catalog.csv",
        "algorithm_summary.csv",
        "family_summary.csv",
        "overall_regime_summary.csv",
        "best_algorithm_by_dataset_evaluation.csv",
        "best_algorithm_by_dataset_f1.csv",
        "best_algorithm_by_dataset_auc.csv",
        "error_report.csv",
        "deep_dive_research_metrics.csv",
    ):
        _move_result_file(RESULTS_DIR / filename, RESULT_TABLES_DIR / filename)

    for algorithm_key in ALGORITHM_ORDER:
        _move_result_file(
            RESULTS_DIR / f"{algorithm_key}_results.csv",
            RESULT_PER_ALGORITHM_TABLES_DIR / f"{algorithm_key}_results.csv",
        )
        _move_result_file(
            RESULTS_DIR / f"{algorithm_key}_benchmark_panel.png",
            RESULT_ALGORITHM_PANEL_DIR /
            f"{algorithm_key}_benchmark_panel.png",
        )

    for filename in (
        "benchmark_overview.png",
        "pareto_frontier.png",
        "metric_heatmap.png",
        "family_evaluation_heatmap.png",
        "family_range_f1_heatmap.png",
        "algorithm_wins.png",
    ):
        _move_result_file(RESULTS_DIR / filename,
                          RESULT_FIGURES_DIR / filename)

    for source in RESULTS_DIR.glob("*__deep_dive__*.png"):
        _move_result_file(source, RESULT_DEEP_DIVE_DIR / source.name)


if str(PROJECT_ROOT / "algorithms") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "algorithms"))


ALGORITHM_METADATA = {
    "isolation_forest": {
        "display": "Isolation Forest",
        "superfamily": "Density-based",
        "category": "Tree",
        "border_color": "#1d4ed8",
    },
    "local_outlier_factor": {
        "display": "Local Outlier Factor",
        "superfamily": "Distance-based",
        "category": "Proximity",
        "border_color": "#ea580c",
    },
    "sand": {
        "display": "SAND",
        "superfamily": "Distance-based",
        "category": "Clustering",
        "border_color": "#15803d",
    },
    "matrix_profile": {
        "display": "Matrix Profile",
        "superfamily": "Distance-based",
        "category": "Discord",
        "border_color": "#0f766e",
    },
    "damp": {
        "display": "DAMP",
        "superfamily": "Distance-based",
        "category": "Discord",
        "border_color": "#b45309",
    },
    "hbos": {
        "display": "HBOS",
        "superfamily": "Density-based",
        "category": "Distribution",
        "border_color": "#9333ea",
    },
    "ocsvm": {
        "display": "OCSVM",
        "superfamily": "Density-based",
        "category": "Distribution",
        "border_color": "#be185d",
    },
    "pca": {
        "display": "PCA",
        "superfamily": "Density-based",
        "category": "Encoding",
        "border_color": "#2563eb",
    },
}
DISPLAY_NAME_MAP = {key: value["display"]
                    for key, value in ALGORITHM_METADATA.items()}
ALGORITHM_ORDER = [
    "isolation_forest",
    "local_outlier_factor",
    "sand",
    "matrix_profile",
    "damp",
    "hbos",
    "ocsvm",
    "pca",
]
ALGORITHM_ENABLE_CONTROL = {
    "isolation_forest": "run_iforest",
    "local_outlier_factor": "run_lof",
    "sand": "run_sand",
    "matrix_profile": "run_matrix_profile",
    "damp": "run_damp",
    "hbos": "run_hbos",
    "ocsvm": "run_ocsvm",
    "pca": "run_pca",
}
DATASET_VARIANT_ORDER = ["raw", "noise", "distorted"]
LENGTH_BUCKET_ORDER = ["short", "medium", "long"]
ANOMALY_RATIO_BUCKET_ORDER = ["sparse", "moderate", "dense"]

PAPER_PRESET_DEFINITIONS: dict[str, dict[str, Any]] = {
    "paper_high_roi": {
        "label": "Paper High ROI Sweep",
        "description": (
            "Focused sweep for the strongest runtime/performance methods plus Isolation Forest for calibration discussion. "
            "Only score-driving parameters are varied so the paper does not claim sensitivity to knobs that only affect backends or post-hoc thresholds."
        ),
        "enabled_algorithms": [
            "isolation_forest",
            "local_outlier_factor",
            "matrix_profile",
            "ocsvm",
        ],
        "variants": {
            "isolation_forest": [
                {
                    "variant_name": "Baseline",
                    "focus": "Stable tree baseline for comparison.",
                    "n_estimators": 200,
                    "max_samples": 256,
                    "max_features": 1.0,
                    "bootstrap": False,
                    "random_state": 42,
                },
                {
                    "variant_name": "Wide Sample",
                    "focus": "More trees and larger sampling for a smoother global isolation score.",
                    "n_estimators": 400,
                    "max_samples": "auto",
                    "max_features": 1.0,
                    "bootstrap": False,
                    "random_state": 42,
                },
                {
                    "variant_name": "Feat 0.6",
                    "focus": "Tests stronger random feature subsampling inside the forest.",
                    "n_estimators": 400,
                    "max_samples": 256,
                    "max_features": 0.6,
                    "bootstrap": False,
                    "random_state": 42,
                },
            ],
            "local_outlier_factor": [
                {
                    "variant_name": "Baseline",
                    "focus": "Balanced neighborhood baseline.",
                    "n_neighbors": 20,
                    "algorithm": "auto",
                    "leaf_size": 30,
                    "metric": "minkowski",
                    "p": 2,
                },
                {
                    "variant_name": "Local k10",
                    "focus": "More sensitive to local shape changes and short anomalies.",
                    "n_neighbors": 10,
                    "algorithm": "auto",
                    "leaf_size": 30,
                    "metric": "minkowski",
                    "p": 2,
                },
                {
                    "variant_name": "Global L1",
                    "focus": "Broader neighborhood with Manhattan distance for more global structure.",
                    "n_neighbors": 50,
                    "algorithm": "auto",
                    "leaf_size": 30,
                    "metric": "manhattan",
                    "p": 1,
                },
            ],
            "matrix_profile": [
                {
                    "variant_name": "Context x1",
                    "focus": "Shortest context, strongest local discord detection.",
                    "subsequence_multiplier": 1,
                },
                {
                    "variant_name": "Context x2",
                    "focus": "Intermediate subsequence context for medium-length anomalies.",
                    "subsequence_multiplier": 2,
                },
                {
                    "variant_name": "Context x4",
                    "focus": "Longer context for broad discord patterns.",
                    "subsequence_multiplier": 4,
                },
            ],
            "ocsvm": [
                {
                    "variant_name": "Baseline",
                    "focus": "Standard RBF novelty boundary with short warm-up.",
                    "kernel": "rbf",
                    "nu": 0.05,
                    "gamma": "scale",
                    "train_fraction": 0.10,
                },
                {
                    "variant_name": "Nu 0.10",
                    "focus": "Tests more permissive abnormal-boundary calibration.",
                    "kernel": "rbf",
                    "nu": 0.10,
                    "gamma": "scale",
                    "train_fraction": 0.10,
                },
                {
                    "variant_name": "Warmup 0.20",
                    "focus": "Uses a longer mostly-normal prefix for model fitting.",
                    "kernel": "rbf",
                    "nu": 0.05,
                    "gamma": "scale",
                    "train_fraction": 0.20,
                },
            ],
        },
    },
    "paper_full_suite": {
        "label": "Paper Full Suite Sweep",
        "description": (
            "Broader sweep across all implemented algorithms with a small but theory-driven set of variants per method. "
            "The sweep keeps runtime and threshold-only knobs fixed so each variant comparison reflects a genuine scoring-behavior change."
        ),
        "enabled_algorithms": ALGORITHM_ORDER,
        "variants": {
            "isolation_forest": [
                {
                    "variant_name": "Baseline",
                    "focus": "Stable tree baseline for comparison.",
                    "n_estimators": 200,
                    "max_samples": 256,
                    "max_features": 1.0,
                    "bootstrap": False,
                    "random_state": 42,
                },
                {
                    "variant_name": "Wide Sample",
                    "focus": "More trees and larger sampling for a smoother global isolation score.",
                    "n_estimators": 400,
                    "max_samples": "auto",
                    "max_features": 1.0,
                    "bootstrap": False,
                    "random_state": 42,
                },
                {
                    "variant_name": "Feat 0.6",
                    "focus": "Tests stronger random feature subsampling inside the forest.",
                    "n_estimators": 400,
                    "max_samples": 256,
                    "max_features": 0.6,
                    "bootstrap": False,
                    "random_state": 42,
                },
            ],
            "local_outlier_factor": [
                {
                    "variant_name": "Baseline",
                    "focus": "Balanced neighborhood baseline.",
                    "n_neighbors": 20,
                    "algorithm": "auto",
                    "leaf_size": 30,
                    "metric": "minkowski",
                    "p": 2,
                },
                {
                    "variant_name": "Local k10",
                    "focus": "More sensitive to local shape changes and short anomalies.",
                    "n_neighbors": 10,
                    "algorithm": "auto",
                    "leaf_size": 30,
                    "metric": "minkowski",
                    "p": 2,
                },
                {
                    "variant_name": "Global L1",
                    "focus": "Broader neighborhood with Manhattan distance for more global structure.",
                    "n_neighbors": 50,
                    "algorithm": "auto",
                    "leaf_size": 30,
                    "metric": "manhattan",
                    "p": 1,
                },
            ],
            "sand": [
                {
                    "variant_name": "Baseline",
                    "focus": "Reference online clustering configuration.",
                    "alpha": 0.5,
                    "init_length": 5000,
                    "batch_size": 2000,
                    "k": 0,
                    "subsequence_multiplier": 4,
                    "overlap": 0,
                },
                {
                    "variant_name": "Adaptive",
                    "focus": "Faster adaptation with shorter context and smaller batches.",
                    "alpha": 0.7,
                    "init_length": 3000,
                    "batch_size": 1000,
                    "k": 0,
                    "subsequence_multiplier": 2,
                    "overlap": 0,
                },
            ],
            "matrix_profile": [
                {
                    "variant_name": "Context x1",
                    "focus": "Shortest context, strongest local discord detection.",
                    "subsequence_multiplier": 1,
                },
                {
                    "variant_name": "Context x2",
                    "focus": "Intermediate subsequence context for medium-length anomalies.",
                    "subsequence_multiplier": 2,
                },
                {
                    "variant_name": "Context x4",
                    "focus": "Longer context for broad discord patterns.",
                    "subsequence_multiplier": 4,
                },
            ],
            "damp": [
                {
                    "variant_name": "Baseline",
                    "focus": "Reference streaming-discord configuration.",
                    "start_index_multiplier": 1.0,
                    "x_lag_multiplier": 0.0,
                },
                {
                    "variant_name": "Delayed Start",
                    "focus": "Longer historical reference before streaming detection starts.",
                    "start_index_multiplier": 2.0,
                    "x_lag_multiplier": 0.0,
                },
                {
                    "variant_name": "Long Lag",
                    "focus": "Searches further back in the stream for similar windows.",
                    "start_index_multiplier": 1.0,
                    "x_lag_multiplier": 8.0,
                },
            ],
            "hbos": [
                {
                    "variant_name": "Baseline",
                    "focus": "Lightweight histogram baseline.",
                    "n_bins": 10,
                    "alpha": 0.10,
                    "tol": 0.50,
                },
                {
                    "variant_name": "Fine Bins",
                    "focus": "Finer density structure through more bins and milder smoothing.",
                    "n_bins": 20,
                    "alpha": 0.05,
                    "tol": 0.50,
                },
                {
                    "variant_name": "Strict Tol",
                    "focus": "Stricter edge handling for stronger outlier penalties.",
                    "n_bins": 10,
                    "alpha": 0.10,
                    "tol": 0.20,
                },
            ],
            "ocsvm": [
                {
                    "variant_name": "Baseline",
                    "focus": "Standard RBF novelty boundary with short warm-up.",
                    "kernel": "rbf",
                    "nu": 0.05,
                    "gamma": "scale",
                    "train_fraction": 0.10,
                },
                {
                    "variant_name": "Nu 0.10",
                    "focus": "Tests more permissive abnormal-boundary calibration.",
                    "kernel": "rbf",
                    "nu": 0.10,
                    "gamma": "scale",
                    "train_fraction": 0.10,
                },
                {
                    "variant_name": "Warmup 0.20",
                    "focus": "Uses a longer mostly-normal prefix for model fitting.",
                    "kernel": "rbf",
                    "nu": 0.05,
                    "gamma": "scale",
                    "train_fraction": 0.20,
                },
                {
                    "variant_name": "Linear",
                    "focus": "Tests whether a simpler linear novelty boundary is sufficient.",
                    "kernel": "linear",
                    "nu": 0.05,
                    "gamma": "scale",
                    "train_fraction": 0.10,
                },
            ],
            "pca": [
                {
                    "variant_name": "Baseline",
                    "focus": "Weighted reconstruction-style baseline.",
                    "n_components": "",
                    "n_selected_components": 0,
                    "whiten": False,
                    "weighted": True,
                    "standardization": True,
                },
                {
                    "variant_name": "Residual 2",
                    "focus": "Focuses scoring on a small set of low-variance components.",
                    "n_components": 0.95,
                    "n_selected_components": 2,
                    "whiten": False,
                    "weighted": True,
                    "standardization": True,
                },
                {
                    "variant_name": "Whitened",
                    "focus": "Tests a whitened non-weighted PCA score.",
                    "n_components": 0.95,
                    "n_selected_components": 0,
                    "whiten": True,
                    "weighted": False,
                    "standardization": True,
                },
            ],
        },
    },
}

ensure_results_layout()

ALGORITHM_REGISTRY = {
    "isolation_forest": ("isolation_forest", "score_time_series"),
    "local_outlier_factor": ("local_outlier_factor", "score_time_series"),
    "sand": ("sand", "score_time_series"),
    "matrix_profile": ("matrix_profile", "score_time_series"),
    "damp": ("damp", "score_time_series"),
    "hbos": ("hbos", "score_time_series"),
    "ocsvm": ("ocsvm", "score_time_series"),
    "pca": ("pca", "score_time_series"),
}

TOOLTIP_TEXT = {
    "variant_label": "Name shown on the subtab for this argument set. Use it to label different experiments of the same algorithm, such as Baseline, Fast, or High Recall.",
    "dataset_limit": "How many prepared datasets to benchmark in this run. Use 0 to process every available dataset. Lower values are useful for smoke tests and fast iteration.",
    "normalization_method": "How raw dataset values are transformed before any algorithm runs. This controls the files created in datasets/normalized and changes the numeric scale each algorithm sees.",
    "clip_quantile": "Optional outlier clipping before normalization. Example: 0.01 clips the bottom 1% and top 1% of raw values. This can reduce extreme spikes, but it can also weaken anomaly contrast.",
    "overwrite_normalized": "Rebuild the normalized dataset files even if cached versions already exist. Turn this on after changing normalization settings when you want to force fresh prepared data.",
    "window_size": "Defines the temporal context length seen by window-based embedding, subsequence scoring, and range-oriented evaluation. Use 0 to keep the current automatic window estimation.",
    "window_stride": "Step between consecutive windows or subsequences. Smaller strides create denser overlap and higher runtime; larger strides reduce overlap and can lower computational cost.",
    "threshold_method": "Defines how anomaly scores become final detections after the scoring stage. Sigma uses a mean-plus-standard-deviation cutoff, quantile uses an upper score percentile, and top-k keeps only the strongest anomalies under a fixed budget.",
    "threshold_value": "Threshold parameter used by the selected threshold method. Sigma is useful for normalized score distributions, quantile is more robust across heterogeneous score scales, and top-k is useful when anomaly count is constrained or benchmark structure is known.",
    "evaluation_mode": "Controls whether detections are judged as anomalous ranges or as anomalous points. Range mode is better for interval anomalies; point mode is stricter and may underrepresent temporal overlap quality.",
    "save_scores": "Save one CSV per dataset and algorithm with the raw score trace. Useful for later inspection, but it creates many files during large benchmark runs.",
    "run_iforest": "Include Isolation Forest in the benchmark. Disable it when you want to compare only the other algorithms or reduce runtime.",
    "run_lof": "Include Local Outlier Factor in the benchmark. Disable it when you want a smaller run or cleaner comparisons.",
    "run_sand": "Include SAND in the benchmark. SAND is usually the slowest option, so disabling it is useful for fast iteration.",
    "run_matrix_profile": "Include Matrix Profile in the benchmark. This adds a discord-based baseline that highlights subsequences with unusually large nearest-neighbor distance.",
    "run_damp": "Include DAMP in the benchmark. This adds another discord-based method designed to handle repeated similar anomalies in an online-style setting.",
    "run_hbos": "Include HBOS in the benchmark. This adds a lightweight histogram-density baseline that is usually faster than the more complex methods.",
    "run_ocsvm": "Include One-Class SVM in the benchmark. This adds a classic boundary-based novelty detector over the windowed representation.",
    "run_pca": "Include PCA in the benchmark. This adds an encoding-based baseline that measures deviation along low-variance principal directions.",
    "if_n_estimators": "Number of trees in the Isolation Forest. More trees usually make the score distribution more stable, but they increase runtime and memory use.",
    "if_contamination": "Isolation Forest estimate of the anomaly fraction in the data. Higher values make the model more willing to isolate points as unusual and can shift score behavior.",
    "if_max_samples": "How many sliding windows each tree trains on. 'auto' lets sklearn choose a default. Smaller samples run faster; larger samples can capture more structure but cost more.",
    "if_max_features": "Fraction of features used when building each tree. Lower values inject more randomness; higher values use more of the window at once and can make trees less diverse.",
    "if_bootstrap": "Whether each tree samples windows with replacement. This can increase tree diversity, but it may also add variance to the final score curve.",
    "if_random_state": "Random seed for Isolation Forest. Keep it fixed for reproducible runs; change it if you want to test sensitivity to random initialization.",
    "lof_neighbors": "How many neighboring windows LOF uses to define local density. Smaller values react to local detail; larger values make the method smoother and more global.",
    "lof_contamination": "Expected anomaly proportion for LOF. This affects the model's outlier calibration and can change how aggressively low-density windows are treated as abnormal.",
    "lof_algorithm": "Neighbor-search backend used by sklearn. This mostly affects performance rather than detection logic; the best choice depends on data size and distance metric.",
    "lof_leaf_size": "Tree leaf size for ball-tree or kd-tree neighbor search. This is a performance knob, not a scoring-theory knob. It can matter when benchmarking many datasets.",
    "lof_metric": "Distance metric for comparing sliding windows in LOF. This changes what 'similarity' means, so it can materially change which windows are treated as outliers.",
    "lof_p": "Power parameter for the Minkowski metric. It matters when the LOF metric is Minkowski: p=1 is Manhattan distance, p=2 is Euclidean distance.",
    "sand_alpha": "Online update weight in SAND. Higher values give more influence to newer behavior, making the detector adapt faster but sometimes become less stable.",
    "sand_init_length": "How much of the time series SAND uses in its initialization phase before online updates continue. Larger values give a stronger starting model but increase startup cost.",
    "sand_batch_size": "Chunk size used as SAND moves through the series online. Larger batches reduce loop overhead but can make updates coarser and raise memory/runtime per step.",
    "sand_k": "Number of nearest subsequences SAND compares against. Use 0 for automatic selection. Larger values usually smooth the score estimate, while smaller values make it more local and sensitive.",
    "sand_subsequence_multiplier": "Multiplier that turns the base window size into the SAND subsequence length. Higher values let SAND compare longer motifs and context, but they also increase runtime.",
    "sand_overlap": "Step size between subsequences in SAND. Use 0 for the default based on window size. Smaller overlap values create denser comparisons and slower runs; larger values are faster but coarser.",
    "mp_subsequence_multiplier": "Multiplier that turns the estimated base window into the Matrix Profile subsequence length. Larger values compare longer motifs and can surface broader discord patterns, but they also increase runtime.",
    "damp_start_index_multiplier": "Multiplier that sets how far DAMP waits before starting its streaming-style discord search. Larger values delay the start and provide a longer historical reference.",
    "damp_x_lag_multiplier": "Optional multiplier for the DAMP lag horizon. Use 0 for the default internal heuristic. Larger values let DAMP search further back in time, which can improve context at the cost of runtime.",
    "hbos_n_bins": "Number of histogram bins used by HBOS for each window position. More bins capture finer density structure, while fewer bins make the model smoother and simpler.",
    "hbos_alpha": "Regularizer added inside the histogram-density score. It prevents numerical issues in sparse bins and slightly smooths the outlier score landscape.",
    "hbos_tol": "How tolerant HBOS is when a value falls just outside the learned histogram edges. Higher tolerance reduces harsh edge penalties; lower tolerance treats outside-bin values as more abnormal.",
    "hbos_contamination": "Expected anomaly proportion passed through to the HBOS configuration. This is mostly useful for aligning experiment settings with the other unsupervised baselines.",
    "ocsvm_kernel": "Kernel used by One-Class SVM to separate normal windows from the origin. This changes the geometry of the learned boundary and can strongly affect what is treated as abnormal.",
    "ocsvm_nu": "Upper bound on training errors and lower bound on the support-vector fraction in One-Class SVM. Larger values make the model more permissive about marking windows as abnormal.",
    "ocsvm_gamma": "Kernel coefficient for non-linear One-Class SVM kernels. Higher values make the boundary more local and sensitive; lower values make it smoother and more global.",
    "ocsvm_train_fraction": "Fraction of the earliest windows used to fit the One-Class SVM before scoring the full series. Lower values assume a shorter mostly-normal warm-up period.",
    "pca_n_components": "Total number of principal components to keep. Leave blank for the default PCA behavior, use a float like 0.95 for explained-variance selection, or an integer for an exact count.",
    "pca_n_selected_components": "How many of the lowest-variance principal components are used for the anomaly score. Use 0 for all retained components. Smaller values focus more on subtle residual structure.",
    "pca_whiten": "Whether PCA should whiten the retained components. Whitening removes scale differences between components and can change how strongly residual directions contribute to the score.",
    "pca_weighted": "Whether the PCA score should weight component distances by explained variance, giving lower-variance directions more influence as in the TSB-UAD-style formulation.",
    "pca_standardization": "Whether windows are standardized before PCA. This usually helps when window features have different scales, but it can also remove some amplitude information.",
}

pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 20)
pd.set_option("display.precision", 4)


@lru_cache(maxsize=1)
def _load_plotting_module():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (14, 8),
            "axes.grid": True,
            "grid.alpha": 0.2,
            "grid.linestyle": "--",
            "savefig.bbox": "tight",
        }
    )
    return plt


@lru_cache(maxsize=1)
def _load_sklearn_metric_functions():
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

    return {
        "average_precision_score": average_precision_score,
        "precision_recall_fscore_support": precision_recall_fscore_support,
        "roc_auc_score": roc_auc_score,
    }


@lru_cache(maxsize=None)
def _load_algorithm_function(algorithm_key: str):
    module_name, function_name = ALGORITHM_REGISTRY[algorithm_key]
    module = importlib.import_module(module_name)
    module = importlib.reload(module)
    return getattr(module, function_name)


def portable_path_str(path: Path | str) -> str:
    candidate = Path(path)
    for base in (PROJECT_ROOT, REPO_ROOT):
        try:
            return str(candidate.relative_to(base))
        except ValueError:
            continue
    return str(candidate)


def result_table_path(filename: str) -> Path:
    return RESULT_TABLES_DIR / filename


def result_per_algorithm_table_path(algorithm_key: str) -> Path:
    return RESULT_PER_ALGORITHM_TABLES_DIR / f"{algorithm_key}_results.csv"


def result_figure_path(filename: str) -> Path:
    return RESULT_FIGURES_DIR / filename


def result_algorithm_panel_path(algorithm_key: str) -> Path:
    return RESULT_ALGORITHM_PANEL_DIR / f"{algorithm_key}_benchmark_panel.png"


def result_algorithm_paper_panel_path(algorithm_key: str) -> Path:
    return RESULT_ALGORITHM_PANEL_DIR / f"{algorithm_key}_paper_panel.png"


def result_algorithm_variant_comparison_path(algorithm_key: str) -> Path:
    return RESULT_ALGORITHM_PANEL_DIR / f"{algorithm_key}_variant_comparison.png"


def result_deep_dive_path(run_id: str, dataset_name: str) -> Path:
    return RESULT_DEEP_DIVE_DIR / f"{run_id}__deep_dive__{dataset_name}.png"


def result_score_path(dataset_name: str, run_id: str) -> Path:
    return RESULT_SCORES_DIR / f"{dataset_name}__{run_id}.csv"


def build_selected_run_parameter_frame(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dataset_limit = "all" if config["dataset_limit"] is None else config["dataset_limit"]
    for run_config in config["selected_runs"]:
        params = dict(run_config["params"])
        row = {
            "dataset_limit": dataset_limit,
            "normalization_method": config["normalization_method"],
            "clip_quantile": config["clip_quantile"],
            "window_size": config["window_size"],
            "window_stride": config["window_stride"],
            "threshold_method": config["threshold_method"],
            "threshold_value": config["threshold_value"],
            "evaluation_mode": config["evaluation_mode"],
            "algorithm": run_config["algorithm"],
            "algorithm_base_display": run_config["algorithm_base_display"],
            "algorithm_display": run_config["algorithm_display"],
            "algorithm_variant": run_config["algorithm_variant"],
            "algorithm_run_id": run_config["algorithm_run_id"],
            "variant_index": run_config["variant_index"],
            "params_json": json.dumps(params, sort_keys=True),
        }
        for key, value in params.items():
            row[f"param__{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["algorithm", "variant_index"]).reset_index(drop=True)


def build_results_layout_frame() -> pd.DataFrame:
    rows = [
        ("results_root", RESULTS_DIR),
        ("tables", RESULT_TABLES_DIR),
        ("tables/per_algorithm", RESULT_PER_ALGORITHM_TABLES_DIR),
        ("figures", RESULT_FIGURES_DIR),
        ("figures/algorithm_panels", RESULT_ALGORITHM_PANEL_DIR),
        ("figures/deep_dives", RESULT_DEEP_DIVE_DIR),
        ("scores", RESULT_SCORES_DIR),
    ]
    return pd.DataFrame(
        [{"output_group": label, "path": portable_path_str(
            path), "exists": path.exists()} for label, path in rows]
    )


def _select_benchmark_dataset_paths(
    prepared_dataset_paths: list[Path],
    dataset_limit: int | None,
    deep_dive_dataset_name: str,
) -> tuple[list[Path], bool]:
    if dataset_limit is None:
        return prepared_dataset_paths, any(path.stem == deep_dive_dataset_name for path in prepared_dataset_paths)

    selected_paths = list(prepared_dataset_paths[:dataset_limit])
    selected_names = {path.stem for path in selected_paths}
    if deep_dive_dataset_name in selected_names:
        return selected_paths, True

    deep_dive_path = next(
        (path for path in prepared_dataset_paths if path.stem == deep_dive_dataset_name), None)
    if deep_dive_path is None:
        return selected_paths, False

    if not selected_paths:
        return [deep_dive_path], True

    # Keep the user-facing dataset limit stable, but guarantee the chosen deep-dive dataset is benchmarked.
    selected_paths[-1] = deep_dive_path
    return selected_paths, True


@lru_cache(maxsize=32)
def _ordered_prepared_dataset_names(
    normalization_method: str,
    clip_quantile: float | None,
) -> tuple[str, ...]:
    _prepared_dataset_dir, prepared_dataset_paths = ensure_normalized_datasets(
        normalization_method,
        clip_quantile,
        overwrite=False,
    )
    ordered_paths = sorted(prepared_dataset_paths,
                           key=lambda path: (path.stat().st_size, path.name))
    return tuple(path.stem for path in ordered_paths)


def _available_deep_dive_dataset_names(
    dataset_names: list[str],
    dataset_limit: int | None,
    normalization_method: str,
    clip_quantile: float | None,
) -> list[str]:
    fallback_names = list(dataset_names)
    if dataset_limit is not None:
        fallback_names = fallback_names[:dataset_limit]

    try:
        ordered_names = list(_ordered_prepared_dataset_names(
            normalization_method, clip_quantile))
    except Exception:
        ordered_names = fallback_names

    if dataset_limit is not None:
        ordered_names = ordered_names[:dataset_limit]
    return ordered_names


def parse_freeform_value(text: str) -> str | int | float:
    text = str(text).strip()
    if text.lower() == "auto":
        return "auto"
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _control_block(title: str, children: list[widgets.Widget], border_color: str) -> widgets.Widget:
    return widgets.VBox(
        [widgets.HTML(
            value=f"<h3 style='margin:0 0 8px 0;color:{border_color};'>{title}</h3>")] + children,
        layout=widgets.Layout(
            border=f"2px solid {border_color}", padding="12px", margin="6px 0 10px 0"),
    )


def _tooltip_icon(text: str) -> widgets.Widget:
    escaped = html.escape(text, quote=True)
    return widgets.HTML(
        value=(
            f"<span title=\"{escaped}\" "
            "style=\"cursor:help; display:inline-block; min-width:28px; padding:0 6px; color:#475569; font-weight:700;\">[?]</span>"
        ),
        layout=widgets.Layout(width="34px"),
    )


def _with_tooltip(widget: widgets.Widget, tooltip_key: str) -> widgets.Widget:
    tooltip_text = TOOLTIP_TEXT[tooltip_key]
    widget.tooltip = tooltip_text
    return widgets.Box(
        [widget, _tooltip_icon(tooltip_text)],
        layout=widgets.Layout(
            display="flex",
            flex_flow="row nowrap",
            align_items="center",
            width="auto",
            overflow="visible",
            margin="0 16px 8px 0",
        ),
    )


def _control_row(children: list[widgets.Widget]) -> widgets.Widget:
    return widgets.Box(
        children,
        layout=widgets.Layout(
            display="flex",
            flex_flow="row wrap",
            align_items="center",
            width="100%",
            overflow="visible",
            margin="0 0 6px 0",
        ),
    )


def _explanation_block(summary: str, tooltip_keys: list[str]) -> widgets.Widget:
    items = "".join(
        f"<li style='margin:0 0 6px 0;'><b>{html.escape(key)}</b>: {html.escape(TOOLTIP_TEXT[key])}</li>"
        for key in tooltip_keys
    )
    body = widgets.HTML(
        value=f"<ul style='margin:8px 0 0 18px; padding:0;'>{items}</ul>"
    )
    accordion = widgets.Accordion(children=[body], selected_index=None)
    accordion.set_title(0, summary)
    return accordion


def _legacy_build_control_panel(dataset_names: list[str]) -> dict[str, Any]:
    controls: dict[str, widgets.Widget] = {}

    controls["dataset_limit"] = widgets.IntText(
        value=0, description="Dataset limit", layout=widgets.Layout(width="220px"))
    controls["normalization_method"] = widgets.Dropdown(
        options=["none", "zscore", "minmax", "robust"],
        value="zscore",
        description="Normalize",
        layout=widgets.Layout(width="240px"),
    )
    controls["clip_quantile"] = widgets.FloatText(
        value=0.0, description="Clip q", layout=widgets.Layout(width="220px"))
    controls["overwrite_normalized"] = widgets.Checkbox(
        value=False, description="Rebuild normalized datasets")
    controls["window_override"] = widgets.IntText(
        value=0, description="Window override", layout=widgets.Layout(width="220px"))
    controls["threshold_std"] = widgets.FloatSlider(
        value=3.0,
        min=0.5,
        max=6.0,
        step=0.5,
        description="Threshold sigma",
        layout=widgets.Layout(width="340px"),
    )
    controls["deep_dive_dataset"] = widgets.Dropdown(
        options=dataset_names,
        value=dataset_names[0] if dataset_names else None,
        description="Deep dive",
        layout=widgets.Layout(width="650px"),
    )
    controls["save_scores"] = widgets.Checkbox(
        value=False, description="Save per-dataset scores")

    controls["run_iforest"] = widgets.Checkbox(
        value=True, description="Isolation Forest")
    controls["run_lof"] = widgets.Checkbox(
        value=True, description="Local Outlier Factor")
    controls["run_sand"] = widgets.Checkbox(value=False, description="SAND")

    controls["if_n_estimators"] = widgets.IntSlider(
        value=200, min=50, max=500, step=50, description="Trees", layout=widgets.Layout(width="320px"))
    controls["if_contamination"] = widgets.FloatSlider(
        value=0.10, min=0.01, max=0.30, step=0.01, readout_format=".2f", description="Contam.", layout=widgets.Layout(width="320px"))
    controls["if_max_samples"] = widgets.Text(
        value="auto", description="Max samples", layout=widgets.Layout(width="240px"))
    controls["if_max_features"] = widgets.FloatSlider(
        value=1.0, min=0.1, max=1.0, step=0.1, readout_format=".1f", description="Max feat.", layout=widgets.Layout(width="320px"))
    controls["if_bootstrap"] = widgets.Checkbox(
        value=False, description="Bootstrap")
    controls["if_random_state"] = widgets.IntText(
        value=42, description="Seed", layout=widgets.Layout(width="180px"))

    controls["lof_neighbors"] = widgets.IntSlider(
        value=20, min=2, max=100, step=1, description="Neighbors", layout=widgets.Layout(width="320px"))
    controls["lof_contamination"] = widgets.FloatSlider(
        value=0.10, min=0.01, max=0.30, step=0.01, readout_format=".2f", description="Contam.", layout=widgets.Layout(width="320px"))
    controls["lof_algorithm"] = widgets.Dropdown(
        options=["auto", "ball_tree", "kd_tree", "brute"], value="auto", description="Search", layout=widgets.Layout(width="260px"))
    controls["lof_leaf_size"] = widgets.IntSlider(
        value=30, min=5, max=100, step=5, description="Leaf size", layout=widgets.Layout(width="320px"))
    controls["lof_metric"] = widgets.Dropdown(options=["minkowski", "euclidean", "manhattan", "chebyshev"],
                                              value="minkowski", description="Metric", layout=widgets.Layout(width="260px"))
    controls["lof_p"] = widgets.IntSlider(
        value=2, min=1, max=5, step=1, description="p", layout=widgets.Layout(width="220px"))

    controls["sand_alpha"] = widgets.FloatSlider(
        value=0.5, min=0.1, max=0.9, step=0.1, readout_format=".1f", description="Alpha", layout=widgets.Layout(width="320px"))
    controls["sand_init_length"] = widgets.IntText(
        value=5000, description="Init length", layout=widgets.Layout(width="220px"))
    controls["sand_batch_size"] = widgets.IntText(
        value=2000, description="Batch size", layout=widgets.Layout(width="220px"))
    controls["sand_k"] = widgets.IntText(
        value=0, description="k (0 auto)", layout=widgets.Layout(width="220px"))
    controls["sand_subsequence_multiplier"] = widgets.IntSlider(
        value=4, min=1, max=8, step=1, description="Subseq x", layout=widgets.Layout(width="320px"))
    controls["sand_overlap"] = widgets.IntText(
        value=0, description="Overlap (0 auto)", layout=widgets.Layout(width="220px"))

    preview_output = widgets.Output()

    def render_preview(*_args: Any) -> None:
        selected = []
        if controls["run_iforest"].value:
            selected.append("Isolation Forest")
        if controls["run_lof"].value:
            selected.append("Local Outlier Factor")
        if controls["run_sand"].value:
            selected.append("SAND")
        preview_frame = pd.DataFrame(
            [
                {
                    "datasets to run": "all available" if controls["dataset_limit"].value <= 0 else controls["dataset_limit"].value,
                    "normalization": controls["normalization_method"].value,
                    "clip_quantile": None if controls["clip_quantile"].value <= 0 else controls["clip_quantile"].value,
                    "window_override": None if controls["window_override"].value <= 0 else controls["window_override"].value,
                    "threshold_sigma": controls["threshold_std"].value,
                    "deep_dive_dataset": controls["deep_dive_dataset"].value,
                    "selected_algorithms": ", ".join(selected) if selected else "none",
                    "save_scores": controls["save_scores"].value,
                }
            ]
        )
        preview_output.clear_output(wait=True)
        with preview_output:
            display(preview_frame)

    for widget in controls.values():
        if hasattr(widget, "observe"):
            widget.observe(render_preview, names="value")

    general_box = _control_block(
        "General Controls",
        [
            _control_row(
                [
                    _with_tooltip(controls["dataset_limit"], "dataset_limit"),
                    _with_tooltip(
                        controls["normalization_method"], "normalization_method"),
                    _with_tooltip(controls["clip_quantile"], "clip_quantile"),
                ]
            ),
            _control_row(
                [
                    _with_tooltip(
                        controls["window_override"], "window_override"),
                    _with_tooltip(controls["threshold_std"], "threshold_std"),
                ]
            ),
            _with_tooltip(controls["deep_dive_dataset"], "deep_dive_dataset"),
            _control_row(
                [
                    _with_tooltip(
                        controls["overwrite_normalized"], "overwrite_normalized"),
                    _with_tooltip(controls["save_scores"], "save_scores"),
                ]
            ),
            _control_row(
                [
                    _with_tooltip(controls["run_iforest"], "run_iforest"),
                    _with_tooltip(controls["run_lof"], "run_lof"),
                    _with_tooltip(controls["run_sand"], "run_sand"),
                ]
            ),
            _explanation_block(
                "Show general control explanations",
                [
                    "dataset_limit",
                    "normalization_method",
                    "clip_quantile",
                    "overwrite_normalized",
                    "window_override",
                    "threshold_std",
                    "deep_dive_dataset",
                    "save_scores",
                    "run_iforest",
                    "run_lof",
                    "run_sand",
                ],
            ),
        ],
        "#334155",
    )

    iforest_box = _control_block(
        "Isolation Forest Knobs",
        [
            _control_row(
                [
                    _with_tooltip(
                        controls["if_n_estimators"], "if_n_estimators"),
                    _with_tooltip(
                        controls["if_contamination"], "if_contamination"),
                ]
            ),
            _control_row(
                [
                    _with_tooltip(
                        controls["if_max_samples"], "if_max_samples"),
                    _with_tooltip(
                        controls["if_max_features"], "if_max_features"),
                ]
            ),
            _control_row(
                [
                    _with_tooltip(controls["if_bootstrap"], "if_bootstrap"),
                    _with_tooltip(
                        controls["if_random_state"], "if_random_state"),
                ]
            ),
            _explanation_block(
                "Show Isolation Forest knob explanations",
                [
                    "if_n_estimators",
                    "if_contamination",
                    "if_max_samples",
                    "if_max_features",
                    "if_bootstrap",
                    "if_random_state",
                ],
            ),
        ],
        "#1d4ed8",
    )

    lof_box = _control_block(
        "Local Outlier Factor Knobs",
        [
            _control_row(
                [
                    _with_tooltip(controls["lof_neighbors"], "lof_neighbors"),
                    _with_tooltip(
                        controls["lof_contamination"], "lof_contamination"),
                ]
            ),
            _control_row(
                [
                    _with_tooltip(controls["lof_algorithm"], "lof_algorithm"),
                    _with_tooltip(controls["lof_leaf_size"], "lof_leaf_size"),
                ]
            ),
            _control_row(
                [
                    _with_tooltip(controls["lof_metric"], "lof_metric"),
                    _with_tooltip(controls["lof_p"], "lof_p"),
                ]
            ),
            _explanation_block(
                "Show Local Outlier Factor knob explanations",
                [
                    "lof_neighbors",
                    "lof_contamination",
                    "lof_algorithm",
                    "lof_leaf_size",
                    "lof_metric",
                    "lof_p",
                ],
            ),
        ],
        "#ea580c",
    )

    sand_box = _control_block(
        "SAND Knobs",
        [
            _control_row(
                [
                    _with_tooltip(controls["sand_alpha"], "sand_alpha"),
                    _with_tooltip(
                        controls["sand_subsequence_multiplier"], "sand_subsequence_multiplier"),
                ]
            ),
            _control_row(
                [
                    _with_tooltip(
                        controls["sand_init_length"], "sand_init_length"),
                    _with_tooltip(
                        controls["sand_batch_size"], "sand_batch_size"),
                ]
            ),
            _control_row(
                [
                    _with_tooltip(controls["sand_k"], "sand_k"),
                    _with_tooltip(controls["sand_overlap"], "sand_overlap"),
                ]
            ),
            _explanation_block(
                "Show SAND knob explanations",
                [
                    "sand_alpha",
                    "sand_init_length",
                    "sand_batch_size",
                    "sand_k",
                    "sand_subsequence_multiplier",
                    "sand_overlap",
                ],
            ),
        ],
        "#15803d",
    )

    render_preview()

    algorithm_tabs = widgets.Tab(
        children=[iforest_box, lof_box, sand_box],
        layout=widgets.Layout(margin="6px 0 10px 0"),
    )
    algorithm_tabs.set_title(0, "Isolation Forest")
    algorithm_tabs.set_title(1, "Local Outlier Factor")
    algorithm_tabs.set_title(2, "SAND")

    panel = widgets.VBox(
        [
            widgets.HTML(
                "<h2 style='margin-top:0;'>Control Panel</h2>"
                "<p>Change the knobs here, then rerun the configuration and benchmark cells below.</p>"
                "<p><b>General controls</b> stay visible, and each algorithm has its own tab so the parameter boundaries are obvious.</p>"
                "<p>Each knob has a <b>[?]</b> marker, and each section also has a visible explanation block you can expand inside the panel.</p>"
            ),
            general_box,
            algorithm_tabs,
            widgets.HTML(
                "<h3 style='margin:8px 0 4px 0;'>Current Selection</h3>"),
            preview_output,
        ]
    )

    return {"controls": controls, "panel": panel}


def _default_variant_name(index: int) -> str:
    return "Baseline" if index == 1 else f"Variant {index}"


def _snapshot_widget_values(widget_map: dict[str, widgets.Widget]) -> dict[str, Any]:
    return {
        key: widget.value
        for key, widget in widget_map.items()
        if hasattr(widget, "value")
    }


def _apply_widget_values(widget_map: dict[str, widgets.Widget], values: dict[str, Any] | None) -> None:
    if not values:
        return
    for key, value in values.items():
        if key in widget_map and hasattr(widget_map[key], "value"):
            widget = widget_map[key]
            if isinstance(widget, widgets.Text):
                widget.value = "" if value is None else str(value)
            else:
                widget.value = value


def _make_if_variant(
    default_name: str,
    initial_values: dict[str, Any] | None,
    register_widget: Any,
) -> dict[str, Any]:
    variant_controls = {
        "variant_name": widgets.Text(value=default_name, description="Tab label", layout=widgets.Layout(width="240px")),
        "n_estimators": widgets.IntSlider(value=200, min=50, max=500, step=50, description="Trees", layout=widgets.Layout(width="320px")),
        "max_samples": widgets.Text(value="auto", description="Max samples", layout=widgets.Layout(width="240px")),
        "max_features": widgets.FloatSlider(value=1.0, min=0.1, max=1.0, step=0.1, readout_format=".1f", description="Max feat.", layout=widgets.Layout(width="320px")),
        "bootstrap": widgets.Checkbox(value=False, description="Bootstrap"),
        "random_state": widgets.IntText(value=42, description="Seed", layout=widgets.Layout(width="180px")),
    }
    _apply_widget_values(variant_controls, initial_values)
    for widget in variant_controls.values():
        register_widget(widget)
    panel = _control_block(
        "Isolation Forest Variant",
        [
            _control_row(
                [_with_tooltip(variant_controls["variant_name"], "variant_label")]),
            _control_row(
                [
                    _with_tooltip(
                        variant_controls["n_estimators"], "if_n_estimators"),
                    _with_tooltip(
                        variant_controls["max_features"], "if_max_features"),
                ]
            ),
            _control_row(
                [
                    _with_tooltip(
                        variant_controls["max_samples"], "if_max_samples"),
                    _with_tooltip(
                        variant_controls["bootstrap"], "if_bootstrap"),
                ]
            ),
            _control_row(
                [
                    _with_tooltip(
                        variant_controls["random_state"], "if_random_state"),
                ]
            ),
            _explanation_block(
                "Show Isolation Forest knob explanations",
                [
                    "variant_label",
                    "if_n_estimators",
                    "if_max_samples",
                    "if_max_features",
                    "if_bootstrap",
                    "if_random_state",
                ],
            ),
        ],
        "#1d4ed8",
    )
    return {"controls": variant_controls, "panel": panel}


def _make_lof_variant(
    default_name: str,
    initial_values: dict[str, Any] | None,
    register_widget: Any,
) -> dict[str, Any]:
    variant_controls = {
        "variant_name": widgets.Text(value=default_name, description="Tab label", layout=widgets.Layout(width="240px")),
        "n_neighbors": widgets.IntSlider(value=20, min=2, max=100, step=1, description="Neighbors", layout=widgets.Layout(width="320px")),
        "algorithm": widgets.Dropdown(options=["auto", "ball_tree", "kd_tree", "brute"], value="auto", description="Search", layout=widgets.Layout(width="260px")),
        "leaf_size": widgets.IntSlider(value=30, min=5, max=100, step=5, description="Leaf size", layout=widgets.Layout(width="320px")),
        "metric": widgets.Dropdown(options=["minkowski", "euclidean", "manhattan", "chebyshev"], value="minkowski", description="Metric", layout=widgets.Layout(width="260px")),
        "p": widgets.IntSlider(value=2, min=1, max=5, step=1, description="p", layout=widgets.Layout(width="220px")),
    }
    _apply_widget_values(variant_controls, initial_values)
    for widget in variant_controls.values():
        register_widget(widget)
    panel = _control_block(
        "Local Outlier Factor Variant",
        [
            _control_row(
                [_with_tooltip(variant_controls["variant_name"], "variant_label")]),
            _control_row(
                [
                    _with_tooltip(
                        variant_controls["n_neighbors"], "lof_neighbors"),
                    _with_tooltip(
                        variant_controls["metric"], "lof_metric"),
                ]
            ),
            _control_row(
                [
                    _with_tooltip(
                        variant_controls["algorithm"], "lof_algorithm"),
                    _with_tooltip(
                        variant_controls["leaf_size"], "lof_leaf_size"),
                ]
            ),
            _control_row(
                [
                    _with_tooltip(variant_controls["p"], "lof_p"),
                ]
            ),
            _explanation_block(
                "Show Local Outlier Factor knob explanations",
                [
                    "variant_label",
                    "lof_neighbors",
                    "lof_algorithm",
                    "lof_leaf_size",
                    "lof_metric",
                    "lof_p",
                ],
            ),
        ],
        "#ea580c",
    )
    return {"controls": variant_controls, "panel": panel}


def _make_sand_variant(
    default_name: str,
    initial_values: dict[str, Any] | None,
    register_widget: Any,
) -> dict[str, Any]:
    variant_controls = {
        "variant_name": widgets.Text(value=default_name, description="Tab label", layout=widgets.Layout(width="240px")),
        "alpha": widgets.FloatSlider(value=0.5, min=0.1, max=0.9, step=0.1, readout_format=".1f", description="Alpha", layout=widgets.Layout(width="320px")),
        "init_length": widgets.IntText(value=5000, description="Init length", layout=widgets.Layout(width="220px")),
        "batch_size": widgets.IntText(value=2000, description="Batch size", layout=widgets.Layout(width="220px")),
        "k": widgets.IntText(value=0, description="k (0 auto)", layout=widgets.Layout(width="220px")),
        "subsequence_multiplier": widgets.IntSlider(value=4, min=1, max=8, step=1, description="Subseq x", layout=widgets.Layout(width="320px")),
        "overlap": widgets.IntText(value=0, description="Overlap (0 auto)", layout=widgets.Layout(width="220px")),
    }
    _apply_widget_values(variant_controls, initial_values)
    for widget in variant_controls.values():
        register_widget(widget)
    panel = _control_block(
        "SAND Variant",
        [
            _control_row(
                [_with_tooltip(variant_controls["variant_name"], "variant_label")]),
            _control_row(
                [
                    _with_tooltip(variant_controls["alpha"], "sand_alpha"),
                    _with_tooltip(
                        variant_controls["subsequence_multiplier"], "sand_subsequence_multiplier"),
                ]
            ),
            _control_row(
                [
                    _with_tooltip(
                        variant_controls["init_length"], "sand_init_length"),
                    _with_tooltip(
                        variant_controls["batch_size"], "sand_batch_size"),
                ]
            ),
            _control_row(
                [
                    _with_tooltip(variant_controls["k"], "sand_k"),
                    _with_tooltip(variant_controls["overlap"], "sand_overlap"),
                ]
            ),
            _explanation_block(
                "Show SAND knob explanations",
                [
                    "variant_label",
                    "sand_alpha",
                    "sand_init_length",
                    "sand_batch_size",
                    "sand_k",
                    "sand_subsequence_multiplier",
                    "sand_overlap",
                ],
            ),
        ],
        "#15803d",
    )
    return {"controls": variant_controls, "panel": panel}


def _make_matrix_profile_variant(
    default_name: str,
    initial_values: dict[str, Any] | None,
    register_widget: Any,
) -> dict[str, Any]:
    variant_controls = {
        "variant_name": widgets.Text(value=default_name, description="Tab label", layout=widgets.Layout(width="240px")),
        "subsequence_multiplier": widgets.IntSlider(
            value=1,
            min=1,
            max=6,
            step=1,
            description="Subseq x",
            layout=widgets.Layout(width="320px"),
        ),
    }
    _apply_widget_values(variant_controls, initial_values)
    for widget in variant_controls.values():
        register_widget(widget)
    panel = _control_block(
        "Matrix Profile Variant",
        [
            _control_row(
                [_with_tooltip(variant_controls["variant_name"], "variant_label")]),
            _control_row(
                [
                    _with_tooltip(
                        variant_controls["subsequence_multiplier"], "mp_subsequence_multiplier"),
                ]
            ),
            _explanation_block(
                "Show Matrix Profile knob explanations",
                [
                    "variant_label",
                    "mp_subsequence_multiplier",
                ],
            ),
        ],
        ALGORITHM_METADATA["matrix_profile"]["border_color"],
    )
    return {"controls": variant_controls, "panel": panel}


def _make_hbos_variant(
    default_name: str,
    initial_values: dict[str, Any] | None,
    register_widget: Any,
) -> dict[str, Any]:
    variant_controls = {
        "variant_name": widgets.Text(value=default_name, description="Tab label", layout=widgets.Layout(width="240px")),
        "n_bins": widgets.IntSlider(value=10, min=4, max=40, step=2, description="Bins", layout=widgets.Layout(width="320px")),
        "alpha": widgets.FloatSlider(value=0.10, min=0.01, max=0.50, step=0.01, readout_format=".2f", description="Alpha", layout=widgets.Layout(width="320px")),
        "tol": widgets.FloatSlider(value=0.50, min=0.10, max=1.00, step=0.05, readout_format=".2f", description="Tol", layout=widgets.Layout(width="320px")),
    }
    _apply_widget_values(variant_controls, initial_values)
    for widget in variant_controls.values():
        register_widget(widget)
    panel = _control_block(
        "HBOS Variant",
        [
            _control_row(
                [_with_tooltip(variant_controls["variant_name"], "variant_label")]),
            _control_row(
                [
                    _with_tooltip(variant_controls["n_bins"], "hbos_n_bins"),
                    _with_tooltip(variant_controls["alpha"], "hbos_alpha"),
                ]
            ),
            _control_row(
                [
                    _with_tooltip(variant_controls["tol"], "hbos_tol"),
                ]
            ),
            _explanation_block(
                "Show HBOS knob explanations",
                [
                    "variant_label",
                    "hbos_n_bins",
                    "hbos_alpha",
                    "hbos_tol",
                ],
            ),
        ],
        ALGORITHM_METADATA["hbos"]["border_color"],
    )
    return {"controls": variant_controls, "panel": panel}


def _make_damp_variant(
    default_name: str,
    initial_values: dict[str, Any] | None,
    register_widget: Any,
) -> dict[str, Any]:
    variant_controls = {
        "variant_name": widgets.Text(value=default_name, description="Tab label", layout=widgets.Layout(width="240px")),
        "start_index_multiplier": widgets.FloatSlider(value=1.0, min=1.0, max=8.0, step=0.5, readout_format=".1f", description="Start x", layout=widgets.Layout(width="320px")),
        "x_lag_multiplier": widgets.FloatText(value=0.0, description="x_lag x", layout=widgets.Layout(width="220px")),
    }
    _apply_widget_values(variant_controls, initial_values)
    for widget in variant_controls.values():
        register_widget(widget)
    panel = _control_block(
        "DAMP Variant",
        [
            _control_row(
                [_with_tooltip(variant_controls["variant_name"], "variant_label")]),
            _control_row(
                [
                    _with_tooltip(
                        variant_controls["start_index_multiplier"], "damp_start_index_multiplier"),
                    _with_tooltip(
                        variant_controls["x_lag_multiplier"], "damp_x_lag_multiplier"),
                ]
            ),
            _explanation_block(
                "Show DAMP knob explanations",
                [
                    "variant_label",
                    "damp_start_index_multiplier",
                    "damp_x_lag_multiplier",
                ],
            ),
        ],
        ALGORITHM_METADATA["damp"]["border_color"],
    )
    return {"controls": variant_controls, "panel": panel}


def _make_ocsvm_variant(
    default_name: str,
    initial_values: dict[str, Any] | None,
    register_widget: Any,
) -> dict[str, Any]:
    variant_controls = {
        "variant_name": widgets.Text(value=default_name, description="Tab label", layout=widgets.Layout(width="240px")),
        "kernel": widgets.Dropdown(options=["rbf", "linear", "poly", "sigmoid"], value="rbf", description="Kernel", layout=widgets.Layout(width="240px")),
        "nu": widgets.FloatSlider(value=0.05, min=0.01, max=0.50, step=0.01, readout_format=".2f", description="Nu", layout=widgets.Layout(width="320px")),
        "gamma": widgets.Text(value="scale", description="Gamma", layout=widgets.Layout(width="220px")),
        "train_fraction": widgets.FloatSlider(value=0.10, min=0.05, max=0.50, step=0.05, readout_format=".2f", description="Train frac", layout=widgets.Layout(width="320px")),
    }
    _apply_widget_values(variant_controls, initial_values)
    for widget in variant_controls.values():
        register_widget(widget)
    panel = _control_block(
        "OCSVM Variant",
        [
            _control_row(
                [_with_tooltip(variant_controls["variant_name"], "variant_label")]),
            _control_row(
                [
                    _with_tooltip(variant_controls["kernel"], "ocsvm_kernel"),
                    _with_tooltip(variant_controls["nu"], "ocsvm_nu"),
                ]
            ),
            _control_row(
                [
                    _with_tooltip(variant_controls["gamma"], "ocsvm_gamma"),
                    _with_tooltip(
                        variant_controls["train_fraction"], "ocsvm_train_fraction"),
                ]
            ),
            _explanation_block(
                "Show OCSVM knob explanations",
                [
                    "variant_label",
                    "ocsvm_kernel",
                    "ocsvm_nu",
                    "ocsvm_gamma",
                    "ocsvm_train_fraction",
                ],
            ),
        ],
        ALGORITHM_METADATA["ocsvm"]["border_color"],
    )
    return {"controls": variant_controls, "panel": panel}


def _make_pca_variant(
    default_name: str,
    initial_values: dict[str, Any] | None,
    register_widget: Any,
) -> dict[str, Any]:
    variant_controls = {
        "variant_name": widgets.Text(value=default_name, description="Tab label", layout=widgets.Layout(width="240px")),
        "n_components": widgets.Text(value="", description="Components", layout=widgets.Layout(width="220px")),
        "n_selected_components": widgets.IntText(value=0, description="Score comps", layout=widgets.Layout(width="220px")),
        "whiten": widgets.Checkbox(value=False, description="Whiten"),
        "weighted": widgets.Checkbox(value=True, description="Weighted"),
        "standardization": widgets.Checkbox(value=True, description="Standardize"),
    }
    _apply_widget_values(variant_controls, initial_values)
    for widget in variant_controls.values():
        register_widget(widget)
    panel = _control_block(
        "PCA Variant",
        [
            _control_row(
                [_with_tooltip(variant_controls["variant_name"], "variant_label")]),
            _control_row(
                [
                    _with_tooltip(
                        variant_controls["n_components"], "pca_n_components"),
                    _with_tooltip(
                        variant_controls["n_selected_components"], "pca_n_selected_components"),
                ]
            ),
            _control_row(
                [
                    _with_tooltip(variant_controls["whiten"], "pca_whiten"),
                    _with_tooltip(
                        variant_controls["weighted"], "pca_weighted"),
                    _with_tooltip(
                        variant_controls["standardization"], "pca_standardization"),
                ]
            ),
            _explanation_block(
                "Show PCA knob explanations",
                [
                    "variant_label",
                    "pca_n_components",
                    "pca_n_selected_components",
                    "pca_whiten",
                    "pca_weighted",
                    "pca_standardization",
                ],
            ),
        ],
        ALGORITHM_METADATA["pca"]["border_color"],
    )
    return {"controls": variant_controls, "panel": panel}


def _build_variant_manager(
    algorithm_key: str,
    factory: Any,
    register_widget: Any,
    render_preview: Any,
) -> dict[str, Any]:
    variant_tabs = widgets.Tab(layout=widgets.Layout(
        width="100%", margin="6px 0 0 0"))
    add_button = widgets.Button(
        description="+", tooltip="Duplicate the current parameter tab", layout=widgets.Layout(width="42px"))
    remove_button = widgets.Button(
        description="-", tooltip="Close the current parameter tab", layout=widgets.Layout(width="42px"))
    header_note = widgets.HTML(
        "<b>Parameter subtabs</b>: use <b>+</b> to duplicate the current argument set and compare versions like browser tabs."
    )
    state = {
        "algorithm_key": algorithm_key,
        "tabs": variant_tabs,
        "variants": [],
        "add_button": add_button,
        "remove_button": remove_button,
    }

    def sync_titles(*_args: Any) -> None:
        variant_tabs.children = [entry["panel"] for entry in state["variants"]]
        for index, entry in enumerate(state["variants"], start=1):
            label = entry["controls"]["variant_name"].value.strip(
            ) or _default_variant_name(index)
            variant_tabs.set_title(index - 1, label)
        remove_button.disabled = len(state["variants"]) <= 1
        if state["variants"] and variant_tabs.selected_index is None:
            variant_tabs.selected_index = 0

    def snapshot_current_variant() -> dict[str, Any] | None:
        if not state["variants"]:
            return None
        current_index = variant_tabs.selected_index if variant_tabs.selected_index is not None else 0
        return _snapshot_widget_values(state["variants"][current_index]["controls"])

    def add_variant(source_values: dict[str, Any] | None = None) -> None:
        new_index = len(state["variants"]) + 1
        initial_values = dict(source_values or {})
        initial_values.setdefault(
            "variant_name", _default_variant_name(new_index))
        entry = factory(_default_variant_name(new_index),
                        initial_values, register_widget)
        entry["controls"]["variant_name"].observe(sync_titles, names="value")
        state["variants"].append(entry)
        sync_titles()

    def replace_variants(variant_values: list[dict[str, Any]]) -> None:
        state["variants"].clear()
        for values in variant_values or [{}]:
            add_variant(values)
        if state["variants"]:
            variant_tabs.selected_index = 0
        render_preview()

    def on_add(_button: widgets.Button) -> None:
        add_variant(snapshot_current_variant())
        variant_tabs.selected_index = len(state["variants"]) - 1
        render_preview()

    def on_remove(_button: widgets.Button) -> None:
        if len(state["variants"]) <= 1:
            return
        current_index = variant_tabs.selected_index if variant_tabs.selected_index is not None else len(
            state["variants"]) - 1
        del state["variants"][current_index]
        sync_titles()
        variant_tabs.selected_index = max(
            0, min(current_index, len(state["variants"]) - 1))
        render_preview()

    add_button.on_click(on_add)
    remove_button.on_click(on_remove)
    add_variant()

    state["container"] = widgets.VBox(
        [
            _control_row([header_note, add_button, remove_button]),
            variant_tabs,
        ],
        layout=widgets.Layout(width="100%"),
    )
    state["replace_variants"] = replace_variants
    return state


def build_control_panel(dataset_names: list[str]) -> dict[str, Any]:
    controls: dict[str, Any] = {}
    preview_output = widgets.Output()
    algorithm_variants: dict[str, Any] = {}
    threshold_defaults = {"sigma": 3.0, "quantile": 0.995, "top_k": 1}
    threshold_value_slot = widgets.VBox()

    def render_preview(*_args: Any) -> None:
        selected = []
        selected_configuration_count = 0
        variant_summary = []

        for algorithm_key, enabled_key in (
            ("isolation_forest", "run_iforest"),
            ("local_outlier_factor", "run_lof"),
            ("sand", "run_sand"),
            ("matrix_profile", "run_matrix_profile"),
            ("damp", "run_damp"),
            ("hbos", "run_hbos"),
            ("ocsvm", "run_ocsvm"),
            ("pca", "run_pca"),
        ):
            if enabled_key in controls and controls[enabled_key].value and algorithm_key in algorithm_variants:
                variant_entries = algorithm_variants[algorithm_key]["variants"]
                variant_count = len(variant_entries)
                selected_configuration_count += variant_count
                selected.append(
                    f"{DISPLAY_NAME_MAP[algorithm_key]} x{variant_count}")
                variant_labels = [entry["controls"]["variant_name"].value.strip(
                ) or f"Variant {index + 1}" for index, entry in enumerate(variant_entries)]
                variant_summary.append(
                    f"{DISPLAY_NAME_MAP[algorithm_key]}: {', '.join(variant_labels)}")

        preview_frame = pd.DataFrame(
            [
                {
                    "datasets to run": "all available" if controls["dataset_limit"].value <= 0 else controls["dataset_limit"].value,
                    "normalization": controls["normalization_method"].value,
                    "clip_quantile": None if controls["clip_quantile"].value <= 0 else controls["clip_quantile"].value,
                    "window_size": None if controls["window_size"].value <= 0 else controls["window_size"].value,
                    "window_stride": max(1, int(controls["window_stride"].value)),
                    "threshold_method": controls["threshold_method"].value,
                    "threshold_value": controls["threshold_value"].value,
                    "evaluation_mode": controls["evaluation_mode"].value,
                    "selected_algorithms": ", ".join(selected) if selected else "none",
                    "selected_configurations": selected_configuration_count,
                    "variant_tabs": " | ".join(variant_summary) if variant_summary else "none",
                    "save_scores": controls["save_scores"].value,
                }
            ]
        )
        preview_output.clear_output(wait=True)
        with preview_output:
            display(preview_frame)

    def register_widget(widget: widgets.Widget) -> None:
        if hasattr(widget, "observe"):
            widget.observe(render_preview, names="value")

    def build_threshold_value_widget(method: str, value: float | int | None = None) -> widgets.Widget:
        if method == "top_k":
            widget = widgets.IntText(
                value=max(
                    1, int(round(threshold_defaults[method] if value is None else value))),
                description="Top-k anomalies",
                layout=widgets.Layout(width="240px"),
            )
        else:
            widget = widgets.FloatText(
                value=float(threshold_defaults[method]
                            if value is None else value),
                description="Threshold sigma" if method == "sigma" else "Threshold quantile",
                layout=widgets.Layout(width="240px"),
            )
        register_widget(widget)
        return widget

    def refresh_threshold_value_control(*_args: Any) -> None:
        method = str(controls["threshold_method"].value)
        next_value = threshold_defaults[method]
        controls["threshold_value"] = build_threshold_value_widget(
            method, next_value)
        threshold_value_slot.children = [_with_tooltip(
            controls["threshold_value"], "threshold_value")]
        render_preview()

    controls["dataset_limit"] = widgets.IntText(
        value=0, description="Dataset limit", layout=widgets.Layout(width="220px"))
    controls["normalization_method"] = widgets.Dropdown(
        options=["none", "zscore", "minmax", "robust"],
        value="zscore",
        description="Normalize",
        layout=widgets.Layout(width="240px"),
    )
    controls["clip_quantile"] = widgets.FloatText(
        value=0.0, description="Clip q", layout=widgets.Layout(width="220px"))
    controls["overwrite_normalized"] = widgets.Checkbox(
        value=False, description="Rebuild normalized datasets")
    controls["window_size"] = widgets.IntText(
        value=0, description="Window size", layout=widgets.Layout(width="220px"))
    controls["window_stride"] = widgets.IntText(
        value=1, description="Window stride", layout=widgets.Layout(width="220px"))
    controls["threshold_method"] = widgets.Dropdown(
        options=["sigma", "quantile", "top_k"],
        value="sigma",
        description="Threshold method",
        layout=widgets.Layout(width="240px"),
    )
    controls["evaluation_mode"] = widgets.Dropdown(
        options=["range", "point"],
        value="range",
        description="Evaluation mode",
        layout=widgets.Layout(width="220px"),
    )
    controls["save_scores"] = widgets.Checkbox(
        value=False, description="Save per-dataset scores")
    controls["run_iforest"] = widgets.Checkbox(
        value=True, description="Isolation Forest")
    controls["run_lof"] = widgets.Checkbox(
        value=True, description="Local Outlier Factor")
    controls["run_sand"] = widgets.Checkbox(value=False, description="SAND")
    controls["run_matrix_profile"] = widgets.Checkbox(
        value=True, description="Matrix Profile")
    controls["run_damp"] = widgets.Checkbox(value=False, description="DAMP")
    controls["run_hbos"] = widgets.Checkbox(value=True, description="HBOS")
    controls["run_ocsvm"] = widgets.Checkbox(value=False, description="OCSVM")
    controls["run_pca"] = widgets.Checkbox(value=False, description="PCA")

    for key in [
        "dataset_limit",
        "normalization_method",
        "clip_quantile",
        "overwrite_normalized",
        "window_size",
        "window_stride",
        "threshold_method",
        "evaluation_mode",
        "save_scores",
        "run_iforest",
        "run_lof",
        "run_sand",
        "run_matrix_profile",
        "run_damp",
        "run_hbos",
        "run_ocsvm",
        "run_pca",
    ]:
        register_widget(controls[key])

    controls["threshold_method"].observe(
        refresh_threshold_value_control, names="value")

    algorithm_variants["isolation_forest"] = _build_variant_manager(
        "isolation_forest", _make_if_variant, register_widget, render_preview)
    algorithm_variants["local_outlier_factor"] = _build_variant_manager(
        "local_outlier_factor", _make_lof_variant, register_widget, render_preview)
    algorithm_variants["sand"] = _build_variant_manager(
        "sand", _make_sand_variant, register_widget, render_preview)
    algorithm_variants["matrix_profile"] = _build_variant_manager(
        "matrix_profile", _make_matrix_profile_variant, register_widget, render_preview)
    algorithm_variants["damp"] = _build_variant_manager(
        "damp", _make_damp_variant, register_widget, render_preview)
    algorithm_variants["hbos"] = _build_variant_manager(
        "hbos", _make_hbos_variant, register_widget, render_preview)
    algorithm_variants["ocsvm"] = _build_variant_manager(
        "ocsvm", _make_ocsvm_variant, register_widget, render_preview)
    algorithm_variants["pca"] = _build_variant_manager(
        "pca", _make_pca_variant, register_widget, render_preview)
    controls["algorithm_variants"] = algorithm_variants

    general_box = _control_block(
        "General Controls",
        [
            _control_row(
                [
                    _with_tooltip(controls["dataset_limit"], "dataset_limit"),
                    _with_tooltip(
                        controls["normalization_method"], "normalization_method"),
                    _with_tooltip(controls["clip_quantile"], "clip_quantile"),
                ]
            ),
            _control_row(
                [
                    _with_tooltip(controls["window_size"], "window_size"),
                    _with_tooltip(controls["window_stride"], "window_stride"),
                ]
            ),
            _control_row(
                [
                    _with_tooltip(
                        controls["threshold_method"], "threshold_method"),
                    threshold_value_slot,
                    _with_tooltip(
                        controls["evaluation_mode"], "evaluation_mode"),
                ]
            ),
            _control_row(
                [
                    _with_tooltip(
                        controls["overwrite_normalized"], "overwrite_normalized"),
                    _with_tooltip(controls["save_scores"], "save_scores"),
                ]
            ),
            _control_row(
                [
                    _with_tooltip(controls["run_iforest"], "run_iforest"),
                    _with_tooltip(controls["run_lof"], "run_lof"),
                    _with_tooltip(controls["run_sand"], "run_sand"),
                    _with_tooltip(
                        controls["run_matrix_profile"], "run_matrix_profile"),
                    _with_tooltip(controls["run_damp"], "run_damp"),
                    _with_tooltip(controls["run_hbos"], "run_hbos"),
                    _with_tooltip(controls["run_ocsvm"], "run_ocsvm"),
                    _with_tooltip(controls["run_pca"], "run_pca"),
                ]
            ),
            _explanation_block(
                "Show general control explanations",
                [
                    "dataset_limit",
                    "normalization_method",
                    "clip_quantile",
                    "overwrite_normalized",
                    "window_size",
                    "window_stride",
                    "threshold_value",
                    "evaluation_mode",
                    "save_scores",
                    "run_iforest",
                    "run_lof",
                    "run_sand",
                    "run_matrix_profile",
                    "run_damp",
                    "run_hbos",
                    "run_ocsvm",
                    "run_pca",
                ],
            ),
        ],
        "#334155",
    )

    refresh_threshold_value_control()
    render_preview()

    algorithm_tabs = widgets.Tab(
        children=[
            algorithm_variants["isolation_forest"]["container"],
            algorithm_variants["local_outlier_factor"]["container"],
            algorithm_variants["sand"]["container"],
            algorithm_variants["matrix_profile"]["container"],
            algorithm_variants["damp"]["container"],
            algorithm_variants["hbos"]["container"],
            algorithm_variants["ocsvm"]["container"],
            algorithm_variants["pca"]["container"],
        ],
        layout=widgets.Layout(margin="6px 0 10px 0"),
    )
    algorithm_tabs.set_title(0, "Isolation Forest")
    algorithm_tabs.set_title(1, "Local Outlier Factor")
    algorithm_tabs.set_title(2, "SAND")
    algorithm_tabs.set_title(3, "Matrix Profile")
    algorithm_tabs.set_title(4, "DAMP")
    algorithm_tabs.set_title(5, "HBOS")
    algorithm_tabs.set_title(6, "OCSVM")
    algorithm_tabs.set_title(7, "PCA")

    panel = widgets.VBox(
        [
            widgets.HTML(
                "<h2 style='margin-top:0;'>Control Panel</h2>"
                "<p>Change the knobs here, then rerun the configuration and benchmark cells below.</p>"
                "<p><b>General controls</b> stay visible, and each algorithm has its own tab so the parameter boundaries are obvious.</p>"
                "<p>Inside every algorithm tab, use the subtab bar like a browser: <b>+</b> duplicates the current argument set so you can benchmark multiple variants of the same algorithm in one run.</p>"
            ),
            general_box,
            algorithm_tabs,
            widgets.HTML(
                "<h3 style='margin:8px 0 4px 0;'>Current Selection</h3>"),
            preview_output,
        ]
    )

    return {"controls": controls, "panel": panel}


def list_paper_presets() -> pd.DataFrame:
    rows = []
    for preset_name, preset in PAPER_PRESET_DEFINITIONS.items():
        rows.append(
            {
                "preset_name": preset_name,
                "label": preset["label"],
                "enabled_algorithms": ", ".join(DISPLAY_NAME_MAP[key] for key in preset["enabled_algorithms"]),
                "variant_count": sum(len(preset["variants"].get(key, [])) for key in preset["enabled_algorithms"]),
                "description": preset["description"],
            }
        )
    return pd.DataFrame(rows).sort_values("preset_name").reset_index(drop=True)


def build_preset_reference_table(preset_name: str) -> pd.DataFrame:
    preset = PAPER_PRESET_DEFINITIONS[preset_name]
    rows = []
    for algorithm_key in preset["enabled_algorithms"]:
        for index, variant in enumerate(preset["variants"].get(algorithm_key, []), start=1):
            params = {key: value for key, value in variant.items() if key not in {
                "variant_name", "focus"}}
            rows.append(
                {
                    "preset_name": preset_name,
                    "algorithm": DISPLAY_NAME_MAP[algorithm_key],
                    "variant_index": index,
                    "variant_name": variant["variant_name"],
                    "focus": variant["focus"],
                    "params_json": json.dumps(params, sort_keys=True),
                }
            )
    return pd.DataFrame(rows)


def apply_paper_experiment_preset(controls: dict[str, Any], preset_name: str = "paper_high_roi") -> None:
    if preset_name not in PAPER_PRESET_DEFINITIONS:
        available = ", ".join(sorted(PAPER_PRESET_DEFINITIONS))
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available presets: {available}")

    preset = PAPER_PRESET_DEFINITIONS[preset_name]
    enabled = set(preset["enabled_algorithms"])

    for algorithm_key in ALGORITHM_ORDER:
        enabled_key = ALGORITHM_ENABLE_CONTROL[algorithm_key]
        controls[enabled_key].value = algorithm_key in enabled
        variant_manager = controls["algorithm_variants"][algorithm_key]
        variant_manager["replace_variants"](
            preset["variants"].get(algorithm_key, [{}]))


def get_run_config(controls: dict[str, Any]) -> dict[str, Any]:
    clip_value = None if controls["clip_quantile"].value <= 0 else float(
        controls["clip_quantile"].value)
    window_size = None if controls["window_size"].value <= 0 else int(
        controls["window_size"].value)
    window_stride = max(1, int(controls["window_stride"].value))
    dataset_limit = None if controls["dataset_limit"].value <= 0 else int(
        controls["dataset_limit"].value)
    threshold_method = str(controls["threshold_method"].value)
    threshold_value = (
        max(1, int(controls["threshold_value"].value))
        if threshold_method == "top_k"
        else float(controls["threshold_value"].value)
    )
    selected_algorithms: list[str] = []
    algorithm_variants: dict[str, list[dict[str, Any]]] = {}
    selected_runs: list[dict[str, Any]] = []

    for algorithm_key, enabled_key in (
        ("isolation_forest", "run_iforest"),
        ("local_outlier_factor", "run_lof"),
        ("sand", "run_sand"),
        ("matrix_profile", "run_matrix_profile"),
        ("damp", "run_damp"),
        ("hbos", "run_hbos"),
        ("ocsvm", "run_ocsvm"),
        ("pca", "run_pca"),
    ):
        variant_entries = controls["algorithm_variants"][algorithm_key]["variants"]
        algorithm_variant_configs = []

        for variant_index, variant_entry in enumerate(variant_entries, start=1):
            variant_controls = variant_entry["controls"]
            variant_name = variant_controls["variant_name"].value.strip(
            ) or _default_variant_name(variant_index)
            if algorithm_key == "isolation_forest":
                params = {
                    "n_estimators": int(variant_controls["n_estimators"].value),
                    "max_samples": parse_freeform_value(variant_controls["max_samples"].value),
                    "max_features": float(variant_controls["max_features"].value),
                    "bootstrap": bool(variant_controls["bootstrap"].value),
                    "random_state": int(variant_controls["random_state"].value),
                }
            elif algorithm_key == "local_outlier_factor":
                params = {
                    "n_neighbors": int(variant_controls["n_neighbors"].value),
                    "algorithm": variant_controls["algorithm"].value,
                    "leaf_size": int(variant_controls["leaf_size"].value),
                    "metric": variant_controls["metric"].value,
                    "p": int(variant_controls["p"].value),
                }
            elif algorithm_key == "sand":
                params = {
                    "alpha": float(variant_controls["alpha"].value),
                    "init_length": int(variant_controls["init_length"].value),
                    "batch_size": int(variant_controls["batch_size"].value),
                    "k": None if variant_controls["k"].value <= 0 else int(variant_controls["k"].value),
                    "subsequence_multiplier": int(variant_controls["subsequence_multiplier"].value),
                    "overlap": None if variant_controls["overlap"].value <= 0 else int(variant_controls["overlap"].value),
                }
            elif algorithm_key == "matrix_profile":
                params = {
                    "subsequence_multiplier": int(variant_controls["subsequence_multiplier"].value),
                }
            elif algorithm_key == "damp":
                params = {
                    "start_index_multiplier": float(variant_controls["start_index_multiplier"].value),
                    "x_lag_multiplier": None if float(variant_controls["x_lag_multiplier"].value) <= 0 else float(variant_controls["x_lag_multiplier"].value),
                }
            elif algorithm_key == "hbos":
                params = {
                    "n_bins": int(variant_controls["n_bins"].value),
                    "alpha": float(variant_controls["alpha"].value),
                    "tol": float(variant_controls["tol"].value),
                }
            elif algorithm_key == "ocsvm":
                params = {
                    "kernel": variant_controls["kernel"].value,
                    "nu": float(variant_controls["nu"].value),
                    "gamma": parse_freeform_value(variant_controls["gamma"].value),
                    "train_fraction": float(variant_controls["train_fraction"].value),
                }
            else:
                n_components_text = variant_controls["n_components"].value.strip(
                )
                params = {
                    "n_components": None if n_components_text == "" else parse_freeform_value(n_components_text),
                    "n_selected_components": None if int(variant_controls["n_selected_components"].value) <= 0 else int(variant_controls["n_selected_components"].value),
                    "whiten": bool(variant_controls["whiten"].value),
                    "weighted": bool(variant_controls["weighted"].value),
                    "standardization": bool(variant_controls["standardization"].value),
                }

            variant_config = {
                "algorithm": algorithm_key,
                "algorithm_base_display": DISPLAY_NAME_MAP[algorithm_key],
                "algorithm_superfamily": ALGORITHM_METADATA[algorithm_key]["superfamily"],
                "algorithm_category": ALGORITHM_METADATA[algorithm_key]["category"],
                "algorithm_display": f"{DISPLAY_NAME_MAP[algorithm_key]} | {variant_name}",
                "algorithm_variant": variant_name,
                "algorithm_run_id": f"{algorithm_key}__tab_{variant_index:02d}",
                "variant_index": variant_index,
                "params": params,
            }
            algorithm_variant_configs.append(variant_config)

        algorithm_variants[algorithm_key] = algorithm_variant_configs
        if controls[enabled_key].value and algorithm_variant_configs:
            selected_algorithms.append(algorithm_key)
            selected_runs.extend(algorithm_variant_configs)

    return {
        "dataset_limit": dataset_limit,
        "normalization_method": controls["normalization_method"].value,
        "clip_quantile": clip_value,
        "overwrite_normalized_datasets": bool(controls["overwrite_normalized"].value),
        "window_size": window_size,
        "window_stride": window_stride,
        "window_override": window_size,
        "threshold_method": threshold_method,
        "threshold_value": threshold_value,
        "threshold_std_multiplier": threshold_value if threshold_method == "sigma" else None,
        "evaluation_mode": str(controls["evaluation_mode"].value),
        "save_per_dataset_scores": bool(controls["save_scores"].value),
        "selected_algorithms": selected_algorithms,
        "selected_runs": selected_runs,
        "algorithm_variants": algorithm_variants,
    }


def normalization_tag(method: str, clip_quantile: float | None) -> str:
    tag = method.lower()
    if clip_quantile is not None:
        clip_label = str(clip_quantile).replace(".", "p")
        tag = f"{tag}_clip_{clip_label}"
    return tag


def parse_dataset_metadata(dataset_path: Path) -> dict[str, int | str]:
    parts = dataset_path.stem.split("_")
    train_end, anomaly_start, anomaly_end = map(int, parts[-3:])
    core_name = "_".join(parts[3:-3])
    variant = "raw"
    clean_name = core_name
    if core_name.startswith("DISTORTED"):
        variant = "distorted"
        clean_name = core_name[len("DISTORTED"):]
    elif core_name.startswith("NOISE"):
        variant = "noise"
        clean_name = core_name[len("NOISE"):]
    family = re.sub(r"\d+$", "", clean_name) or clean_name
    return {
        "dataset_name": dataset_path.stem,
        "dataset_sequence": int(parts[0]),
        "dataset_core_name": core_name,
        "dataset_clean_name": clean_name,
        "family": family,
        "variant": variant,
        "train_end": train_end,
        "anomaly_start": anomaly_start,
        "anomaly_end": anomaly_end,
    }


def load_raw_values_from_file(raw_dataset_path: Path) -> np.ndarray:
    text = raw_dataset_path.read_text(encoding="utf-8", errors="ignore")
    values = np.fromstring(text, sep=" ")
    if values.size == 0:
        raise ValueError(
            f"Could not parse numeric values from {raw_dataset_path}")
    return values.astype(float)


def build_labels(series_length: int, anomaly_start: int, anomaly_end: int) -> np.ndarray:
    labels = np.zeros(series_length, dtype=int)
    start_index = max(0, anomaly_start - 1)
    end_index = min(series_length, anomaly_end)
    labels[start_index:end_index] = 1
    return labels


def apply_normalization(values: np.ndarray, method: str = "zscore", clip_quantile: float | None = None) -> np.ndarray:
    method = method.lower()
    supported = {"none", "zscore", "minmax", "robust"}
    if method not in supported:
        raise ValueError(f"Unsupported normalization method: {method}")

    transformed = np.asarray(values, dtype=float).copy()
    if clip_quantile is not None:
        if not (0.0 < clip_quantile < 0.5):
            raise ValueError("clip_quantile must be between 0 and 0.5.")
        lower, upper = np.quantile(
            transformed, [clip_quantile, 1.0 - clip_quantile])
        transformed = np.clip(transformed, lower, upper)

    if method == "none":
        return transformed
    if method == "minmax":
        minimum = float(np.min(transformed))
        maximum = float(np.max(transformed))
        return np.zeros_like(transformed) if np.isclose(minimum, maximum) else (transformed - minimum) / (maximum - minimum)
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


def ensure_raw_datasets_available() -> list[Path]:
    source_paths = sorted(LEGACY_VIRGIN_DIR.glob("*.txt"))
    if not source_paths:
        raise FileNotFoundError(
            f"No raw datasets found in {LEGACY_VIRGIN_DIR}")
    for source_path in source_paths:
        target_path = RAW_DATASET_DIR / source_path.name
        if not target_path.exists():
            shutil.copy2(source_path, target_path)
    return sorted(RAW_DATASET_DIR.glob("*.txt"))


def write_normalized_dataset(
    raw_dataset_path: Path,
    output_dir: Path,
    method: str,
    clip_quantile: float | None,
    overwrite: bool = False,
) -> Path:
    output_path = output_dir / raw_dataset_path.name
    if output_path.exists() and not overwrite:
        return output_path
    metadata = parse_dataset_metadata(raw_dataset_path)
    raw_values = load_raw_values_from_file(raw_dataset_path)
    labels = build_labels(
        len(raw_values), metadata["anomaly_start"], metadata["anomaly_end"])
    normalized_values = apply_normalization(
        raw_values, method=method, clip_quantile=clip_quantile)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([normalized_values, labels])
    np.savetxt(output_path, data, delimiter=",", fmt=["%.10g", "%d"])
    return output_path


def ensure_normalized_datasets(
    method: str,
    clip_quantile: float | None,
    overwrite: bool = False,
) -> tuple[Path, list[Path]]:
    raw_dataset_paths = ensure_raw_datasets_available()
    output_dir = NORMALIZED_DATASET_ROOT / \
        normalization_tag(method, clip_quantile)
    prepared_paths = [write_normalized_dataset(
        path, output_dir, method, clip_quantile, overwrite=overwrite) for path in raw_dataset_paths]
    return output_dir, sorted(prepared_paths)


def load_prepared_dataset(prepared_dataset_path: Path) -> dict[str, Any]:
    frame = pd.read_csv(prepared_dataset_path, header=None,
                        names=["value", "label"])
    metadata = parse_dataset_metadata(prepared_dataset_path)
    metadata["dataset_path"] = str(prepared_dataset_path)
    metadata["values"] = frame["value"].to_numpy(dtype=float)
    metadata["labels"] = frame["label"].to_numpy(dtype=int)
    return metadata


def estimate_window_size(values: np.ndarray, default: int = 125, max_lag: int = 400) -> int:
    values = np.asarray(values, dtype=float).ravel()
    values = values[: min(20_000, values.size)]
    fallback = min(default, max(4, min(300, max(4, values.size // 4))))
    if values.size < 8:
        return fallback
    centered = values - values.mean()
    denominator = float(np.dot(centered, centered))
    usable_max_lag = min(max_lag, values.size - 2)
    if denominator == 0 or usable_max_lag <= 3:
        return fallback
    autocorrelation = np.array([float(np.dot(
        centered[:-lag], centered[lag:]) / denominator) for lag in range(3, usable_max_lag + 1)])
    candidate_lags = []
    for index in range(1, len(autocorrelation) - 1):
        if autocorrelation[index] > autocorrelation[index - 1] and autocorrelation[index] > autocorrelation[index + 1]:
            lag = index + 3
            if 4 <= lag <= 300:
                candidate_lags.append(lag)
    if not candidate_lags:
        return fallback
    return max(candidate_lags, key=lambda lag: autocorrelation[lag - 3])


def safe_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    return float(_load_sklearn_metric_functions()["roc_auc_score"](labels, scores))


def safe_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    return float(_load_sklearn_metric_functions()["average_precision_score"](labels, scores))


def precision_at_k(labels: np.ndarray, scores: np.ndarray) -> float:
    k = int(labels.sum())
    if k <= 0:
        return float("nan")
    top_indices = np.argsort(scores)[-k:]
    return float(labels[top_indices].sum() / k)


def _positive_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start_index: int | None = None
    for index, value in enumerate(np.asarray(mask, dtype=bool)):
        if value and start_index is None:
            start_index = index
        elif not value and start_index is not None:
            segments.append((start_index, index))
            start_index = None
    if start_index is not None:
        segments.append((start_index, len(mask)))
    return segments


def apply_threshold_strategy(
    scores: np.ndarray,
    threshold_method: str,
    threshold_value: float,
    window_size: int,
    window_stride: int,
    evaluation_mode: str,
) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=float).ravel()
    if scores.size == 0:
        return {
            "predictions": np.zeros(0, dtype=int),
            "score_threshold": float("nan"),
            "threshold_method": threshold_method,
            "threshold_value": threshold_value,
        }

    method = str(threshold_method).lower()
    evaluation_mode = str(evaluation_mode).lower()
    if method == "sigma":
        sigma = float(threshold_value)
        threshold = float(scores.mean() + sigma * scores.std())
        predictions = (scores >= threshold).astype(int)
        return {
            "predictions": predictions,
            "score_threshold": threshold,
            "threshold_method": method,
            "threshold_value": sigma,
        }

    if method == "quantile":
        quantile = float(threshold_value)
        quantile = min(max(quantile, 0.0), 1.0)
        threshold = float(np.quantile(scores, quantile))
        predictions = (scores >= threshold).astype(int)
        return {
            "predictions": predictions,
            "score_threshold": threshold,
            "threshold_method": method,
            "threshold_value": quantile,
        }

    if method != "top_k":
        raise ValueError(f"Unsupported threshold method: {threshold_method}")

    top_k = max(1, int(round(float(threshold_value))))
    predictions = np.zeros(scores.size, dtype=int)
    ranked_indices = np.argsort(scores)[::-1]
    selected_indices: list[int] = []

    if evaluation_mode == "point":
        selected_indices = ranked_indices[: min(top_k, scores.size)].tolist()
        predictions[selected_indices] = 1
    else:
        left = math.ceil((window_size - 1) / 2)
        right = (window_size - 1) // 2
        for index in ranked_indices:
            start_index = max(0, int(index) - left)
            end_index = min(scores.size, int(index) + right + 1)
            if predictions[start_index:end_index].any():
                continue
            predictions[start_index:end_index] = 1
            selected_indices.append(int(index))
            if len(selected_indices) >= top_k:
                break
        if not selected_indices:
            selected_indices = ranked_indices[:1].tolist()
            predictions[selected_indices] = 1

    score_threshold = float(
        np.min(scores[selected_indices])) if selected_indices else float("nan")
    return {
        "predictions": predictions,
        "score_threshold": score_threshold,
        "threshold_method": method,
        "threshold_value": float(top_k),
        "decision_window_size": int(window_size),
        "decision_window_stride": int(window_stride),
    }


@lru_cache(maxsize=1)
def _load_tsb_uad_components() -> dict[str, Any] | None:
    try:
        from TSB_UAD.vus.affiliation.generics import convert_vector_to_events
        from TSB_UAD.vus.affiliation.metrics import pr_from_events
        from TSB_UAD.vus.basic_metrics import basic_metricor
        from TSB_UAD.vus.metrics import get_metrics as get_tsb_metrics
    except Exception:
        return None
    return {
        "basic_metricor": basic_metricor,
        "convert_vector_to_events": convert_vector_to_events,
        "pr_from_events": pr_from_events,
        "get_tsb_metrics": get_tsb_metrics,
    }


def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold_method: str,
    threshold_value: float,
    window_size: int,
    evaluation_mode: str,
    window_stride: int = 1,
) -> dict[str, float]:
    decision = apply_threshold_strategy(
        scores,
        threshold_method=threshold_method,
        threshold_value=threshold_value,
        window_size=window_size,
        window_stride=window_stride,
        evaluation_mode=evaluation_mode,
    )
    predictions = decision["predictions"]
    metric_functions = _load_sklearn_metric_functions()
    precision, recall, f1, _ = metric_functions["precision_recall_fscore_support"](
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    metrics = {
        "roc_auc": safe_roc_auc(labels, scores),
        "average_precision": safe_average_precision(labels, scores),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "precision_at_k": precision_at_k(labels, scores),
        "score_threshold": float(decision["score_threshold"]),
        "threshold_method": str(decision["threshold_method"]),
        "threshold_value": float(decision["threshold_value"]),
        "evaluation_mode": str(evaluation_mode),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    components = _load_tsb_uad_components()
    if components is None:
        metrics.update(
            {
                "range_precision": float("nan"),
                "range_recall": float("nan"),
                "range_f1": float("nan"),
                "existence_reward": float("nan"),
                "overlap_reward": float("nan"),
                "affiliation_precision": float("nan"),
                "affiliation_recall": float("nan"),
                "tsb_uad_window": int(window_size),
                "evaluation_precision": float(precision),
                "evaluation_recall": float(recall),
                "evaluation_f1": float(f1),
            }
        )
        return metrics

    try:
        grader = components["basic_metricor"]()
        range_recall, existence_reward, overlap_reward = grader.range_recall_new(
            labels, predictions, alpha=0.2)
        range_precision = grader.range_recall_new(
            predictions, labels, alpha=0)[0]
        range_f1 = (
            0.0
            if np.isclose(range_precision + range_recall, 0.0)
            else float(2 * range_precision * range_recall / (range_precision + range_recall))
        )
    except Exception:
        range_precision = float("nan")
        range_recall = float("nan")
        range_f1 = float("nan")
        existence_reward = float("nan")
        overlap_reward = float("nan")

    try:
        events_pred = components["convert_vector_to_events"](
            predictions.astype(np.float32))
        events_gt = components["convert_vector_to_events"](labels)
        affiliation = components["pr_from_events"](
            events_pred, events_gt, (0, len(predictions)))
        affiliation_precision = float(affiliation["Affiliation_Precision"])
        affiliation_recall = float(affiliation["Affiliation_Recall"])
    except Exception:
        affiliation_precision = float("nan")
        affiliation_recall = float("nan")

    metrics.update(
        {
            "range_precision": float(range_precision),
            "range_recall": float(range_recall),
            "range_f1": float(range_f1),
            "existence_reward": float(existence_reward),
            "overlap_reward": float(overlap_reward),
            "affiliation_precision": affiliation_precision,
            "affiliation_recall": affiliation_recall,
            "tsb_uad_window": int(window_size),
        }
    )
    if str(evaluation_mode).lower() == "range" and np.isfinite(range_precision) and np.isfinite(range_recall) and np.isfinite(range_f1):
        metrics["evaluation_precision"] = float(range_precision)
        metrics["evaluation_recall"] = float(range_recall)
        metrics["evaluation_f1"] = float(range_f1)
    else:
        metrics["evaluation_precision"] = float(precision)
        metrics["evaluation_recall"] = float(recall)
        metrics["evaluation_f1"] = float(f1)
    return metrics


def compute_surface_metrics(labels: np.ndarray, scores: np.ndarray, window_size: int) -> dict[str, float]:
    components = _load_tsb_uad_components()
    if components is None:
        return {
            "range_auc_roc": float("nan"),
            "range_auc_pr": float("nan"),
            "vus_roc": float("nan"),
            "vus_pr": float("nan"),
        }
    try:
        range_metrics = components["get_tsb_metrics"](
            scores, labels, metric="range_auc", slidingWindow=window_size)
        vus_metrics = components["get_tsb_metrics"](
            scores, labels, metric="vus", slidingWindow=window_size)
        return {
            "range_auc_roc": float(range_metrics.get("R_AUC_ROC", float("nan"))),
            "range_auc_pr": float(range_metrics.get("R_AUC_PR", float("nan"))),
            "vus_roc": float(vus_metrics.get("VUS_ROC", float("nan"))),
            "vus_pr": float(vus_metrics.get("VUS_PR", float("nan"))),
        }
    except Exception:
        return {
            "range_auc_roc": float("nan"),
            "range_auc_pr": float("nan"),
            "vus_roc": float("nan"),
            "vus_pr": float("nan"),
        }


def save_scores_if_needed(dataset_name: str, run_id: str, labels: np.ndarray, scores: np.ndarray, enabled: bool) -> None:
    if not enabled:
        return
    RESULT_SCORES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"label": labels, "score": scores}).to_csv(
        result_score_path(dataset_name, run_id), index=False)


def build_dataset_catalog(results_frame: pd.DataFrame) -> pd.DataFrame:
    catalog = results_frame.sort_values(
        ["dataset_sequence", "algorithm"]).drop_duplicates("dataset_name").copy()
    catalog["anomaly_ratio"] = catalog["anomaly_count"] / \
        catalog["series_length"]
    return catalog[
        [
            "dataset_name",
            "dataset_sequence",
            "family",
            "variant",
            "series_length",
            "anomaly_count",
            "anomaly_ratio",
            "train_end",
            "anomaly_start",
            "anomaly_end",
            "window_size",
            "window_stride",
            "normalization_method",
            "threshold_method",
            "threshold_value",
            "evaluation_mode",
            "prepared_dataset_dir",
        ]
    ].reset_index(drop=True)


def add_analysis_regime_columns(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    if enriched.empty:
        return enriched
    enriched["anomaly_ratio"] = enriched["anomaly_count"] / \
        enriched["series_length"]
    enriched["length_bucket"] = pd.cut(
        enriched["series_length"],
        bins=[-np.inf, 5_000, 20_000, np.inf],
        labels=LENGTH_BUCKET_ORDER,
    )
    enriched["anomaly_ratio_bucket"] = pd.cut(
        enriched["anomaly_ratio"],
        bins=[-np.inf, 0.01, 0.05, np.inf],
        labels=ANOMALY_RATIO_BUCKET_ORDER,
    )
    return enriched


def _evaluation_metric_spec(frame: pd.DataFrame | None = None, evaluation_mode: str | None = None) -> dict[str, str]:
    mode = str(
        evaluation_mode
        if evaluation_mode is not None
        else (frame["evaluation_mode"].iloc[0] if frame is not None and not frame.empty else "range")
    ).lower()
    label = "Range F1" if mode == "range" else "Point F1"
    return {
        "mode": mode,
        "metric_column": "evaluation_f1",
        "mean_column": "mean_evaluation_f1",
        "label": label,
        "mean_label": f"Mean {label}",
    }


def build_overall_regime_summary(results_frame: pd.DataFrame) -> pd.DataFrame:
    enriched = add_analysis_regime_columns(results_frame)
    rows = []
    for regime_column in ("variant", "length_bucket", "anomaly_ratio_bucket"):
        summary = (
            enriched.groupby(
                ["algorithm_display", regime_column], as_index=False)
            .agg(
                dataset_count=("dataset_name", "nunique"),
                mean_roc_auc=("roc_auc", "mean"),
                mean_f1=("f1", "mean"),
                mean_evaluation_f1=("evaluation_f1", "mean"),
                mean_range_f1=("range_f1", "mean"),
                mean_runtime_seconds=("runtime_seconds", "mean"),
            )
            .rename(columns={regime_column: "regime_value"})
        )
        summary["regime_dimension"] = regime_column
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)[
        [
            "regime_dimension",
            "regime_value",
            "algorithm_display",
            "dataset_count",
            "mean_roc_auc",
            "mean_f1",
            "mean_evaluation_f1",
            "mean_range_f1",
            "mean_runtime_seconds",
        ]
    ].sort_values(
        ["regime_dimension", "regime_value", "mean_evaluation_f1", "mean_roc_auc"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)


def summarize_algorithms(results_frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results_frame.groupby(
            [
                "algorithm",
                "algorithm_base_display",
                "algorithm_superfamily",
                "algorithm_category",
                "algorithm_display",
                "algorithm_variant",
                "algorithm_run_id",
                "evaluation_mode",
            ],
            as_index=False,
        )
        .agg(
            run_count=("dataset_name", "count"),
            success_count=("error", lambda series: int((series == "").sum())),
            mean_roc_auc=("roc_auc", "mean"),
            median_roc_auc=("roc_auc", "median"),
            mean_average_precision=("average_precision", "mean"),
            median_average_precision=("average_precision", "median"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            median_f1=("f1", "median"),
            mean_evaluation_precision=("evaluation_precision", "mean"),
            mean_evaluation_recall=("evaluation_recall", "mean"),
            mean_evaluation_f1=("evaluation_f1", "mean"),
            median_evaluation_f1=("evaluation_f1", "median"),
            mean_precision_at_k=("precision_at_k", "mean"),
            mean_range_precision=("range_precision", "mean"),
            mean_range_recall=("range_recall", "mean"),
            mean_range_f1=("range_f1", "mean"),
            mean_affiliation_precision=("affiliation_precision", "mean"),
            mean_affiliation_recall=("affiliation_recall", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            median_runtime_seconds=("runtime_seconds", "median"),
        )
        .sort_values(["mean_evaluation_f1", "mean_roc_auc", "mean_f1"], ascending=False)
        .reset_index(drop=True)
    )
    summary["success_rate"] = summary["success_count"] / summary["run_count"]
    return summary


def summarize_families(results_frame: pd.DataFrame) -> pd.DataFrame:
    return (
        results_frame.groupby(
            ["family", "algorithm_display", "algorithm_superfamily",
                "algorithm_category", "evaluation_mode"],
            as_index=False,
        )
        .agg(
            dataset_count=("dataset_name", "nunique"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_f1=("f1", "mean"),
            mean_evaluation_f1=("evaluation_f1", "mean"),
            mean_range_f1=("range_f1", "mean"),
            mean_affiliation_precision=("affiliation_precision", "mean"),
            mean_affiliation_recall=("affiliation_recall", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
        )
        .sort_values(["dataset_count", "mean_evaluation_f1", "mean_f1"], ascending=[False, False, False])
        .reset_index(drop=True)
    )


def build_best_algorithm_table(results_frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    candidate_frame = results_frame.dropna(subset=[metric]).copy()
    if candidate_frame.empty:
        return pd.DataFrame(columns=["dataset_name", "best_algorithm_display", f"best_{metric}"])
    best_indices = candidate_frame.groupby("dataset_name")[metric].idxmax()
    winners = candidate_frame.loc[
        best_indices,
        [
            "dataset_name",
            "algorithm",
            "algorithm_base_display",
            "algorithm_superfamily",
            "algorithm_category",
            "algorithm_display",
            "algorithm_variant",
            "window_size",
            "window_stride",
            "threshold_method",
            "threshold_value",
            "evaluation_mode",
            metric,
        ],
    ].copy()
    winners.rename(
        columns={
            "algorithm": "best_algorithm",
            "algorithm_base_display": "best_algorithm_base_display",
            "algorithm_superfamily": "best_algorithm_superfamily",
            "algorithm_category": "best_algorithm_category",
            "algorithm_display": "best_algorithm_display",
            "algorithm_variant": "best_algorithm_variant",
            "window_size": "best_window_size",
            "window_stride": "best_window_stride",
            "threshold_method": "best_threshold_method",
            "threshold_value": "best_threshold_value",
            "evaluation_mode": "best_evaluation_mode",
            metric: f"best_{metric}",
        },
        inplace=True,
    )
    return winners.sort_values("dataset_name").reset_index(drop=True)


def build_algorithm_section_tables(results_frame: pd.DataFrame, algorithm_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = results_frame.loc[results_frame["algorithm"]
                               == algorithm_key].copy()
    if subset.empty:
        return pd.DataFrame(), pd.DataFrame()
    summary = (
        subset.groupby(["algorithm_display", "algorithm_variant",
                       "algorithm_run_id", "evaluation_mode"], as_index=False)
        .agg(
            runs=("dataset_name", "count"),
            success_rate=("error", lambda series: float(
                (series == "").mean())),
            mean_roc_auc=("roc_auc", "mean"),
            mean_average_precision=("average_precision", "mean"),
            mean_f1=("f1", "mean"),
            mean_evaluation_f1=("evaluation_f1", "mean"),
            mean_range_f1=("range_f1", "mean"),
            mean_affiliation_precision=("affiliation_precision", "mean"),
            mean_affiliation_recall=("affiliation_recall", "mean"),
            median_f1=("f1", "median"),
            median_evaluation_f1=("evaluation_f1", "median"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
        )
        .sort_values(["mean_evaluation_f1", "mean_roc_auc", "mean_f1"], ascending=False)
        .reset_index(drop=True)
    )
    top_rows = subset.sort_values(["evaluation_f1", "evaluation_recall", "roc_auc"], ascending=False)[
        [
            "algorithm_display",
            "dataset_name",
            "evaluation_mode",
            "roc_auc",
            "average_precision",
            "precision",
            "recall",
            "f1",
            "evaluation_precision",
            "evaluation_recall",
            "evaluation_f1",
            "range_f1",
            "affiliation_precision",
            "affiliation_recall",
            "runtime_seconds",
        ]
    ].head(12)
    return summary, top_rows.reset_index(drop=True)


def build_variant_config_table(config: dict[str, Any], algorithm_key: str) -> pd.DataFrame:
    variant_rows = config["algorithm_variants"].get(algorithm_key, [])
    if not variant_rows:
        return pd.DataFrame()
    rows = []
    for row in variant_rows:
        rows.append(
            {
                "algorithm_display": row["algorithm_display"],
                "algorithm_variant": row["algorithm_variant"],
                **row["params"],
            }
        )
    return pd.DataFrame(rows)


def _summarized_param_columns(frame: pd.DataFrame) -> list[str]:
    param_columns = [
        column
        for column in frame.columns
        if column.startswith("param__") and not frame[column].isna().all()
    ]
    varying_columns = [
        column for column in param_columns if frame[column].dropna().nunique() > 1
    ]
    return varying_columns if varying_columns else param_columns


def build_algorithm_parameter_effect_table(results_frame: pd.DataFrame, algorithm_key: str) -> pd.DataFrame:
    subset = results_frame.loc[results_frame["algorithm"]
                               == algorithm_key].copy()
    if subset.empty:
        return pd.DataFrame()
    param_columns = _summarized_param_columns(subset)
    group_columns = ["algorithm_display", "algorithm_variant", *param_columns]
    summary = (
        subset.groupby(group_columns, as_index=False)
        .agg(
            dataset_count=("dataset_name", "nunique"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_f1=("f1", "mean"),
            mean_evaluation_f1=("evaluation_f1", "mean"),
            median_evaluation_f1=("evaluation_f1", "median"),
            mean_range_f1=("range_f1", "mean"),
            success_rate=("error", lambda series: float((series == "").mean())),
            mean_runtime_seconds=("runtime_seconds", "mean"),
        )
        .sort_values(["mean_evaluation_f1", "mean_roc_auc", "mean_runtime_seconds"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    return summary


def build_algorithm_regime_table(results_frame: pd.DataFrame, algorithm_key: str, regime_column: str) -> pd.DataFrame:
    subset = add_analysis_regime_columns(
        results_frame.loc[results_frame["algorithm"] == algorithm_key].copy())
    if subset.empty:
        return pd.DataFrame()
    return (
        subset.groupby(["algorithm_display", "algorithm_variant",
                       regime_column], as_index=False)
        .agg(
            dataset_count=("dataset_name", "nunique"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_f1=("f1", "mean"),
            mean_evaluation_f1=("evaluation_f1", "mean"),
            mean_range_f1=("range_f1", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
        )
        .sort_values([regime_column, "mean_evaluation_f1", "mean_roc_auc"], ascending=[True, False, False])
        .reset_index(drop=True)
    )


def _ordered_regime_pivot(
    results_frame: pd.DataFrame,
    algorithm_key: str,
    regime_column: str,
    order: list[str],
    metric: str = "range_f1",
) -> pd.DataFrame:
    subset = add_analysis_regime_columns(
        results_frame.loc[results_frame["algorithm"] == algorithm_key].copy())
    if subset.empty:
        return pd.DataFrame()
    grouped = (
        subset.groupby(["algorithm_display", regime_column],
                       as_index=False)[metric]
        .mean()
        .pivot(index="algorithm_display", columns=regime_column, values=metric)
    )
    present_columns = [column for column in order if column in grouped.columns]
    return grouped.reindex(columns=present_columns).fillna(0.0)


def select_deep_dive_variant(
    results_frame: pd.DataFrame,
    deep_dive_payload: dict[str, Any] | None,
    algorithm_key: str,
) -> tuple[pd.Series | None, np.ndarray | None]:
    if deep_dive_payload is None:
        return None, None
    subset = results_frame.loc[
        (results_frame["dataset_name"] ==
         deep_dive_payload["dataset"]["dataset_name"])
        & (results_frame["algorithm"] == algorithm_key)
    ].sort_values(["evaluation_f1", "evaluation_recall", "roc_auc", "runtime_seconds"], ascending=[False, False, False, True])
    if subset.empty:
        return None, None
    metric_row = subset.iloc[0]
    score_values = deep_dive_payload["scores"].get(
        metric_row["algorithm_run_id"])
    return metric_row, score_values


def build_deep_dive_research_table(
    results_frame: pd.DataFrame,
    deep_dive_payload: dict[str, Any] | None,
    algorithm_keys: list[str],
) -> pd.DataFrame:
    if deep_dive_payload is None:
        return pd.DataFrame()

    rows = []
    for algorithm_key in algorithm_keys:
        metric_row, score_values = select_deep_dive_variant(
            results_frame, deep_dive_payload, algorithm_key)
        if metric_row is None or score_values is None:
            continue
        surface_metrics = compute_surface_metrics(
            deep_dive_payload["dataset"]["labels"],
            score_values,
            int(metric_row["window_size"]),
        )
        rows.append(
            {
                "algorithm_display": metric_row["algorithm_display"],
                "algorithm_superfamily": metric_row["algorithm_superfamily"],
                "algorithm_category": metric_row["algorithm_category"],
                "roc_auc": metric_row["roc_auc"],
                "average_precision": metric_row["average_precision"],
                "f1": metric_row["f1"],
                "range_f1": metric_row["range_f1"],
                "affiliation_precision": metric_row["affiliation_precision"],
                "affiliation_recall": metric_row["affiliation_recall"],
                **surface_metrics,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["range_f1", "vus_pr", "roc_auc"],
        ascending=False,
    ).reset_index(drop=True)


def select_algorithm_showcase(results_frame: pd.DataFrame, algorithm_key: str) -> pd.Series | None:
    subset = results_frame.loc[results_frame["algorithm"]
                               == algorithm_key].copy()
    if subset.empty:
        return None
    subset = subset.sort_values(
        ["evaluation_recall", "evaluation_f1", "overlap_reward",
            "roc_auc", "f1", "runtime_seconds"],
        ascending=[False, False, False, False, False, True],
        na_position="last",
    )
    return subset.iloc[0]


def build_algorithm_showcase(
    config: dict[str, Any],
    prepared_dataset_dir: Path,
    results_frame: pd.DataFrame,
    algorithm_key: str,
) -> dict[str, Any] | None:
    metric_row = select_algorithm_showcase(results_frame, algorithm_key)
    if metric_row is None:
        return None

    run_config = next(
        (entry for entry in config["selected_runs"]
         if entry["algorithm_run_id"] == metric_row["algorithm_run_id"]),
        None,
    )
    if run_config is None:
        return None

    prepared_dataset_path = Path(
        prepared_dataset_dir) / f"{metric_row['dataset_name']}.txt"
    raw_dataset_path = RAW_DATASET_DIR / f"{metric_row['dataset_name']}.txt"
    if not prepared_dataset_path.exists() or not raw_dataset_path.exists():
        return None

    dataset = load_prepared_dataset(prepared_dataset_path)
    raw_values = load_raw_values_from_file(raw_dataset_path)
    scores = run_algorithm(
        algorithm_key,
        dataset["values"],
        int(metric_row["window_size"]),
        run_config["params"],
        window_stride=int(metric_row["window_stride"]),
    )

    summary = pd.DataFrame(
        [
            {
                "showcase_dataset": metric_row["dataset_name"],
                "algorithm_display": metric_row["algorithm_display"],
                "window_size": metric_row["window_size"],
                "window_stride": metric_row["window_stride"],
                "threshold_method": metric_row["threshold_method"],
                "threshold_value": metric_row["threshold_value"],
                "evaluation_mode": metric_row["evaluation_mode"],
                "evaluation_recall": metric_row["evaluation_recall"],
                "evaluation_f1": metric_row["evaluation_f1"],
                "roc_auc": metric_row["roc_auc"],
                "f1": metric_row["f1"],
                "range_f1": metric_row["range_f1"],
                "runtime_seconds": metric_row["runtime_seconds"],
            }
        ]
    )

    decision = apply_threshold_strategy(
        scores,
        threshold_method=str(metric_row["threshold_method"]),
        threshold_value=float(metric_row["threshold_value"]),
        window_size=int(metric_row["window_size"]),
        window_stride=int(metric_row["window_stride"]),
        evaluation_mode=str(metric_row["evaluation_mode"]),
    )

    return {
        "dataset": dataset,
        "raw_values": raw_values,
        "scores": scores,
        "predictions": decision["predictions"],
        "metric_row": metric_row,
        "summary": summary,
        "run_config": run_config,
    }


def build_algorithm_variant_comparison(
    config: dict[str, Any],
    prepared_dataset_dir: Path,
    results_frame: pd.DataFrame,
    algorithm_key: str,
) -> dict[str, Any] | None:
    subset = results_frame.loc[results_frame["algorithm"] == algorithm_key].copy()
    if subset.empty:
        return None

    variant_run_configs = [
        entry for entry in config["selected_runs"] if entry["algorithm"] == algorithm_key
    ]
    if not variant_run_configs:
        return None

    dataset_ranking = (
        subset.groupby("dataset_name", as_index=False)
        .agg(
            variant_count=("algorithm_run_id", "nunique"),
            evaluation_spread=("evaluation_f1", lambda series: float(series.max() - series.min()) if not series.dropna().empty else float("nan")),
            best_evaluation_f1=("evaluation_f1", "max"),
            mean_evaluation_f1=("evaluation_f1", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
        )
        .sort_values(
            ["variant_count", "evaluation_spread", "best_evaluation_f1", "mean_roc_auc"],
            ascending=[False, False, False, False],
        )
        .reset_index(drop=True)
    )
    if dataset_ranking.empty:
        return None

    dataset_name = str(dataset_ranking.iloc[0]["dataset_name"])
    prepared_dataset_path = Path(prepared_dataset_dir) / f"{dataset_name}.txt"
    raw_dataset_path = RAW_DATASET_DIR / f"{dataset_name}.txt"
    if not prepared_dataset_path.exists() or not raw_dataset_path.exists():
        return None

    dataset = load_prepared_dataset(prepared_dataset_path)
    raw_values = load_raw_values_from_file(raw_dataset_path)
    variant_payloads: list[dict[str, Any]] = []

    for run_config in variant_run_configs:
        metric_subset = subset.loc[
            (subset["dataset_name"] == dataset_name)
            & (subset["algorithm_run_id"] == run_config["algorithm_run_id"])
        ].copy()
        if metric_subset.empty:
            continue
        metric_row = metric_subset.sort_values(
            ["evaluation_f1", "evaluation_recall", "roc_auc", "runtime_seconds"],
            ascending=[False, False, False, True],
        ).iloc[0]
        scores = run_algorithm(
            algorithm_key,
            dataset["values"],
            int(metric_row["window_size"]),
            run_config["params"],
            window_stride=int(metric_row["window_stride"]),
        )
        decision = apply_threshold_strategy(
            scores,
            threshold_method=str(metric_row["threshold_method"]),
            threshold_value=float(metric_row["threshold_value"]),
            window_size=int(metric_row["window_size"]),
            window_stride=int(metric_row["window_stride"]),
            evaluation_mode=str(metric_row["evaluation_mode"]),
        )
        variant_payloads.append(
            {
                "run_config": run_config,
                "metric_row": metric_row,
                "scores": scores,
                "predictions": decision["predictions"],
            }
        )

    if not variant_payloads:
        return None

    variant_summary = pd.DataFrame(
        [
            {
                "comparison_dataset": dataset_name,
                "algorithm_display": payload["metric_row"]["algorithm_display"],
                "algorithm_variant": payload["metric_row"]["algorithm_variant"],
                "window_size": payload["metric_row"]["window_size"],
                "window_stride": payload["metric_row"]["window_stride"],
                "roc_auc": payload["metric_row"]["roc_auc"],
                "average_precision": payload["metric_row"]["average_precision"],
                "evaluation_precision": payload["metric_row"]["evaluation_precision"],
                "evaluation_recall": payload["metric_row"]["evaluation_recall"],
                "evaluation_f1": payload["metric_row"]["evaluation_f1"],
                "range_f1": payload["metric_row"]["range_f1"],
                "runtime_seconds": payload["metric_row"]["runtime_seconds"],
            }
            for payload in variant_payloads
        ]
    ).sort_values(
        ["evaluation_f1", "roc_auc", "runtime_seconds"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    selection_summary = dataset_ranking.head(1).rename(
        columns={"dataset_name": "comparison_dataset"}
    )

    return {
        "dataset": dataset,
        "raw_values": raw_values,
        "selection_summary": selection_summary,
        "summary": variant_summary,
        "variants": variant_payloads,
    }


def plot_algorithm_benchmark_panel(results_frame: pd.DataFrame, algorithm_key: str, save_path: Path | None = None) -> plt.Figure | None:
    subset = results_frame.loc[results_frame["algorithm"]
                               == algorithm_key].copy()
    if subset.empty:
        return None
    plt = _load_plotting_module()
    evaluation_mode = str(subset["evaluation_mode"].iloc[0]).lower()
    metric_label = "Range F1" if evaluation_mode == "range" else "Point F1"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    palette = plt.get_cmap("tab10")
    for color_index, (display_name, frame) in enumerate(subset.groupby("algorithm_display")):
        color = palette(color_index % 10)
        axes[0].hist(frame["evaluation_f1"].dropna(), bins=20, alpha=0.45,
                     label=display_name, color=color, edgecolor="white")
        axes[1].scatter(frame["runtime_seconds"], frame["evaluation_f1"],
                        alpha=0.7, color=color, label=display_name)
    axes[0].set_title(
        f"{DISPLAY_NAME_MAP[algorithm_key]} | {metric_label} distribution")
    axes[0].set_xlabel(metric_label)
    axes[0].set_ylabel("Dataset count")
    axes[0].legend()
    axes[1].set_title(
        f"{DISPLAY_NAME_MAP[algorithm_key]} | Runtime vs {metric_label}")
    axes[1].set_xlabel("Runtime (seconds)")
    axes[1].set_ylabel(metric_label)
    axes[1].legend()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)
    return fig


def _draw_metric_heatmap(ax: Any, frame: pd.DataFrame, title: str, cmap: str) -> Any:
    if frame.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        ax.set_title(title)
        return None
    image = ax.imshow(frame.to_numpy(), cmap=cmap,
                      aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks(range(len(frame.columns)))
    ax.set_xticklabels(frame.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(frame.index)))
    ax.set_yticklabels(frame.index)
    for row_index in range(frame.shape[0]):
        for col_index in range(frame.shape[1]):
            ax.text(col_index, row_index,
                    f"{frame.iloc[row_index, col_index]:.2f}", ha="center", va="center", fontsize=9)
    return image


def plot_algorithm_paper_panel(results_frame: pd.DataFrame, algorithm_key: str, save_path: Path | None = None) -> plt.Figure | None:
    subset = results_frame.loc[results_frame["algorithm"]
                               == algorithm_key].copy()
    if subset.empty:
        return None

    plt = _load_plotting_module()
    metric_spec = _evaluation_metric_spec(subset)
    variant_pivot = _ordered_regime_pivot(
        results_frame, algorithm_key, "variant", DATASET_VARIANT_ORDER, metric=metric_spec["metric_column"])
    length_pivot = _ordered_regime_pivot(
        results_frame, algorithm_key, "length_bucket", LENGTH_BUCKET_ORDER, metric=metric_spec["metric_column"])
    anomaly_pivot = _ordered_regime_pivot(
        results_frame, algorithm_key, "anomaly_ratio_bucket", ANOMALY_RATIO_BUCKET_ORDER, metric=metric_spec["metric_column"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    image = _draw_metric_heatmap(
        axes[0, 0],
        variant_pivot,
        f"{DISPLAY_NAME_MAP[algorithm_key]} | {metric_spec['mean_label']} by dataset variant",
        "YlGnBu",
    )
    _draw_metric_heatmap(
        axes[0, 1],
        length_pivot,
        f"{DISPLAY_NAME_MAP[algorithm_key]} | {metric_spec['mean_label']} by series length",
        "YlOrBr",
    )
    _draw_metric_heatmap(
        axes[1, 0],
        anomaly_pivot,
        f"{DISPLAY_NAME_MAP[algorithm_key]} | {metric_spec['mean_label']} by anomaly ratio",
        "PuBuGn",
    )

    for display_name, frame in subset.groupby("algorithm_display"):
        axes[1, 1].scatter(frame["runtime_seconds"],
                           frame[metric_spec["metric_column"]], alpha=0.7, label=display_name)
    axes[1, 1].set_title(
        f"{DISPLAY_NAME_MAP[algorithm_key]} | Runtime vs {metric_spec['label']}")
    axes[1, 1].set_xlabel("Runtime (seconds)")
    axes[1, 1].set_ylabel(metric_spec["label"])
    axes[1, 1].legend()

    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)
    return fig


def plot_algorithm_variant_comparison(
    comparison_payload: dict[str, Any],
    algorithm_key: str,
    context_points: int = 1200,
    save_path: Path | None = None,
) -> plt.Figure | None:
    variants = comparison_payload.get("variants", [])
    if not variants:
        return None

    plt = _load_plotting_module()
    dataset = comparison_payload["dataset"]
    raw_values = np.asarray(comparison_payload["raw_values"], dtype=float)
    normalized_values = np.asarray(dataset["values"], dtype=float)

    start_index = max(0, int(dataset["anomaly_start"]) - context_points)
    end_index = min(len(normalized_values), int(dataset["anomaly_end"]) + context_points + 1)
    columns = 2 if len(variants) > 1 else 1
    variant_rows = max(1, math.ceil(len(variants) / columns))
    figure_height = 5.5 + (3.4 * variant_rows)

    fig = plt.figure(figsize=(18 if columns == 2 else 14, figure_height), constrained_layout=True)
    grid = fig.add_gridspec(variant_rows + 2, columns)
    raw_ax = fig.add_subplot(grid[0, :])
    normalized_ax = fig.add_subplot(grid[1, :], sharex=raw_ax)
    variant_axes = [
        fig.add_subplot(grid[row + 2, column], sharex=raw_ax)
        for row in range(variant_rows)
        for column in range(columns)
    ]

    title = (
        f"{DISPLAY_NAME_MAP[algorithm_key]} | side-by-side variant comparison | "
        f"{dataset['dataset_name']}"
    )
    fig.suptitle(title, fontsize=16, y=1.02)

    ground_truth_patch = raw_ax.axvspan(
        int(dataset["anomaly_start"]) - start_index,
        int(dataset["anomaly_end"]) - start_index,
        color="tomato",
        alpha=0.2,
        label="Ground-truth anomaly",
    )
    normalized_ax.axvspan(
        int(dataset["anomaly_start"]) - start_index,
        int(dataset["anomaly_end"]) - start_index,
        color="tomato",
        alpha=0.2,
    )

    raw_ax.plot(raw_values[start_index:end_index], color="black", linewidth=1.0)
    raw_ax.set_title("raw signal")
    raw_ax.set_ylabel("raw")
    raw_ax.legend([ground_truth_patch], ["Ground-truth anomaly"], loc="upper right")

    normalized_ax.plot(normalized_values[start_index:end_index], color="#0f766e", linewidth=1.0)
    normalized_ax.set_title("normalized signal")
    normalized_ax.set_ylabel("normalized")

    visible_score_max = 0.0
    for payload in variants:
        current_max = float(np.nanmax(payload["scores"][start_index:end_index]))
        visible_score_max = max(visible_score_max, current_max)
    visible_score_max = 1.0 if visible_score_max <= 0 else visible_score_max * 1.05

    for axis_index, axis in enumerate(variant_axes):
        if axis_index >= len(variants):
            axis.set_axis_off()
            continue
        payload = variants[axis_index]
        metric_row = payload["metric_row"]
        threshold = float(metric_row["score_threshold"])
        prediction_mask = np.asarray(payload["predictions"], dtype=bool)
        predicted_segments = [
            (max(segment_start, start_index), min(segment_end, end_index))
            for segment_start, segment_end in _positive_segments(prediction_mask)
            if segment_end > start_index and segment_start < end_index
        ]
        axis.axvspan(
            int(dataset["anomaly_start"]) - start_index,
            int(dataset["anomaly_end"]) - start_index,
            color="tomato",
            alpha=0.2,
        )
        for segment_start, segment_end in predicted_segments:
            axis.axvspan(
                segment_start - start_index,
                segment_end - start_index,
                color="#8b5cf6",
                alpha=0.10,
            )
        axis.plot(payload["scores"][start_index:end_index], color="#7c3aed", linewidth=1.1)
        axis.axhline(threshold, color="tomato", linestyle="--", linewidth=1.0)
        axis.set_ylim(0, visible_score_max)
        axis.set_title(
            f"{metric_row['algorithm_variant']} | "
            f"Eval F1={metric_row['evaluation_f1']:.2f} | "
            f"ROC AUC={metric_row['roc_auc']:.2f}"
        )
        axis.set_ylabel("score")
        axis.set_xlabel("time index")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)
    return fig


def plot_algorithm_deep_dive(
    raw_values: np.ndarray,
    normalized_values: np.ndarray,
    dataset: dict[str, Any],
    score_values: np.ndarray,
    predictions: np.ndarray,
    metric_row: pd.Series,
    algorithm_key: str,
    context_points: int,
    save_path: Path | None = None,
) -> plt.Figure:
    plt = _load_plotting_module()
    start_index = max(0, int(dataset["anomaly_start"]) - context_points)
    end_index = min(len(normalized_values), int(
        dataset["anomaly_end"]) + context_points + 1)
    threshold = float(metric_row["score_threshold"])
    prediction_mask = np.asarray(predictions, dtype=bool)
    predicted_segments = [
        (max(segment_start, start_index), min(segment_end, end_index))
        for segment_start, segment_end in _positive_segments(prediction_mask)
        if segment_end > start_index and segment_start < end_index
    ]

    fig, axes = plt.subplots(3, 1, figsize=(
        16, 10), sharex=True, constrained_layout=True)
    fig.suptitle(
        f"{DISPLAY_NAME_MAP[algorithm_key]} | showcase dataset: {dataset['dataset_name']}",
        fontsize=16,
        y=1.02,
    )

    ground_truth_patch = None
    predicted_patch = None
    for axis in axes:
        ground_truth_patch = axis.axvspan(
            int(dataset["anomaly_start"]) - start_index,
            int(dataset["anomaly_end"]) - start_index,
            color="tomato",
            alpha=0.2,
            label="Ground-truth anomaly",
        )
        for segment_start, segment_end in predicted_segments:
            predicted_patch = axis.axvspan(
                segment_start - start_index,
                segment_end - start_index,
                color="#8b5cf6",
                alpha=0.10,
                label="Predicted above threshold",
            )

    axes[0].plot(raw_values[start_index:end_index], color="black", linewidth=1)
    axes[0].set_title("raw signal")
    axes[0].set_ylabel("raw")
    legend_handles = [
        ground_truth_patch] if ground_truth_patch is not None else []
    legend_labels = [
        "Ground-truth anomaly"] if ground_truth_patch is not None else []
    if predicted_patch is not None:
        legend_handles.append(predicted_patch)
        legend_labels.append("Predicted above threshold")
    if legend_handles:
        axes[0].legend(legend_handles, legend_labels, loc="upper right")

    axes[1].plot(normalized_values[start_index:end_index],
                 color="#0f766e", linewidth=1)
    axes[1].set_title("normalized signal")
    axes[1].set_ylabel("normalized")

    axes[2].plot(score_values[start_index:end_index],
                 color="#7c3aed", linewidth=1.2)
    axes[2].axhline(threshold, color="tomato", linestyle="--", linewidth=1)
    axes[2].set_title(
        "score | "
        f"ROC AUC={metric_row['roc_auc']:.2f}, "
        f"F1={metric_row['f1']:.2f}, "
        f"Eval Recall={metric_row['evaluation_recall']:.2f}, "
        f"Eval F1={metric_row['evaluation_f1']:.2f}, "
        f"runtime={metric_row['runtime_seconds']:.2f}s"
    )
    axes[2].set_ylabel("score")
    axes[2].set_xlabel("time index")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)
    return fig


def render_algorithm_report(notebook_state: dict[str, Any], algorithm_key: str, context_points: int = 1200) -> None:
    ns = notebook_state["ns"]
    config = notebook_state["config"]
    benchmark = notebook_state["benchmark"]

    if algorithm_key not in config["selected_algorithms"]:
        print(
            f"{DISPLAY_NAME_MAP[algorithm_key]} is disabled in the control panel.")
        return

    results = benchmark["results"]
    context = notebook_state["context"]

    variant_config = ns.build_variant_config_table(config, algorithm_key)
    parameter_effects = ns.build_algorithm_parameter_effect_table(
        results, algorithm_key)
    dataset_variant_summary = ns.build_algorithm_regime_table(
        results, algorithm_key, "variant")
    length_summary = ns.build_algorithm_regime_table(
        results, algorithm_key, "length_bucket")
    anomaly_ratio_summary = ns.build_algorithm_regime_table(
        results, algorithm_key, "anomaly_ratio_bucket")
    section_summary, section_top = ns.build_algorithm_section_tables(
        results, algorithm_key)

    display(variant_config)
    display(parameter_effects)
    display(dataset_variant_summary)
    display(length_summary)
    display(anomaly_ratio_summary)
    display(section_summary)
    display(section_top)

    parameter_effects.to_csv(ns.result_table_path(
        f"{algorithm_key}_parameter_effects.csv"), index=False)
    dataset_variant_summary.to_csv(ns.result_table_path(
        f"{algorithm_key}_dataset_variant_summary.csv"), index=False)
    length_summary.to_csv(ns.result_table_path(
        f"{algorithm_key}_length_summary.csv"), index=False)
    anomaly_ratio_summary.to_csv(ns.result_table_path(
        f"{algorithm_key}_anomaly_ratio_summary.csv"), index=False)

    fig = ns.plot_algorithm_benchmark_panel(
        results, algorithm_key, ns.result_algorithm_panel_path(algorithm_key))
    if fig is not None:
        plt = ns._load_plotting_module()
        plt.show()

    paper_fig = ns.plot_algorithm_paper_panel(
        results, algorithm_key, ns.result_algorithm_paper_panel_path(algorithm_key))
    if paper_fig is not None:
        plt = ns._load_plotting_module()
        plt.show()

    variant_comparison = ns.build_algorithm_variant_comparison(
        config,
        context["prepared_dataset_dir"],
        results,
        algorithm_key,
    )
    if variant_comparison is not None and len(variant_comparison["variants"]) > 1:
        display(variant_comparison["selection_summary"])
        display(variant_comparison["summary"])
        variant_comparison["summary"].to_csv(
            ns.result_table_path(f"{algorithm_key}_variant_comparison.csv"),
            index=False,
        )
        comparison_fig = ns.plot_algorithm_variant_comparison(
            variant_comparison,
            algorithm_key,
            context_points=context_points,
            save_path=ns.result_algorithm_variant_comparison_path(algorithm_key),
        )
        if comparison_fig is not None:
            plt = ns._load_plotting_module()
            plt.show()

    showcase = ns.build_algorithm_showcase(
        config,
        context["prepared_dataset_dir"],
        results,
        algorithm_key,
    )
    if showcase is None:
        print(f"No showcase case available for {DISPLAY_NAME_MAP[algorithm_key]} in the current run.")
        return

    display(showcase["summary"])
    print(f"Showcase variant: {showcase['metric_row']['algorithm_display']}")
    fig = ns.plot_algorithm_deep_dive(
        showcase["raw_values"],
        showcase["dataset"]["values"],
        showcase["dataset"],
        showcase["scores"],
        showcase["predictions"],
        showcase["metric_row"],
        algorithm_key,
        context_points=context_points,
        save_path=ns.result_deep_dive_path(
            showcase["metric_row"]["algorithm_run_id"], showcase["dataset"]["dataset_name"]),
    )
    plt = ns._load_plotting_module()
    plt.show()


def run_algorithm(
    algorithm_key: str,
    values: np.ndarray,
    window_size: int,
    params: dict[str, Any],
    window_stride: int = 1,
) -> np.ndarray:
    cleaned_params = {key: value for key,
                      value in params.items() if value is not None}
    algorithm_function = _load_algorithm_function(algorithm_key)
    return np.asarray(
        algorithm_function(values, window_size, window_stride=max(
            1, int(window_stride)), **cleaned_params),
        dtype=float,
    ).ravel()


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    hours, remaining_minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {remaining_minutes}m {remaining_seconds}s"
    if minutes > 0:
        return f"{minutes}m {remaining_seconds}s"
    return f"{remaining_seconds}s"


def _progress_html(inner_html: str) -> widgets.HTML:
    return widgets.HTML(
        value=(
            "<div style='white-space: normal; overflow-wrap: anywhere; word-break: break-word; "
            "max-width: 100%; line-height: 1.45;'>"
            f"{inner_html}"
            "</div>"
        ),
        layout=widgets.Layout(width="100%"),
    )


def prepare_run_context(config: dict[str, Any]) -> dict[str, Any]:
    if not config["selected_runs"]:
        raise ValueError(
            "Select at least one algorithm in the control panel before running the notebook.")

    raw_dataset_paths = ensure_raw_datasets_available()
    prepared_dataset_dir, prepared_dataset_paths = ensure_normalized_datasets(
        config["normalization_method"],
        config["clip_quantile"],
        overwrite=config["overwrite_normalized_datasets"],
    )
    prepared_dataset_paths = sorted(
        prepared_dataset_paths, key=lambda path: (path.stat().st_size, path.name))
    benchmark_dataset_paths = prepared_dataset_paths[: config["dataset_limit"]
                                                     ] if config["dataset_limit"] is not None else prepared_dataset_paths

    run_config_frame = pd.DataFrame(
        [
            {
                "dataset_limit": "all" if config["dataset_limit"] is None else config["dataset_limit"],
                "normalization_method": config["normalization_method"],
                "clip_quantile": config["clip_quantile"],
                "window_size": config["window_size"],
                "window_stride": config["window_stride"],
                "threshold_method": config["threshold_method"],
                "threshold_value": config["threshold_value"],
                "evaluation_mode": config["evaluation_mode"],
                "selected_algorithms": ", ".join(config["selected_algorithms"]),
                "selected_configurations": len(config["selected_runs"]),
                "variant_tabs": " | ".join(
                    f"{DISPLAY_NAME_MAP[algorithm_key]}: {', '.join(entry['algorithm_variant'] for entry in variant_rows)}"
                    for algorithm_key, variant_rows in config["algorithm_variants"].items()
                    if variant_rows
                ),
                "prepared_dataset_dir": portable_path_str(prepared_dataset_dir),
            }
        ]
    )

    preparation_summary = pd.DataFrame(
        [
            {
                "legacy_virgin_count": len(list(LEGACY_VIRGIN_DIR.glob("*.txt"))),
                "workspace_raw_count": len(raw_dataset_paths),
                "workspace_normalized_count": len(prepared_dataset_paths),
                "benchmark_count": len(benchmark_dataset_paths),
                "normalization_method": config["normalization_method"],
                "clip_quantile": config["clip_quantile"],
                "window_size": config["window_size"],
                "window_stride": config["window_stride"],
                "threshold_method": config["threshold_method"],
                "threshold_value": config["threshold_value"],
                "evaluation_mode": config["evaluation_mode"],
                "normalized_dataset_dir": portable_path_str(prepared_dataset_dir),
            }
        ]
    )
    selected_run_parameters = build_selected_run_parameter_frame(config)

    run_config_frame.to_csv(result_table_path(
        "run_configuration.csv"), index=False)
    selected_run_parameters.to_csv(result_table_path(
        "selected_run_parameters.csv"), index=False)
    preparation_summary.to_csv(result_table_path(
        "dataset_preparation_summary.csv"), index=False)

    return {
        "raw_dataset_paths": raw_dataset_paths,
        "prepared_dataset_dir": prepared_dataset_dir,
        "prepared_dataset_paths": prepared_dataset_paths,
        "benchmark_dataset_paths": benchmark_dataset_paths,
        "run_config_frame": run_config_frame,
        "selected_run_parameters": selected_run_parameters,
        "preparation_summary": preparation_summary,
    }


def run_benchmark(
    config: dict[str, Any],
    prepared_dataset_dir: Path,
    benchmark_dataset_paths: list[Path],
    show_progress: bool = True,
) -> dict[str, Any]:
    records = []
    selected_runs = config["selected_runs"]
    total_dataset_count = len(benchmark_dataset_paths)
    total_run_count = total_dataset_count * len(selected_runs)
    benchmark_started_at = time.perf_counter()
    completed_runs = 0
    recent_messages: list[str] = []

    progress_bar = None
    progress_summary = None
    progress_status = None
    progress_recent = None

    if show_progress:
        progress_bar = widgets.IntProgress(
            value=0,
            min=0,
            max=max(total_run_count, 1),
            description="Runs",
            bar_style="info",
            layout=widgets.Layout(width="100%"),
        )
        progress_summary = _progress_html(
            (
                f"<b>Benchmark queued</b>: {total_dataset_count} datasets x {len(selected_runs)} configurations "
                f"= {total_run_count} runs"
            )
        )
        progress_status = _progress_html("<b>Status</b>: waiting to start...")
        progress_recent = _progress_html("<b>Recent runs</b>: none yet")
        display(
            widgets.VBox(
                [
                    progress_summary,
                    progress_bar,
                    progress_status,
                    progress_recent,
                ],
                layout=widgets.Layout(width="100%", overflow="hidden"),
            )
        )

    for dataset_index, prepared_dataset_path in enumerate(benchmark_dataset_paths, start=1):
        dataset = load_prepared_dataset(prepared_dataset_path)
        values = dataset["values"]
        labels = dataset["labels"]
        window_size = config["window_size"] if config["window_size"] is not None else estimate_window_size(
            values)
        window_stride = max(1, int(config["window_stride"]))

        for algorithm_index, run_config in enumerate(selected_runs, start=1):
            algorithm_key = run_config["algorithm"]
            if show_progress and progress_status is not None:
                elapsed_seconds = time.perf_counter() - benchmark_started_at
                sand_note = ""
                if algorithm_key == "sand":
                    sand_note = (
                        "<br><b>Note</b>: SAND is the slowest option on long series because it runs repeated "
                        "K-Shape clustering and matrix-profile distance calculations. "
                        "The <code>alpha</code> value changes adaptation behavior, but it is not the main runtime driver."
                    )
                elif algorithm_key == "damp":
                    sand_note = (
                        "<br><b>Note</b>: DAMP can also take time on long series because it repeatedly runs "
                        "matrix-profile similarity searches while moving through the stream."
                    )
                progress_status.value = (
                    "<div style='white-space: normal; overflow-wrap: anywhere; word-break: break-word; max-width: 100%; line-height: 1.45;'>"
                    f"<b>Status</b><br>"
                    f"<b>Dataset</b>: {dataset_index}/{total_dataset_count} | {html.escape(dataset['dataset_name'])}<br>"
                    f"<b>Configuration</b>: {algorithm_index}/{len(selected_runs)} | {html.escape(run_config['algorithm_display'])}<br>"
                    f"<b>Window</b>: {window_size} | <b>Stride</b>: {window_stride} | "
                    f"<b>Threshold</b>: {html.escape(str(config['threshold_method']))}={html.escape(str(config['threshold_value']))} | "
                    f"<b>Eval</b>: {html.escape(str(config['evaluation_mode']))}<br>"
                    f"<b>Completed</b>: {completed_runs}/{total_run_count} runs | "
                    f"<b>Elapsed</b>: {_format_duration(elapsed_seconds)}"
                    f"{sand_note}"
                    "</div>"
                )
            start_time = time.perf_counter()
            try:
                scores = run_algorithm(
                    algorithm_key,
                    values,
                    window_size,
                    run_config["params"],
                    window_stride=window_stride,
                )
                runtime_seconds = time.perf_counter() - start_time
                metrics = compute_metrics(
                    labels,
                    scores,
                    threshold_method=str(config["threshold_method"]),
                    threshold_value=float(config["threshold_value"]),
                    window_size=window_size,
                    evaluation_mode=str(config["evaluation_mode"]),
                    window_stride=window_stride,
                )
                error_message = ""
                save_scores_if_needed(
                    dataset["dataset_name"], run_config["algorithm_run_id"], labels, scores, config["save_per_dataset_scores"])
            except Exception as error:
                scores = np.array([], dtype=float)
                runtime_seconds = time.perf_counter() - start_time
                metrics = {
                    "roc_auc": float("nan"),
                    "average_precision": float("nan"),
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "f1": float("nan"),
                    "precision_at_k": float("nan"),
                    "range_precision": float("nan"),
                    "range_recall": float("nan"),
                    "range_f1": float("nan"),
                    "evaluation_precision": float("nan"),
                    "evaluation_recall": float("nan"),
                    "evaluation_f1": float("nan"),
                    "existence_reward": float("nan"),
                    "overlap_reward": float("nan"),
                    "affiliation_precision": float("nan"),
                    "affiliation_recall": float("nan"),
                    "tsb_uad_window": window_size,
                    "score_threshold": float("nan"),
                    "threshold_method": str(config["threshold_method"]),
                    "threshold_value": float(config["threshold_value"]),
                    "evaluation_mode": str(config["evaluation_mode"]),
                    "tp": float("nan"),
                    "tn": float("nan"),
                    "fp": float("nan"),
                    "fn": float("nan"),
                }
                error_message = str(error)

            records.append(
                {
                    "dataset_name": dataset["dataset_name"],
                    "dataset_sequence": dataset["dataset_sequence"],
                    "family": dataset["family"],
                    "variant": dataset["variant"],
                    "algorithm": algorithm_key,
                    "algorithm_base_display": run_config["algorithm_base_display"],
                    "algorithm_superfamily": run_config["algorithm_superfamily"],
                    "algorithm_category": run_config["algorithm_category"],
                    "algorithm_display": run_config["algorithm_display"],
                    "algorithm_variant": run_config["algorithm_variant"],
                    "algorithm_run_id": run_config["algorithm_run_id"],
                    "variant_index": run_config["variant_index"],
                    "window_size": window_size,
                    "window_stride": window_stride,
                    "series_length": len(values),
                    "anomaly_count": int(labels.sum()),
                    "train_end": dataset["train_end"],
                    "anomaly_start": dataset["anomaly_start"],
                    "anomaly_end": dataset["anomaly_end"],
                    "normalization_method": config["normalization_method"],
                    "threshold_method": str(config["threshold_method"]),
                    "threshold_value": float(config["threshold_value"]),
                    "evaluation_mode": str(config["evaluation_mode"]),
                    "prepared_dataset_dir": portable_path_str(prepared_dataset_dir),
                    "runtime_seconds": runtime_seconds,
                    "score_mean": float(scores.mean()) if scores.size else float("nan"),
                    "score_std": float(scores.std()) if scores.size else float("nan"),
                    "params_json": json.dumps(run_config["params"], sort_keys=True),
                    "error": error_message,
                    **{f"param__{key}": value for key, value in run_config["params"].items()},
                    **metrics,
                }
            )
            completed_runs += 1

            if show_progress and progress_bar is not None and progress_summary is not None and progress_recent is not None:
                elapsed_seconds = time.perf_counter() - benchmark_started_at
                average_seconds_per_run = elapsed_seconds / \
                    max(completed_runs, 1)
                remaining_runs = max(total_run_count - completed_runs, 0)
                eta_seconds = average_seconds_per_run * remaining_runs
                progress_bar.value = completed_runs
                progress_summary.value = (
                    "<div style='white-space: normal; overflow-wrap: anywhere; word-break: break-word; max-width: 100%; line-height: 1.45;'>"
                    f"<b>Progress</b>: {completed_runs}/{total_run_count} runs | "
                    f"<b>Datasets</b>: {dataset_index}/{total_dataset_count} | "
                    f"<b>Elapsed</b>: {_format_duration(elapsed_seconds)} | "
                    f"<b>ETA</b>: {_format_duration(eta_seconds)}"
                    "</div>"
                )
                recent_line = (
                    f"{completed_runs}/{total_run_count} | "
                    f"{dataset['dataset_name']} | "
                    f"{run_config['algorithm_display']} | "
                    f"{runtime_seconds:.2f}s"
                )
                if error_message:
                    recent_line += f" | error: {error_message}"
                recent_messages = [recent_line] + recent_messages[:4]
                progress_recent.value = (
                    "<div style='white-space: normal; overflow-wrap: anywhere; word-break: break-word; max-width: 100%; line-height: 1.45;'>"
                    "<b>Recent runs</b><br>"
                    + "<br>".join(html.escape(line)
                                  for line in recent_messages)
                    + "</div>"
                )

    results = pd.DataFrame.from_records(records)
    dataset_catalog = build_dataset_catalog(results)
    algorithm_summary = summarize_algorithms(results)
    family_summary = summarize_families(results)
    overall_regime_summary = build_overall_regime_summary(results)
    best_by_evaluation = build_best_algorithm_table(results, "evaluation_f1")
    best_by_f1 = build_best_algorithm_table(results, "f1")
    best_by_auc = build_best_algorithm_table(results, "roc_auc")
    errors = results.loc[results["error"] != ""].copy()

    results.to_csv(result_table_path("benchmark_results.csv"), index=False)
    dataset_catalog.to_csv(result_table_path(
        "dataset_catalog.csv"), index=False)
    algorithm_summary.to_csv(result_table_path(
        "algorithm_summary.csv"), index=False)
    family_summary.to_csv(result_table_path("family_summary.csv"), index=False)
    overall_regime_summary.to_csv(result_table_path(
        "overall_regime_summary.csv"), index=False)
    best_by_evaluation.to_csv(result_table_path(
        "best_algorithm_by_dataset_evaluation.csv"), index=False)
    best_by_f1.to_csv(result_table_path(
        "best_algorithm_by_dataset_f1.csv"), index=False)
    best_by_auc.to_csv(result_table_path(
        "best_algorithm_by_dataset_auc.csv"), index=False)
    errors.to_csv(result_table_path("error_report.csv"), index=False)

    for algorithm_key in config["selected_algorithms"]:
        results.loc[results["algorithm"] == algorithm_key].to_csv(
            result_per_algorithm_table_path(algorithm_key), index=False)

    overview = pd.DataFrame(
        [
            {
                "dataset_count": dataset_catalog["dataset_name"].nunique(),
                "algorithm_count": len(config["selected_algorithms"]),
                "configuration_count": len(selected_runs),
                "run_count": len(results),
                "median_series_length": dataset_catalog["series_length"].median(),
                "median_anomaly_ratio": dataset_catalog["anomaly_ratio"].median(),
                "median_window_size": dataset_catalog["window_size"].median(),
                "window_stride": config["window_stride"],
                "threshold_method": config["threshold_method"],
                "threshold_value": config["threshold_value"],
                "evaluation_mode": config["evaluation_mode"],
                "error_count": len(errors),
                "normalization_method": config["normalization_method"],
            }
        ]
    )

    if show_progress and progress_bar is not None and progress_summary is not None and progress_status is not None:
        total_elapsed = time.perf_counter() - benchmark_started_at
        progress_bar.value = max(total_run_count, progress_bar.value)
        progress_bar.bar_style = "warning" if not errors.empty else "success"
        progress_summary.value = (
            "<div style='white-space: normal; overflow-wrap: anywhere; word-break: break-word; max-width: 100%; line-height: 1.45;'>"
            f"<b>Benchmark complete</b>: {completed_runs}/{total_run_count} runs finished | "
            f"<b>Total time</b>: {_format_duration(total_elapsed)}"
            "</div>"
        )
        progress_status.value = (
            "<div style='white-space: normal; overflow-wrap: anywhere; word-break: break-word; max-width: 100%; line-height: 1.45;'>"
            f"<b>Finished</b>: {dataset_catalog['dataset_name'].nunique()} datasets processed | "
            f"{len(errors)} runs with errors"
            "</div>"
        )

    return {
        "results": results,
        "dataset_catalog": dataset_catalog,
        "algorithm_summary": algorithm_summary,
        "family_summary": family_summary,
        "overall_regime_summary": overall_regime_summary,
        "best_by_evaluation": best_by_evaluation,
        "best_by_f1": best_by_f1,
        "best_by_auc": best_by_auc,
        "errors": errors,
        "overview": overview,
    }
