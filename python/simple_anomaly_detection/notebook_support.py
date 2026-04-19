from __future__ import annotations

import html
import importlib
import json
import math
import re
import shutil
import sys
import time
import types
import warnings
from difflib import get_close_matches
from datetime import datetime, timezone
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
RESULT_THESIS_FIGURES_DIR = RESULT_FIGURES_DIR / "thesis"
RESULT_SCORES_DIR = RESULTS_DIR / "scores"
RESULT_RUN_SESSION_DIR = RESULTS_DIR / "run_sessions"
HIGH_ROI_NOTES_SOURCE_PATH = PROJECT_ROOT / "high_roi_algorithm_notes.md"
HIGH_ROI_NOTES_RESULT_PATH = RESULT_TABLES_DIR / "high_roi_algorithm_notes.md"
THESIS_FIGURE_CATALOG_PATH = RESULT_TABLES_DIR / "thesis_figure_catalog.csv"
THESIS_FIGURE_CAPTIONS_PATH = RESULT_TABLES_DIR / "thesis_figure_captions.md"
DEFAULT_RUN_NAME = "active_benchmark"
RUN_SESSION_MANIFEST_FILENAME = "session_manifest.json"
RUN_SESSION_CONTROL_STATE_FILENAME = "control_state.json"

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
        RESULT_THESIS_FIGURES_DIR,
        RESULT_SCORES_DIR,
        RESULT_RUN_SESSION_DIR,
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

VARIANT_METADATA_KEYS = {
    "variant_name",
    "focus",
    "variant_family",
    "ablation_parameter",
    "ablation_label",
    "ablation_role",
}


AUTO_ABLATION_BLUEPRINTS: dict[str, dict[str, Any]] = {
    "isolation_forest": {
        "ablations": [
            {
                "parameter": "n_estimators",
                "variant_name": "Trees 400",
                "focus": "Measures whether a larger forest improves score stability enough to justify the runtime.",
                "role": "score_driver",
                "changes": {"n_estimators": 400},
            },
            {
                "parameter": "max_samples",
                "variant_name": "Max samples auto",
                "focus": "Measures how a broader training sample per tree changes the global isolation pattern.",
                "role": "score_driver",
                "changes": {"max_samples": "auto"},
            },
            {
                "parameter": "max_features",
                "variant_name": "Max feat 0.6",
                "focus": "Measures the effect of stronger random feature subsampling inside the forest.",
                "role": "score_driver",
                "changes": {"max_features": 0.6},
            },
            {
                "parameter": "bootstrap",
                "variant_name": "Bootstrap on",
                "focus": "Measures whether sampling windows with replacement changes the tree ensemble enough to alter detections.",
                "role": "score_driver",
                "changes": {"bootstrap": True},
            },
            {
                "parameter": "random_state",
                "variant_name": "Seed 7",
                "focus": "Measures sensitivity to stochastic initialization rather than score geometry.",
                "role": "stability",
                "changes": {"random_state": 7},
            },
        ],
    },
    "local_outlier_factor": {
        "ablations": [
            {
                "parameter": "n_neighbors",
                "variant_name": "Neighbors 10",
                "focus": "Makes LOF more local so short or sharp anomalies have more leverage.",
                "role": "score_driver",
                "changes": {"n_neighbors": 10},
            },
            {
                "parameter": "algorithm",
                "variant_name": "Search brute",
                "focus": "Measures the neighbor-search backend while leaving the density formula unchanged.",
                "role": "backend",
                "changes": {"algorithm": "brute"},
            },
            {
                "parameter": "leaf_size",
                "variant_name": "Leaf size 60",
                "focus": "Measures the tree-search backend tuning rather than a scoring-theory change.",
                "role": "backend",
                "changes": {"leaf_size": 60},
            },
            {
                "parameter": "metric",
                "variant_name": "Metric manhattan",
                "focus": "Measures how redefining window similarity changes local density estimates.",
                "role": "score_driver",
                "changes": {"metric": "manhattan"},
            },
            {
                "parameter": "p",
                "variant_name": "p = 1",
                "focus": "Measures the Minkowski exponent directly while keeping the metric family the same.",
                "role": "score_driver",
                "changes": {"p": 1},
            },
        ],
    },
    "sand": {
        "ablations": [
            {
                "parameter": "alpha",
                "variant_name": "Alpha 0.7",
                "focus": "Measures faster adaptation to recent behavior in the online updates.",
                "role": "score_driver",
                "changes": {"alpha": 0.7},
            },
            {
                "parameter": "init_length",
                "variant_name": "Init 3000",
                "focus": "Measures a shorter initialization phase before online updates take over.",
                "role": "score_driver",
                "changes": {"init_length": 3000},
            },
            {
                "parameter": "batch_size",
                "variant_name": "Batch 1000",
                "focus": "Measures finer-grained online updates at the cost of more update steps.",
                "role": "score_driver",
                "changes": {"batch_size": 1000},
            },
            {
                "parameter": "k",
                "variant_name": "k = 3",
                "focus": "Measures a more local nearest-subsequence comparison.",
                "role": "score_driver",
                "changes": {"k": 3},
            },
            {
                "parameter": "subsequence_multiplier",
                "variant_name": "Subseq x2",
                "focus": "Measures a shorter subsequence context while keeping the rest of SAND fixed.",
                "role": "score_driver",
                "changes": {"subsequence_multiplier": 2},
            },
            {
                "parameter": "overlap",
                "variant_name": "Overlap 64",
                "focus": "Measures a coarser explicit subsequence step rather than the auto overlap heuristic.",
                "role": "score_driver",
                "changes": {"overlap": 64},
            },
        ],
    },
    "matrix_profile": {
        "baseline": {
            "variant_name": "Baseline",
            "focus": "Balanced discord context for one-factor ablation.",
            "subsequence_multiplier": 2,
        },
        "ablations": [
            {
                "parameter": "subsequence_multiplier",
                "variant_name": "Subseq x1",
                "focus": "Measures a shorter discord context focused on local deviations.",
                "role": "score_driver",
                "changes": {"subsequence_multiplier": 1},
            },
            {
                "parameter": "subsequence_multiplier",
                "variant_name": "Subseq x4",
                "focus": "Measures a broader discord context for long anomalous structure.",
                "role": "score_driver",
                "changes": {"subsequence_multiplier": 4},
            },
        ],
    },
    "damp": {
        "ablations": [
            {
                "parameter": "start_index_multiplier",
                "variant_name": "Start x2",
                "focus": "Measures the effect of waiting longer before the backward search begins.",
                "role": "score_driver",
                "changes": {"start_index_multiplier": 2.0},
            },
            {
                "parameter": "x_lag_multiplier",
                "variant_name": "x_lag x8",
                "focus": "Measures a much deeper historical search horizon during backward matching.",
                "role": "score_driver",
                "changes": {"x_lag_multiplier": 8.0},
            },
        ],
    },
    "hbos": {
        "ablations": [
            {
                "parameter": "n_bins",
                "variant_name": "Bins 20",
                "focus": "Measures finer histogram resolution at every window position.",
                "role": "score_driver",
                "changes": {"n_bins": 20},
            },
            {
                "parameter": "alpha",
                "variant_name": "Alpha 0.05",
                "focus": "Measures milder smoothing inside the log-density score.",
                "role": "score_driver",
                "changes": {"alpha": 0.05},
            },
            {
                "parameter": "tol",
                "variant_name": "Tol 0.2",
                "focus": "Measures stricter out-of-range penalties near histogram edges.",
                "role": "score_driver",
                "changes": {"tol": 0.2},
            },
        ],
    },
    "ocsvm": {
        "ablations": [
            {
                "parameter": "kernel",
                "variant_name": "Kernel linear",
                "focus": "Measures whether a linear novelty boundary is sufficient in the embedded window space.",
                "role": "score_driver",
                "changes": {"kernel": "linear"},
            },
            {
                "parameter": "nu",
                "variant_name": "Nu 0.10",
                "focus": "Measures a more permissive novelty boundary.",
                "role": "score_driver",
                "changes": {"nu": 0.10},
            },
            {
                "parameter": "gamma",
                "variant_name": "Gamma 0.1",
                "focus": "Measures a more local nonlinear boundary than the default `scale` heuristic.",
                "role": "score_driver",
                "changes": {"gamma": 0.1},
            },
            {
                "parameter": "train_fraction",
                "variant_name": "Train frac 0.20",
                "focus": "Measures a longer mostly-normal warmup segment for fitting the boundary.",
                "role": "score_driver",
                "changes": {"train_fraction": 0.20},
            },
        ],
    },
    "pca": {
        "ablations": [
            {
                "parameter": "n_components",
                "variant_name": "Components 0.95",
                "focus": "Measures explained-variance truncation rather than retaining the default PCA basis.",
                "role": "score_driver",
                "changes": {"n_components": 0.95},
            },
            {
                "parameter": "n_selected_components",
                "variant_name": "Score comps 2",
                "focus": "Measures scoring on a narrow set of trailing low-variance components.",
                "role": "score_driver",
                "changes": {"n_selected_components": 2},
            },
            {
                "parameter": "whiten",
                "variant_name": "Whiten on",
                "focus": "Measures whitening of retained components before the PCA score is computed.",
                "role": "score_driver",
                "changes": {"whiten": True},
            },
            {
                "parameter": "weighted",
                "variant_name": "Weighted off",
                "focus": "Measures the score without explained-variance weighting.",
                "role": "score_driver",
                "changes": {"weighted": False},
            },
            {
                "parameter": "standardization",
                "variant_name": "Standardize off",
                "focus": "Measures PCA on raw normalized windows without per-feature standardization.",
                "role": "score_driver",
                "changes": {"standardization": False},
            },
        ],
    },
}


def _variant_param_payload(variant: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in variant.items()
        if key not in VARIANT_METADATA_KEYS
    }


def _default_baseline_variant(algorithm_key: str) -> dict[str, Any]:
    for preset_name in ("paper_full_suite", "paper_high_roi"):
        preset = PAPER_PRESET_DEFINITIONS.get(preset_name, {})
        for variant in preset.get("variants", {}).get(algorithm_key, []):
            if str(variant.get("variant_name", "")).strip().lower() == "baseline":
                return dict(variant)
    raise KeyError(f"No baseline variant found for {algorithm_key}.")


def _build_auto_ablation_variants() -> dict[str, list[dict[str, Any]]]:
    variants: dict[str, list[dict[str, Any]]] = {}
    for algorithm_key in ALGORITHM_ORDER:
        blueprint = AUTO_ABLATION_BLUEPRINTS[algorithm_key]
        baseline_variant = dict(blueprint.get("baseline") or _default_baseline_variant(algorithm_key))
        baseline_params = _variant_param_payload(baseline_variant)
        baseline_variant.update(
            {
                "variant_name": baseline_variant.get("variant_name", "Baseline"),
                "focus": baseline_variant.get("focus", "Reference baseline used for one-knob-at-a-time ablation."),
                "variant_family": "baseline",
                "ablation_parameter": "baseline",
                "ablation_label": "Baseline",
                "ablation_role": "baseline",
            }
        )
        algorithm_variants = [baseline_variant]

        for ablation in blueprint.get("ablations", []):
            variant = {
                **baseline_params,
                **ablation["changes"],
                "variant_name": ablation["variant_name"],
                "focus": ablation["focus"],
                "variant_family": "ablation",
                "ablation_parameter": ablation["parameter"],
                "ablation_label": ablation["variant_name"],
                "ablation_role": ablation["role"],
            }
            algorithm_variants.append(variant)
        variants[algorithm_key] = algorithm_variants
    return variants


PAPER_PRESET_DEFINITIONS["auto_ablation"] = {
    "label": "Auto Ablation Sweep",
    "description": (
        "True one-factor-at-a-time ablation. Each enabled algorithm gets one baseline plus single-knob changes, "
        "so you can defend which arguments materially change the score and which mainly affect runtime or stability."
    ),
    "enabled_algorithms": ALGORITHM_ORDER,
    "variants": _build_auto_ablation_variants(),
}

VARIANT_MODE_LABELS = {
    "manual": "Manual subtabs",
    "paper_high_roi": "Auto: paper_high_roi",
    "paper_full_suite": "Auto: paper_full_suite",
    "auto_ablation": "Auto: auto_ablation",
}

ABLATION_ROLE_STYLES = {
    "score_driver": {"label": "Score-driving", "color": "#1d4ed8"},
    "backend": {"label": "Backend/runtime", "color": "#64748b"},
    "stability": {"label": "Stability", "color": "#7c3aed"},
    "baseline": {"label": "Baseline", "color": "#0f766e"},
    "": {"label": "Variant", "color": "#334155"},
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
    "run_name": "Stable name for this saved benchmark session. Keep the same name when you want the notebook to reopen the same configuration and continue from its last successful checkpoint.",
    "saved_run_selector": "Pick a saved benchmark session from disk. Loading it restores the control-panel settings and reconnects resume mode to that session's checkpoint tables.",
    "variant_mode": "How argument combinations are sourced for the run. Manual uses the visible subtabs exactly as edited. Auto modes ignore the live subtab values at run time and expand each enabled algorithm into preset combinations from paper_high_roi, paper_full_suite, or auto_ablation so parameter impact is measured automatically.",
    "dataset_limit": "How many prepared datasets to benchmark in this run. Use 0 to process every available dataset. Lower values are useful for smoke tests and fast iteration.",
    "batch_size": "How many selected datasets to process in this notebook run. Use 0 to process every selected dataset. When resume is enabled, this becomes the size of each resumable batch.",
    "resume_from_existing": "Continue from successful rows already saved for the current run name. The notebook skips completed dataset/configuration pairs, retries failed or incomplete ones, and appends the new rows to both the active results tables and the saved run-session checkpoint.",
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

SHARED_PIPELINE_STEPS = [
    "The raw time series is optionally clipped by `Clip q`, then normalized by `Normalize` before any detector sees it.",
    "Each detector receives the normalized 1D series together with a base `window_size` and `window_stride` from the general controls.",
    "Every detector returns a continuous anomaly score trace, and the notebook compares detectors on that score trace first.",
    "Each algorithm module rescales its own score trace to `[0, 1]` before returning it to the notebook.",
    "`Threshold method` and `Threshold value` are applied after scoring to turn the continuous trace into binary detections.",
    "`Evaluation mode` changes only the metric calculation, not the detector score itself.",
]

GENERAL_CONTROL_REFERENCE = [
    {
        "label": "Run name",
        "effect": "Binds the current setup to a saved session folder under results/run_sessions so you can reload the exact controls later and continue from the last successful checkpoint.",
    },
    {
        "label": "Argument mode",
        "effect": "Selects whether the run uses the visible subtabs exactly (`manual`) or replaces them with curated multi-variant sweeps from `paper_high_roi`, `paper_full_suite`, or `auto_ablation` at run time.",
    },
    {
        "label": "Dataset limit",
        "effect": "Filters how many prepared datasets are benchmarked. It does not change anomaly scores on any individual dataset.",
    },
    {
        "label": "Batch size",
        "effect": "Caps how many of the selected datasets are processed in the current notebook run. With resume enabled, it defines the size of each resumable batch.",
    },
    {
        "label": "Resume from existing",
        "effect": "Reuses successful benchmark rows already saved under the current run name, skips completed dataset/configuration pairs, and continues from the next incomplete work for the same benchmark setup.",
    },
    {
        "label": "Normalize",
        "effect": "Changes the numeric scale seen by every detector before windowing. This can materially change distance-based, density-based, and boundary-based scores.",
    },
    {
        "label": "Clip q",
        "effect": "Clamps extreme tails before normalization. This can suppress large spikes, which may reduce false positives or weaken genuine anomaly contrast.",
    },
    {
        "label": "Window size",
        "effect": "Sets the base temporal context used for sliding-window embedding. Matrix Profile, DAMP, and SAND also derive their subsequence lengths or warm-up positions from this base length.",
    },
    {
        "label": "Window stride",
        "effect": "Controls how densely windows or subsequences are sampled. Larger strides reduce overlap and runtime; most detectors then interpolate the lower-frequency window scores back onto the original time axis.",
    },
    {
        "label": "Threshold method",
        "effect": "Chooses the post-scoring rule that converts the normalized score trace into binary anomalies. It does not alter the detector's raw scoring process.",
    },
    {
        "label": "Threshold value",
        "effect": "Provides the numeric cutoff for the selected threshold rule. It changes which high-score regions are finally marked anomalous, not how the score trace is produced.",
    },
    {
        "label": "Evaluation mode",
        "effect": "Chooses whether metrics are computed as interval overlap (`range`) or exact point hits (`point`). It affects reported scores only.",
    },
    {
        "label": "Rebuild normalized datasets",
        "effect": "Forces regeneration of cached normalized CSV files. This is a data-preparation control and does not change detector math by itself.",
    },
    {
        "label": "Save per-dataset scores",
        "effect": "Writes the returned score traces to `results/scores/`. It does not change scoring or thresholding.",
    },
    {
        "label": "Algorithm checkboxes",
        "effect": "Choose which detectors are included in the benchmark. In auto modes they also act as a filter over the preset sweeps.",
    },
]

ALGORITHM_REFERENCE = {
    "isolation_forest": {
        "summary": "Fits an Isolation Forest on sliding windows and uses the negated `score_samples` output as the anomaly score.",
        "process_steps": [
            "Build overlapping rolling windows from the normalized series with length `window_size` and step `window_stride`.",
            "Fit `sklearn.ensemble.IsolationForest` on those windows.",
            "Compute `window_scores = -model.score_samples(windows)`, so windows isolated more easily by the trees receive higher anomaly scores.",
            "Min-max normalize the window scores to `[0, 1]`.",
            "Align the window scores back to the original time axis by center padding when stride is `1`, otherwise by interpolation across window centers.",
        ],
        "controls": [
            {
                "label": "Trees",
                "param": "n_estimators",
                "effect": "Adds more trees to the forest. More trees usually stabilize the average isolation score, but increase runtime and memory.",
            },
            {
                "label": "Max samples",
                "param": "max_samples",
                "effect": "Controls how many windows each tree is fit on. Smaller samples make each tree more local and cheaper; larger samples expose more global structure. `'auto'` delegates the sample count to sklearn's default rule.",
            },
            {
                "label": "Max feat.",
                "param": "max_features",
                "effect": "Controls what fraction of the window positions each tree can split on. Lower values add stronger feature subsampling; higher values let each tree use more of the full temporal context.",
            },
            {
                "label": "Bootstrap",
                "param": "bootstrap",
                "effect": "Switches tree training from sampling without replacement to sampling with replacement, which changes how much repeated windows can influence each tree.",
            },
            {
                "label": "Seed",
                "param": "random_state",
                "effect": "Fixes the random draws used by the forest so score traces are reproducible across runs.",
            },
        ],
    },
    "local_outlier_factor": {
        "summary": "Computes LOF on sliding windows and uses the negated `negative_outlier_factor_` as the anomaly score.",
        "process_steps": [
            "Build rolling windows from the normalized series.",
            "Clamp `n_neighbors` into the valid range with `effective_neighbors = max(2, min(n_neighbors, len(windows) - 1))`.",
            "Fit `sklearn.neighbors.LocalOutlierFactor` on the windows and call `fit_predict` to populate the local density ratios.",
            "Compute `window_scores = -model.negative_outlier_factor_`, so windows with much lower local density than their neighbors score higher.",
            "Min-max normalize and align the window scores back onto the original time axis.",
        ],
        "controls": [
            {
                "label": "Neighbors",
                "param": "n_neighbors",
                "effect": "Changes the size of the neighborhood used to estimate local density. Smaller values make the score more local and sensitive; larger values make it smoother and more global.",
            },
            {
                "label": "Search",
                "param": "algorithm",
                "effect": "Changes only the sklearn nearest-neighbor backend (`auto`, `ball_tree`, `kd_tree`, `brute`). It mainly affects runtime, not the density formula itself.",
            },
            {
                "label": "Leaf size",
                "param": "leaf_size",
                "effect": "Tunes the search-tree backend used by LOF. This is a performance knob rather than a scoring-logic knob.",
            },
            {
                "label": "Metric",
                "param": "metric",
                "effect": "Changes the distance function used between windows, which directly changes who counts as a neighbor and therefore the local density ratio.",
            },
            {
                "label": "p",
                "param": "p",
                "effect": "Changes the exponent of the Minkowski distance. It only has scoring impact when `Metric = minkowski`; `p=1` is Manhattan and `p=2` is Euclidean.",
            },
        ],
    },
    "sand": {
        "summary": "Runs the legacy SAND online detector on a subsequence representation derived from the base window size and uses `decision_scores_` as the anomaly trace.",
        "process_steps": [
            "Set `pattern_length = window_size` and compute `subsequence_length = max(subsequence_multiplier * window_size, window_size + 1)`, capped by the series length.",
            "Resolve the subsequence step size from `Overlap`; when the UI leaves `Overlap = 0`, the code uses `window_size` when stride is `1` and uses `window_stride` otherwise.",
            "Clamp `Init length` and `Batch size` so both are at least long enough for one subsequence.",
            "Clamp `k` to the smallest number of subsequences available across the initialization block and all online batches. When the UI leaves `k = 0`, the call omits `k` and the implementation falls back to SAND's default of `6`, then clamps it.",
            "Fit `SAND(...).fit(..., online=True, alpha=..., init_length=..., batch_size=..., overlaping_rate=overlap)` on the normalized series.",
            "Use `model.decision_scores_`, min-max normalize them, and resize the result back to the original series length.",
        ],
        "controls": [
            {
                "label": "Alpha",
                "param": "alpha",
                "effect": "Controls how strongly new batches influence the online update. Higher values adapt faster to recent behavior; lower values keep more inertia from earlier batches.",
            },
            {
                "label": "Init length",
                "param": "init_length",
                "effect": "Controls how much initial history is used before the online updates continue. Larger values give SAND a longer starting reference set.",
            },
            {
                "label": "Batch size",
                "param": "batch_size",
                "effect": "Controls how much data SAND ingests per online update. Larger batches reduce update frequency but make each update coarser and heavier.",
            },
            {
                "label": "k",
                "param": "k",
                "effect": "Sets how many neighboring subsequences SAND compares. Smaller values keep the score local; larger values smooth it. `0` means use the implementation default and then clamp it to what the data can support.",
            },
            {
                "label": "Subseq x",
                "param": "subsequence_multiplier",
                "effect": "Scales the subsequence length relative to `window_size`. Larger values make SAND compare longer contexts.",
            },
            {
                "label": "Overlap",
                "param": "overlap",
                "effect": "Sets the subsequence step used by SAND's online fit. Smaller steps create denser comparisons; larger steps reduce overlap and runtime. `0` means infer the step from the shared window settings.",
            },
        ],
    },
    "matrix_profile": {
        "summary": "Uses the first column of `stumpy.stump` as a discord score, so windows whose nearest neighbor is far away score highly.",
        "process_steps": [
            "Compute `subsequence_length = max(4, window_size * subsequence_multiplier)`, capped below the series length.",
            "Run `stumpy.stump(values, m=subsequence_length)` and keep the first column of the returned matrix profile.",
            "Subsample the profile by `window_stride` when stride is greater than `1`.",
            "Normalize the profile to `[0, 1]`, replacing non-finite values before scaling.",
            "Align the subsequence scores back to the original time axis around each subsequence center.",
        ],
        "controls": [
            {
                "label": "Subseq x",
                "param": "subsequence_multiplier",
                "effect": "Scales the discord subsequence length relative to `window_size`. Larger values look for broader anomalous contexts; smaller values focus on shorter local discords.",
            },
        ],
    },
    "damp": {
        "summary": "Runs the DAMP streaming-discord search and uses backward nearest-neighbor distances as the anomaly score.",
        "process_steps": [
            "Set the DAMP start position `sp_index = max(window_size + 1, round(window_size * start_index_multiplier) + 1)`.",
            "Resolve `x_lag`; when the UI leaves it at `0`, DAMP falls back to its internal heuristic `2^ceil(log2(8 * window_size))`.",
            "From `sp_index` onward, compare each current window against historical reference windows using repeated MASS nearest-neighbor searches. The returned nearest-neighbor distance is the discord score.",
            "Use DAMP's forward-pruning pass to skip future windows that already have a close enough match.",
            "Subsample by `window_stride`, normalize the resulting profile, and align it back onto the original time axis.",
        ],
        "controls": [
            {
                "label": "Start x",
                "param": "start_index_multiplier",
                "effect": "Moves the first scored window later in time by a multiple of `window_size`. Larger values delay scoring and give the method more history before the backward search starts.",
            },
            {
                "label": "x_lag x",
                "param": "x_lag_multiplier",
                "effect": "Sets how far back the backward search can look, as a multiple of `window_size`. Larger values widen the historical search horizon; `0` means use DAMP's internal heuristic instead.",
            },
        ],
    },
    "hbos": {
        "summary": "Builds one histogram per window position and scores each window by the negative sum of log histogram densities.",
        "process_steps": [
            "Build rolling windows from the normalized series.",
            "For each feature position inside the window, build a histogram with `n_bins` bins over that column across all windows.",
            "For each window value, look up the corresponding histogram-bin density and compute `log2(hist + alpha)`.",
            "If a value falls outside the learned histogram range, use `Tol` to decide whether to borrow the nearest edge-bin density or assign the minimum-density penalty.",
            "Sum the negative log densities across all window positions to get the window anomaly score, then normalize and align it back to the time axis.",
        ],
        "controls": [
            {
                "label": "Bins",
                "param": "n_bins",
                "effect": "Changes the histogram resolution at each window position. More bins capture finer structure; fewer bins smooth the density estimate.",
            },
            {
                "label": "Alpha",
                "param": "alpha",
                "effect": "Adds smoothing inside `log2(hist + alpha)`, preventing empty bins from producing undefined scores and softening sparse-bin penalties.",
            },
            {
                "label": "Tol",
                "param": "tol",
                "effect": "Controls how far outside a histogram edge a value can land before it receives the harsh minimum-density penalty instead of the nearest edge-bin density.",
            },
        ],
    },
    "ocsvm": {
        "summary": "Fits One-Class SVM on the earliest normalized windows and scores all windows with the negated `decision_function`.",
        "process_steps": [
            "Build rolling windows from the normalized series.",
            "Apply row-wise min-max scaling so every window is independently mapped into `[0, 1]` before model fitting.",
            "Choose the training prefix from the earliest windows using `train_fraction`, clamped so at least `8` windows are used when available.",
            "Fit `sklearn.svm.OneClassSVM` on that prefix only.",
            "Compute `window_scores = -model.decision_function(all_windows)` so windows farther outside the learned boundary score higher.",
            "Normalize and align the score trace back to the original time axis.",
        ],
        "controls": [
            {
                "label": "Kernel",
                "param": "kernel",
                "effect": "Changes the geometry of the decision boundary in window space. `linear` uses a flat boundary, while `rbf`, `poly`, and `sigmoid` introduce nonlinear boundaries.",
            },
            {
                "label": "Nu",
                "param": "nu",
                "effect": "Sets the One-Class SVM regularization level that bounds training errors and support-vector fraction. Larger values usually make the model more willing to score windows as abnormal.",
            },
            {
                "label": "Gamma",
                "param": "gamma",
                "effect": "Controls the locality of nonlinear kernels. Higher values make the boundary respond to finer local variation; lower values make it smoother. `linear` largely ignores this setting.",
            },
            {
                "label": "Train frac",
                "param": "train_fraction",
                "effect": "Sets how much of the earliest series prefix is assumed to be mostly normal and therefore used for fitting before scoring the full series.",
            },
        ],
    },
    "pca": {
        "summary": "Fits PCA on sliding windows and scores each window by its distance to the selected principal-component vectors used by the current implementation.",
        "process_steps": [
            "Build rolling windows from the normalized series.",
            "If `Standardize` is enabled, standardize each window position across the full window matrix with `StandardScaler` before PCA.",
            "Fit `sklearn.decomposition.PCA` with the requested `Components` and `Whiten` settings.",
            "Select the trailing principal-component vectors with `model.components_[-effective_selected:, :]`, where `effective_selected` comes from `Score comps` and defaults to all retained components when `0` is entered.",
            "Score each window with `sum(cdist(transformed_windows, selected_components) / selected_weights, axis=1)`. When `Weighted` is enabled, `selected_weights` comes from `explained_variance_ratio_`, so lower-variance components contribute more strongly because they divide by smaller values.",
            "Normalize and align the resulting score trace back to the original time axis.",
        ],
        "controls": [
            {
                "label": "Components",
                "param": "n_components",
                "effect": "Sets how many principal components PCA retains. Blank means sklearn's default behavior; a float such as `0.95` uses explained variance; an integer keeps an exact count.",
            },
            {
                "label": "Score comps",
                "param": "n_selected_components",
                "effect": "Chooses how many trailing principal-component vectors are used in the score. Smaller values focus the score on the lowest-variance directions; `0` means score against all retained components.",
            },
            {
                "label": "Whiten",
                "param": "whiten",
                "effect": "Requests PCA whitening, which rescales the retained components before the distance calculation.",
            },
            {
                "label": "Weighted",
                "param": "weighted",
                "effect": "Divides each component distance by its explained-variance ratio. This gives low-variance components more leverage in the final anomaly score.",
            },
            {
                "label": "Standardize",
                "param": "standardization",
                "effect": "Applies featurewise standardization before PCA so each position inside the window contributes on a comparable scale.",
            },
        ],
    },
}

pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 20)
pd.set_option("display.precision", 4)

DEFAULT_PLOT_STYLE = {
    "figure.figsize": (14, 8),
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linestyle": "--",
    "savefig.bbox": "tight",
}

THESIS_PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "semibold",
    "axes.titlesize": 14,
    "axes.labelsize": 11.5,
    "axes.edgecolor": "#334155",
    "axes.linewidth": 0.9,
    "grid.alpha": 0.28,
    "grid.color": "#cbd5e1",
    "grid.linestyle": "--",
    "font.family": "DejaVu Serif",
    "font.size": 11,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.autolayout": False,
}


@lru_cache(maxsize=1)
def _load_plotting_module():
    import matplotlib.pyplot as plt

    plt.rcParams.update(DEFAULT_PLOT_STYLE)
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


def result_algorithm_ablation_panel_path(algorithm_key: str) -> Path:
    return RESULT_ALGORITHM_PANEL_DIR / f"{algorithm_key}_ablation_panel.png"


def result_deep_dive_path(run_id: str, dataset_name: str) -> Path:
    return RESULT_DEEP_DIVE_DIR / f"{run_id}__deep_dive__{dataset_name}.png"


def result_thesis_figure_path(filename: str) -> Path:
    return RESULT_THESIS_FIGURES_DIR / filename


def result_score_path(dataset_name: str, run_id: str) -> Path:
    return RESULT_SCORES_DIR / f"{dataset_name}__{run_id}.csv"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def saved_run_session_id(run_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(run_name).strip().lower()).strip("-")
    return slug or "run"


def saved_run_session_dir(session_id: str) -> Path:
    return RESULT_RUN_SESSION_DIR / session_id


def saved_run_session_tables_dir(session_id: str) -> Path:
    return saved_run_session_dir(session_id) / "tables"


def saved_run_session_manifest_path(session_id: str) -> Path:
    return saved_run_session_dir(session_id) / RUN_SESSION_MANIFEST_FILENAME


def saved_run_session_control_state_path(session_id: str) -> Path:
    return saved_run_session_dir(session_id) / RUN_SESSION_CONTROL_STATE_FILENAME


def saved_run_session_table_path(session_id: str, filename: str) -> Path:
    return saved_run_session_tables_dir(session_id) / filename


def saved_run_session_figures_dir(session_id: str) -> Path:
    return saved_run_session_dir(session_id) / "figures"


def saved_run_session_figure_path(session_id: str, filename: str) -> Path:
    return saved_run_session_figures_dir(session_id) / filename


def saved_run_session_algorithm_panel_dir(session_id: str) -> Path:
    return saved_run_session_figures_dir(session_id) / "algorithm_panels"


def saved_run_session_algorithm_panel_path(session_id: str, algorithm_key: str) -> Path:
    return saved_run_session_algorithm_panel_dir(session_id) / f"{algorithm_key}_benchmark_panel.png"


def saved_run_session_algorithm_paper_panel_path(session_id: str, algorithm_key: str) -> Path:
    return saved_run_session_algorithm_panel_dir(session_id) / f"{algorithm_key}_paper_panel.png"


def saved_run_session_algorithm_variant_comparison_path(session_id: str, algorithm_key: str) -> Path:
    return saved_run_session_algorithm_panel_dir(session_id) / f"{algorithm_key}_variant_comparison.png"


def saved_run_session_algorithm_ablation_panel_path(session_id: str, algorithm_key: str) -> Path:
    return saved_run_session_algorithm_panel_dir(session_id) / f"{algorithm_key}_ablation_panel.png"


def saved_run_session_deep_dive_dir(session_id: str) -> Path:
    return saved_run_session_figures_dir(session_id) / "deep_dives"


def saved_run_session_deep_dive_path(session_id: str, run_id: str, dataset_name: str) -> Path:
    return saved_run_session_deep_dive_dir(session_id) / f"{run_id}__deep_dive__{dataset_name}.png"


def saved_run_session_thesis_figures_dir(session_id: str) -> Path:
    return saved_run_session_figures_dir(session_id) / "thesis"


def saved_run_session_thesis_figure_path(session_id: str, filename: str) -> Path:
    return saved_run_session_thesis_figures_dir(session_id) / filename


def saved_run_session_thesis_figure_catalog_path(session_id: str) -> Path:
    return saved_run_session_table_path(session_id, "thesis_figure_catalog.csv")


def saved_run_session_thesis_figure_captions_path(session_id: str) -> Path:
    return saved_run_session_table_path(session_id, "thesis_figure_captions.md")


def _session_supports_global_fallback(session_id: str | None) -> bool:
    return session_id in (None, saved_run_session_id(DEFAULT_RUN_NAME))


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric_value = float(value)
        return None if not math.isfinite(numeric_value) else numeric_value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _read_resume_table_if_exists(filename: str, session_id: str | None = None) -> pd.DataFrame | None:
    if session_id:
        session_frame = _read_table_if_exists(saved_run_session_table_path(session_id, filename))
        if session_frame is not None:
            return session_frame
    if _session_supports_global_fallback(session_id):
        return _read_table_if_exists(result_table_path(filename))
    return None


def _write_table_artifact(
    frame: pd.DataFrame,
    filename: str,
    session_id: str | None = None,
    *,
    write_global: bool = True,
) -> None:
    if write_global:
        frame.to_csv(result_table_path(filename), index=False)
    if session_id:
        session_path = saved_run_session_table_path(session_id, filename)
        session_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(session_path, index=False)


def build_selected_run_parameter_frame(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dataset_limit = "all" if config["dataset_limit"] is None else config["dataset_limit"]
    for run_config in config["selected_runs"]:
        params = dict(run_config["params"])
        row = {
            "dataset_limit": dataset_limit,
            "variant_mode": config["variant_mode"],
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
            "variant_origin": run_config["variant_origin"],
            "variant_source": run_config["variant_source"],
            "variant_focus": run_config["variant_focus"],
            "variant_family": run_config["variant_family"],
            "ablation_parameter": run_config["ablation_parameter"],
            "ablation_label": run_config["ablation_label"],
            "ablation_role": run_config["ablation_role"],
            "params_json": json.dumps(params, sort_keys=True),
        }
        for key, value in params.items():
            row[f"param__{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["algorithm", "variant_index"]).reset_index(drop=True)


def build_results_layout_frame() -> pd.DataFrame:
    rows = [
        ("notes/source", HIGH_ROI_NOTES_SOURCE_PATH),
        ("notes/results_copy", HIGH_ROI_NOTES_RESULT_PATH),
        ("results_root", RESULTS_DIR),
        ("tables", RESULT_TABLES_DIR),
        ("tables/per_algorithm", RESULT_PER_ALGORITHM_TABLES_DIR),
        ("figures", RESULT_FIGURES_DIR),
        ("figures/algorithm_panels", RESULT_ALGORITHM_PANEL_DIR),
        ("figures/deep_dives", RESULT_DEEP_DIVE_DIR),
        ("figures/thesis", RESULT_THESIS_FIGURES_DIR),
        ("scores", RESULT_SCORES_DIR),
        ("run_sessions", RESULT_RUN_SESSION_DIR),
        ("tables/thesis_figure_catalog.csv", THESIS_FIGURE_CATALOG_PATH),
        ("tables/thesis_figure_captions.md", THESIS_FIGURE_CAPTIONS_PATH),
    ]
    return pd.DataFrame(
        [{"output_group": label, "path": portable_path_str(
            path), "exists": path.exists()} for label, path in rows]
    )


RESULT_RUN_KEY_COLUMNS = [
    "dataset_name",
    "algorithm_run_id",
    "normalization_method",
    "threshold_method",
    "threshold_value",
    "evaluation_mode",
    "window_size",
    "window_stride",
    "prepared_dataset_dir",
]

PERSISTED_GENERAL_CONTROL_KEYS = [
    "run_name",
    "dataset_limit",
    "batch_size",
    "resume_from_existing",
    "variant_mode",
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
]

ALGORITHM_CONTROL_KEYS = [
    "run_iforest",
    "run_lof",
    "run_sand",
    "run_matrix_profile",
    "run_damp",
    "run_hbos",
    "run_ocsvm",
    "run_pca",
]


def _read_table_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _normalize_comparison_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            return ""
        return format(numeric_value, ".12g")
    return str(value)


def _normalize_comparison_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    normalized = frame.reindex(columns=columns).copy()
    for column in columns:
        normalized[column] = normalized[column].map(_normalize_comparison_value)
    if columns:
        normalized = normalized.sort_values(columns).reset_index(drop=True)
    return normalized


def _validate_resume_compatibility(
    current_selected_run_parameters: pd.DataFrame,
    *,
    session_id: str | None = None,
) -> None:
    existing_results = _read_resume_table_if_exists("benchmark_results.csv", session_id)
    if existing_results is None or existing_results.empty:
        return

    existing_selected_run_parameters = _read_resume_table_if_exists(
        "selected_run_parameters.csv",
        session_id,
    )
    if existing_selected_run_parameters is None:
        raise ValueError(
            "Cannot resume because benchmark_results.csv already exists but selected_run_parameters.csv is missing. "
            "Clear results/tables or disable Resume from existing before starting a different benchmark."
        )

    comparison_columns = sorted(
        (set(existing_selected_run_parameters.columns) | set(current_selected_run_parameters.columns))
        - {"dataset_limit"}
    )
    existing_normalized = _normalize_comparison_frame(
        existing_selected_run_parameters, comparison_columns
    )
    current_normalized = _normalize_comparison_frame(
        current_selected_run_parameters, comparison_columns
    )
    if existing_normalized.equals(current_normalized):
        return

    raise ValueError(
        "Cannot resume because the saved benchmark tables were produced with different settings. "
        "Keep the same normalization, window, threshold, evaluation, and algorithm variant configuration across batches, "
        "or clear results/tables before starting a new batch series."
    )


def _successful_results_frame(results_frame: pd.DataFrame) -> pd.DataFrame:
    if results_frame.empty or "error" not in results_frame.columns:
        return pd.DataFrame(columns=results_frame.columns)
    success_mask = results_frame["error"].fillna("").astype(str) == ""
    return results_frame.loc[success_mask].copy()


def _result_run_key_series(results_frame: pd.DataFrame) -> pd.Series:
    key_frame = results_frame.reindex(columns=RESULT_RUN_KEY_COLUMNS).copy()
    for column in RESULT_RUN_KEY_COLUMNS:
        key_frame[column] = key_frame[column].map(_normalize_comparison_value)
    return key_frame.agg("||".join, axis=1)


def _result_run_key(
    dataset_name: str,
    algorithm_run_id: str,
    normalization_method: str,
    threshold_method: str,
    threshold_value: float | int,
    evaluation_mode: str,
    window_size: int,
    window_stride: int,
    prepared_dataset_dir: str,
) -> str:
    return "||".join(
        _normalize_comparison_value(value)
        for value in (
            dataset_name,
            algorithm_run_id,
            normalization_method,
            threshold_method,
            threshold_value,
            evaluation_mode,
            window_size,
            window_stride,
            prepared_dataset_dir,
        )
    )


def _dataset_completion_run_keys(
    prepared_dataset_path: Path,
    prepared_dataset_dir: Path,
    config: dict[str, Any],
    selected_runs: list[dict[str, Any]],
) -> set[str]:
    dataset = load_prepared_dataset(prepared_dataset_path)
    values = dataset["values"]
    window_size = config["window_size"] if config["window_size"] is not None else estimate_window_size(values)
    window_stride = max(1, int(config["window_stride"]))
    prepared_dataset_dir_value = portable_path_str(prepared_dataset_dir)
    return {
        _result_run_key(
            dataset_name=dataset["dataset_name"],
            algorithm_run_id=run_config["algorithm_run_id"],
            normalization_method=config["normalization_method"],
            threshold_method=str(config["threshold_method"]),
            threshold_value=float(config["threshold_value"]),
            evaluation_mode=str(config["evaluation_mode"]),
            window_size=window_size,
            window_stride=window_stride,
            prepared_dataset_dir=prepared_dataset_dir_value,
        )
        for run_config in selected_runs
    }


def _completed_dataset_names(
    results_frame: pd.DataFrame,
    selected_runs: list[dict[str, Any]],
    prepared_dataset_dir: Path,
    selected_dataset_paths: list[Path],
    config: dict[str, Any],
) -> set[str]:
    if results_frame.empty or not selected_runs or not selected_dataset_paths:
        return set()
    successful = _successful_results_frame(results_frame)
    if successful.empty:
        return set()
    completed_run_keys = set(_result_run_key_series(successful))
    completed_datasets: set[str] = set()
    for prepared_dataset_path in selected_dataset_paths:
        expected_run_keys = _dataset_completion_run_keys(
            prepared_dataset_path,
            prepared_dataset_dir,
            config,
            selected_runs,
        )
        if expected_run_keys and expected_run_keys.issubset(completed_run_keys):
            completed_datasets.add(prepared_dataset_path.stem)
    return completed_datasets


def _merge_benchmark_results(existing_results: pd.DataFrame, new_results: pd.DataFrame) -> pd.DataFrame:
    if existing_results.empty and new_results.empty:
        return pd.DataFrame()
    if existing_results.empty:
        merged = new_results.copy()
    elif new_results.empty:
        merged = existing_results.copy()
    else:
        merged = pd.concat([existing_results, new_results], ignore_index=True, sort=False)

    merged["__run_key"] = _result_run_key_series(merged)
    merged = merged.drop_duplicates("__run_key", keep="last").drop(columns="__run_key")

    sort_columns = [
        column
        for column in ("dataset_sequence", "dataset_name", "algorithm_display", "variant_index")
        if column in merged.columns
    ]
    if sort_columns:
        merged = merged.sort_values(sort_columns).reset_index(drop=True)
    else:
        merged = merged.reset_index(drop=True)
    return merged


def _write_benchmark_checkpoint(
    results_frame: pd.DataFrame,
    session_id: str | None = None,
    *,
    write_global: bool = True,
) -> None:
    _write_table_artifact(
        results_frame,
        "benchmark_results.csv",
        session_id,
        write_global=write_global,
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


def _format_inline_value(value: Any) -> str:
    if value is None:
        return "auto"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _format_params_inline(params: dict[str, Any]) -> str:
    return ", ".join(
        f"{key}={_format_inline_value(value)}" for key, value in params.items()
    )


def _algorithm_implementation_path(algorithm_key: str) -> str:
    module_name, _function_name = ALGORITHM_REGISTRY[algorithm_key]
    return f"algorithms/{module_name}.py"


def _algorithm_auto_variant_rows(algorithm_key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for preset_name, preset in PAPER_PRESET_DEFINITIONS.items():
        for variant in preset["variants"].get(algorithm_key, []):
            params = {
                key: value
                for key, value in variant.items()
                if key not in VARIANT_METADATA_KEYS
            }
            rows.append(
                {
                    "preset_name": preset_name,
                    "variant_name": variant["variant_name"],
                    "focus": variant.get("focus", ""),
                    "params": params,
                }
            )
    return rows


def _shared_pipeline_html() -> str:
    steps = "".join(
        f"<li style='margin:0 0 6px 0;'>{html.escape(step)}</li>"
        for step in SHARED_PIPELINE_STEPS
    )
    controls = "".join(
        (
            f"<li style='margin:0 0 6px 0;'><b>{html.escape(item['label'])}</b>: "
            f"{html.escape(item['effect'])}</li>"
        )
        for item in GENERAL_CONTROL_REFERENCE
    )
    return (
        "<div>"
        "<p style='margin:0 0 8px 0;'><b>Shared scoring pipeline</b></p>"
        f"<ol style='margin:0 0 10px 18px; padding:0;'>{steps}</ol>"
        "<p style='margin:0 0 8px 0;'><b>General controls in the screenshot</b></p>"
        f"<ul style='margin:0 0 0 18px; padding:0;'>{controls}</ul>"
        "</div>"
    )


def _algorithm_reference_html(algorithm_key: str) -> str:
    reference = ALGORITHM_REFERENCE[algorithm_key]
    process = "".join(
        f"<li style='margin:0 0 6px 0;'>{html.escape(step)}</li>"
        for step in reference["process_steps"]
    )
    controls = "".join(
        (
            f"<li style='margin:0 0 6px 0;'><b>{html.escape(item['label'])}</b> "
            f"(<code>{html.escape(item['param'])}</code>): {html.escape(item['effect'])}</li>"
        )
        for item in reference["controls"]
    )
    variants = _algorithm_auto_variant_rows(algorithm_key)
    variant_items = "".join(
        (
            f"<li style='margin:0 0 6px 0;'><b>{html.escape(row['preset_name'])}</b> -> "
            f"<b>{html.escape(row['variant_name'])}</b>: "
            f"{html.escape(row['focus'])} "
            f"(<code>{html.escape(_format_params_inline(row['params']))}</code>)</li>"
        )
        for row in variants
    )
    auto_html = (
        "<p style='margin:10px 0 8px 0;'><b>Auto sweep variants</b></p>"
        f"<ul style='margin:0 0 0 18px; padding:0;'>{variant_items}</ul>"
        if variant_items
        else "<p style='margin:10px 0 0 0;'><b>Auto sweep variants</b>: none configured.</p>"
    )
    return (
        "<div>"
        f"<p style='margin:0 0 8px 0;'><b>Implementation</b>: "
        f"<code>{html.escape(_algorithm_implementation_path(algorithm_key))}</code></p>"
        f"<p style='margin:0 0 8px 0;'>{html.escape(reference['summary'])}</p>"
        "<p style='margin:0 0 8px 0;'><b>How the score is produced</b></p>"
        f"<ol style='margin:0 0 10px 18px; padding:0;'>{process}</ol>"
        "<p style='margin:0 0 8px 0;'><b>Visible controls in this tab</b></p>"
        f"<ul style='margin:0 0 0 18px; padding:0;'>{controls}</ul>"
        f"{auto_html}"
        "</div>"
    )


def _rich_reference_block(summary: str, html_value: str) -> widgets.Widget:
    body = widgets.HTML(value=html_value)
    accordion = widgets.Accordion(children=[body], selected_index=None)
    accordion.set_title(0, summary)
    return accordion


def _general_process_block() -> widgets.Widget:
    return _rich_reference_block(
        "Show shared pipeline and general-control impact",
        _shared_pipeline_html(),
    )


def _algorithm_process_block(algorithm_key: str) -> widgets.Widget:
    return _rich_reference_block(
        f"Show how {DISPLAY_NAME_MAP[algorithm_key]} scores anomalies and how each knob changes it",
        _algorithm_reference_html(algorithm_key),
    )


def build_algorithm_reference_overview() -> pd.DataFrame:
    rows = []
    for algorithm_key in ALGORITHM_ORDER:
        auto_variants = _algorithm_auto_variant_rows(algorithm_key)
        preset_counts = {
            preset_name: sum(1 for row in auto_variants if row["preset_name"] == preset_name)
            for preset_name in PAPER_PRESET_DEFINITIONS
        }
        rows.append(
            {
                "algorithm": DISPLAY_NAME_MAP[algorithm_key],
                "implementation": _algorithm_implementation_path(algorithm_key),
                "summary": ALGORITHM_REFERENCE[algorithm_key]["summary"],
                "visible_controls": ", ".join(
                    item["label"] for item in ALGORITHM_REFERENCE[algorithm_key]["controls"]
                ),
                "paper_high_roi_variants": preset_counts.get("paper_high_roi", 0),
                "paper_full_suite_variants": preset_counts.get("paper_full_suite", 0),
            }
        )
    return pd.DataFrame(rows)


def build_high_roi_algorithm_notes_markdown() -> str:
    lines = [
        "# High-ROI Algorithm Notes",
        "",
        "This file is generated from `notebook_support.py` and the detector implementations in `python/simple_anomaly_detection/algorithms/`.",
        "",
        "## Where the algorithm logic is stated",
        "",
        "- Shared run orchestration, thresholding, and benchmark aggregation live in `python/simple_anomaly_detection/notebook_support.py`.",
        "- The detector scoring logic itself lives in the per-algorithm modules listed below.",
        "- The notebook UI now mirrors the same explanation in each algorithm tab under the new process/knob accordion.",
        "",
        "## Shared scoring pipeline",
        "",
    ]
    lines.extend(f"1. {step}" for step in SHARED_PIPELINE_STEPS)
    lines.extend(
        [
            "",
            "## General controls visible in the notebook",
            "",
        ]
    )
    for item in GENERAL_CONTROL_REFERENCE:
        lines.append(f"- `{item['label']}`: {item['effect']}")
    lines.extend(
        [
            "",
            "## Automatic sweep mode",
            "",
            "- `manual`: use the visible subtabs exactly as edited.",
            "- `paper_high_roi`: automatically benchmark the curated high-return variants from `PAPER_PRESET_DEFINITIONS` for the enabled high-ROI algorithms.",
            "- `paper_full_suite`: automatically benchmark the broader appendix-style variants from `PAPER_PRESET_DEFINITIONS` for every enabled algorithm that has a configured sweep.",
            "- `auto_ablation`: automatically benchmark one baseline plus one-knob-at-a-time ablations so parameter claims are paired against a fixed reference.",
            "- In auto modes, the algorithm checkboxes still filter what runs, but the current subtab values are ignored at run time.",
            "",
            "## Algorithm-by-algorithm reference",
            "",
        ]
    )

    for algorithm_key in ALGORITHM_ORDER:
        reference = ALGORITHM_REFERENCE[algorithm_key]
        lines.extend(
            [
                f"### {DISPLAY_NAME_MAP[algorithm_key]}",
                "",
                f"- Implementation: `{_algorithm_implementation_path(algorithm_key)}`",
                f"- Summary: {reference['summary']}",
                "",
                "How the score is produced:",
            ]
        )
        lines.extend(
            f"{index}. {step}"
            for index, step in enumerate(reference["process_steps"], start=1)
        )
        lines.extend(
            [
                "",
                "Visible controls and exact effect:",
            ]
        )
        for item in reference["controls"]:
            lines.append(
                f"- `{item['label']}` (`{item['param']}`): {item['effect']}"
            )
        lines.extend(
            [
                "",
                "Auto sweep variants:",
            ]
        )
        auto_variants = _algorithm_auto_variant_rows(algorithm_key)
        if auto_variants:
            for row in auto_variants:
                lines.append(
                    f"- `{row['preset_name']}` -> `{row['variant_name']}`: {row['focus']} "
                    f"(`{_format_params_inline(row['params'])}`)"
                )
        else:
            lines.append("- None configured.")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_high_roi_algorithm_notes() -> dict[str, str]:
    notes_markdown = build_high_roi_algorithm_notes_markdown()
    HIGH_ROI_NOTES_SOURCE_PATH.write_text(notes_markdown, encoding="utf-8")
    HIGH_ROI_NOTES_RESULT_PATH.write_text(notes_markdown, encoding="utf-8")
    return {
        "source_path": portable_path_str(HIGH_ROI_NOTES_SOURCE_PATH),
        "results_path": portable_path_str(HIGH_ROI_NOTES_RESULT_PATH),
    }


def _build_variant_config(
    algorithm_key: str,
    variant_index: int,
    variant_name: str,
    params: dict[str, Any],
    *,
    variant_origin: str,
    variant_source: str,
    variant_focus: str | None = None,
    variant_family: str = "variant",
    ablation_parameter: str | None = None,
    ablation_label: str | None = None,
    ablation_role: str | None = None,
) -> dict[str, Any]:
    return {
        "algorithm": algorithm_key,
        "algorithm_base_display": DISPLAY_NAME_MAP[algorithm_key],
        "algorithm_superfamily": ALGORITHM_METADATA[algorithm_key]["superfamily"],
        "algorithm_category": ALGORITHM_METADATA[algorithm_key]["category"],
        "algorithm_display": f"{DISPLAY_NAME_MAP[algorithm_key]} | {variant_name}",
        "algorithm_variant": variant_name,
        "algorithm_run_id": f"{algorithm_key}__tab_{variant_index:02d}",
        "variant_index": variant_index,
        "variant_origin": variant_origin,
        "variant_source": variant_source,
        "variant_focus": variant_focus or "",
        "variant_family": variant_family,
        "ablation_parameter": ablation_parameter or "",
        "ablation_label": ablation_label or "",
        "ablation_role": ablation_role or "",
        "params": params,
    }


def _normalize_variant_params(algorithm_key: str, raw_params: dict[str, Any]) -> dict[str, Any]:
    if algorithm_key == "isolation_forest":
        return {
            "n_estimators": int(raw_params["n_estimators"]),
            "max_samples": parse_freeform_value(raw_params["max_samples"]),
            "max_features": float(raw_params["max_features"]),
            "bootstrap": bool(raw_params["bootstrap"]),
            "random_state": int(raw_params["random_state"]),
        }
    if algorithm_key == "local_outlier_factor":
        return {
            "n_neighbors": int(raw_params["n_neighbors"]),
            "algorithm": str(raw_params["algorithm"]),
            "leaf_size": int(raw_params["leaf_size"]),
            "metric": str(raw_params["metric"]),
            "p": int(raw_params["p"]),
        }
    if algorithm_key == "sand":
        return {
            "alpha": float(raw_params["alpha"]),
            "init_length": int(raw_params["init_length"]),
            "batch_size": int(raw_params["batch_size"]),
            "k": None if raw_params.get("k") is None or int(raw_params["k"]) <= 0 else int(raw_params["k"]),
            "subsequence_multiplier": int(raw_params["subsequence_multiplier"]),
            "overlap": None if raw_params.get("overlap") is None or int(raw_params["overlap"]) <= 0 else int(raw_params["overlap"]),
        }
    if algorithm_key == "matrix_profile":
        return {
            "subsequence_multiplier": int(raw_params["subsequence_multiplier"]),
        }
    if algorithm_key == "damp":
        x_lag_multiplier = raw_params.get("x_lag_multiplier")
        return {
            "start_index_multiplier": float(raw_params["start_index_multiplier"]),
            "x_lag_multiplier": None if x_lag_multiplier is None or float(x_lag_multiplier) <= 0 else float(x_lag_multiplier),
        }
    if algorithm_key == "hbos":
        return {
            "n_bins": int(raw_params["n_bins"]),
            "alpha": float(raw_params["alpha"]),
            "tol": float(raw_params["tol"]),
        }
    if algorithm_key == "ocsvm":
        return {
            "kernel": str(raw_params["kernel"]),
            "nu": float(raw_params["nu"]),
            "gamma": parse_freeform_value(raw_params["gamma"]),
            "train_fraction": float(raw_params["train_fraction"]),
        }

    n_components_value = raw_params.get("n_components")
    n_components_text = "" if n_components_value is None else str(n_components_value).strip()
    return {
        "n_components": None if n_components_text == "" else parse_freeform_value(n_components_text),
        "n_selected_components": None if raw_params.get("n_selected_components") is None or int(raw_params["n_selected_components"]) <= 0 else int(raw_params["n_selected_components"]),
        "whiten": bool(raw_params["whiten"]),
        "weighted": bool(raw_params["weighted"]),
        "standardization": bool(raw_params["standardization"]),
    }


def _manual_variant_configs_for_algorithm(controls: dict[str, Any], algorithm_key: str) -> list[dict[str, Any]]:
    variant_entries = controls["algorithm_variants"][algorithm_key]["variants"]
    algorithm_variant_configs: list[dict[str, Any]] = []
    for variant_index, variant_entry in enumerate(variant_entries, start=1):
        variant_controls = variant_entry["controls"]
        variant_name = variant_controls["variant_name"].value.strip() or _default_variant_name(variant_index)
        raw_params = {
            key: widget.value for key, widget in variant_controls.items() if key != "variant_name"
        }
        params = _normalize_variant_params(algorithm_key, raw_params)
        algorithm_variant_configs.append(
            _build_variant_config(
                algorithm_key,
                variant_index,
                variant_name,
                params,
                variant_origin="manual",
                variant_source="manual",
                variant_family="manual",
            )
        )
    return algorithm_variant_configs


def _auto_variant_configs_for_algorithm(algorithm_key: str, preset_name: str) -> list[dict[str, Any]]:
    preset = PAPER_PRESET_DEFINITIONS[preset_name]
    rows: list[dict[str, Any]] = []
    for variant_index, variant in enumerate(preset["variants"].get(algorithm_key, []), start=1):
        raw_params = {
            key: value
            for key, value in variant.items()
            if key not in VARIANT_METADATA_KEYS
        }
        rows.append(
            _build_variant_config(
                algorithm_key,
                variant_index,
                variant["variant_name"],
                _normalize_variant_params(algorithm_key, raw_params),
                variant_origin="auto",
                variant_source=preset_name,
                variant_focus=variant.get("focus"),
                variant_family=str(variant.get("variant_family", "variant")),
                ablation_parameter=variant.get("ablation_parameter"),
                ablation_label=variant.get("ablation_label"),
                ablation_role=variant.get("ablation_role"),
            )
        )
    return rows


def _resolve_effective_variants_from_controls(controls: dict[str, Any]) -> dict[str, Any]:
    variant_mode = str(controls.get("variant_mode").value) if controls.get("variant_mode") is not None else "manual"
    enabled_algorithms = [
        algorithm_key
        for algorithm_key in ALGORITHM_ORDER
        if controls[ALGORITHM_ENABLE_CONTROL[algorithm_key]].value
    ]

    algorithm_variants: dict[str, list[dict[str, Any]]] = {}
    selected_algorithms: list[str] = []
    selected_runs: list[dict[str, Any]] = []

    if variant_mode == "manual":
        for algorithm_key in enabled_algorithms:
            configs = _manual_variant_configs_for_algorithm(controls, algorithm_key)
            if configs:
                algorithm_variants[algorithm_key] = configs
                selected_algorithms.append(algorithm_key)
                selected_runs.extend(configs)
        return {
            "variant_mode": variant_mode,
            "selected_algorithms": selected_algorithms,
            "algorithm_variants": algorithm_variants,
            "selected_runs": selected_runs,
            "auto_preset_name": None,
            "auto_filtered_out": [],
        }

    preset = PAPER_PRESET_DEFINITIONS[variant_mode]
    preset_enabled = set(preset["enabled_algorithms"])
    auto_filtered_out = [
        algorithm_key
        for algorithm_key in enabled_algorithms
        if algorithm_key not in preset_enabled
    ]
    for algorithm_key in enabled_algorithms:
        if algorithm_key not in preset_enabled:
            continue
        configs = _auto_variant_configs_for_algorithm(algorithm_key, variant_mode)
        if configs:
            algorithm_variants[algorithm_key] = configs
            selected_algorithms.append(algorithm_key)
            selected_runs.extend(configs)

    return {
        "variant_mode": variant_mode,
        "selected_algorithms": selected_algorithms,
        "algorithm_variants": algorithm_variants,
        "selected_runs": selected_runs,
        "auto_preset_name": variant_mode,
        "auto_filtered_out": auto_filtered_out,
    }


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


def snapshot_control_panel_state(controls: dict[str, Any]) -> dict[str, Any]:
    general = {
        key: controls[key].value
        for key in PERSISTED_GENERAL_CONTROL_KEYS
        if key in controls and hasattr(controls[key], "value")
    }
    if "threshold_value" in controls and hasattr(controls["threshold_value"], "value"):
        general["threshold_value"] = controls["threshold_value"].value

    variants: dict[str, list[dict[str, Any]]] = {}
    if "algorithm_variants" in controls:
        for algorithm_key in ALGORITHM_ORDER:
            manager = controls["algorithm_variants"].get(algorithm_key)
            if manager is None:
                continue
            variants[algorithm_key] = [
                _snapshot_widget_values(entry["controls"])
                for entry in manager["variants"]
            ]

    return {
        "schema_version": 1,
        "saved_at": utc_now_iso(),
        "general": _json_safe(general),
        "variants": _json_safe(variants),
    }


def apply_control_panel_state(controls: dict[str, Any], payload: dict[str, Any]) -> None:
    general = dict(payload.get("general", {}))
    variants = dict(payload.get("variants", {}))

    threshold_method = general.pop("threshold_method", None)
    threshold_value = general.pop("threshold_value", None)

    for key in PERSISTED_GENERAL_CONTROL_KEYS:
        if key in ("threshold_method",):
            continue
        if key in general and key in controls and hasattr(controls[key], "value"):
            controls[key].value = general[key]

    if threshold_method is not None and "threshold_method" in controls:
        controls["threshold_method"].value = threshold_method
    if threshold_value is not None and "threshold_value" in controls:
        controls["threshold_value"].value = threshold_value

    if "algorithm_variants" in controls:
        for algorithm_key in ALGORITHM_ORDER:
            manager = controls["algorithm_variants"].get(algorithm_key)
            if manager is None:
                continue
            manager["replace_variants"](variants.get(algorithm_key) or [{}])


def build_control_state_from_tables(
    run_configuration_frame: pd.DataFrame | None,
    selected_run_parameters_frame: pd.DataFrame | None,
    *,
    run_name: str,
) -> dict[str, Any] | None:
    if run_configuration_frame is None or run_configuration_frame.empty:
        return None
    if selected_run_parameters_frame is None or selected_run_parameters_frame.empty:
        return None

    run_row = run_configuration_frame.iloc[0]

    def _value_from_frame(value: Any, fallback: Any = None) -> Any:
        try:
            if pd.isna(value):
                return fallback
        except TypeError:
            pass
        return value

    def _bool_from_frame(value: Any, fallback: bool = False) -> bool:
        normalized = _value_from_frame(value, fallback)
        if isinstance(normalized, str):
            return normalized.strip().lower() in {"1", "true", "yes", "y"}
        return bool(normalized)

    dataset_limit_value = _value_from_frame(run_row.get("dataset_limit"))
    batch_size_value = _value_from_frame(run_row.get("batch_size"))
    clip_quantile_value = _value_from_frame(run_row.get("clip_quantile"), 0.0)
    window_size_value = _value_from_frame(run_row.get("window_size"), 0)
    threshold_value = _value_from_frame(run_row.get("threshold_value"), 3.0)

    general = {
        "run_name": run_name,
        "dataset_limit": 0 if dataset_limit_value in (None, "", "all") else int(float(dataset_limit_value)),
        "batch_size": 0 if batch_size_value in (None, "", "all selected") else int(float(batch_size_value)),
        "resume_from_existing": _bool_from_frame(run_row.get("resume_from_existing"), True),
        "variant_mode": str(_value_from_frame(run_row.get("variant_mode"), "manual")),
        "normalization_method": str(_value_from_frame(run_row.get("normalization_method"), "zscore")),
        "clip_quantile": 0.0 if clip_quantile_value in (None, "") else float(clip_quantile_value),
        "overwrite_normalized": False,
        "window_size": 0 if window_size_value in (None, "") else int(float(window_size_value)),
        "window_stride": int(float(_value_from_frame(run_row.get("window_stride"), 1))),
        "threshold_method": str(_value_from_frame(run_row.get("threshold_method"), "sigma")),
        "threshold_value": max(1, int(float(threshold_value))) if str(_value_from_frame(run_row.get("threshold_method"), "sigma")) == "top_k" else float(threshold_value),
        "evaluation_mode": str(_value_from_frame(run_row.get("evaluation_mode"), "range")),
        "save_scores": False,
        **{key: False for key in ALGORITHM_CONTROL_KEYS},
    }

    variants: dict[str, list[dict[str, Any]]] = {algorithm_key: [] for algorithm_key in ALGORITHM_ORDER}
    sorted_rows = selected_run_parameters_frame.sort_values(["algorithm", "variant_index"]).reset_index(drop=True)
    for row in sorted_rows.to_dict(orient="records"):
        algorithm_key = str(row.get("algorithm"))
        if algorithm_key not in variants:
            continue
        try:
            params = json.loads(str(row.get("params_json", "{}")))
        except json.JSONDecodeError:
            params = {}
        variant_values = {"variant_name": row.get("algorithm_variant") or row.get("algorithm_display") or "Variant"}
        variant_values.update(params)
        variants[algorithm_key].append(variant_values)
        enable_key = ALGORITHM_ENABLE_CONTROL.get(algorithm_key)
        if enable_key:
            general[enable_key] = True

    return {
        "schema_version": 1,
        "saved_at": utc_now_iso(),
        "general": _json_safe(general),
        "variants": _json_safe(variants),
    }


def load_saved_run_session_payload(session_id: str) -> dict[str, Any] | None:
    manifest = _read_json_if_exists(saved_run_session_manifest_path(session_id)) or {}
    payload = _read_json_if_exists(saved_run_session_control_state_path(session_id))
    if payload is not None:
        return payload

    session_run_configuration = _read_table_if_exists(saved_run_session_table_path(session_id, "run_configuration.csv"))
    session_selected_runs = _read_table_if_exists(saved_run_session_table_path(session_id, "selected_run_parameters.csv"))
    payload = build_control_state_from_tables(
        session_run_configuration,
        session_selected_runs,
        run_name=str(manifest.get("run_name") or session_id),
    )
    if payload is not None:
        return payload

    if not _session_supports_global_fallback(session_id):
        return None

    run_configuration_frame = _read_table_if_exists(result_table_path("run_configuration.csv"))
    selected_run_parameters_frame = _read_table_if_exists(result_table_path("selected_run_parameters.csv"))
    return build_control_state_from_tables(
        run_configuration_frame,
        selected_run_parameters_frame,
        run_name=DEFAULT_RUN_NAME,
    )


def load_saved_run_session_into_controls(controls: dict[str, Any], session_id: str) -> dict[str, Any]:
    payload = load_saved_run_session_payload(session_id)
    if payload is None:
        raise FileNotFoundError(f"No saved run session found for '{session_id}'.")
    apply_control_panel_state(controls, payload)
    return payload


def _write_saved_run_control_state(session_id: str, control_state: dict[str, Any]) -> None:
    _write_json(saved_run_session_control_state_path(session_id), control_state)


def _build_run_manifest_base(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "session_id": config["session_id"],
        "run_name": config["run_name"],
        "variant_mode": config["variant_mode"],
        "normalization_method": config["normalization_method"],
        "clip_quantile": config["clip_quantile"],
        "window_size": config["window_size"],
        "window_stride": config["window_stride"],
        "threshold_method": config["threshold_method"],
        "threshold_value": config["threshold_value"],
        "evaluation_mode": config["evaluation_mode"],
        "batch_size": "all selected" if config["batch_size"] is None else config["batch_size"],
        "dataset_limit": "all" if config["dataset_limit"] is None else config["dataset_limit"],
        "selected_algorithms": list(config["selected_algorithms"]),
        "selected_algorithm_labels": [DISPLAY_NAME_MAP[key] for key in config["selected_algorithms"]],
        "selected_run_count": len(config["selected_runs"]),
        "tables_dir": portable_path_str(saved_run_session_tables_dir(config["session_id"])),
        "control_state_path": portable_path_str(saved_run_session_control_state_path(config["session_id"])),
    }


def update_saved_run_manifest(config: dict[str, Any], **updates: Any) -> dict[str, Any]:
    session_id = config["session_id"]
    manifest_path = saved_run_session_manifest_path(session_id)
    existing_manifest = _read_json_if_exists(manifest_path) or {}
    timestamp = utc_now_iso()
    manifest = {
        **existing_manifest,
        **_build_run_manifest_base(config),
        **_json_safe(updates),
        "updated_at": timestamp,
    }
    manifest.setdefault("created_at", existing_manifest.get("created_at", timestamp))
    _write_json(manifest_path, manifest)
    return manifest


def list_saved_run_sessions() -> list[dict[str, Any]]:
    ensure_results_layout()
    manifests: list[dict[str, Any]] = []
    for manifest_path in RESULT_RUN_SESSION_DIR.glob(f"*/{RUN_SESSION_MANIFEST_FILENAME}"):
        manifest = _read_json_if_exists(manifest_path)
        if manifest is not None:
            manifests.append(manifest)

    default_session = saved_run_session_id(DEFAULT_RUN_NAME)
    has_default_manifest = any(manifest.get("session_id") == default_session for manifest in manifests)
    if not has_default_manifest:
        legacy_files = [
            result_table_path("run_configuration.csv"),
            result_table_path("selected_run_parameters.csv"),
            result_table_path("benchmark_results.csv"),
        ]
        existing_legacy_files = [path for path in legacy_files if path.exists()]
        if existing_legacy_files:
            latest_path = max(existing_legacy_files, key=lambda path: path.stat().st_mtime)
            manifests.append(
                {
                    "schema_version": 1,
                    "session_id": default_session,
                    "run_name": DEFAULT_RUN_NAME,
                    "status": "legacy_global",
                    "updated_at": datetime.fromtimestamp(latest_path.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
                    "tables_dir": portable_path_str(RESULT_TABLES_DIR),
                    "control_state_path": portable_path_str(result_table_path("selected_run_parameters.csv")),
                }
            )

    manifests.sort(
        key=lambda manifest: str(manifest.get("updated_at", "")),
        reverse=True,
    )
    return manifests


def _session_match_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def resolve_saved_run_session(run_name_or_session_id: str | None = None) -> dict[str, Any]:
    manifests = list_saved_run_sessions()
    if not manifests:
        raise FileNotFoundError("No saved run sessions were found under results/run_sessions.")

    if run_name_or_session_id is None or not str(run_name_or_session_id).strip():
        running = [manifest for manifest in manifests if str(manifest.get("status", "")).lower() == "running"]
        return running[0] if running else manifests[0]

    query = str(run_name_or_session_id).strip()
    query_match = _session_match_key(query)
    query_slug = saved_run_session_id(query)

    exact_session = [
        manifest
        for manifest in manifests
        if str(manifest.get("session_id", "")).strip().lower() in {query.lower(), query_slug.lower()}
    ]
    if exact_session:
        return exact_session[0]

    exact_name = [
        manifest
        for manifest in manifests
        if _session_match_key(manifest.get("run_name", "")) == query_match
        or _session_match_key(manifest.get("session_id", "")) == query_match
    ]
    if exact_name:
        return exact_name[0]

    partial_name = [
        manifest
        for manifest in manifests
        if query_match
        and (
            query_match in _session_match_key(manifest.get("run_name", ""))
            or query_match in _session_match_key(manifest.get("session_id", ""))
        )
    ]
    if partial_name:
        return partial_name[0]

    fuzzy_candidates: dict[str, dict[str, Any]] = {}
    for manifest in manifests:
        for candidate in (manifest.get("run_name", ""), manifest.get("session_id", "")):
            key = _session_match_key(candidate)
            if key:
                fuzzy_candidates.setdefault(key, manifest)
    close_matches = get_close_matches(query_match, list(fuzzy_candidates), n=1, cutoff=0.65)
    if close_matches:
        return fuzzy_candidates[close_matches[0]]

    available = ", ".join(
        f"{manifest.get('run_name') or manifest.get('session_id')} [{manifest.get('session_id')}]"
        for manifest in manifests[:8]
    )
    raise FileNotFoundError(
        f"No saved run matched '{query}'. Available runs: {available}"
    )


def _saved_frame_value(row: pd.Series | None, key: str, fallback: Any = None) -> Any:
    if row is None or key not in row.index:
        return fallback
    value = row[key]
    try:
        if pd.isna(value):
            return fallback
    except TypeError:
        pass
    return value


def _saved_int_or_none(value: Any) -> int | None:
    if value in (None, "", "all", "all selected"):
        return None
    return int(float(value))


def _saved_float_or_none(value: Any) -> float | None:
    if value in (None, "", "nan"):
        return None
    return float(value)


def _saved_bool(value: Any, fallback: bool = False) -> bool:
    if value is None:
        return fallback
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _resolve_project_path(path_value: Any) -> Path | None:
    if path_value is None:
        return None
    candidate_text = str(path_value).strip()
    if not candidate_text:
        return None
    candidate = Path(candidate_text)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _read_live_table_if_exists(
    path: Path,
    retries: int = 6,
    delay_seconds: float = 0.25,
) -> pd.DataFrame | None:
    if not path.exists():
        return None

    last_parser_error: Exception | None = None
    for attempt in range(max(retries, 1)):
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            if attempt == retries - 1:
                return pd.DataFrame()
        except pd.errors.ParserError as exc:
            last_parser_error = exc
            if attempt == retries - 1:
                break
        time.sleep(delay_seconds)

    if last_parser_error is not None:
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    raise last_parser_error or RuntimeError(f"Could not read {path}")


def _build_saved_run_variants(
    selected_run_parameters_frame: pd.DataFrame,
) -> tuple[list[str], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    selected_runs: list[dict[str, Any]] = []
    algorithm_variants: dict[str, list[dict[str, Any]]] = {algorithm_key: [] for algorithm_key in ALGORITHM_ORDER}

    if selected_run_parameters_frame.empty:
        return [], {}, []

    sortable_frame = selected_run_parameters_frame.copy()
    sortable_frame["__algorithm_order__"] = sortable_frame["algorithm"].map(
        {algorithm_key: index for index, algorithm_key in enumerate(ALGORITHM_ORDER)}
    )
    sortable_frame["__algorithm_order__"] = sortable_frame["__algorithm_order__"].fillna(len(ALGORITHM_ORDER))

    for row in sortable_frame.sort_values(["__algorithm_order__", "variant_index"]).to_dict(orient="records"):
        algorithm_key = str(row.get("algorithm") or "").strip()
        if algorithm_key not in ALGORITHM_ORDER:
            continue
        params_json = row.get("params_json", "{}")
        try:
            params = json.loads(str(params_json))
        except json.JSONDecodeError:
            params = {}

        variant_index_value = row.get("variant_index")
        try:
            variant_index = int(float(variant_index_value))
        except (TypeError, ValueError):
            variant_index = len(algorithm_variants[algorithm_key]) + 1

        variant_name = str(
            row.get("algorithm_variant")
            or row.get("algorithm_display")
            or _default_variant_name(variant_index)
        )
        variant_config = {
            "algorithm": algorithm_key,
            "algorithm_base_display": str(
                row.get("algorithm_base_display")
                or DISPLAY_NAME_MAP[algorithm_key]
            ),
            "algorithm_superfamily": str(
                row.get("algorithm_superfamily")
                or ALGORITHM_METADATA[algorithm_key]["superfamily"]
            ),
            "algorithm_category": str(
                row.get("algorithm_category")
                or ALGORITHM_METADATA[algorithm_key]["category"]
            ),
            "algorithm_display": str(
                row.get("algorithm_display")
                or f"{DISPLAY_NAME_MAP[algorithm_key]} | {variant_name}"
            ),
            "algorithm_variant": variant_name,
            "algorithm_run_id": str(
                row.get("algorithm_run_id")
                or f"{algorithm_key}__tab_{variant_index:02d}"
            ),
            "variant_index": variant_index,
            "variant_origin": str(row.get("variant_origin") or ""),
            "variant_source": str(row.get("variant_source") or ""),
            "variant_focus": str(row.get("variant_focus") or ""),
            "variant_family": str(row.get("variant_family") or "variant"),
            "ablation_parameter": str(row.get("ablation_parameter") or ""),
            "ablation_label": str(row.get("ablation_label") or ""),
            "ablation_role": str(row.get("ablation_role") or ""),
            "params": params,
        }
        algorithm_variants[algorithm_key].append(variant_config)
        selected_runs.append(variant_config)

    selected_algorithms = [
        algorithm_key
        for algorithm_key in ALGORITHM_ORDER
        if algorithm_variants.get(algorithm_key)
    ]
    algorithm_variants = {
        algorithm_key: algorithm_variants[algorithm_key]
        for algorithm_key in selected_algorithms
    }
    return selected_algorithms, algorithm_variants, selected_runs


def load_saved_run_config(run_name_or_session_id: str | None = None) -> dict[str, Any]:
    manifest = resolve_saved_run_session(run_name_or_session_id)
    session_id = str(manifest["session_id"])
    run_configuration_frame = _read_resume_table_if_exists("run_configuration.csv", session_id)
    selected_run_parameters_frame = _read_resume_table_if_exists("selected_run_parameters.csv", session_id)
    if selected_run_parameters_frame is None or selected_run_parameters_frame.empty:
        raise FileNotFoundError(
            f"No selected_run_parameters.csv was found for session '{session_id}'."
        )

    control_state = load_saved_run_session_payload(session_id)
    run_row = None if run_configuration_frame is None or run_configuration_frame.empty else run_configuration_frame.iloc[0]
    selected_algorithms, algorithm_variants, selected_runs = _build_saved_run_variants(selected_run_parameters_frame)

    variant_mode = str(
        _saved_frame_value(run_row, "variant_mode", manifest.get("variant_mode", "manual"))
    )
    threshold_method = str(
        _saved_frame_value(run_row, "threshold_method", manifest.get("threshold_method", "sigma"))
    )
    threshold_value_raw = _saved_frame_value(
        run_row,
        "threshold_value",
        manifest.get("threshold_value", 3.0),
    )
    threshold_value = (
        max(1, int(float(threshold_value_raw)))
        if threshold_method == "top_k"
        else float(threshold_value_raw)
    )

    return {
        "run_name": str(
            _saved_frame_value(run_row, "run_name", manifest.get("run_name", session_id))
        ),
        "session_id": session_id,
        "dataset_limit": _saved_int_or_none(
            _saved_frame_value(run_row, "dataset_limit", manifest.get("dataset_limit"))
        ),
        "batch_size": _saved_int_or_none(
            _saved_frame_value(run_row, "batch_size", manifest.get("batch_size"))
        ),
        "resume_from_existing": _saved_bool(
            _saved_frame_value(run_row, "resume_from_existing", True),
            True,
        ),
        "variant_mode": variant_mode,
        "normalization_method": str(
            _saved_frame_value(
                run_row,
                "normalization_method",
                manifest.get("normalization_method", "zscore"),
            )
        ),
        "clip_quantile": _saved_float_or_none(
            _saved_frame_value(run_row, "clip_quantile", manifest.get("clip_quantile"))
        ),
        "overwrite_normalized_datasets": False,
        "window_size": _saved_int_or_none(
            _saved_frame_value(run_row, "window_size", manifest.get("window_size"))
        ),
        "window_stride": max(
            1,
            int(
                float(
                    _saved_frame_value(
                        run_row,
                        "window_stride",
                        manifest.get("window_stride", 1),
                    )
                )
            ),
        ),
        "window_override": _saved_int_or_none(
            _saved_frame_value(run_row, "window_size", manifest.get("window_size"))
        ),
        "threshold_method": threshold_method,
        "threshold_value": threshold_value,
        "threshold_std_multiplier": threshold_value if threshold_method == "sigma" else None,
        "evaluation_mode": str(
            _saved_frame_value(
                run_row,
                "evaluation_mode",
                manifest.get("evaluation_mode", "range"),
            )
        ),
        "save_per_dataset_scores": False,
        "selected_algorithms": selected_algorithms,
        "selected_runs": selected_runs,
        "algorithm_variants": algorithm_variants,
        "auto_preset_name": variant_mode if variant_mode in PAPER_PRESET_DEFINITIONS else None,
        "auto_filtered_out": [],
        "control_state": control_state,
    }


def load_saved_run_context(
    run_name_or_session_id: str | None = None,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = load_saved_run_config(run_name_or_session_id) if config is None else config
    session_id = str(config["session_id"])
    run_configuration_frame = _read_resume_table_if_exists("run_configuration.csv", session_id)
    preparation_frame = _read_resume_table_if_exists("dataset_preparation_summary.csv", session_id)

    prepared_dataset_dir = None
    if preparation_frame is not None and not preparation_frame.empty:
        prepared_dataset_dir = _resolve_project_path(
            _saved_frame_value(preparation_frame.iloc[0], "normalized_dataset_dir")
        )
    if prepared_dataset_dir is None and run_configuration_frame is not None and not run_configuration_frame.empty:
        prepared_dataset_dir = _resolve_project_path(
            _saved_frame_value(run_configuration_frame.iloc[0], "prepared_dataset_dir")
        )
    if prepared_dataset_dir is None or not prepared_dataset_dir.exists():
        prepared_dataset_dir, _ = ensure_normalized_datasets(
            config["normalization_method"],
            config["clip_quantile"],
            overwrite=False,
        )
    else:
        ensure_raw_datasets_available()

    return {
        "prepared_dataset_dir": prepared_dataset_dir,
        "session_manifest": resolve_saved_run_session(session_id),
    }


def build_saved_run_benchmark(
    run_name_or_session_id: str | None = None,
    *,
    config: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    persist_tables: bool = False,
    write_global: bool = False,
) -> dict[str, Any]:
    config = load_saved_run_config(run_name_or_session_id) if config is None else config
    context = load_saved_run_context(config=config) if context is None else context
    manifest = resolve_saved_run_session(config["session_id"])
    session_id = str(config["session_id"])

    results = _read_live_table_if_exists(
        saved_run_session_table_path(session_id, "benchmark_results.csv")
    )
    if (results is None or results.empty) and _session_supports_global_fallback(session_id):
        results = _read_live_table_if_exists(result_table_path("benchmark_results.csv"))
    if results is None or results.empty:
        raise ValueError(
            f"No benchmark checkpoint rows are available yet for session '{session_id}'."
        )

    dataset_catalog = build_dataset_catalog(results)
    algorithm_summary = summarize_algorithms(results)
    family_summary = summarize_families(results)
    overall_regime_summary = build_overall_regime_summary(results)
    best_by_evaluation = build_best_algorithm_table(results, "evaluation_f1")
    best_by_f1 = build_best_algorithm_table(results, "f1")
    best_by_auc = build_best_algorithm_table(results, "roc_auc")
    if "error" in results.columns:
        errors = results.loc[results["error"].fillna("").astype(str) != ""].copy()
    else:
        errors = pd.DataFrame()

    completed_dataset_count = int(dataset_catalog["dataset_name"].nunique())
    selected_dataset_count = int(manifest.get("selected_dataset_count") or completed_dataset_count)
    pending_dataset_count = max(selected_dataset_count - completed_dataset_count, 0)
    selected_run_count = int(manifest.get("selected_run_count") or len(config["selected_runs"]))
    overview = pd.DataFrame(
        [
            {
                "dataset_count": completed_dataset_count,
                "selected_dataset_count": selected_dataset_count,
                "completed_dataset_count": completed_dataset_count,
                "pending_dataset_count": pending_dataset_count,
                "algorithm_count": len(config["selected_algorithms"]),
                "configuration_count": len(config["selected_runs"]),
                "selected_run_count": selected_run_count,
                "run_count": len(results),
                "batch_dataset_count": manifest.get("batch_dataset_count", completed_dataset_count),
                "run_name": config["run_name"],
                "session_id": session_id,
                "session_status": manifest.get("status", "saved"),
                "executed_run_count": manifest.get("executed_run_count", len(results)),
                "skipped_existing_run_count": manifest.get("skipped_existing_run_count", 0),
                "resume_from_existing": bool(config.get("resume_from_existing")),
                "batch_size": "all selected" if config.get("batch_size") is None else config["batch_size"],
                "median_series_length": dataset_catalog["series_length"].median(),
                "median_anomaly_ratio": dataset_catalog["anomaly_ratio"].median(),
                "median_window_size": dataset_catalog["window_size"].median(),
                "window_stride": config["window_stride"],
                "threshold_method": config["threshold_method"],
                "threshold_value": config["threshold_value"],
                "evaluation_mode": config["evaluation_mode"],
                "error_count": len(errors),
                "normalization_method": config["normalization_method"],
                "progress_fraction": (
                    completed_dataset_count / selected_dataset_count
                    if selected_dataset_count > 0
                    else float("nan")
                ),
            }
        ]
    )

    if persist_tables:
        _write_benchmark_checkpoint(results, session_id, write_global=write_global)
        _write_table_artifact(dataset_catalog, "dataset_catalog.csv", session_id, write_global=write_global)
        _write_table_artifact(algorithm_summary, "algorithm_summary.csv", session_id, write_global=write_global)
        _write_table_artifact(family_summary, "family_summary.csv", session_id, write_global=write_global)
        _write_table_artifact(
            overall_regime_summary,
            "overall_regime_summary.csv",
            session_id,
            write_global=write_global,
        )
        _write_table_artifact(
            best_by_evaluation,
            "best_algorithm_by_dataset_evaluation.csv",
            session_id,
            write_global=write_global,
        )
        _write_table_artifact(
            best_by_f1,
            "best_algorithm_by_dataset_f1.csv",
            session_id,
            write_global=write_global,
        )
        _write_table_artifact(
            best_by_auc,
            "best_algorithm_by_dataset_auc.csv",
            session_id,
            write_global=write_global,
        )
        _write_table_artifact(errors, "error_report.csv", session_id, write_global=write_global)
        _write_table_artifact(overview, "snapshot_overview.csv", session_id, write_global=write_global)

        for algorithm_key in config["selected_algorithms"]:
            algorithm_results = results.loc[results["algorithm"] == algorithm_key].copy()
            if write_global:
                algorithm_results.to_csv(result_per_algorithm_table_path(algorithm_key), index=False)
            session_algorithm_table = saved_run_session_table_path(
                session_id,
                f"per_algorithm/{algorithm_key}_results.csv",
            )
            session_algorithm_table.parent.mkdir(parents=True, exist_ok=True)
            algorithm_results.to_csv(session_algorithm_table, index=False)

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
        "batch_results": pd.DataFrame(),
        "executed_run_count": int(manifest.get("executed_run_count", len(results))),
        "skipped_existing_run_count": int(manifest.get("skipped_existing_run_count", 0)),
        "selected_dataset_count": selected_dataset_count,
        "completed_dataset_count": completed_dataset_count,
        "pending_dataset_count": pending_dataset_count,
        "is_partial": pending_dataset_count > 0 or str(manifest.get("status", "")).lower() == "running",
        "session_manifest": manifest,
    }


def load_saved_run_notebook_state(
    run_name_or_session_id: str | None = None,
    *,
    persist_tables: bool = False,
    write_global: bool = False,
) -> dict[str, Any]:
    config = load_saved_run_config(run_name_or_session_id)
    context = load_saved_run_context(config=config)
    benchmark = build_saved_run_benchmark(
        config=config,
        context=context,
        persist_tables=persist_tables,
        write_global=write_global,
    )
    ns_module = sys.modules.get(__name__) or types.SimpleNamespace(**globals())
    return {
        "ns": ns_module,
        "config": config,
        "context": context,
        "benchmark": benchmark,
        "snapshot_extracted_at": utc_now_iso(),
    }


def _notebook_state_run_reference(notebook_state: dict[str, Any]) -> str | None:
    config = notebook_state.get("config")
    if isinstance(config, dict):
        session_id = str(config.get("session_id") or "").strip()
        if session_id:
            return session_id
        run_name = str(config.get("run_name") or "").strip()
        if run_name:
            return run_name

    controls = notebook_state.get("controls")
    if isinstance(controls, dict):
        run_name_control = controls.get("run_name")
        if run_name_control is not None and hasattr(run_name_control, "value"):
            run_name = str(run_name_control.value or "").strip()
            if run_name:
                return run_name

    return None


def hydrate_notebook_state_from_saved_run(
    notebook_state: dict[str, Any],
    run_name_or_session_id: str | None = None,
    *,
    persist_tables: bool = True,
    write_global: bool = False,
) -> dict[str, Any]:
    resolved_run = run_name_or_session_id or _notebook_state_run_reference(notebook_state)
    snapshot = load_saved_run_notebook_state(
        resolved_run,
        persist_tables=persist_tables,
        write_global=write_global,
    )
    notebook_state["config"] = snapshot["config"]
    notebook_state["context"] = snapshot["context"]
    notebook_state["benchmark"] = snapshot["benchmark"]
    notebook_state["snapshot_extracted_at"] = snapshot["snapshot_extracted_at"]
    if "ns" not in notebook_state:
        notebook_state["ns"] = snapshot["ns"]
    return notebook_state


def ensure_notebook_state_benchmark(
    notebook_state: dict[str, Any],
    *,
    refresh_from_saved_run: bool = False,
    persist_tables: bool = True,
    write_global: bool = False,
) -> dict[str, Any]:
    has_config = isinstance(notebook_state.get("config"), dict)
    has_context = isinstance(notebook_state.get("context"), dict)
    benchmark = notebook_state.get("benchmark")
    has_benchmark = isinstance(benchmark, dict) and benchmark.get("results") is not None

    if has_config and has_context and has_benchmark and not refresh_from_saved_run:
        return notebook_state

    try:
        return hydrate_notebook_state_from_saved_run(
            notebook_state,
            persist_tables=persist_tables,
            write_global=write_global,
        )
    except Exception:
        if has_config and has_context and has_benchmark:
            return notebook_state
        raise


def bootstrap_notebook_state(
    saved_run_name_or_session_id: str | None = None,
    *,
    hydrate_benchmark: bool = False,
    persist_tables: bool = True,
    write_global: bool = False,
) -> dict[str, Any]:
    dataset_name_source = RAW_DATASET_DIR if any(
        RAW_DATASET_DIR.glob("*.txt")
    ) else LEGACY_VIRGIN_DIR
    dataset_names = [path.stem for path in sorted(dataset_name_source.glob("*.txt"))]
    panel_bundle = build_control_panel(dataset_names)
    ns_module = sys.modules.get(__name__) or types.SimpleNamespace(**globals())

    state = {
        "project_root": PROJECT_ROOT,
        "ns": ns_module,
        "controls": panel_bundle["controls"],
        "panel_bundle": panel_bundle,
        "config": None,
        "context": None,
        "benchmark": None,
    }

    try:
        manifest = resolve_saved_run_session(saved_run_name_or_session_id)
        load_saved_run_session_into_controls(
            state["controls"],
            str(manifest["session_id"]),
        )
    except FileNotFoundError:
        if hydrate_benchmark:
            raise

    if hydrate_benchmark:
        ensure_notebook_state_benchmark(
            state,
            refresh_from_saved_run=True,
            persist_tables=persist_tables,
            write_global=write_global,
        )

    return state


def _recover_saved_run_progress_counts(
    session_id: str,
) -> tuple[int, int]:
    manifest = _read_json_if_exists(saved_run_session_manifest_path(session_id)) or {}
    executed_run_count = int(manifest.get("executed_run_count") or 0)
    skipped_existing_run_count = int(manifest.get("skipped_existing_run_count") or 0)

    snapshot_overview = _read_table_if_exists(
        saved_run_session_table_path(session_id, "snapshot_overview.csv")
    )
    if snapshot_overview is not None and not snapshot_overview.empty:
        row = snapshot_overview.iloc[0]
        executed_run_count = max(
            executed_run_count,
            int(_saved_frame_value(row, "executed_run_count", 0) or 0),
        )
        skipped_existing_run_count = max(
            skipped_existing_run_count,
            int(_saved_frame_value(row, "skipped_existing_run_count", 0) or 0),
        )

    return executed_run_count, skipped_existing_run_count


def _count_completed_run_prefix(
    benchmark_dataset_paths: list[Path],
    selected_runs: list[dict[str, Any]],
    completed_run_keys: set[str],
    config: dict[str, Any],
    prepared_dataset_dir_value: str,
) -> int:
    if not benchmark_dataset_paths or not selected_runs or not completed_run_keys:
        return 0

    prefix_count = 0
    for prepared_dataset_path in benchmark_dataset_paths:
        dataset = load_prepared_dataset(prepared_dataset_path)
        values = dataset["values"]
        window_size = (
            config["window_size"]
            if config["window_size"] is not None
            else estimate_window_size(values)
        )
        window_stride = max(1, int(config["window_stride"]))

        for run_config in selected_runs:
            current_run_key = _result_run_key(
                dataset_name=dataset["dataset_name"],
                algorithm_run_id=run_config["algorithm_run_id"],
                normalization_method=config["normalization_method"],
                threshold_method=str(config["threshold_method"]),
                threshold_value=float(config["threshold_value"]),
                evaluation_mode=str(config["evaluation_mode"]),
                window_size=window_size,
                window_stride=window_stride,
                prepared_dataset_dir=prepared_dataset_dir_value,
            )
            if current_run_key not in completed_run_keys:
                return prefix_count
            prefix_count += 1

    return prefix_count


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
            _algorithm_process_block("isolation_forest"),
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
            _algorithm_process_block("local_outlier_factor"),
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
            _algorithm_process_block("sand"),
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
            _algorithm_process_block("matrix_profile"),
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
            _algorithm_process_block("hbos"),
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
            _algorithm_process_block("damp"),
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
            _algorithm_process_block("ocsvm"),
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
            _algorithm_process_block("pca"),
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
    saved_run_status = widgets.HTML()
    saved_run_selector = widgets.Dropdown(
        options=[("No saved runs yet", "")],
        value="",
        description="Saved run",
        layout=widgets.Layout(width="520px"),
    )
    refresh_saved_runs_button = widgets.Button(
        description="Refresh saved runs",
        tooltip="Reload the saved-run list from results/run_sessions.",
        layout=widgets.Layout(width="160px"),
    )
    save_run_button = widgets.Button(
        description="Save current run",
        tooltip="Write the current control-panel state to this run name.",
        layout=widgets.Layout(width="150px"),
    )
    load_run_button = widgets.Button(
        description="Load selected run",
        tooltip="Restore the selected saved run into the control panel.",
        layout=widgets.Layout(width="150px"),
    )
    load_latest_button = widgets.Button(
        description="Load latest resume",
        tooltip="Restore the most recently updated saved run, preferring incomplete sessions.",
        layout=widgets.Layout(width="150px"),
    )

    def render_preview(*_args: Any) -> None:
        if "algorithm_variants" not in controls or "variant_mode" not in controls:
            return

        effective = _resolve_effective_variants_from_controls(controls)
        selected_configuration_count = len(effective["selected_runs"])
        selected = [
            f"{DISPLAY_NAME_MAP[algorithm_key]} x{len(rows)}"
            for algorithm_key, rows in effective["algorithm_variants"].items()
            if rows
        ]
        variant_summary = [
            f"{DISPLAY_NAME_MAP[algorithm_key]}: {', '.join(row['algorithm_variant'] for row in rows)}"
            for algorithm_key, rows in effective["algorithm_variants"].items()
            if rows
        ]
        auto_filtered_out = ", ".join(
            DISPLAY_NAME_MAP[key] for key in effective["auto_filtered_out"]
        )

        preview_frame = pd.DataFrame(
            [
                {
                    "run_name": controls["run_name"].value.strip() or DEFAULT_RUN_NAME,
                    "argument_mode": VARIANT_MODE_LABELS.get(effective["variant_mode"], effective["variant_mode"]),
                    "dataset scope": "all available" if controls["dataset_limit"].value <= 0 else controls["dataset_limit"].value,
                    "batch_size": "all selected" if controls["batch_size"].value <= 0 else controls["batch_size"].value,
                    "resume_batches": bool(controls["resume_from_existing"].value),
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
                    "auto_filtered_out": auto_filtered_out or None,
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

    def saved_run_option_label(manifest: dict[str, Any]) -> str:
        run_name = str(manifest.get("run_name") or manifest.get("session_id") or "saved_run")
        status = str(manifest.get("status") or "saved")
        pending = manifest.get("pending_dataset_count")
        selected_count = manifest.get("selected_dataset_count")
        updated_at = str(manifest.get("updated_at") or "")[:19].replace("T", " ")
        pending_text = ""
        if pending is not None:
            pending_text = f" | pending {pending}"
            if selected_count is not None:
                pending_text += f"/{selected_count}"
        return f"{run_name} | {status}{pending_text} | {updated_at}".strip(" |")

    def render_saved_run_summary(selected_session_id: str | None = None) -> None:
        manifests = list_saved_run_sessions()
        if not manifests:
            saved_run_status.value = (
                "<div style='line-height:1.45;'><b>Saved runs</b>: none yet. "
                "Use <b>Run name</b> plus <b>Save current run</b> to persist a resumable benchmark setup.</div>"
            )
            return

        chosen_manifest = next(
            (manifest for manifest in manifests if manifest.get("session_id") == selected_session_id),
            manifests[0],
        )
        selected_algorithms = ", ".join(chosen_manifest.get("selected_algorithm_labels", [])) or ", ".join(
            chosen_manifest.get("selected_algorithms", [])
        )
        pending_count = chosen_manifest.get("pending_dataset_count")
        completed_count = chosen_manifest.get("completed_dataset_count")
        error_count = chosen_manifest.get("error_count")
        saved_run_status.value = (
            "<div style='line-height:1.45; white-space:normal;'>"
            f"<b>Saved run</b>: {html.escape(str(chosen_manifest.get('run_name') or chosen_manifest.get('session_id')))}"
            f" | <b>Status</b>: {html.escape(str(chosen_manifest.get('status', 'saved')))}"
            f" | <b>Updated</b>: {html.escape(str(chosen_manifest.get('updated_at', '')))}<br>"
            f"<b>Algorithms</b>: {html.escape(selected_algorithms or 'not recorded')}<br>"
            f"<b>Completed datasets</b>: {html.escape(str(completed_count if completed_count is not None else 'n/a'))}"
            f" | <b>Pending</b>: {html.escape(str(pending_count if pending_count is not None else 'n/a'))}"
            f" | <b>Errors</b>: {html.escape(str(error_count if error_count is not None else 0))}"
            "</div>"
        )

    def refresh_saved_run_options(*_args: Any) -> None:
        manifests = list_saved_run_sessions()
        options = [(saved_run_option_label(manifest), str(manifest["session_id"])) for manifest in manifests]
        if not options:
            options = [("No saved runs yet", "")]
        current_value = saved_run_selector.value
        current_run_id = saved_run_session_id(controls["run_name"].value.strip() or DEFAULT_RUN_NAME)
        saved_run_selector.options = options
        available_ids = {value for _label, value in options}
        if current_value in available_ids:
            saved_run_selector.value = current_value
        elif current_run_id in available_ids:
            saved_run_selector.value = current_run_id
        else:
            saved_run_selector.value = options[0][1]
        render_saved_run_summary(saved_run_selector.value)

    def save_current_run(_button: widgets.Button) -> None:
        try:
            config = get_run_config(controls)
            _validate_resume_compatibility(
                build_selected_run_parameter_frame(config),
                session_id=config["session_id"],
            )
            _write_saved_run_control_state(config["session_id"], config["control_state"])
            update_saved_run_manifest(config, status="saved")
            refresh_saved_run_options()
            saved_run_selector.value = config["session_id"]
            render_saved_run_summary(config["session_id"])
        except Exception as error:
            saved_run_status.value = (
                "<div style='line-height:1.45; color:#991b1b;'><b>Save failed</b>: "
                f"{html.escape(str(error))}</div>"
            )

    def load_selected_run(_button: widgets.Button) -> None:
        session_id = str(saved_run_selector.value or "").strip()
        if not session_id:
            render_saved_run_summary()
            return
        try:
            load_saved_run_session_into_controls(controls, session_id)
            refresh_saved_run_options()
            saved_run_selector.value = session_id
            render_saved_run_summary(session_id)
        except Exception as error:
            saved_run_status.value = (
                "<div style='line-height:1.45; color:#991b1b;'><b>Load failed</b>: "
                f"{html.escape(str(error))}</div>"
            )

    def load_latest_run(_button: widgets.Button) -> None:
        manifests = list_saved_run_sessions()
        if not manifests:
            render_saved_run_summary()
            return
        preferred = next(
            (
                manifest
                for manifest in manifests
                if str(manifest.get("status", "")) not in {"complete", "complete_with_errors", "saved"}
            ),
            manifests[0],
        )
        saved_run_selector.value = str(preferred["session_id"])
        load_selected_run(_button)

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
    controls["run_name"] = widgets.Text(
        value=DEFAULT_RUN_NAME,
        description="Run name",
        layout=widgets.Layout(width="280px"),
    )
    controls["batch_size"] = widgets.IntText(
        value=0, description="Batch size", layout=widgets.Layout(width="220px"))
    controls["resume_from_existing"] = widgets.Checkbox(
        value=False, description="Resume from existing results")
    controls["variant_mode"] = widgets.Dropdown(
        options=[
            (VARIANT_MODE_LABELS["manual"], "manual"),
            (VARIANT_MODE_LABELS["paper_high_roi"], "paper_high_roi"),
            (VARIANT_MODE_LABELS["paper_full_suite"], "paper_full_suite"),
            (VARIANT_MODE_LABELS["auto_ablation"], "auto_ablation"),
        ],
        value="manual",
        description="Argument mode",
        layout=widgets.Layout(width="280px"),
    )
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
    controls["saved_run_selector"] = saved_run_selector

    for key in [
        "run_name",
        "dataset_limit",
        "batch_size",
        "resume_from_existing",
        "variant_mode",
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
    saved_run_selector.observe(
        lambda change: render_saved_run_summary(change["new"]), names="value")
    refresh_saved_runs_button.on_click(refresh_saved_run_options)
    save_run_button.on_click(save_current_run)
    load_run_button.on_click(load_selected_run)
    load_latest_button.on_click(load_latest_run)

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
                    _with_tooltip(controls["run_name"], "run_name"),
                    _with_tooltip(saved_run_selector, "saved_run_selector"),
                ]
            ),
            _control_row(
                [
                    refresh_saved_runs_button,
                    load_run_button,
                    load_latest_button,
                    save_run_button,
                ]
            ),
            _control_row([saved_run_status]),
            _control_row(
                [
                    _with_tooltip(controls["dataset_limit"], "dataset_limit"),
                    _with_tooltip(controls["batch_size"], "batch_size"),
                    _with_tooltip(controls["resume_from_existing"], "resume_from_existing"),
                    _with_tooltip(controls["variant_mode"], "variant_mode"),
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
                    "run_name",
                    "saved_run_selector",
                    "variant_mode",
                    "dataset_limit",
                    "batch_size",
                    "resume_from_existing",
                    "normalization_method",
                    "clip_quantile",
                    "overwrite_normalized",
                    "window_size",
                    "window_stride",
                    "threshold_method",
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
            _general_process_block(),
        ],
        "#334155",
    )

    refresh_threshold_value_control()
    refresh_saved_run_options()
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
                "<p>Set <b>Argument mode</b> to an auto sweep if you want the notebook to benchmark multiple parameter combinations for you. Use <b>auto_ablation</b> when you need one-knob-at-a-time evidence for a defense. In <b>manual</b> mode, the subtab bar behaves like a browser: <b>+</b> duplicates the current argument set so you can compare variants yourself.</p>"
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
    if "variant_mode" in controls:
        controls["variant_mode"].value = preset_name


def get_run_config(controls: dict[str, Any]) -> dict[str, Any]:
    run_name = str(controls["run_name"].value).strip() or DEFAULT_RUN_NAME
    clip_value = None if controls["clip_quantile"].value <= 0 else float(
        controls["clip_quantile"].value)
    window_size = None if controls["window_size"].value <= 0 else int(
        controls["window_size"].value)
    window_stride = max(1, int(controls["window_stride"].value))
    dataset_limit = None if controls["dataset_limit"].value <= 0 else int(
        controls["dataset_limit"].value)
    batch_size = None if controls["batch_size"].value <= 0 else int(
        controls["batch_size"].value)
    threshold_method = str(controls["threshold_method"].value)
    threshold_value = (
        max(1, int(controls["threshold_value"].value))
        if threshold_method == "top_k"
        else float(controls["threshold_value"].value)
    )
    effective_variants = _resolve_effective_variants_from_controls(controls)

    return {
        "run_name": run_name,
        "session_id": saved_run_session_id(run_name),
        "dataset_limit": dataset_limit,
        "batch_size": batch_size,
        "resume_from_existing": bool(controls["resume_from_existing"].value),
        "variant_mode": effective_variants["variant_mode"],
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
        "selected_algorithms": effective_variants["selected_algorithms"],
        "selected_runs": effective_variants["selected_runs"],
        "algorithm_variants": effective_variants["algorithm_variants"],
        "auto_preset_name": effective_variants["auto_preset_name"],
        "auto_filtered_out": effective_variants["auto_filtered_out"],
        "control_state": snapshot_control_panel_state(controls),
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
                "variant_origin": row["variant_origin"],
                "variant_source": row["variant_source"],
                "variant_focus": row["variant_focus"],
                "variant_family": row["variant_family"],
                "ablation_parameter": row["ablation_parameter"],
                "ablation_label": row["ablation_label"],
                "ablation_role": row["ablation_role"],
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


def build_algorithm_report_narrative(results_frame: pd.DataFrame, algorithm_key: str) -> str:
    subset = add_analysis_regime_columns(
        results_frame.loc[results_frame["algorithm"] == algorithm_key].copy()
    )
    if subset.empty:
        return ""

    variant_summary = (
        subset.groupby("algorithm_variant", as_index=False)
        .agg(
            dataset_count=("dataset_name", "nunique"),
            mean_evaluation_f1=("evaluation_f1", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
        )
        .sort_values(["mean_evaluation_f1", "mean_roc_auc", "mean_runtime_seconds"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    best_variant = variant_summary.iloc[0]
    fastest_variant = variant_summary.sort_values("mean_runtime_seconds", ascending=True).iloc[0]

    family_summary = (
        subset.groupby("variant", as_index=False)["evaluation_f1"]
        .mean()
        .rename(columns={"evaluation_f1": "mean_evaluation_f1"})
        .sort_values("mean_evaluation_f1", ascending=False)
        .reset_index(drop=True)
    )
    strongest_family = family_summary.iloc[0]
    weakest_family = family_summary.iloc[-1]

    length_summary = (
        subset.groupby("length_bucket", as_index=False)["evaluation_f1"]
        .mean()
        .rename(columns={"evaluation_f1": "mean_evaluation_f1"})
        .sort_values("mean_evaluation_f1", ascending=False)
        .reset_index(drop=True)
    )
    strongest_length = length_summary.iloc[0]

    if (subset["variant_source"].astype(str) == "auto_ablation").any():
        ablation_sentence = (
            "Use the ablation section to defend knob-level claims: those rows are paired against the same baseline on the same dataset, "
            "so they are the controlled sensitivity evidence. The parameter-effects table is still useful, but it is only a ranking across the currently run variants."
        )
    else:
        ablation_sentence = (
            "This run does not include paired one-knob ablations. If you need defensible knob-level claims, rerun this algorithm with "
            "`Argument mode = auto_ablation` so every control is measured against a fixed baseline."
        )

    return (
        "<div style='margin:8px 0 12px 0; padding:10px 12px; border:1px solid #cbd5e1; border-radius:8px; background:#f8fafc; line-height:1.45;'>"
        f"<p style='margin:0 0 8px 0;'><b>{html.escape(DISPLAY_NAME_MAP[algorithm_key])} report guide</b>: "
        "start with the variant table to verify exactly which settings were run, then cite the parameter-effects and regime tables for aggregate performance, "
        "and use the figures to show runtime tradeoffs and concrete score behavior on shared series.</p>"
        "<ul style='margin:0 0 0 18px; padding:0;'>"
        f"<li><b>Best average variant in this run</b>: {html.escape(str(best_variant['algorithm_variant']))} "
        f"with mean evaluation F1 {best_variant['mean_evaluation_f1']:.3f}, ROC AUC {best_variant['mean_roc_auc']:.3f}, "
        f"across {int(best_variant['dataset_count'])} datasets.</li>"
        f"<li><b>Fastest average variant</b>: {html.escape(str(fastest_variant['algorithm_variant']))} "
        f"at {fastest_variant['mean_runtime_seconds']:.3f}s mean runtime.</li>"
        f"<li><b>Strongest dataset family for this algorithm in the current run</b>: {html.escape(str(strongest_family['variant']))} "
        f"(mean evaluation F1 {strongest_family['mean_evaluation_f1']:.3f}); weakest family: {html.escape(str(weakest_family['variant']))} "
        f"(mean evaluation F1 {weakest_family['mean_evaluation_f1']:.3f}).</li>"
        f"<li><b>Best length regime</b>: {html.escape(str(strongest_length['length_bucket']))} "
        f"(mean evaluation F1 {strongest_length['mean_evaluation_f1']:.3f}).</li>"
        f"<li>{html.escape(ablation_sentence)}</li>"
        "</ul>"
        "</div>"
    )


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int = 400, seed: int = 42) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan"), float("nan")
    if clean.size == 1:
        return float(clean[0]), float(clean[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, clean.size, size=(n_boot, clean.size))
    boot_means = clean[indices].mean(axis=1)
    return float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975))


def build_algorithm_ablation_delta_frame(results_frame: pd.DataFrame, algorithm_key: str) -> pd.DataFrame:
    subset = results_frame.loc[
        (results_frame["algorithm"] == algorithm_key)
        & (results_frame["variant_source"] == "auto_ablation")
    ].copy()
    if subset.empty:
        return pd.DataFrame()

    baseline = subset.loc[subset["variant_family"] == "baseline"].copy()
    ablations = subset.loc[subset["variant_family"] == "ablation"].copy()
    if baseline.empty or ablations.empty:
        return pd.DataFrame()

    baseline = baseline[
        [
            "dataset_name",
            "algorithm_display",
            "algorithm_variant",
            "evaluation_f1",
            "roc_auc",
            "runtime_seconds",
            "evaluation_precision",
            "evaluation_recall",
            "range_f1",
        ]
    ].rename(
        columns={
            "algorithm_display": "baseline_algorithm_display",
            "algorithm_variant": "baseline_algorithm_variant",
            "evaluation_f1": "baseline_evaluation_f1",
            "roc_auc": "baseline_roc_auc",
            "runtime_seconds": "baseline_runtime_seconds",
            "evaluation_precision": "baseline_evaluation_precision",
            "evaluation_recall": "baseline_evaluation_recall",
            "range_f1": "baseline_range_f1",
        }
    )

    merged = ablations.merge(baseline, on="dataset_name", how="inner")
    if merged.empty:
        return pd.DataFrame()

    merged["delta_evaluation_f1"] = merged["evaluation_f1"] - merged["baseline_evaluation_f1"]
    merged["delta_roc_auc"] = merged["roc_auc"] - merged["baseline_roc_auc"]
    merged["delta_runtime_seconds"] = merged["runtime_seconds"] - merged["baseline_runtime_seconds"]
    merged["runtime_ratio"] = np.where(
        np.isfinite(merged["baseline_runtime_seconds"]) & (merged["baseline_runtime_seconds"] > 0),
        merged["runtime_seconds"] / merged["baseline_runtime_seconds"],
        np.nan,
    )
    merged["delta_evaluation_precision"] = merged["evaluation_precision"] - merged["baseline_evaluation_precision"]
    merged["delta_evaluation_recall"] = merged["evaluation_recall"] - merged["baseline_evaluation_recall"]
    merged["delta_range_f1"] = merged["range_f1"] - merged["baseline_range_f1"]
    return add_analysis_regime_columns(merged)


def build_algorithm_ablation_impact_table(results_frame: pd.DataFrame, algorithm_key: str) -> pd.DataFrame:
    deltas = build_algorithm_ablation_delta_frame(results_frame, algorithm_key)
    if deltas.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    group_columns = [
        "algorithm_display",
        "algorithm_variant",
        "ablation_parameter",
        "ablation_label",
        "ablation_role",
    ]
    for keys, frame in deltas.groupby(group_columns, dropna=False):
        (
            algorithm_display,
            algorithm_variant,
            ablation_parameter,
            ablation_label,
            ablation_role,
        ) = keys
        eval_ci_low, eval_ci_high = _bootstrap_mean_ci(frame["delta_evaluation_f1"].to_numpy(dtype=float))
        auc_ci_low, auc_ci_high = _bootstrap_mean_ci(frame["delta_roc_auc"].to_numpy(dtype=float))
        rows.append(
            {
                "algorithm_display": algorithm_display,
                "algorithm_variant": algorithm_variant,
                "ablation_parameter": ablation_parameter,
                "ablation_label": ablation_label,
                "ablation_role": ablation_role,
                "dataset_count": frame["dataset_name"].nunique(),
                "mean_baseline_evaluation_f1": frame["baseline_evaluation_f1"].mean(),
                "mean_variant_evaluation_f1": frame["evaluation_f1"].mean(),
                "mean_delta_evaluation_f1": frame["delta_evaluation_f1"].mean(),
                "delta_evaluation_f1_ci_low": eval_ci_low,
                "delta_evaluation_f1_ci_high": eval_ci_high,
                "mean_delta_roc_auc": frame["delta_roc_auc"].mean(),
                "delta_roc_auc_ci_low": auc_ci_low,
                "delta_roc_auc_ci_high": auc_ci_high,
                "mean_delta_runtime_seconds": frame["delta_runtime_seconds"].mean(),
                "median_runtime_ratio": frame["runtime_ratio"].median(),
                "mean_runtime_ratio": frame["runtime_ratio"].mean(),
                "win_rate_evaluation_f1": float((frame["delta_evaluation_f1"] > 0).mean()),
                "loss_rate_evaluation_f1": float((frame["delta_evaluation_f1"] < 0).mean()),
                "mean_delta_evaluation_precision": frame["delta_evaluation_precision"].mean(),
                "mean_delta_evaluation_recall": frame["delta_evaluation_recall"].mean(),
                "mean_delta_range_f1": frame["delta_range_f1"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mean_delta_evaluation_f1", "win_rate_evaluation_f1", "mean_delta_roc_auc"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_algorithm_ablation_regime_table(results_frame: pd.DataFrame, algorithm_key: str, regime_column: str) -> pd.DataFrame:
    deltas = build_algorithm_ablation_delta_frame(results_frame, algorithm_key)
    if deltas.empty:
        return pd.DataFrame()
    return (
        deltas.groupby(["algorithm_display", "ablation_parameter", "ablation_role", regime_column], as_index=False)
        .agg(
            dataset_count=("dataset_name", "nunique"),
            mean_delta_evaluation_f1=("delta_evaluation_f1", "mean"),
            mean_delta_roc_auc=("delta_roc_auc", "mean"),
            mean_runtime_ratio=("runtime_ratio", "mean"),
        )
        .sort_values([regime_column, "mean_delta_evaluation_f1"], ascending=[True, False])
        .reset_index(drop=True)
    )


def build_algorithm_ablation_narrative(results_frame: pd.DataFrame, algorithm_key: str) -> str:
    impact = build_algorithm_ablation_impact_table(results_frame, algorithm_key)
    if impact.empty:
        return ""

    strongest_gain = impact.sort_values(
        ["mean_delta_evaluation_f1", "win_rate_evaluation_f1", "mean_delta_roc_auc"],
        ascending=[False, False, False],
    ).iloc[0]
    strongest_loss = impact.sort_values(
        ["mean_delta_evaluation_f1", "mean_delta_roc_auc"],
        ascending=[True, True],
    ).iloc[0]
    cheapest = impact.sort_values(
        ["mean_runtime_ratio", "mean_delta_evaluation_f1"],
        ascending=[True, False],
    ).iloc[0]
    most_expensive = impact.sort_values(
        ["mean_runtime_ratio", "mean_delta_evaluation_f1"],
        ascending=[False, False],
    ).iloc[0]

    backend_subset = impact.loc[impact["ablation_role"] == "backend"].copy()
    backend_sentence = ""
    if not backend_subset.empty:
        backend_best = backend_subset.loc[
            (backend_subset["mean_delta_evaluation_f1"].abs() + (backend_subset["mean_runtime_ratio"] - 1.0).abs()).idxmin()
        ]
        backend_sentence = (
            f"<li><b>Backend-only evidence</b>: {html.escape(str(backend_best['algorithm_variant']))} "
            f"moved mean evaluation F1 by {backend_best['mean_delta_evaluation_f1']:+.3f} "
            f"with runtime ratio {backend_best['mean_runtime_ratio']:.2f}.</li>"
        )

    return (
        "<div style='margin:6px 0 10px 0; line-height:1.45;'>"
        "<p style='margin:0 0 8px 0;'><b>How to read this ablation section</b>: "
        "every non-baseline row changes one visible algorithm argument while keeping preprocessing, thresholding, and evaluation fixed. "
        "All deltas are paired against the same baseline on the same dataset.</p>"
        "<ul style='margin:0 0 0 18px; padding:0;'>"
        f"<li><b>Strongest positive shift</b>: {html.escape(str(strongest_gain['algorithm_variant']))} "
        f"changed mean evaluation F1 by {strongest_gain['mean_delta_evaluation_f1']:+.3f} "
        f"(95% CI {strongest_gain['delta_evaluation_f1_ci_low']:+.3f} to {strongest_gain['delta_evaluation_f1_ci_high']:+.3f}) "
        f"with win rate {strongest_gain['win_rate_evaluation_f1']:.1%}.</li>"
        f"<li><b>Strongest negative shift</b>: {html.escape(str(strongest_loss['algorithm_variant']))} "
        f"changed mean evaluation F1 by {strongest_loss['mean_delta_evaluation_f1']:+.3f}.</li>"
        f"<li><b>Cheapest change</b>: {html.escape(str(cheapest['algorithm_variant']))} ran at "
        f"{cheapest['mean_runtime_ratio']:.2f}x baseline runtime.</li>"
        f"<li><b>Most expensive change</b>: {html.escape(str(most_expensive['algorithm_variant']))} ran at "
        f"{most_expensive['mean_runtime_ratio']:.2f}x baseline runtime.</li>"
        f"{backend_sentence}"
        "</ul>"
        "</div>"
    )


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
                "variant_index": payload["metric_row"]["variant_index"],
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
        ["variant_index", "evaluation_f1", "roc_auc"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    variant_payloads = sorted(
        variant_payloads,
        key=lambda payload: int(payload["metric_row"]["variant_index"]),
    )

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


def _draw_signed_heatmap(ax: Any, frame: pd.DataFrame, title: str, cmap: str = "RdBu_r") -> Any:
    if frame.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        ax.set_title(title)
        return None
    max_abs = float(np.nanmax(np.abs(frame.to_numpy(dtype=float))))
    max_abs = max(max_abs, 1e-6)
    image = ax.imshow(frame.to_numpy(), cmap=cmap, aspect="auto", vmin=-max_abs, vmax=max_abs)
    ax.set_title(title)
    ax.set_xticks(range(len(frame.columns)))
    ax.set_xticklabels(frame.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(frame.index)))
    ax.set_yticklabels(frame.index)
    for row_index in range(frame.shape[0]):
        for col_index in range(frame.shape[1]):
            ax.text(col_index, row_index, f"{frame.iloc[row_index, col_index]:+.2f}", ha="center", va="center", fontsize=9)
    return image


def _save_figure_bundle(
    fig: Any,
    save_path: Path,
    formats: tuple[str, ...] = ("png", "pdf"),
    dpi: int = 240,
) -> dict[str, Path]:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[str, Path] = {}
    for fmt in formats:
        target = save_path.with_suffix(f".{fmt}")
        kwargs: dict[str, Any] = {"facecolor": "white"}
        if fmt.lower() in {"png", "jpg", "jpeg", "tif", "tiff"}:
            kwargs["dpi"] = dpi
        fig.savefig(target, **kwargs)
        saved_paths[fmt.lower()] = target
    return saved_paths


def _pareto_frontier_frame(algorithm_summary: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    tradeoff_frame = (
        algorithm_summary[["algorithm_display", "mean_runtime_seconds", "mean_evaluation_f1"]]
        .dropna()
        .sort_values("mean_runtime_seconds")
        .reset_index(drop=True)
    )
    frontier: list[tuple[float, float]] = []
    best_seen = -np.inf
    for row in tradeoff_frame.itertuples():
        if row.mean_evaluation_f1 >= best_seen:
            frontier.append((float(row.mean_runtime_seconds), float(row.mean_evaluation_f1)))
            best_seen = float(row.mean_evaluation_f1)
    return tradeoff_frame, np.asarray(frontier, dtype=float) if frontier else np.empty((0, 2), dtype=float)


def _winner_count_frame(benchmark: dict[str, Any]) -> pd.DataFrame:
    algorithm_summary = benchmark["algorithm_summary"]
    best_by_evaluation = benchmark["best_by_evaluation"]
    best_by_auc = benchmark["best_by_auc"]
    labels = algorithm_summary["algorithm_display"]
    return pd.DataFrame(
        {
            "algorithm_display": labels,
            "evaluation_wins": best_by_evaluation["best_algorithm_display"].value_counts().reindex(labels).fillna(0).astype(int).to_numpy(),
            "auc_wins": best_by_auc["best_algorithm_display"].value_counts().reindex(labels).fillna(0).astype(int).to_numpy(),
        }
    )


def plot_benchmark_overview_panel(benchmark: dict[str, Any], save_path: Path | None = None) -> plt.Figure:
    plt = _load_plotting_module()
    dataset_catalog = benchmark["dataset_catalog"]
    algorithm_summary = benchmark["algorithm_summary"]
    metric_spec = _evaluation_metric_spec(benchmark["results"])
    tradeoff_frame, pareto_points = _pareto_frontier_frame(algorithm_summary)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)

    axes[0, 0].hist(dataset_catalog["series_length"], bins=24, color="#4c78a8", edgecolor="white")
    axes[0, 0].set_title("Dataset length distribution")
    axes[0, 0].set_xlabel("Series length")
    axes[0, 0].set_ylabel("Dataset count")

    axes[0, 1].hist(dataset_catalog["anomaly_ratio"], bins=24, color="#72b7b2", edgecolor="white")
    axes[0, 1].set_title("Anomaly ratio distribution")
    axes[0, 1].set_xlabel("Anomaly ratio")
    axes[0, 1].set_ylabel("Dataset count")

    for row in tradeoff_frame.itertuples():
        axes[1, 0].scatter(row.mean_runtime_seconds, row.mean_evaluation_f1, s=90, alpha=0.9, color="#4c78a8")
        axes[1, 0].annotate(
            row.algorithm_display,
            (row.mean_runtime_seconds, row.mean_evaluation_f1),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )
    if len(pareto_points) > 0:
        axes[1, 0].plot(
            pareto_points[:, 0],
            pareto_points[:, 1],
            color="#e45756",
            linewidth=1.8,
            label="Pareto frontier",
        )
        axes[1, 0].legend()
    axes[1, 0].set_title(f"Runtime vs {metric_spec['mean_label']}")
    axes[1, 0].set_xlabel("Mean runtime (seconds)")
    axes[1, 0].set_ylabel(metric_spec["mean_label"])

    bar_positions = np.arange(len(algorithm_summary))
    bar_width = 0.38
    axes[1, 1].bar(
        bar_positions - bar_width / 2,
        algorithm_summary["mean_evaluation_f1"],
        width=bar_width,
        label=metric_spec["mean_label"],
        color="#f58518",
    )
    axes[1, 1].bar(
        bar_positions + bar_width / 2,
        algorithm_summary["mean_roc_auc"],
        width=bar_width,
        label="Mean ROC AUC",
        color="#54a24b",
    )
    axes[1, 1].set_xticks(bar_positions)
    axes[1, 1].set_xticklabels(algorithm_summary["algorithm_display"], rotation=25, ha="right")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_title("Average accuracy by configuration")
    axes[1, 1].set_ylabel("Metric value")
    axes[1, 1].legend()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)
    return fig


def plot_pareto_frontier_panel(benchmark: dict[str, Any], save_path: Path | None = None) -> plt.Figure | None:
    plt = _load_plotting_module()
    tradeoff_frame, pareto_points = _pareto_frontier_frame(benchmark["algorithm_summary"])
    if tradeoff_frame.empty or len(pareto_points) == 0:
        return None

    metric_spec = _evaluation_metric_spec(benchmark["results"])
    fig, ax = plt.subplots(figsize=(9.5, 6.5), constrained_layout=True)
    for row in tradeoff_frame.itertuples():
        ax.scatter(row.mean_runtime_seconds, row.mean_evaluation_f1, s=90, alpha=0.9, color="#4c78a8")
        ax.annotate(
            row.algorithm_display,
            (row.mean_runtime_seconds, row.mean_evaluation_f1),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )
    ax.plot(pareto_points[:, 0], pareto_points[:, 1], color="#e45756", linewidth=1.8, label="Pareto frontier")
    ax.set_title(f"Pareto frontier | mean runtime vs {metric_spec['mean_label']}")
    ax.set_xlabel("Mean runtime (seconds)")
    ax.set_ylabel(metric_spec["mean_label"])
    ax.legend(loc="best")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)
    return fig


def plot_metric_heatmap_panel(benchmark: dict[str, Any], save_path: Path | None = None) -> plt.Figure:
    plt = _load_plotting_module()
    algorithm_summary = benchmark["algorithm_summary"]
    metric_spec = _evaluation_metric_spec(benchmark["results"])
    heatmap_frame = algorithm_summary.set_index("algorithm_display")[
        [
            "mean_roc_auc",
            "mean_average_precision",
            "mean_f1",
            "mean_evaluation_f1",
            "mean_range_precision",
            "mean_range_recall",
            "mean_range_f1",
            "mean_affiliation_precision",
            "mean_affiliation_recall",
        ]
    ]
    heatmap_labels = [
        "ROC AUC",
        "Avg Precision",
        "F1",
        metric_spec["mean_label"],
        "Range Precision",
        "Range Recall",
        "Range F1",
        "Affil. Precision",
        "Affil. Recall",
    ]

    fig, ax = plt.subplots(figsize=(14, 5.4), constrained_layout=True)
    image = ax.imshow(heatmap_frame.to_numpy(), cmap="YlOrBr", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(heatmap_labels)))
    ax.set_xticklabels(heatmap_labels, rotation=20, ha="right")
    ax.set_yticks(range(len(heatmap_frame.index)))
    ax.set_yticklabels(heatmap_frame.index)
    ax.set_title("Metric heatmap by configuration")
    for row_index in range(heatmap_frame.shape[0]):
        for col_index in range(heatmap_frame.shape[1]):
            ax.text(col_index, row_index, f"{heatmap_frame.iloc[row_index, col_index]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)
    return fig


def plot_family_evaluation_heatmap_panel(benchmark: dict[str, Any], save_path: Path | None = None) -> plt.Figure:
    plt = _load_plotting_module()
    family_summary = benchmark["family_summary"]
    metric_spec = _evaluation_metric_spec(benchmark["results"])
    family_heatmap = family_summary.pivot(index="family", columns="algorithm_display", values="mean_evaluation_f1").fillna(0.0)

    fig, ax = plt.subplots(figsize=(16, max(6, 0.35 * len(family_heatmap.index))), constrained_layout=True)
    image = ax.imshow(family_heatmap.to_numpy(), cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(family_heatmap.columns)))
    ax.set_xticklabels(family_heatmap.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(family_heatmap.index)))
    ax.set_yticklabels(family_heatmap.index)
    ax.set_title(f"{metric_spec['mean_label']} by dataset family and configuration")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)
    return fig


def plot_algorithm_wins_panel(benchmark: dict[str, Any], save_path: Path | None = None) -> plt.Figure:
    plt = _load_plotting_module()
    metric_spec = _evaluation_metric_spec(benchmark["results"])
    wins = _winner_count_frame(benchmark)
    win_metric_label = metric_spec["label"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    axes[0].bar(wins["algorithm_display"], wins["evaluation_wins"], color="#4c78a8")
    axes[0].set_title(f"Configuration wins by {win_metric_label}")
    axes[0].set_ylabel("Dataset wins")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(wins["algorithm_display"], wins["auc_wins"], color="#e45756")
    axes[1].set_title("Configuration wins by ROC AUC")
    axes[1].set_ylabel("Dataset wins")
    axes[1].tick_params(axis="x", rotation=25)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)
    return fig


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


def build_ablation_overview_table(results_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for algorithm_key in ALGORITHM_ORDER:
        impact = build_algorithm_ablation_impact_table(results_frame, algorithm_key)
        if impact.empty:
            continue
        impact = impact.copy()
        impact.insert(0, "algorithm", DISPLAY_NAME_MAP[algorithm_key])
        impact.insert(1, "algorithm_key", algorithm_key)
        rows.append(impact)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(
        ["mean_delta_evaluation_f1", "win_rate_evaluation_f1", "mean_runtime_ratio"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def build_ablation_overview_narrative(results_frame: pd.DataFrame) -> str:
    overview = build_ablation_overview_table(results_frame)
    if overview.empty:
        return ""

    strongest_gain = overview.loc[overview["mean_delta_evaluation_f1"].idxmax()]
    strongest_loss = overview.loc[overview["mean_delta_evaluation_f1"].idxmin()]
    efficient_positive = overview.loc[
        (overview["mean_delta_evaluation_f1"] > 0)
        & (overview["mean_runtime_ratio"] <= 1.05)
    ].copy()
    if efficient_positive.empty:
        efficiency_sentence = "No positive ablation in this run improved evaluation F1 without a noticeable runtime increase."
    else:
        best_tradeoff = efficient_positive.sort_values(
            ["mean_delta_evaluation_f1", "mean_runtime_ratio"],
            ascending=[False, True],
        ).iloc[0]
        efficiency_sentence = (
            f"Best low-cost positive ablation: {best_tradeoff['algorithm']} | {best_tradeoff['algorithm_variant']} "
            f"shifted mean evaluation F1 by {best_tradeoff['mean_delta_evaluation_f1']:+.3f} at {best_tradeoff['mean_runtime_ratio']:.2f}x baseline runtime."
        )

    return (
        "<div style='margin:8px 0 12px 0; padding:10px 12px; border:1px solid #cbd5e1; border-radius:8px; background:#f8fafc; line-height:1.45;'>"
        "<p style='margin:0 0 8px 0;'><b>Global ablation reading guide</b>: "
        "every non-baseline variant changes one knob at a time, so these plots are the best evidence for which controls materially help, hurt, or mostly change runtime.</p>"
        "<ul style='margin:0 0 0 18px; padding:0;'>"
        f"<li><b>Strongest global gain</b>: {html.escape(str(strongest_gain['algorithm']))} | {html.escape(str(strongest_gain['algorithm_variant']))} "
        f"with mean evaluation F1 delta {strongest_gain['mean_delta_evaluation_f1']:+.3f} "
        f"(95% CI {strongest_gain['delta_evaluation_f1_ci_low']:+.3f} to {strongest_gain['delta_evaluation_f1_ci_high']:+.3f}).</li>"
        f"<li><b>Strongest global loss</b>: {html.escape(str(strongest_loss['algorithm']))} | {html.escape(str(strongest_loss['algorithm_variant']))} "
        f"with mean evaluation F1 delta {strongest_loss['mean_delta_evaluation_f1']:+.3f}.</li>"
        f"<li>{html.escape(efficiency_sentence)}</li>"
        "<li>Use the confidence intervals on the bar charts to separate stable shifts from noisy ones, then use the runtime scatter to show whether the gain is worth the cost.</li>"
        "</ul>"
        "</div>"
    )


def _ci_error_arrays(frame: pd.DataFrame, mean_col: str, low_col: str, high_col: str) -> np.ndarray:
    lower = (frame[mean_col] - frame[low_col]).clip(lower=0).fillna(0.0).to_numpy(dtype=float)
    upper = (frame[high_col] - frame[mean_col]).clip(lower=0).fillna(0.0).to_numpy(dtype=float)
    return np.vstack([lower, upper])


def plot_ablation_overview_panel(results_frame: pd.DataFrame, save_path: Path | None = None) -> plt.Figure | None:
    overview = build_ablation_overview_table(results_frame)
    if overview.empty:
        return None

    plt = _load_plotting_module()
    positive = overview.head(12).sort_values("mean_delta_evaluation_f1", ascending=True)
    negative = overview.sort_values("mean_delta_evaluation_f1", ascending=True).head(12)
    best_by_algorithm = overview.sort_values("mean_delta_evaluation_f1", ascending=False).groupby("algorithm", as_index=False).head(1)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)

    for ax, frame, title in (
        (axes[0, 0], positive, "Top positive ablations by mean evaluation F1 delta"),
        (axes[0, 1], negative, "Top negative ablations by mean evaluation F1 delta"),
    ):
        labels = [f"{row.algorithm} | {row.algorithm_variant}" for row in frame.itertuples()]
        colors = [ABLATION_ROLE_STYLES.get(str(row.ablation_role), ABLATION_ROLE_STYLES[""])["color"] for row in frame.itertuples()]
        ax.barh(
            labels,
            frame["mean_delta_evaluation_f1"],
            color=colors,
            edgecolor="white",
        )
        ax.errorbar(
            frame["mean_delta_evaluation_f1"],
            labels,
            xerr=_ci_error_arrays(
                frame,
                "mean_delta_evaluation_f1",
                "delta_evaluation_f1_ci_low",
                "delta_evaluation_f1_ci_high",
            ),
            fmt="none",
            ecolor="#0f172a",
            elinewidth=1.0,
            capsize=3,
        )
        ax.axvline(0.0, color="#0f172a", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("Mean paired delta in evaluation F1 (95% CI)")

    scatter_colors = [
        ABLATION_ROLE_STYLES.get(str(role), ABLATION_ROLE_STYLES[""])["color"]
        for role in overview["ablation_role"]
    ]
    axes[1, 0].scatter(
        overview["mean_runtime_ratio"],
        overview["mean_delta_evaluation_f1"],
        s=90,
        alpha=0.85,
        color=scatter_colors,
    )
    for row in overview.itertuples():
        axes[1, 0].annotate(
            f"{row.algorithm} | {row.algorithm_variant}",
            (row.mean_runtime_ratio, row.mean_delta_evaluation_f1),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )
    axes[1, 0].axhline(0.0, color="#0f172a", linewidth=1.0)
    axes[1, 0].axvline(1.0, color="#475569", linestyle="--", linewidth=1.0)
    axes[1, 0].set_title("Runtime tradeoff vs paired evaluation F1 delta")
    axes[1, 0].set_xlabel("Mean runtime ratio vs baseline")
    axes[1, 0].set_ylabel("Mean paired delta in evaluation F1")

    best_labels = [f"{row.algorithm} | {row.algorithm_variant}" for row in best_by_algorithm.itertuples()]
    best_colors = [ABLATION_ROLE_STYLES.get(str(row.ablation_role), ABLATION_ROLE_STYLES[""])["color"] for row in best_by_algorithm.itertuples()]
    axes[1, 1].barh(
        best_labels,
        best_by_algorithm["mean_delta_evaluation_f1"],
        color=best_colors,
        edgecolor="white",
    )
    axes[1, 1].errorbar(
        best_by_algorithm["mean_delta_evaluation_f1"],
        best_labels,
        xerr=_ci_error_arrays(
            best_by_algorithm,
            "mean_delta_evaluation_f1",
            "delta_evaluation_f1_ci_low",
            "delta_evaluation_f1_ci_high",
        ),
        fmt="none",
        ecolor="#0f172a",
        elinewidth=1.0,
        capsize=3,
    )
    axes[1, 1].axvline(0.0, color="#0f172a", linewidth=1.0)
    axes[1, 1].set_title("Best ablation per algorithm")
    axes[1, 1].set_xlabel("Mean paired delta in evaluation F1 (95% CI)")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)
    return fig


def plot_algorithm_ablation_panel(results_frame: pd.DataFrame, algorithm_key: str, save_path: Path | None = None) -> plt.Figure | None:
    impact = build_algorithm_ablation_impact_table(results_frame, algorithm_key)
    if impact.empty:
        return None

    regime = build_algorithm_ablation_regime_table(results_frame, algorithm_key, "variant")
    plt = _load_plotting_module()
    ordered = impact.sort_values("mean_delta_evaluation_f1", ascending=True).reset_index(drop=True)
    labels = ordered["algorithm_variant"].tolist()
    positions = np.arange(len(labels))
    colors = [
        ABLATION_ROLE_STYLES.get(str(role), ABLATION_ROLE_STYLES[""])["color"]
        for role in ordered["ablation_role"]
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)

    eval_err_low = ordered["mean_delta_evaluation_f1"] - ordered["delta_evaluation_f1_ci_low"]
    eval_err_high = ordered["delta_evaluation_f1_ci_high"] - ordered["mean_delta_evaluation_f1"]
    axes[0, 0].barh(positions, ordered["mean_delta_evaluation_f1"], color=colors, edgecolor="white")
    axes[0, 0].errorbar(
        ordered["mean_delta_evaluation_f1"],
        positions,
        xerr=np.vstack([eval_err_low.to_numpy(dtype=float), eval_err_high.to_numpy(dtype=float)]),
        fmt="none",
        ecolor="#0f172a",
        elinewidth=1.0,
        capsize=3,
    )
    axes[0, 0].axvline(0.0, color="#0f172a", linewidth=1.0)
    axes[0, 0].set_yticks(positions)
    axes[0, 0].set_yticklabels(labels)
    axes[0, 0].set_title(f"{DISPLAY_NAME_MAP[algorithm_key]} | Mean paired delta in evaluation F1")
    axes[0, 0].set_xlabel("Delta vs baseline")

    auc_err_low = ordered["mean_delta_roc_auc"] - ordered["delta_roc_auc_ci_low"]
    auc_err_high = ordered["delta_roc_auc_ci_high"] - ordered["mean_delta_roc_auc"]
    axes[0, 1].barh(positions, ordered["mean_delta_roc_auc"], color=colors, edgecolor="white")
    axes[0, 1].errorbar(
        ordered["mean_delta_roc_auc"],
        positions,
        xerr=np.vstack([auc_err_low.to_numpy(dtype=float), auc_err_high.to_numpy(dtype=float)]),
        fmt="none",
        ecolor="#0f172a",
        elinewidth=1.0,
        capsize=3,
    )
    axes[0, 1].axvline(0.0, color="#0f172a", linewidth=1.0)
    axes[0, 1].set_yticks(positions)
    axes[0, 1].set_yticklabels(labels)
    axes[0, 1].set_title(f"{DISPLAY_NAME_MAP[algorithm_key]} | Mean paired delta in ROC AUC")
    axes[0, 1].set_xlabel("Delta vs baseline")

    axes[1, 0].scatter(
        ordered["mean_runtime_ratio"],
        ordered["mean_delta_evaluation_f1"],
        s=100,
        alpha=0.9,
        color=colors,
    )
    for row in ordered.itertuples():
        axes[1, 0].annotate(
            row.algorithm_variant,
            (row.mean_runtime_ratio, row.mean_delta_evaluation_f1),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )
    axes[1, 0].axhline(0.0, color="#0f172a", linewidth=1.0)
    axes[1, 0].axvline(1.0, color="#475569", linestyle="--", linewidth=1.0)
    axes[1, 0].set_title(f"{DISPLAY_NAME_MAP[algorithm_key]} | Runtime tradeoff against mean delta in evaluation F1")
    axes[1, 0].set_xlabel("Mean runtime ratio vs baseline")
    axes[1, 0].set_ylabel("Mean delta in evaluation F1")

    regime_pivot = (
        regime.pivot(index="algorithm_display", columns="variant", values="mean_delta_evaluation_f1")
        if not regime.empty
        else pd.DataFrame()
    )
    regime_pivot = regime_pivot.reindex(
        [f"{DISPLAY_NAME_MAP[algorithm_key]} | {label}" for label in labels]
    )
    image = _draw_signed_heatmap(
        axes[1, 1],
        regime_pivot,
        f"{DISPLAY_NAME_MAP[algorithm_key]} | Mean paired delta in evaluation F1 by dataset variant",
    )
    if image is not None:
        fig.colorbar(image, ax=axes[1, 1], fraction=0.046, pad=0.04)

    legend_handles = []
    legend_labels = []
    for role in ordered["ablation_role"].drop_duplicates():
        style = ABLATION_ROLE_STYLES.get(str(role), ABLATION_ROLE_STYLES[""])
        legend_handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=style["color"], markersize=10))
        legend_labels.append(style["label"])
    if legend_handles:
        axes[1, 0].legend(legend_handles, legend_labels, loc="best")

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
    columns = 2
    variant_rows = max(1, math.ceil(len(variants) / columns))
    figure_height = 5.5 + (3.2 * variant_rows)

    fig = plt.figure(figsize=(18, figure_height), constrained_layout=True)
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
        axis.set_ylim(0.0, 1.05)
        axis.set_title(
            f"Variant {int(metric_row['variant_index'])}: {metric_row['algorithm_variant']} | "
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


def _slugify_label(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower()).strip("_")
    return text or "item"


def _append_thesis_figure_row(
    rows: list[dict[str, Any]],
    figure_id: str,
    figure_group: str,
    title: str,
    caption: str,
    saved_paths: dict[str, Path],
    algorithm: str = "",
) -> None:
    png_path = saved_paths.get("png")
    pdf_path = saved_paths.get("pdf")
    rows.append(
        {
            "figure_id": figure_id,
            "figure_group": figure_group,
            "algorithm": algorithm,
            "title": title,
            "png_path": portable_path_str(png_path) if png_path else "",
            "pdf_path": portable_path_str(pdf_path) if pdf_path else "",
            "caption": caption,
        }
    )


def build_thesis_figure_caption_markdown(catalog: pd.DataFrame) -> str:
    lines = [
        "# Thesis Figure Captions",
        "",
        "This file is generated automatically from the current benchmark state.",
        "",
    ]
    for row in catalog.itertuples():
        lines.extend(
            [
                f"## {row.figure_id}",
                f"- Title: {row.title}",
                f"- PNG: {row.png_path}",
                f"- PDF: {row.pdf_path}",
                "",
                row.caption,
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def _snapshot_figure_path(
    filename: str,
    session_id: str | None,
    *,
    write_global: bool,
) -> Path:
    return result_figure_path(filename) if write_global or not session_id else saved_run_session_figure_path(session_id, filename)


def _snapshot_algorithm_panel_path(
    algorithm_key: str,
    session_id: str | None,
    panel_kind: str,
    *,
    write_global: bool,
) -> Path:
    if write_global or not session_id:
        if panel_kind == "benchmark":
            return result_algorithm_panel_path(algorithm_key)
        if panel_kind == "paper":
            return result_algorithm_paper_panel_path(algorithm_key)
        if panel_kind == "variant_comparison":
            return result_algorithm_variant_comparison_path(algorithm_key)
        if panel_kind == "ablation":
            return result_algorithm_ablation_panel_path(algorithm_key)
    else:
        if panel_kind == "benchmark":
            return saved_run_session_algorithm_panel_path(session_id, algorithm_key)
        if panel_kind == "paper":
            return saved_run_session_algorithm_paper_panel_path(session_id, algorithm_key)
        if panel_kind == "variant_comparison":
            return saved_run_session_algorithm_variant_comparison_path(session_id, algorithm_key)
        if panel_kind == "ablation":
            return saved_run_session_algorithm_ablation_panel_path(session_id, algorithm_key)
    raise ValueError(f"Unknown panel kind: {panel_kind}")


def _snapshot_deep_dive_path(
    session_id: str | None,
    run_id: str,
    dataset_name: str,
    *,
    write_global: bool,
) -> Path:
    return result_deep_dive_path(run_id, dataset_name) if write_global or not session_id else saved_run_session_deep_dive_path(session_id, run_id, dataset_name)


def _snapshot_thesis_figure_path(
    filename: str,
    session_id: str | None,
    *,
    write_global: bool,
) -> Path:
    return result_thesis_figure_path(filename) if write_global or not session_id else saved_run_session_thesis_figure_path(session_id, filename)


def _snapshot_thesis_catalog_path(
    session_id: str | None,
    *,
    write_global: bool,
) -> Path:
    return THESIS_FIGURE_CATALOG_PATH if write_global or not session_id else saved_run_session_thesis_figure_catalog_path(session_id)


def _snapshot_thesis_captions_path(
    session_id: str | None,
    *,
    write_global: bool,
) -> Path:
    return THESIS_FIGURE_CAPTIONS_PATH if write_global or not session_id else saved_run_session_thesis_figure_captions_path(session_id)


def _close_figure(fig: Any) -> None:
    if fig is None:
        return
    plt = _load_plotting_module()
    plt.close(fig)


def export_algorithm_section_artifacts(
    notebook_state: dict[str, Any],
    algorithm_key: str,
    *,
    session_id: str | None = None,
    write_global: bool = False,
    context_points: int = 1200,
) -> dict[str, Any]:
    ns = notebook_state["ns"]
    config = notebook_state.get("config")
    context = notebook_state.get("context")
    benchmark = notebook_state.get("benchmark")
    if config is None or context is None or benchmark is None:
        raise ValueError("Run the benchmark cells before exporting algorithm section artifacts.")

    resolved_session_id = session_id or config.get("session_id")
    results = benchmark["results"]
    if algorithm_key not in config["selected_algorithms"]:
        raise ValueError(f"Algorithm '{algorithm_key}' is not enabled in this run.")

    variant_config = ns.build_variant_config_table(config, algorithm_key)
    parameter_effects = ns.build_algorithm_parameter_effect_table(results, algorithm_key)
    dataset_variant_summary = ns.build_algorithm_regime_table(results, algorithm_key, "variant")
    length_summary = ns.build_algorithm_regime_table(results, algorithm_key, "length_bucket")
    anomaly_ratio_summary = ns.build_algorithm_regime_table(results, algorithm_key, "anomaly_ratio_bucket")
    section_summary, section_top = ns.build_algorithm_section_tables(results, algorithm_key)
    ablation_impact = ns.build_algorithm_ablation_impact_table(results, algorithm_key)
    ablation_dataset_variant = ns.build_algorithm_ablation_regime_table(results, algorithm_key, "variant")
    ablation_length = ns.build_algorithm_ablation_regime_table(results, algorithm_key, "length_bucket")

    _write_table_artifact(
        variant_config,
        f"{algorithm_key}_variant_configuration.csv",
        resolved_session_id,
        write_global=write_global,
    )
    _write_table_artifact(
        parameter_effects,
        f"{algorithm_key}_parameter_effects.csv",
        resolved_session_id,
        write_global=write_global,
    )
    _write_table_artifact(
        dataset_variant_summary,
        f"{algorithm_key}_dataset_variant_summary.csv",
        resolved_session_id,
        write_global=write_global,
    )
    _write_table_artifact(
        length_summary,
        f"{algorithm_key}_length_summary.csv",
        resolved_session_id,
        write_global=write_global,
    )
    _write_table_artifact(
        anomaly_ratio_summary,
        f"{algorithm_key}_anomaly_ratio_summary.csv",
        resolved_session_id,
        write_global=write_global,
    )
    _write_table_artifact(
        section_summary,
        f"{algorithm_key}_section_summary.csv",
        resolved_session_id,
        write_global=write_global,
    )
    _write_table_artifact(
        section_top,
        f"{algorithm_key}_top_cases.csv",
        resolved_session_id,
        write_global=write_global,
    )
    if not ablation_impact.empty:
        _write_table_artifact(
            ablation_impact,
            f"{algorithm_key}_ablation_impacts.csv",
            resolved_session_id,
            write_global=write_global,
        )
        _write_table_artifact(
            ablation_dataset_variant,
            f"{algorithm_key}_ablation_dataset_variant_summary.csv",
            resolved_session_id,
            write_global=write_global,
        )
        _write_table_artifact(
            ablation_length,
            f"{algorithm_key}_ablation_length_summary.csv",
            resolved_session_id,
            write_global=write_global,
        )

    benchmark_fig = ns.plot_algorithm_benchmark_panel(
        results,
        algorithm_key,
        _snapshot_algorithm_panel_path(
            algorithm_key,
            resolved_session_id,
            "benchmark",
            write_global=write_global,
        ),
    )
    _close_figure(benchmark_fig)

    paper_fig = ns.plot_algorithm_paper_panel(
        results,
        algorithm_key,
        _snapshot_algorithm_panel_path(
            algorithm_key,
            resolved_session_id,
            "paper",
            write_global=write_global,
        ),
    )
    _close_figure(paper_fig)

    ablation_fig = ns.plot_algorithm_ablation_panel(
        results,
        algorithm_key,
        _snapshot_algorithm_panel_path(
            algorithm_key,
            resolved_session_id,
            "ablation",
            write_global=write_global,
        ),
    )
    _close_figure(ablation_fig)

    comparison_payload = ns.build_algorithm_variant_comparison(
        config,
        context["prepared_dataset_dir"],
        results,
        algorithm_key,
    )
    comparison_dataset = None
    if comparison_payload is not None and len(comparison_payload["variants"]) > 1:
        comparison_dataset = str(comparison_payload["dataset"]["dataset_name"])
        _write_table_artifact(
            comparison_payload["selection_summary"],
            f"{algorithm_key}_variant_comparison_selection.csv",
            resolved_session_id,
            write_global=write_global,
        )
        _write_table_artifact(
            comparison_payload["summary"],
            f"{algorithm_key}_variant_comparison.csv",
            resolved_session_id,
            write_global=write_global,
        )
        comparison_fig = ns.plot_algorithm_variant_comparison(
            comparison_payload,
            algorithm_key,
            context_points=context_points,
            save_path=_snapshot_algorithm_panel_path(
                algorithm_key,
                resolved_session_id,
                "variant_comparison",
                write_global=write_global,
            ),
        )
        _close_figure(comparison_fig)

    showcase = ns.build_algorithm_showcase(
        config,
        context["prepared_dataset_dir"],
        results,
        algorithm_key,
    )
    showcase_dataset = None
    if showcase is not None:
        showcase_dataset = str(showcase["dataset"]["dataset_name"])
        _write_table_artifact(
            showcase["summary"],
            f"{algorithm_key}_showcase_summary.csv",
            resolved_session_id,
            write_global=write_global,
        )
        showcase_fig = ns.plot_algorithm_deep_dive(
            showcase["raw_values"],
            showcase["dataset"]["values"],
            showcase["dataset"],
            showcase["scores"],
            showcase["predictions"],
            showcase["metric_row"],
            algorithm_key,
            context_points=context_points,
            save_path=_snapshot_deep_dive_path(
                resolved_session_id,
                str(showcase["metric_row"]["algorithm_run_id"]),
                showcase_dataset,
                write_global=write_global,
            ),
        )
        _close_figure(showcase_fig)

    return {
        "algorithm_key": algorithm_key,
        "comparison_dataset": comparison_dataset,
        "showcase_dataset": showcase_dataset,
    }


def export_thesis_figure_pack(
    notebook_state: dict[str, Any],
    context_points: int = 1200,
    *,
    session_id: str | None = None,
    write_global: bool = True,
) -> dict[str, Any]:
    notebook_state = ensure_notebook_state_benchmark(
        notebook_state,
        refresh_from_saved_run=True,
        persist_tables=True,
        write_global=False,
    )
    ns = notebook_state["ns"]
    config = notebook_state.get("config")
    context = notebook_state.get("context")
    benchmark = notebook_state.get("benchmark")
    if config is None or context is None or benchmark is None:
        raise ValueError("Run the benchmark cells before exporting the thesis figure pack.")

    resolved_session_id = session_id or config.get("session_id")
    ensure_results_layout()
    figure_dir = RESULT_THESIS_FIGURES_DIR if write_global or not resolved_session_id else saved_run_session_thesis_figures_dir(resolved_session_id)
    figure_dir.mkdir(parents=True, exist_ok=True)

    plt = _load_plotting_module()
    results = benchmark["results"]
    metric_spec = _evaluation_metric_spec(results)
    algorithm_summary = benchmark["algorithm_summary"]
    family_summary = benchmark["family_summary"]
    dataset_catalog = benchmark["dataset_catalog"]
    best_config = algorithm_summary.iloc[0] if not algorithm_summary.empty else None
    rows: list[dict[str, Any]] = []

    with plt.rc_context(THESIS_PLOT_STYLE):
        overview_fig = plot_benchmark_overview_panel(benchmark)
        overview_paths = _save_figure_bundle(
            overview_fig,
            _snapshot_thesis_figure_path(
                "benchmark_overview.png",
                resolved_session_id,
                write_global=write_global,
            ),
        )
        _append_thesis_figure_row(
            rows,
            figure_id="benchmark_overview",
            figure_group="overview",
            title="Benchmark overview",
            caption=(
                f"Benchmark overview across {len(dataset_catalog)} benchmarked datasets and {len(config['selected_runs'])} algorithm configurations. "
                f"The top row summarizes dataset coverage in series length and anomaly ratio. The bottom-left panel shows the runtime versus {metric_spec['mean_label']} tradeoff with the Pareto frontier, "
                f"and the bottom-right panel compares mean {metric_spec['label']} and mean ROC AUC across configurations. "
                + (
                    f"In this run, the strongest average configuration was {best_config['algorithm_display']} "
                    f"with mean {metric_spec['label']} {best_config['mean_evaluation_f1']:.3f} and mean ROC AUC {best_config['mean_roc_auc']:.3f}."
                    if best_config is not None
                    else ""
                )
            ),
            saved_paths=overview_paths,
        )
        plt.close(overview_fig)

        pareto_fig = plot_pareto_frontier_panel(benchmark)
        if pareto_fig is not None:
            pareto_paths = _save_figure_bundle(
                pareto_fig,
                _snapshot_thesis_figure_path(
                    "pareto_frontier.png",
                    resolved_session_id,
                    write_global=write_global,
                ),
            )
            _append_thesis_figure_row(
                rows,
                figure_id="pareto_frontier",
                figure_group="overview",
                title="Pareto frontier",
                caption=(
                    f"Pareto frontier for the current benchmark, using mean runtime and mean {metric_spec['label']} as the competing objectives. "
                    "Points on the frontier are the non-dominated configurations that maximize accuracy without being strictly slower and worse than another option."
                ),
                saved_paths=pareto_paths,
            )
            plt.close(pareto_fig)

        metric_heatmap_fig = plot_metric_heatmap_panel(benchmark)
        metric_heatmap_paths = _save_figure_bundle(
            metric_heatmap_fig,
            _snapshot_thesis_figure_path(
                "metric_heatmap.png",
                resolved_session_id,
                write_global=write_global,
            ),
        )
        _append_thesis_figure_row(
            rows,
            figure_id="metric_heatmap",
            figure_group="overview",
            title="Metric heatmap by configuration",
            caption=(
                f"Metric heatmap across all benchmarked configurations. The figure places mean {metric_spec['label']}, ROC AUC, average precision, classical F1, range metrics, and affiliation metrics on one aligned grid "
                "so broad winners and methods with metric-specific strengths are visible at a glance."
            ),
            saved_paths=metric_heatmap_paths,
        )
        plt.close(metric_heatmap_fig)

        family_heatmap_fig = plot_family_evaluation_heatmap_panel(benchmark)
        family_heatmap_paths = _save_figure_bundle(
            family_heatmap_fig,
            _snapshot_thesis_figure_path(
                "family_evaluation_heatmap.png",
                resolved_session_id,
                write_global=write_global,
            ),
        )
        best_family = family_summary.sort_values("mean_evaluation_f1", ascending=False).iloc[0] if not family_summary.empty else None
        _append_thesis_figure_row(
            rows,
            figure_id="family_evaluation_heatmap",
            figure_group="overview",
            title="Evaluation metric by dataset family",
            caption=(
                f"Mean {metric_spec['label']} by dataset family and configuration. This view shows where each method family is strongest instead of averaging away regime structure."
                + (
                    f" The strongest family-specific result in this run was {best_family['algorithm_display']} on {best_family['family']} with mean {metric_spec['label']} {best_family['mean_evaluation_f1']:.3f}."
                    if best_family is not None
                    else ""
                )
            ),
            saved_paths=family_heatmap_paths,
        )
        plt.close(family_heatmap_fig)

        wins_fig = plot_algorithm_wins_panel(benchmark)
        wins_paths = _save_figure_bundle(
            wins_fig,
            _snapshot_thesis_figure_path(
                "algorithm_wins.png",
                resolved_session_id,
                write_global=write_global,
            ),
        )
        wins = _winner_count_frame(benchmark)
        evaluation_winner = wins.sort_values("evaluation_wins", ascending=False).iloc[0] if not wins.empty else None
        _append_thesis_figure_row(
            rows,
            figure_id="algorithm_wins",
            figure_group="overview",
            title="Per-dataset win counts",
            caption=(
                f"Per-dataset win counts for the two most defensible leaderboard views in this notebook: best {metric_spec['label']} and best ROC AUC. "
                + (
                    f"{evaluation_winner['algorithm_display']} won the most datasets by {metric_spec['label']} with {int(evaluation_winner['evaluation_wins'])} wins."
                    if evaluation_winner is not None
                    else ""
                )
            ),
            saved_paths=wins_paths,
        )
        plt.close(wins_fig)

        if config["variant_mode"] == "auto_ablation":
            ablation_fig = plot_ablation_overview_panel(results)
            ablation_overview = build_ablation_overview_table(results)
            if ablation_fig is not None and not ablation_overview.empty:
                ablation_paths = _save_figure_bundle(
                    ablation_fig,
                    _snapshot_thesis_figure_path(
                        "ablation_overview.png",
                        resolved_session_id,
                        write_global=write_global,
                    ),
                )
                strongest_gain = ablation_overview.loc[ablation_overview["mean_delta_evaluation_f1"].idxmax()]
                _append_thesis_figure_row(
                    rows,
                    figure_id="ablation_overview",
                    figure_group="ablation",
                    title="Global one-factor-at-a-time ablation overview",
                    caption=(
                        f"One-factor-at-a-time ablation overview across all enabled algorithms. Each non-baseline variant changes one visible algorithm control while preprocessing, thresholding, and evaluation remain fixed. "
                        "Bar-chart error bars show bootstrap 95% confidence intervals over paired dataset deltas, and the scatter plot separates runtime cost from accuracy shift. "
                        f"The strongest positive shift in this run was {strongest_gain['algorithm']} | {strongest_gain['algorithm_variant']} with mean {metric_spec['label']} delta {strongest_gain['mean_delta_evaluation_f1']:+.3f}."
                    ),
                    saved_paths=ablation_paths,
                )
                plt.close(ablation_fig)

        for algorithm_key in config["selected_algorithms"]:
            subset = results.loc[results["algorithm"] == algorithm_key].copy()
            if subset.empty:
                continue

            display_name = DISPLAY_NAME_MAP[algorithm_key]
            summary = (
                subset.groupby("algorithm_display", as_index=False)
                .agg(
                    dataset_count=("dataset_name", "nunique"),
                    mean_evaluation_f1=("evaluation_f1", "mean"),
                    mean_roc_auc=("roc_auc", "mean"),
                    mean_runtime_seconds=("runtime_seconds", "mean"),
                )
                .sort_values(["mean_evaluation_f1", "mean_roc_auc", "mean_runtime_seconds"], ascending=[False, False, True])
                .reset_index(drop=True)
            )
            best_variant = summary.iloc[0]

            paper_fig = plot_algorithm_paper_panel(results, algorithm_key)
            if paper_fig is not None:
                paper_paths = _save_figure_bundle(
                    paper_fig,
                    _snapshot_thesis_figure_path(
                        f"{algorithm_key}_paper_panel.png",
                        resolved_session_id,
                        write_global=write_global,
                    ),
                )
                _append_thesis_figure_row(
                    rows,
                    figure_id=f"{algorithm_key}_paper_panel",
                    figure_group="algorithm",
                    algorithm=display_name,
                    title=f"{display_name} paper panel",
                    caption=(
                        f"Paper panel for {display_name}. The three heatmaps summarize mean {metric_spec['label']} by dataset variant, series length, and anomaly-ratio regime, while the runtime scatter shows which configured variants convert extra cost into measurable gains. "
                        f"In this run, the strongest {display_name} variant was {best_variant['algorithm_display']} with mean {metric_spec['label']} {best_variant['mean_evaluation_f1']:.3f} and mean ROC AUC {best_variant['mean_roc_auc']:.3f}."
                    ),
                    saved_paths=paper_paths,
                )
                plt.close(paper_fig)

            ablation_impact = build_algorithm_ablation_impact_table(results, algorithm_key)
            ablation_fig = plot_algorithm_ablation_panel(results, algorithm_key)
            if ablation_fig is not None and not ablation_impact.empty:
                ablation_best = ablation_impact.sort_values("mean_delta_evaluation_f1", ascending=False).iloc[0]
                ablation_paths = _save_figure_bundle(
                    ablation_fig,
                    _snapshot_thesis_figure_path(
                        f"{algorithm_key}_ablation_panel.png",
                        resolved_session_id,
                        write_global=write_global,
                    ),
                )
                _append_thesis_figure_row(
                    rows,
                    figure_id=f"{algorithm_key}_ablation_panel",
                    figure_group="ablation",
                    algorithm=display_name,
                    title=f"{display_name} ablation panel",
                    caption=(
                        f"One-factor-at-a-time ablation panel for {display_name}. Horizontal bars report paired deltas against the algorithm baseline, error bars show bootstrap 95% confidence intervals, the runtime scatter isolates cost, and the heatmap shows regime sensitivity by dataset variant. "
                        f"The strongest positive ablation for {display_name} in this run was {ablation_best['algorithm_variant']} with mean {metric_spec['label']} delta {ablation_best['mean_delta_evaluation_f1']:+.3f}."
                    ),
                    saved_paths=ablation_paths,
                )
                plt.close(ablation_fig)

            comparison_payload = build_algorithm_variant_comparison(
                config,
                context["prepared_dataset_dir"],
                results,
                algorithm_key,
            )
            if comparison_payload is not None and len(comparison_payload["variants"]) > 1:
                comparison_fig = plot_algorithm_variant_comparison(
                    comparison_payload,
                    algorithm_key,
                    context_points=context_points,
                )
                if comparison_fig is not None:
                    dataset_name = comparison_payload["dataset"]["dataset_name"]
                    comparison_paths = _save_figure_bundle(
                        comparison_fig,
                        _snapshot_thesis_figure_path(
                            f"{algorithm_key}_variant_comparison_{_slugify_label(dataset_name)}.png",
                            resolved_session_id,
                            write_global=write_global,
                        ),
                    )
                    _append_thesis_figure_row(
                        rows,
                        figure_id=f"{algorithm_key}_variant_comparison",
                        figure_group="algorithm",
                        algorithm=display_name,
                        title=f"{display_name} side-by-side variant comparison",
                        caption=(
                            f"Side-by-side comparison of {len(comparison_payload['variants'])} {display_name} variants on the shared dataset {dataset_name}. "
                            "The top two panels show the raw and normalized signal with the ground-truth anomaly, and the lower panels compare per-variant score traces and threshold crossings on exactly the same time range."
                        ),
                        saved_paths=comparison_paths,
                    )
                    plt.close(comparison_fig)

            showcase = build_algorithm_showcase(
                config,
                context["prepared_dataset_dir"],
                results,
                algorithm_key,
            )
            if showcase is not None:
                deep_dive_fig = plot_algorithm_deep_dive(
                    showcase["raw_values"],
                    showcase["dataset"]["values"],
                    showcase["dataset"],
                    showcase["scores"],
                    showcase["predictions"],
                    showcase["metric_row"],
                    algorithm_key,
                    context_points=context_points,
                )
                showcase_name = showcase["dataset"]["dataset_name"]
                deep_dive_paths = _save_figure_bundle(
                    deep_dive_fig,
                    _snapshot_thesis_figure_path(
                        f"{algorithm_key}_showcase_{_slugify_label(showcase_name)}.png",
                        resolved_session_id,
                        write_global=write_global,
                    ),
                )
                _append_thesis_figure_row(
                    rows,
                    figure_id=f"{algorithm_key}_showcase",
                    figure_group="deep_dive",
                    algorithm=display_name,
                    title=f"{display_name} showcase deep dive",
                    caption=(
                        f"Showcase deep dive for {display_name} on {showcase_name}. The panels align the raw signal, normalized signal, and anomaly score for the strongest selected variant on that dataset, making it possible to show exactly where the detector crosses threshold relative to the ground-truth anomaly interval. "
                        f"The selected showcase variant was {showcase['metric_row']['algorithm_display']} with {metric_spec['label']} {showcase['metric_row']['evaluation_f1']:.3f}, ROC AUC {showcase['metric_row']['roc_auc']:.3f}, and runtime {showcase['metric_row']['runtime_seconds']:.3f}s."
                    ),
                    saved_paths=deep_dive_paths,
                )
                plt.close(deep_dive_fig)

    catalog = pd.DataFrame(rows)
    catalog_path = _snapshot_thesis_catalog_path(
        resolved_session_id,
        write_global=write_global,
    )
    captions_path = _snapshot_thesis_captions_path(
        resolved_session_id,
        write_global=write_global,
    )
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    captions_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(catalog_path, index=False)
    captions_path.write_text(
        build_thesis_figure_caption_markdown(catalog),
        encoding="utf-8",
    )
    return {
        "catalog": catalog,
        "catalog_path": catalog_path,
        "captions_path": captions_path,
        "figure_dir": figure_dir,
    }


def export_saved_run_snapshot_artifacts(
    run_name_or_session_id: str | None = None,
    *,
    include_algorithm_sections: bool = True,
    include_thesis_pack: bool = False,
    write_global: bool = False,
    context_points: int = 1200,
) -> dict[str, Any]:
    notebook_state = load_saved_run_notebook_state(
        run_name_or_session_id,
        persist_tables=True,
        write_global=write_global,
    )
    config = notebook_state["config"]
    benchmark = notebook_state["benchmark"]
    session_id = str(config["session_id"])

    overview_fig = plot_benchmark_overview_panel(
        benchmark,
        _snapshot_figure_path(
            "benchmark_overview.png",
            session_id,
            write_global=write_global,
        ),
    )
    _close_figure(overview_fig)

    pareto_fig = plot_pareto_frontier_panel(
        benchmark,
        _snapshot_figure_path(
            "pareto_frontier.png",
            session_id,
            write_global=write_global,
        ),
    )
    _close_figure(pareto_fig)

    metric_heatmap_fig = plot_metric_heatmap_panel(
        benchmark,
        _snapshot_figure_path(
            "metric_heatmap.png",
            session_id,
            write_global=write_global,
        ),
    )
    _close_figure(metric_heatmap_fig)

    family_heatmap_fig = plot_family_evaluation_heatmap_panel(
        benchmark,
        _snapshot_figure_path(
            "family_evaluation_heatmap.png",
            session_id,
            write_global=write_global,
        ),
    )
    _close_figure(family_heatmap_fig)

    wins_fig = plot_algorithm_wins_panel(
        benchmark,
        _snapshot_figure_path(
            "algorithm_wins.png",
            session_id,
            write_global=write_global,
        ),
    )
    _close_figure(wins_fig)

    ablation_fig = plot_ablation_overview_panel(
        benchmark["results"],
        _snapshot_figure_path(
            "ablation_overview.png",
            session_id,
            write_global=write_global,
        ),
    )
    _close_figure(ablation_fig)

    exported_algorithms: list[dict[str, Any]] = []
    if include_algorithm_sections:
        for algorithm_key in config["selected_algorithms"]:
            exported_algorithms.append(
                export_algorithm_section_artifacts(
                    notebook_state,
                    algorithm_key,
                    session_id=session_id,
                    write_global=write_global,
                    context_points=context_points,
                )
            )

    thesis_payload = None
    if include_thesis_pack:
        thesis_payload = export_thesis_figure_pack(
            notebook_state,
            context_points=context_points,
            session_id=session_id,
            write_global=write_global,
        )

    return {
        "run_name": config["run_name"],
        "session_id": session_id,
        "artifact_root": RESULTS_DIR if write_global else saved_run_session_dir(session_id),
        "completed_dataset_count": benchmark["completed_dataset_count"],
        "selected_dataset_count": benchmark["selected_dataset_count"],
        "pending_dataset_count": benchmark["pending_dataset_count"],
        "is_partial": benchmark["is_partial"],
        "exported_algorithms": exported_algorithms,
        "thesis_payload": thesis_payload,
        "snapshot_extracted_at": notebook_state["snapshot_extracted_at"],
    }


def render_ablation_overview(notebook_state: dict[str, Any]) -> None:
    notebook_state = ensure_notebook_state_benchmark(
        notebook_state,
        refresh_from_saved_run=True,
        persist_tables=True,
        write_global=False,
    )
    ns = notebook_state["ns"]
    config = notebook_state["config"]
    benchmark = notebook_state["benchmark"]
    results = benchmark["results"]

    if config["variant_mode"] != "auto_ablation":
        print("Ablation overview is most meaningful when Argument mode is set to auto_ablation.")
        return

    overview = ns.build_ablation_overview_table(results)
    if overview.empty:
        print("No ablation deltas are available in the current run.")
        return

    display(HTML(ns.build_ablation_overview_narrative(results)))
    interpretation = HTML(
        "<div style='margin:6px 0 10px 0; line-height:1.45;'>"
        "<p style='margin:0 0 8px 0;'><b>Ablation interpretation guide</b></p>"
        "<ul style='margin:0 0 0 18px; padding:0;'>"
        "<li>Every non-baseline row changes one visible algorithm argument while holding preprocessing, thresholding, and evaluation fixed.</li>"
        "<li>All deltas are paired against the matching baseline on the same dataset, so positive values mean the single knob change helped on average.</li>"
        "<li><b>Mean runtime ratio</b> above 1.0 means the ablation is slower than baseline; below 1.0 means it is faster.</li>"
        "<li>Backend/runtime ablations are especially useful when you need to show that a control changes cost more than accuracy.</li>"
        "</ul>"
        "</div>"
    )
    display(interpretation)
    display(overview)

    fig = ns.plot_ablation_overview_panel(
        results,
        save_path=ns.result_figure_path("ablation_overview.png"),
    )
    if fig is not None:
        plt = ns._load_plotting_module()
        plt.show()


def render_algorithm_report(notebook_state: dict[str, Any], algorithm_key: str, context_points: int = 1200) -> None:
    notebook_state = ensure_notebook_state_benchmark(
        notebook_state,
        refresh_from_saved_run=True,
        persist_tables=True,
        write_global=False,
    )
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
    ablation_impact = ns.build_algorithm_ablation_impact_table(
        results, algorithm_key)
    ablation_dataset_variant = ns.build_algorithm_ablation_regime_table(
        results, algorithm_key, "variant")
    ablation_length = ns.build_algorithm_ablation_regime_table(
        results, algorithm_key, "length_bucket")
    ablation_narrative = ns.build_algorithm_ablation_narrative(
        results, algorithm_key)
    report_narrative = ns.build_algorithm_report_narrative(
        results, algorithm_key)

    display(HTML(report_narrative))
    display(variant_config)
    display(parameter_effects)
    display(dataset_variant_summary)
    display(length_summary)
    display(anomaly_ratio_summary)
    display(section_summary)
    display(section_top)
    if not ablation_impact.empty:
        display(HTML(ablation_narrative))
        display(ablation_impact)
        display(ablation_dataset_variant)
        display(ablation_length)

    parameter_effects.to_csv(ns.result_table_path(
        f"{algorithm_key}_parameter_effects.csv"), index=False)
    dataset_variant_summary.to_csv(ns.result_table_path(
        f"{algorithm_key}_dataset_variant_summary.csv"), index=False)
    length_summary.to_csv(ns.result_table_path(
        f"{algorithm_key}_length_summary.csv"), index=False)
    anomaly_ratio_summary.to_csv(ns.result_table_path(
        f"{algorithm_key}_anomaly_ratio_summary.csv"), index=False)
    if not ablation_impact.empty:
        ablation_impact.to_csv(ns.result_table_path(
            f"{algorithm_key}_ablation_impacts.csv"), index=False)
        ablation_dataset_variant.to_csv(ns.result_table_path(
            f"{algorithm_key}_ablation_dataset_variant_summary.csv"), index=False)
        ablation_length.to_csv(ns.result_table_path(
            f"{algorithm_key}_ablation_length_summary.csv"), index=False)

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

    ablation_fig = ns.plot_algorithm_ablation_panel(
        results, algorithm_key, ns.result_algorithm_ablation_panel_path(algorithm_key))
    if ablation_fig is not None:
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
        if config["variant_mode"] != "manual":
            preset = config["variant_mode"]
            supported = ", ".join(
                DISPLAY_NAME_MAP[key]
                for key in PAPER_PRESET_DEFINITIONS[preset]["enabled_algorithms"]
            )
            raise ValueError(
                f"No enabled algorithms have auto variants under '{preset}'. "
                f"Enable one of: {supported}, or switch Argument mode back to manual."
            )
        raise ValueError(
            "Select at least one algorithm in the control panel before running the notebook.")

    notes_info = write_high_roi_algorithm_notes()
    raw_dataset_paths = ensure_raw_datasets_available()
    prepared_dataset_dir, prepared_dataset_paths = ensure_normalized_datasets(
        config["normalization_method"],
        config["clip_quantile"],
        overwrite=config["overwrite_normalized_datasets"],
    )
    prepared_dataset_paths = sorted(
        prepared_dataset_paths, key=lambda path: (path.stat().st_size, path.name))
    selected_dataset_paths = list(
        prepared_dataset_paths[: config["dataset_limit"]]
        if config["dataset_limit"] is not None
        else prepared_dataset_paths
    )
    selected_run_parameters = build_selected_run_parameter_frame(config)
    session_results = _read_resume_table_if_exists(
        "benchmark_results.csv",
        config["session_id"],
    )
    if session_results is not None and not session_results.empty:
        _validate_resume_compatibility(
            selected_run_parameters,
            session_id=config["session_id"],
        )
    _write_saved_run_control_state(config["session_id"], config["control_state"])

    existing_results = pd.DataFrame()
    completed_dataset_names: set[str] = set()
    if config["resume_from_existing"]:
        _validate_resume_compatibility(
            selected_run_parameters,
            session_id=config["session_id"],
        )
        existing_results = _read_resume_table_if_exists(
            "benchmark_results.csv",
            config["session_id"],
        )
        if existing_results is None:
            existing_results = pd.DataFrame()
        completed_dataset_names = _completed_dataset_names(
            existing_results,
            config["selected_runs"],
            prepared_dataset_dir,
            selected_dataset_paths,
            config,
        )

    pending_dataset_paths = [
        path for path in selected_dataset_paths if path.stem not in completed_dataset_names
    ]
    batch_source_paths = pending_dataset_paths if config["resume_from_existing"] else selected_dataset_paths
    benchmark_dataset_paths = list(
        batch_source_paths[: config["batch_size"]]
        if config["batch_size"] is not None
        else batch_source_paths
    )
    current_batch_names = [path.stem for path in benchmark_dataset_paths]

    run_config_frame = pd.DataFrame(
        [
            {
                "dataset_limit": "all" if config["dataset_limit"] is None else config["dataset_limit"],
                "batch_size": "all selected" if config["batch_size"] is None else config["batch_size"],
                "resume_from_existing": config["resume_from_existing"],
                "variant_mode": config["variant_mode"],
                "auto_preset_name": config["auto_preset_name"],
                "run_name": config["run_name"],
                "session_id": config["session_id"],
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
                "auto_filtered_out": ", ".join(DISPLAY_NAME_MAP[key] for key in config["auto_filtered_out"]) or None,
                "selected_dataset_count": len(selected_dataset_paths),
                "pending_dataset_count": len(pending_dataset_paths),
                "planned_batch_count": len(benchmark_dataset_paths),
                "next_batch_first_dataset": current_batch_names[0] if current_batch_names else None,
                "next_batch_last_dataset": current_batch_names[-1] if current_batch_names else None,
                "prepared_dataset_dir": portable_path_str(prepared_dataset_dir),
                "notes_source_path": notes_info["source_path"],
                "notes_results_path": notes_info["results_path"],
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
                "selected_dataset_count": len(selected_dataset_paths),
                "batch_size": "all selected" if config["batch_size"] is None else config["batch_size"],
                "resume_from_existing": config["resume_from_existing"],
                "completed_dataset_count": len(completed_dataset_names),
                "pending_dataset_count": len(pending_dataset_paths),
                "current_batch_count": len(benchmark_dataset_paths),
                "next_batch_first_dataset": current_batch_names[0] if current_batch_names else None,
                "next_batch_last_dataset": current_batch_names[-1] if current_batch_names else None,
                "variant_mode": config["variant_mode"],
                "auto_preset_name": config["auto_preset_name"],
                "run_name": config["run_name"],
                "session_id": config["session_id"],
                "normalization_method": config["normalization_method"],
                "clip_quantile": config["clip_quantile"],
                "window_size": config["window_size"],
                "window_stride": config["window_stride"],
                "threshold_method": config["threshold_method"],
                "threshold_value": config["threshold_value"],
                "evaluation_mode": config["evaluation_mode"],
                "normalized_dataset_dir": portable_path_str(prepared_dataset_dir),
                "notes_source_path": notes_info["source_path"],
                "notes_results_path": notes_info["results_path"],
            }
        ]
    )

    _write_table_artifact(
        run_config_frame,
        "run_configuration.csv",
        config["session_id"],
    )
    _write_table_artifact(
        selected_run_parameters,
        "selected_run_parameters.csv",
        config["session_id"],
    )
    _write_table_artifact(
        preparation_summary,
        "dataset_preparation_summary.csv",
        config["session_id"],
    )
    update_saved_run_manifest(
        config,
        status="resume_ready" if completed_dataset_names else "prepared",
        selected_dataset_count=len(selected_dataset_paths),
        completed_dataset_count=len(completed_dataset_names),
        pending_dataset_count=len(pending_dataset_paths),
        planned_batch_count=len(benchmark_dataset_paths),
        current_batch_count=len(benchmark_dataset_paths),
        next_batch_first_dataset=current_batch_names[0] if current_batch_names else None,
        next_batch_last_dataset=current_batch_names[-1] if current_batch_names else None,
        existing_result_row_count=len(existing_results),
    )

    return {
        "raw_dataset_paths": raw_dataset_paths,
        "prepared_dataset_dir": prepared_dataset_dir,
        "prepared_dataset_paths": prepared_dataset_paths,
        "selected_dataset_paths": selected_dataset_paths,
        "benchmark_dataset_paths": benchmark_dataset_paths,
        "run_config_frame": run_config_frame,
        "selected_run_parameters": selected_run_parameters,
        "preparation_summary": preparation_summary,
        "notes_info": notes_info,
        "completed_dataset_names": sorted(completed_dataset_names),
        "pending_dataset_names": [path.stem for path in pending_dataset_paths],
    }


def run_benchmark(
    config: dict[str, Any],
    prepared_dataset_dir: Path,
    benchmark_dataset_paths: list[Path],
    show_progress: bool = True,
) -> dict[str, Any]:
    records = []
    checkpoint_results = pd.DataFrame()
    selected_runs = config["selected_runs"]
    total_dataset_count = len(benchmark_dataset_paths)
    total_run_count = total_dataset_count * len(selected_runs)
    benchmark_started_at = time.perf_counter()
    recent_messages: list[str] = []
    existing_results = pd.DataFrame()
    completed_run_keys: set[str] = set()
    prepared_dataset_dir_value = portable_path_str(prepared_dataset_dir)
    executed_runs, skipped_existing_runs = _recover_saved_run_progress_counts(
        config["session_id"],
    )
    completed_runs = 0
    already_accounted_run_count = 0

    if config.get("resume_from_existing"):
        _validate_resume_compatibility(
            build_selected_run_parameter_frame(config),
            session_id=config["session_id"],
        )
        existing_results = _read_resume_table_if_exists(
            "benchmark_results.csv",
            config["session_id"],
        )
        if existing_results is None:
            existing_results = pd.DataFrame()
        successful_existing_results = _successful_results_frame(existing_results)
        if not successful_existing_results.empty:
            completed_run_keys = set(_result_run_key_series(successful_existing_results))
    already_accounted_run_count = _count_completed_run_prefix(
        benchmark_dataset_paths,
        selected_runs,
        completed_run_keys,
        config,
        prepared_dataset_dir_value,
    )
    completed_runs = already_accounted_run_count
    checkpoint_results = existing_results.copy()
    update_saved_run_manifest(
        config,
        status="running",
        batch_dataset_count=total_dataset_count,
        executed_run_count=executed_runs,
        skipped_existing_run_count=skipped_existing_runs,
        existing_result_row_count=len(existing_results),
    )

    progress_bar = None
    progress_summary = None
    progress_status = None
    progress_recent = None

    if show_progress:
        progress_bar = widgets.IntProgress(
            value=completed_runs,
            min=0,
            max=max(total_run_count, 1),
            description="Runs",
            bar_style="info",
            layout=widgets.Layout(width="100%"),
        )
        remaining_checks = max(total_run_count - completed_runs, 0)
        progress_summary = _progress_html(
            (
                f"<b>Resume</b>: checked {completed_runs}/{total_run_count} | "
                f"<b>Executed</b>: {executed_runs} | <b>Skipped</b>: {skipped_existing_runs} | "
                f"<b>Remaining</b>: {remaining_checks}"
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
            run_check_index = ((dataset_index - 1) * len(selected_runs)) + algorithm_index
            algorithm_key = run_config["algorithm"]
            current_run_key = _result_run_key(
                dataset_name=dataset["dataset_name"],
                algorithm_run_id=run_config["algorithm_run_id"],
                normalization_method=config["normalization_method"],
                threshold_method=str(config["threshold_method"]),
                threshold_value=float(config["threshold_value"]),
                evaluation_mode=str(config["evaluation_mode"]),
                window_size=window_size,
                window_stride=window_stride,
                prepared_dataset_dir=prepared_dataset_dir_value,
            )
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
                        f"<b>Checked</b>: {completed_runs}/{total_run_count} | "
                        f"<b>Executed</b>: {executed_runs} | <b>Skipped</b>: {skipped_existing_runs} | "
                    f"<b>Elapsed</b>: {_format_duration(elapsed_seconds)}"
                    f"{sand_note}"
                    "</div>"
                )
            already_accounted = run_check_index <= already_accounted_run_count
            if current_run_key in completed_run_keys:
                if already_accounted:
                    continue
                completed_runs += 1
                skipped_existing_runs += 1

                completed_dataset_count = (
                    checkpoint_results.loc[checkpoint_results["error"].fillna("").astype(str) == "", "dataset_name"].nunique()
                    if not checkpoint_results.empty and "dataset_name" in checkpoint_results.columns
                    else 0
                )
                selected_dataset_count = int(
                    (_read_json_if_exists(saved_run_session_manifest_path(config["session_id"])) or {}).get(
                        "selected_dataset_count",
                        total_dataset_count,
                    )
                    or total_dataset_count
                )
                update_saved_run_manifest(
                    config,
                    status="running",
                    batch_dataset_count=total_dataset_count,
                    executed_run_count=executed_runs,
                    skipped_existing_run_count=skipped_existing_runs,
                    selected_dataset_count=selected_dataset_count,
                    completed_dataset_count=completed_dataset_count,
                    pending_dataset_count=max(selected_dataset_count - completed_dataset_count, 0),
                    last_dataset_name=dataset["dataset_name"],
                    last_algorithm_display=run_config["algorithm_display"],
                    error_count=int(checkpoint_results["error"].fillna("").astype(str).ne("").sum()) if "error" in checkpoint_results.columns else 0,
                    result_row_count=len(checkpoint_results),
                )

                if show_progress and progress_bar is not None and progress_summary is not None and progress_recent is not None:
                    elapsed_seconds = time.perf_counter() - benchmark_started_at
                    average_seconds_per_check = elapsed_seconds / max(completed_runs, 1)
                    remaining_checks = max(total_run_count - completed_runs, 0)
                    eta_seconds = average_seconds_per_check * remaining_checks
                    progress_bar.value = completed_runs
                    progress_summary.value = (
                        "<div style='white-space: normal; overflow-wrap: anywhere; word-break: break-word; max-width: 100%; line-height: 1.45;'>"
                        f"<b>Progress</b>: checked {completed_runs}/{total_run_count} | "
                        f"<b>Executed</b>: {executed_runs} | <b>Skipped</b>: {skipped_existing_runs} | "
                        f"<b>Datasets</b>: {dataset_index}/{total_dataset_count} | "
                        f"<b>Elapsed</b>: {_format_duration(elapsed_seconds)} | "
                        f"<b>ETA</b>: {_format_duration(eta_seconds)}"
                        "</div>"
                    )
                    recent_line = (
                        f"{completed_runs}/{total_run_count} | "
                        f"{dataset['dataset_name']} | "
                        f"{run_config['algorithm_display']} | skipped existing"
                    )
                    recent_messages = [recent_line] + recent_messages[:4]
                    progress_recent.value = (
                        "<div style='white-space: normal; overflow-wrap: anywhere; word-break: break-word; max-width: 100%; line-height: 1.45;'>"
                        "<b>Recent runs</b><br>"
                        + "<br>".join(html.escape(line)
                                      for line in recent_messages)
                        + "</div>"
                    )
                continue
            if already_accounted:
                already_accounted_run_count = run_check_index - 1
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
                    "variant_origin": run_config["variant_origin"],
                    "variant_source": run_config["variant_source"],
                    "variant_focus": run_config["variant_focus"],
                    "variant_family": run_config["variant_family"],
                    "ablation_parameter": run_config["ablation_parameter"],
                    "ablation_label": run_config["ablation_label"],
                    "ablation_role": run_config["ablation_role"],
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
                    "prepared_dataset_dir": prepared_dataset_dir_value,
                    "runtime_seconds": runtime_seconds,
                    "score_mean": float(scores.mean()) if scores.size else float("nan"),
                    "score_std": float(scores.std()) if scores.size else float("nan"),
                    "params_json": json.dumps(run_config["params"], sort_keys=True),
                    "error": error_message,
                    **{f"param__{key}": value for key, value in run_config["params"].items()},
                    **metrics,
                }
            )
            checkpoint_results = _merge_benchmark_results(
                checkpoint_results,
                pd.DataFrame.from_records([records[-1]]),
            )
            _write_benchmark_checkpoint(checkpoint_results, config["session_id"])
            completed_runs += 1
            executed_runs += 1
            completed_dataset_count = (
                checkpoint_results.loc[checkpoint_results["error"].fillna("").astype(str) == "", "dataset_name"].nunique()
                if not checkpoint_results.empty and "dataset_name" in checkpoint_results.columns
                else 0
            )
            selected_dataset_count = int(
                (_read_json_if_exists(saved_run_session_manifest_path(config["session_id"])) or {}).get(
                    "selected_dataset_count",
                    total_dataset_count,
                )
                or total_dataset_count
            )
            update_saved_run_manifest(
                config,
                status="running",
                batch_dataset_count=total_dataset_count,
                executed_run_count=executed_runs,
                skipped_existing_run_count=skipped_existing_runs,
                selected_dataset_count=selected_dataset_count,
                completed_dataset_count=completed_dataset_count,
                pending_dataset_count=max(selected_dataset_count - completed_dataset_count, 0),
                last_dataset_name=dataset["dataset_name"],
                last_algorithm_display=run_config["algorithm_display"],
                error_count=int(checkpoint_results["error"].fillna("").astype(str).ne("").sum()) if "error" in checkpoint_results.columns else 0,
                result_row_count=len(checkpoint_results),
            )

            if show_progress and progress_bar is not None and progress_summary is not None and progress_recent is not None:
                elapsed_seconds = time.perf_counter() - benchmark_started_at
                average_seconds_per_check = elapsed_seconds / \
                    max(completed_runs, 1)
                remaining_runs = max(total_run_count - completed_runs, 0)
                eta_seconds = average_seconds_per_check * remaining_runs
                progress_bar.value = completed_runs
                progress_summary.value = (
                    "<div style='white-space: normal; overflow-wrap: anywhere; word-break: break-word; max-width: 100%; line-height: 1.45;'>"
                    f"<b>Progress</b>: checked {completed_runs}/{total_run_count} | "
                    f"<b>Executed</b>: {executed_runs} | <b>Skipped</b>: {skipped_existing_runs} | "
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

    batch_results = pd.DataFrame.from_records(records)
    results = checkpoint_results if not checkpoint_results.empty else _merge_benchmark_results(existing_results, batch_results)
    if results.empty:
        raise ValueError(
            "No benchmark runs are available for execution. Check the selected algorithms and batch settings."
        )
    dataset_catalog = build_dataset_catalog(results)
    algorithm_summary = summarize_algorithms(results)
    family_summary = summarize_families(results)
    overall_regime_summary = build_overall_regime_summary(results)
    best_by_evaluation = build_best_algorithm_table(results, "evaluation_f1")
    best_by_f1 = build_best_algorithm_table(results, "f1")
    best_by_auc = build_best_algorithm_table(results, "roc_auc")
    errors = results.loc[results["error"] != ""].copy()

    _write_benchmark_checkpoint(results, config["session_id"])
    _write_table_artifact(dataset_catalog, "dataset_catalog.csv", config["session_id"])
    _write_table_artifact(algorithm_summary, "algorithm_summary.csv", config["session_id"])
    _write_table_artifact(family_summary, "family_summary.csv", config["session_id"])
    _write_table_artifact(overall_regime_summary, "overall_regime_summary.csv", config["session_id"])
    _write_table_artifact(best_by_evaluation, "best_algorithm_by_dataset_evaluation.csv", config["session_id"])
    _write_table_artifact(best_by_f1, "best_algorithm_by_dataset_f1.csv", config["session_id"])
    _write_table_artifact(best_by_auc, "best_algorithm_by_dataset_auc.csv", config["session_id"])
    _write_table_artifact(errors, "error_report.csv", config["session_id"])

    for algorithm_key in config["selected_algorithms"]:
        algorithm_results = results.loc[results["algorithm"] == algorithm_key]
        algorithm_results.to_csv(result_per_algorithm_table_path(algorithm_key), index=False)
        session_algorithm_table = saved_run_session_table_path(
            config["session_id"],
            f"per_algorithm/{algorithm_key}_results.csv",
        )
        session_algorithm_table.parent.mkdir(parents=True, exist_ok=True)
        algorithm_results.to_csv(session_algorithm_table, index=False)

    overview = pd.DataFrame(
        [
            {
                "dataset_count": dataset_catalog["dataset_name"].nunique(),
                "algorithm_count": len(config["selected_algorithms"]),
                "configuration_count": len(selected_runs),
                "run_count": len(results),
                "batch_dataset_count": total_dataset_count,
                "run_name": config["run_name"],
                "session_id": config["session_id"],
                "executed_run_count": executed_runs,
                "skipped_existing_run_count": skipped_existing_runs,
                "checked_run_count": completed_runs,
                "resume_from_existing": bool(config.get("resume_from_existing")),
                "batch_size": "all selected" if config.get("batch_size") is None else config["batch_size"],
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
            f"<b>Benchmark complete</b>: checked {completed_runs}/{total_run_count} | "
            f"<b>Executed</b>: {executed_runs} | <b>Skipped</b>: {skipped_existing_runs} | "
            f"<b>Total time</b>: {_format_duration(total_elapsed)}"
            "</div>"
        )
        progress_status.value = (
            "<div style='white-space: normal; overflow-wrap: anywhere; word-break: break-word; max-width: 100%; line-height: 1.45;'>"
            f"<b>Finished</b>: {dataset_catalog['dataset_name'].nunique()} datasets processed | "
            f"{len(errors)} runs with errors"
            "</div>"
        )

    update_saved_run_manifest(
        config,
        status="complete_with_errors" if not errors.empty else "complete",
        batch_dataset_count=total_dataset_count,
        selected_dataset_count=dataset_catalog["dataset_name"].nunique(),
        completed_dataset_count=dataset_catalog["dataset_name"].nunique(),
        pending_dataset_count=0,
        executed_run_count=executed_runs,
        skipped_existing_run_count=skipped_existing_runs,
        result_row_count=len(results),
        error_count=len(errors),
        current_batch_count=total_dataset_count,
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
        "batch_results": batch_results,
        "executed_run_count": executed_runs,
        "skipped_existing_run_count": skipped_existing_runs,
    }
