from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


NOTEBOOK_PATH = Path(__file__).resolve().parent / "run_anomaly_detection.ipynb"
ALGORITHM_SECTIONS = [
    ("isolation_forest", "Isolation Forest"),
    ("local_outlier_factor", "Local Outlier Factor"),
    ("sand", "SAND"),
    ("matrix_profile", "Matrix Profile"),
    ("damp", "DAMP"),
    ("hbos", "HBOS"),
    ("ocsvm", "OCSVM"),
    ("pca", "PCA"),
]


def lines(text: str) -> list[str]:
    return dedent(text).strip().splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines(text)}


def _algorithm_cells(algorithm_key: str, display_name: str) -> list[dict]:
    return [
        markdown(
            f"""
            ## On Run: Show {display_name} Paper Report

            This next cell renders the paper-facing report for {display_name}: configured variants, parameter-effect tables, regime-aware summaries by dataset type, the benchmark panel, the paper panel, and the best showcase plot.
            """
        ),
        code(
            f"""
            state = NOTEBOOK_STATE
            state["ns"].render_algorithm_report(state, "{algorithm_key}")
            """
        ),
    ]


def build_notebook() -> dict:
    cells: list[dict] = [
        markdown(
            """
            # Simple Anomaly Detection Analysis

            This notebook is organized for paper-facing experiments rather than one-off benchmarking.

            Main ideas:

            - All cross-cell state stays inside `NOTEBOOK_STATE`.
            - The paper presets create multiple named variants per algorithm so the notebook can show parameter effects instead of a single baseline.
            - Paper-facing sweeps vary only score-driving parameters; backend-only or threshold-only knobs stay fixed inside the presets.
            - Benchmark outputs are saved into `results/tables/`, `results/figures/`, and `results/scores/`.

            Default workflow:

            1. Run the dependency and setup cells.
            2. Optionally apply `paper_high_roi` or `paper_full_suite`.
            3. Run the preparation and benchmark cells.
            4. Inspect the regime summary and winner tables.
            5. Open the per-algorithm report cells for the paper figures.
            """
        ),
        markdown(
            """
            ## Control Reference

            **General**
            - `Dataset limit`: how many prepared datasets to benchmark. `0` means all datasets.
            - `Normalize`: preprocessing applied before any algorithm runs.
            - `Clip q`: optional quantile clipping before normalization.
            - `Window size`: base temporal context length for subsequence-aware processing. `0` keeps automatic estimation.
            - `Window stride`: step between consecutive windows or subsequences. Larger strides reduce overlap and runtime.
            - `Threshold method` with `Threshold value`: turns continuous scores into binary detections for the paper metrics.
            - `Evaluation mode`: `range` emphasizes overlap with anomaly intervals, while `point` is stricter and judges exact indices.
            - `Rebuild normalized datasets`: regenerates cached prepared datasets.
            - `Save per-dataset scores`: writes raw score traces to `results/scores/`.

            **Paper sweep rules**
            - `paper_high_roi`: short, paper-friendly sweep on the highest-return methods plus Isolation Forest for calibration discussion.
            - `paper_full_suite`: broader appendix-ready sweep across every implemented method.
            - Presets keep runtime-only knobs fixed and vary only parameters that materially change each algorithm's scoring behavior in this framework.

            **Algorithm sweep highlights**
            - Isolation Forest: `Trees`, `Max samples`, `Max feat.`, `Bootstrap`.
            - Local Outlier Factor: `Neighbors`, `Metric`, `p`.
            - SAND: `Alpha`, `Init length`, `Batch size`, `k`, `Subseq x`, `Overlap`.
            - Matrix Profile: `Subseq x`.
            - DAMP: `Start x`, `x_lag x`.
            - HBOS: `Bins`, `Alpha`, `Tol`.
            - OCSVM: `Kernel`, `Nu`, `Gamma`, `Train frac`.
            - PCA: `Components`, `Score comps`, `Whiten`, `Weighted`, `Standardize`.
            """
        ),
        markdown(
            """
            ## On Run: Check Whether Notebook Dependencies Are Installed And Install Any Missing Ones

            This next cell checks for `ipywidgets`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `stumpy`, and `TSB-UAD`, then installs any missing package into the current Python environment.
            """
        ),
        code(
            """
            import importlib.util
            import subprocess
            import sys

            required = {
                "ipywidgets": "ipywidgets",
                "numpy": "numpy",
                "pandas": "pandas",
                "matplotlib": "matplotlib",
                "sklearn": "scikit-learn",
                "stumpy": "stumpy",
                "TSB_UAD": "TSB-UAD",
            }
            missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
            if missing:
                print("Installing missing packages:", missing)
                subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            else:
                print("Notebook dependencies are available.")
            """
        ),
        markdown(
            """
            ## On Run: Locate The Project, Reload Notebook Support, And Render The Interactive Control Panel

            This next cell finds the `simple_anomaly_detection` folder, reloads the support module, loads dataset names, renders the control panel, and initializes `NOTEBOOK_STATE`.
            """
        ),
        code(
            """
            import importlib
            from pathlib import Path
            import sys

            import pandas as pd
            from IPython.display import display

            search_bases = [Path.cwd(), *Path.cwd().parents]
            candidate_roots = []
            for base in search_bases:
                candidate_roots.extend(
                    [
                        base,
                        base / "simple_anomaly_detection",
                        base / "python" / "simple_anomaly_detection",
                    ]
                )
            project_root = next((root for root in candidate_roots if (root / "notebook_support.py").exists()), None)
            if project_root is None:
                raise FileNotFoundError("Could not locate the simple_anomaly_detection project root.")

            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            import notebook_support as ns
            ns = importlib.reload(ns)

            dataset_name_source = ns.RAW_DATASET_DIR if any(ns.RAW_DATASET_DIR.glob("*.txt")) else ns.LEGACY_VIRGIN_DIR
            dataset_names = [path.stem for path in sorted(dataset_name_source.glob("*.txt"))]

            panel_bundle = ns.build_control_panel(dataset_names)
            NOTEBOOK_STATE = {
                "project_root": project_root,
                "ns": ns,
                "controls": panel_bundle["controls"],
                "panel_bundle": panel_bundle,
                "config": None,
                "context": None,
                "benchmark": None,
            }

            display(panel_bundle["panel"])
            display(ns.build_results_layout_frame())
            display(ns.list_paper_presets())
            print(f"Project root: {ns.PROJECT_ROOT}")
            print(f"Legacy virgin source: {ns.LEGACY_VIRGIN_DIR}")
            print(f"Results root: {ns.RESULTS_DIR}")
            print(f"Results tables: {ns.RESULT_TABLES_DIR}")
            print(f"Results figures: {ns.RESULT_FIGURES_DIR}")
            print(f"Score traces: {ns.RESULT_SCORES_DIR}")
            """
        ),
        markdown(
            """
            ## Optional: Apply A Reproducible Paper Preset

            Edit `preset_name` below if you want a different paper experiment layout, then rerun the preparation and benchmark cells.
            """
        ),
        code(
            """
            preset_name = "paper_high_roi"
            state = NOTEBOOK_STATE
            state["ns"].apply_paper_experiment_preset(state["controls"], preset_name)
            display(state["ns"].build_preset_reference_table(preset_name))
            print(f"Applied preset: {preset_name}")
            """
        ),
        markdown(
            """
            ## On Run: Read The Current Widget Values, Prepare Raw Datasets, And Build Normalized Benchmark Inputs

            This next cell reads the current control-panel settings, ensures the raw datasets exist in the workspace, generates normalized labeled datasets if needed, and shows the exact paper run configuration.
            """
        ),
        code(
            """
            state = NOTEBOOK_STATE
            state["config"] = state["ns"].get_run_config(state["controls"])
            state["context"] = state["ns"].prepare_run_context(state["config"])

            display(state["context"]["run_config_frame"])
            display(state["context"]["selected_run_parameters"])
            display(state["context"]["preparation_summary"])
            display(pd.DataFrame({"sample_raw_dataset_file": [path.name for path in state["context"]["raw_dataset_paths"][:10]]}))
            print(f"Prepared datasets will be loaded from: {state['context']['prepared_dataset_dir']}")
            print(f"Benchmark dataset count: {len(state['context']['benchmark_dataset_paths'])}")
            """
        ),
        markdown(
            """
            ## On Run: Execute Every Selected Algorithm Configuration Across The Prepared Datasets

            This next cell refreshes the configuration from the widgets, runs every selected algorithm variant, writes the benchmark outputs into `results/tables/`, and stores all benchmark objects inside `NOTEBOOK_STATE["benchmark"]`.
            """
        ),
        code(
            """
            state = NOTEBOOK_STATE
            state["config"] = state["ns"].get_run_config(state["controls"])
            state["context"] = state["ns"].prepare_run_context(state["config"])
            state["benchmark"] = state["ns"].run_benchmark(
                state["config"],
                state["context"]["prepared_dataset_dir"],
                state["context"]["benchmark_dataset_paths"],
            )

            display(state["benchmark"]["overview"])
            display(state["benchmark"]["algorithm_summary"])
            display(state["benchmark"]["dataset_catalog"].head(12))
            if not state["benchmark"]["errors"].empty:
                display(state["benchmark"]["errors"][["dataset_name", "algorithm_display", "error"]])

            print(f"Saved benchmark outputs to: {state['ns'].RESULT_TABLES_DIR}")
            """
        ),
        markdown(
            """
            ## On Run: Show The Paper-Oriented Regime Summaries And Winners

            This next cell displays the family summary, the regime-aware summary table, the best configuration by the selected evaluation metric, and the ROC-AUC winners table.
            """
        ),
        code(
            """
            state = NOTEBOOK_STATE
            display(state["benchmark"]["family_summary"])
            display(state["benchmark"]["overall_regime_summary"])
            display(state["benchmark"]["best_by_evaluation"].head(15))
            display(state["benchmark"]["best_by_auc"].head(15))
            """
        ),
    ]

    for algorithm_key, display_name in ALGORITHM_SECTIONS:
        cells.extend(_algorithm_cells(algorithm_key, display_name))

    cells.extend(
        [
            markdown(
                """
                ## On Run: Build The Cross-Configuration Comparison Charts And Save The Final Overview Images

                This next cell builds the overall comparison charts across all selected configurations, saves the overview images into `results/figures/`, and displays the winner tables together with the regime-aware benchmark table.
                """
            ),
            code(
                """
                import matplotlib.pyplot as plt
                import numpy as np

                state = NOTEBOOK_STATE
                ns = state["ns"]
                benchmark = state["benchmark"]
                results = benchmark["results"]
                dataset_catalog = benchmark["dataset_catalog"]
                algorithm_summary = benchmark["algorithm_summary"]
                family_summary = benchmark["family_summary"]
                overall_regime_summary = benchmark["overall_regime_summary"]
                best_by_evaluation = benchmark["best_by_evaluation"]
                best_by_auc = benchmark["best_by_auc"]

                evaluation_mode = str(results["evaluation_mode"].iloc[0]).lower()
                evaluation_metric_label = "Mean Range F1" if evaluation_mode == "range" else "Mean Point F1"
                win_metric_label = "Range F1" if evaluation_mode == "range" else "Point F1"

                fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)

                axes[0, 0].hist(dataset_catalog["series_length"], bins=24, color="#4c78a8", edgecolor="white")
                axes[0, 0].set_title("Dataset Length Distribution")
                axes[0, 0].set_xlabel("Series length")
                axes[0, 0].set_ylabel("Dataset count")

                axes[0, 1].hist(dataset_catalog["anomaly_ratio"], bins=24, color="#72b7b2", edgecolor="white")
                axes[0, 1].set_title("Anomaly Ratio Distribution")
                axes[0, 1].set_xlabel("Anomaly ratio")
                axes[0, 1].set_ylabel("Dataset count")

                tradeoff_frame = algorithm_summary[["algorithm_display", "mean_runtime_seconds", "mean_evaluation_f1"]].dropna().sort_values("mean_runtime_seconds")
                pareto_points = []
                best_seen = -np.inf
                for row in tradeoff_frame.itertuples():
                    axes[1, 0].scatter(row.mean_runtime_seconds, row.mean_evaluation_f1, s=90, alpha=0.9, color="#4c78a8")
                    axes[1, 0].annotate(
                        row.algorithm_display,
                        (row.mean_runtime_seconds, row.mean_evaluation_f1),
                        textcoords="offset points",
                        xytext=(6, 6),
                        fontsize=9,
                    )
                    if row.mean_evaluation_f1 >= best_seen:
                        pareto_points.append((row.mean_runtime_seconds, row.mean_evaluation_f1))
                        best_seen = row.mean_evaluation_f1
                has_pareto_points = len(pareto_points) > 0
                if pareto_points:
                    pareto_points = np.asarray(pareto_points)
                    axes[1, 0].plot(
                        pareto_points[:, 0],
                        pareto_points[:, 1],
                        color="#e45756",
                        linewidth=1.6,
                        label="Pareto frontier",
                    )
                axes[1, 0].set_title(f"Runtime vs {evaluation_metric_label}")
                axes[1, 0].set_xlabel("Mean runtime (seconds)")
                axes[1, 0].set_ylabel(evaluation_metric_label)
                if has_pareto_points:
                    axes[1, 0].legend()

                bar_positions = np.arange(len(algorithm_summary))
                bar_width = 0.38
                axes[1, 1].bar(bar_positions - bar_width / 2, algorithm_summary["mean_evaluation_f1"], width=bar_width, label=evaluation_metric_label, color="#f58518")
                axes[1, 1].bar(bar_positions + bar_width / 2, algorithm_summary["mean_roc_auc"], width=bar_width, label="Mean ROC AUC", color="#54a24b")
                axes[1, 1].set_xticks(bar_positions)
                axes[1, 1].set_xticklabels(algorithm_summary["algorithm_display"], rotation=25, ha="right")
                axes[1, 1].set_ylim(0, 1.05)
                axes[1, 1].set_title("Average Accuracy by Configuration")
                axes[1, 1].set_ylabel("Metric value")
                axes[1, 1].legend()

                overview_path = ns.result_figure_path("benchmark_overview.png")
                fig.savefig(overview_path, dpi=160)
                plt.show()
                print(f"Saved: {overview_path}")

                if has_pareto_points:
                    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
                    for row in tradeoff_frame.itertuples():
                        ax.scatter(row.mean_runtime_seconds, row.mean_evaluation_f1, s=90, alpha=0.9, color="#4c78a8")
                        ax.annotate(
                            row.algorithm_display,
                            (row.mean_runtime_seconds, row.mean_evaluation_f1),
                            textcoords="offset points",
                            xytext=(6, 6),
                            fontsize=9,
                        )
                    ax.plot(pareto_points[:, 0], pareto_points[:, 1], color="#e45756", linewidth=1.6)
                    ax.set_title(f"Pareto Frontier | Mean runtime vs {evaluation_metric_label}")
                    ax.set_xlabel("Mean runtime (seconds)")
                    ax.set_ylabel(evaluation_metric_label)
                    pareto_path = ns.result_figure_path("pareto_frontier.png")
                    fig.savefig(pareto_path, dpi=160)
                    plt.show()
                    print(f"Saved: {pareto_path}")

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
                    evaluation_metric_label,
                    "Range Precision",
                    "Range Recall",
                    "Range F1",
                    "Affil. Precision",
                    "Affil. Recall",
                ]

                fig, ax = plt.subplots(figsize=(14, 5.2), constrained_layout=True)
                image = ax.imshow(heatmap_frame.to_numpy(), cmap="YlOrBr", aspect="auto")
                ax.set_xticks(range(len(heatmap_labels)))
                ax.set_xticklabels(heatmap_labels, rotation=20, ha="right")
                ax.set_yticks(range(len(heatmap_frame.index)))
                ax.set_yticklabels(heatmap_frame.index)
                ax.set_title("TSB-UAD-Aligned Metric Heatmap by Configuration")
                for row_index in range(heatmap_frame.shape[0]):
                    for col_index in range(heatmap_frame.shape[1]):
                        ax.text(col_index, row_index, f"{heatmap_frame.iloc[row_index, col_index]:.2f}", ha="center", va="center", fontsize=10)
                fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
                heatmap_path = ns.result_figure_path("metric_heatmap.png")
                fig.savefig(heatmap_path, dpi=160)
                plt.show()
                print(f"Saved: {heatmap_path}")

                family_heatmap = family_summary.pivot(index="family", columns="algorithm_display", values="mean_evaluation_f1").fillna(0.0)
                fig, ax = plt.subplots(figsize=(16, max(6, 0.35 * len(family_heatmap.index))), constrained_layout=True)
                image = ax.imshow(family_heatmap.to_numpy(), cmap="YlGnBu", aspect="auto")
                ax.set_xticks(range(len(family_heatmap.columns)))
                ax.set_xticklabels(family_heatmap.columns, rotation=30, ha="right")
                ax.set_yticks(range(len(family_heatmap.index)))
                ax.set_yticklabels(family_heatmap.index)
                ax.set_title(f"{evaluation_metric_label} by Dataset Family and Configuration")
                fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
                family_heatmap_path = ns.result_figure_path("family_evaluation_heatmap.png")
                fig.savefig(family_heatmap_path, dpi=160)
                plt.show()
                print(f"Saved: {family_heatmap_path}")

                evaluation_wins = best_by_evaluation["best_algorithm_display"].value_counts().reindex(algorithm_summary["algorithm_display"]).fillna(0)
                auc_wins = best_by_auc["best_algorithm_display"].value_counts().reindex(algorithm_summary["algorithm_display"]).fillna(0)

                fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
                axes[0].bar(evaluation_wins.index, evaluation_wins.values, color="#4c78a8")
                axes[0].set_title(f"Configuration Wins by {win_metric_label}")
                axes[0].set_ylabel("Dataset wins")
                axes[0].tick_params(axis="x", rotation=25)

                axes[1].bar(auc_wins.index, auc_wins.values, color="#e45756")
                axes[1].set_title("Configuration Wins by ROC AUC")
                axes[1].set_ylabel("Dataset wins")
                axes[1].tick_params(axis="x", rotation=25)

                wins_path = ns.result_figure_path("algorithm_wins.png")
                fig.savefig(wins_path, dpi=160)
                plt.show()

                display(best_by_evaluation.head(15))
                display(best_by_auc.head(15))
                display(overall_regime_summary)
                print(f"Saved: {wins_path}")
                """
            ),
        ]
    )

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.13",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_PATH.write_text(json.dumps(build_notebook(), indent=2), encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
