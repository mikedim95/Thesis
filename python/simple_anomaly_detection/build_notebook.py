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

            This next cell renders the paper-facing report for {display_name}: a plain-language interpretation block, the configured variants, parameter-effect tables, regime-aware summaries by dataset type, the benchmark panel, the paper panel, and, when `Argument mode = auto_ablation`, paired baseline-vs-knob delta tables plus an ablation panel that shows effect size, confidence intervals, runtime tradeoff, and regime sensitivity. It also renders a fixed-layout side-by-side variant comparison graph on one shared dataset and the best showcase plot.
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
            - `Argument mode` can now switch between manual subtabs and automatic preset sweeps.
            - `auto_ablation` creates one baseline plus one-knob-at-a-time variants so each parameter claim is paired against a fixed reference.
            - Paper-facing sweeps vary only score-driving parameters; backend-only or threshold-only knobs stay fixed inside the presets.
            - Benchmark outputs are saved into `results/tables/`, `results/figures/`, and `results/scores/`.
            - The notebook can export a fixed-style thesis figure pack with matching caption text in one step.

            Default workflow:

            1. Run the dependency and setup cells.
            2. Optionally apply `paper_high_roi`, `paper_full_suite`, or `auto_ablation`.
            3. Run the preparation and benchmark cells.
            4. Inspect the regime summary and winner tables.
            5. Open the per-algorithm report cells for the paper figures.
            6. Run the thesis export cell to write PNG/PDF figures and captions.
            """
        ),
        markdown(
            """
            ## Control Reference

            **General**
            - `Argument mode`: `manual` uses the current subtabs, while the auto modes expand enabled algorithms into preset parameter combinations.
            - `Dataset limit`: how many prepared datasets to benchmark. `0` means all datasets.
            - `Batch size`: how many of the selected datasets to process in the current run. `0` means the whole selected scope.
            - `Resume from existing results`: continue from successful rows already saved in `results/tables/benchmark_results.csv` and automatically take the next incomplete batch.
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
            - `auto_ablation`: one baseline plus one-knob-at-a-time variants, intended for defensible sensitivity claims and presentation-ready parameter analysis.
            - In auto modes, the notebook ignores the live subtab values at run time and uses the preset variants instead.
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

            This next cell finds the `simple_anomaly_detection` folder, reloads the support module, loads dataset names, renders the control panel, writes the algorithm-mechanics notes, and initializes `NOTEBOOK_STATE`.
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

            notes_info = ns.write_high_roi_algorithm_notes()
            display(panel_bundle["panel"])
            display(ns.build_results_layout_frame())
            display(ns.list_paper_presets())
            display(ns.build_algorithm_reference_overview())
            print(f"Project root: {ns.PROJECT_ROOT}")
            print(f"Legacy virgin source: {ns.LEGACY_VIRGIN_DIR}")
            print(f"Results root: {ns.RESULTS_DIR}")
            print(f"Results tables: {ns.RESULT_TABLES_DIR}")
            print(f"Results figures: {ns.RESULT_FIGURES_DIR}")
            print(f"Score traces: {ns.RESULT_SCORES_DIR}")
            print(f"Algorithm notes: {notes_info['source_path']}")
            print(f"Notes results copy: {notes_info['results_path']}")
            """
        ),
        markdown(
            """
            ## Optional: Apply A Reproducible Automatic Experiment Layout

            Edit `preset_name` below if you want a different automatic experiment layout. This also switches `Argument mode` to the selected auto sweep, then rerun the preparation and benchmark cells.
            """
        ),
        code(
            """
            preset_name = "auto_ablation"
            state = NOTEBOOK_STATE
            state["ns"].apply_paper_experiment_preset(state["controls"], preset_name)
            display(state["ns"].build_preset_reference_table(preset_name))
            print(f"Applied preset: {preset_name}")
            """
        ),
        markdown(
            """
            ## On Run: Read The Current Widget Values, Prepare Raw Datasets, And Build Normalized Benchmark Inputs

            This next cell reads the current control-panel settings, ensures the raw datasets exist in the workspace, generates normalized labeled datasets if needed, and shows the exact paper run configuration together with the current batch plan.
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

            This next cell refreshes the configuration from the widgets, runs the current dataset batch for every selected algorithm variant, writes the benchmark outputs into `results/tables/`, and stores the accumulated benchmark objects inside `NOTEBOOK_STATE["benchmark"]`.
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
            ## On Run: Explain How To Read The Ablation Outputs

            This next cell is most useful when `Argument mode = auto_ablation`. It explains that each non-baseline variant changes one knob at a time, shows paired delta tables against the baseline on the same dataset, adds a short written summary of the strongest global gain/loss, and renders a high-level ablation overview figure with confidence intervals so you can defend which controls materially help, hurt, or mainly affect runtime.
            """
        ),
        code(
            """
            state = NOTEBOOK_STATE
            state["ns"].render_ablation_overview(state)
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

                This next cell builds the overall comparison charts across all selected configurations by calling the reusable plotting helpers in `notebook_support.py`, saves the overview images into `results/figures/`, and displays the winner tables together with the regime-aware benchmark table.
                """
            ),
            code(
                """
                state = NOTEBOOK_STATE
                ns = state["ns"]
                benchmark = state["benchmark"]
                plt = ns._load_plotting_module()

                figure_jobs = [
                    ("benchmark_overview.png", lambda: ns.plot_benchmark_overview_panel(benchmark, ns.result_figure_path("benchmark_overview.png"))),
                    ("pareto_frontier.png", lambda: ns.plot_pareto_frontier_panel(benchmark, ns.result_figure_path("pareto_frontier.png"))),
                    ("metric_heatmap.png", lambda: ns.plot_metric_heatmap_panel(benchmark, ns.result_figure_path("metric_heatmap.png"))),
                    ("family_evaluation_heatmap.png", lambda: ns.plot_family_evaluation_heatmap_panel(benchmark, ns.result_figure_path("family_evaluation_heatmap.png"))),
                    ("algorithm_wins.png", lambda: ns.plot_algorithm_wins_panel(benchmark, ns.result_figure_path("algorithm_wins.png"))),
                ]

                for filename, builder in figure_jobs:
                    fig = builder()
                    if fig is not None:
                        plt.show()
                        plt.close(fig)
                        print(f"Saved: {ns.result_figure_path(filename)}")

                display(benchmark["best_by_evaluation"].head(15))
                display(benchmark["best_by_auc"].head(15))
                display(benchmark["overall_regime_summary"])
                """
            ),
            markdown(
                """
                ## Optional: Export A Thesis-Ready Figure Pack

                This next cell exports a fixed-style set of thesis-ready figures into `results/figures/thesis/` as both PNG and PDF, then writes a figure catalog plus ready-to-paste caption text into `results/tables/thesis_figure_catalog.csv` and `results/tables/thesis_figure_captions.md`.
                """
            ),
            code(
                """
                state = NOTEBOOK_STATE
                thesis_export = state["ns"].export_thesis_figure_pack(state)
                display(thesis_export["catalog"])
                print(f"Saved thesis figures to: {thesis_export['figure_dir']}")
                print(f"Saved figure catalog to: {thesis_export['catalog_path']}")
                print(f"Saved captions to: {thesis_export['captions_path']}")
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
