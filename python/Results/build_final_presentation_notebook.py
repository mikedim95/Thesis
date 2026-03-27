from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


MODULE_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = MODULE_DIR / "final_anomaly_detection_presentation.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


def build_notebook():
    cells = [
        md(
            """
            # Anomaly Detection Presentation Notebook

            This notebook is designed to be **obvious and rerunnable** in three stages:

            1. **Beginning**: start from the dataset, show normalization and preparation.
            2. **Middle**: show where the files live, which algorithm arguments are active, and how parameter tweaks change behaviour.
            3. **End**: run the three algorithms and show the anomaly detection graphs.

            The benchmark context still comes from the finalized report files already in the repo, but the core demo part of the notebook is now parameter-driven so you can edit one control cell and rerun.
            """
        ),
        code(
            """
            import importlib.util
            import subprocess
            import sys

            required = {
                "numpy": "numpy",
                "pandas": "pandas",
                "matplotlib": "matplotlib",
                "sklearn": "scikit-learn",
                "scipy": "scipy",
                "nbformat": "nbformat",
            }
            missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
            if missing:
                print("Installing missing packages:", missing)
                subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            else:
                print("Core analysis dependencies are available.")
            """
        ),
        code(
            """
            from pathlib import Path
            import sys
            import pandas as pd

            candidate_dirs = [
                Path.cwd(),
                Path.cwd() / "python" / "Results",
                Path.cwd().parent,
                Path.cwd().parent / "python" / "Results",
            ]
            for candidate in candidate_dirs:
                if (candidate / "anomaly_presentation_support.py").exists():
                    if str(candidate) not in sys.path:
                        sys.path.insert(0, str(candidate))
                    break
            else:
                raise FileNotFoundError("Could not locate anomaly_presentation_support.py from the current working directory.")

            from anomaly_presentation_support import (
                ALGORITHM_SOURCE_MAP,
                DEMO_DATASETS,
                FIGURE_OUTPUT_DIR,
                PIPELINE_FILE_MAP,
                algorithm_overview_frame,
                algorithm_parameter_frame,
                archived_dataset_metrics,
                build_algorithm_summary,
                build_dataset_catalog,
                build_dataset_deep_dive_frame,
                build_dataset_segment_frame,
                build_key_findings,
                default_algorithm_parameters,
                load_all_results,
                plot_algorithm_cards,
                plot_dataset_deep_dive,
                plot_dataset_landscape,
                plot_live_demo_suite,
                plot_median_metric_heatmap,
                plot_parameter_sweep,
                plot_preprocessing_story,
                plot_runtime_analysis,
                plot_win_heatmap,
                prepare_dataset_bundle,
                repo_layout_frame,
                run_demo_suite,
                set_presentation_style,
                sweep_parameter,
            )

            set_presentation_style()
            pd.set_option("display.max_colwidth", 120)
            pd.set_option("display.precision", 3)

            results = load_all_results()
            dataset_catalog = build_dataset_catalog(results)
            algorithm_summary = build_algorithm_summary(results)
            findings = build_key_findings(algorithm_summary, dataset_catalog)
            FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            print(f"Benchmark rows loaded: {len(results):,}")
            print(f"Unique datasets: {dataset_catalog['dataset_name'].nunique():,}")
            print(f"Figure export directory: {FIGURE_OUTPUT_DIR}")
            """
        ),
        md(
            """
            ## Beginning

            The first stage answers: **what is the input, where is it stored, and how is it normalized before detection starts?**
            """
        ),
        code(
            """
            display(repo_layout_frame())
            """
        ),
        code(
            """
            # BEGINNING CONTROL CELL
            # Change these values first, then rerun the next 2-3 cells.

            dataset_name = DEMO_DATASETS[0]
            normalization_method = "zscore"   # one of: none, zscore, minmax, robust
            clip_quantile = None              # example: 0.01 to clip extreme values before normalization
            window_override = None            # set an integer to override the automatic sliding window

            prepared = prepare_dataset_bundle(
                dataset_name=dataset_name,
                normalization_method=normalization_method,
                clip_quantile=clip_quantile,
                window_override=window_override,
            )

            display(pd.DataFrame([prepared["metadata"]]).T.rename(columns={0: "Value"}))
            display(prepared["statistics"].round(3))
            """
        ),
        code(
            """
            _ = plot_preprocessing_story(
                prepared,
                FIGURE_OUTPUT_DIR / "01_beginning_preprocessing.png",
            )
            """
        ),
        md(
            """
            The notebook is now at the **prepared dataset** stage: the raw series is visible, the normalized series is visible, and the selected sliding-window length is fixed for the next stage unless you override it.
            """
        ),
        md(
            """
            ## Deep Dive Dataset Analysis

            This section stays focused on the **currently selected dataset** rather than the whole benchmark. It makes the dataset structure explicit: where training ends, how wide the labeled anomaly is, how the prepared values differ from the raw signal, and how the anomaly segment compares with the train and post-anomaly segments.
            """
        ),
        code(
            """
            display(build_dataset_deep_dive_frame(prepared))
            display(build_dataset_segment_frame(prepared).round(3))
            """
        ),
        code(
            """
            _ = plot_dataset_deep_dive(
                prepared,
                FIGURE_OUTPUT_DIR / "02_dataset_deep_dive.png",
            )
            """
        ),
        md(
            """
            The deep-dive section is the place to explain the dataset itself before talking about the algorithms: signal length, anomaly sparsity, segment statistics, and the local anomaly context are all visible here.
            """
        ),
        md(
            """
            ## Middle

            The middle stage answers: **which files implement each algorithm, which arguments are active, and what happens when those arguments change?**
            """
        ),
        code(
            """
            _ = plot_algorithm_cards(algorithm_summary, FIGURE_OUTPUT_DIR / "02_algorithm_cards.png")
            display(algorithm_overview_frame(algorithm_summary).round(3))
            display(findings)
            """
        ),
        code(
            """
            # MIDDLE CONTROL CELL
            # Edit any parameter below and rerun the next cells.

            algorithm_params = default_algorithm_parameters()

            algorithm_params["IForest"].update(
                {
                    "n_estimators": 100,
                    "max_samples": "auto",
                    "contamination": 0.1,
                    "max_features": 1.0,
                    "bootstrap": False,
                    "random_state": 42,
                }
            )

            algorithm_params["LOF"].update(
                {
                    "n_neighbors": 20,
                    "contamination": 0.1,
                    "algorithm": "auto",
                    "leaf_size": 30,
                    "metric": "minkowski",
                    "p": 2,
                }
            )

            algorithm_params["SAND"].update(
                {
                    "k": None,                    # None = infer from series length
                    "alpha": 0.5,
                    "init_length": 5000,
                    "batch_size": 2000,
                    "subsequence_multiplier": 4,
                    "overlapping_rate": None,    # None = use subsequence length
                    "prefer_new_implementation": True,
                }
            )

            include_sand = True
            threshold_std_multiplier = 3.0

            display(algorithm_parameter_frame(algorithm_params))
            """
        ),
        code(
            """
            configured_demo = run_demo_suite(
                dataset_name=dataset_name,
                results=results,
                include_sand=include_sand,
                normalization_method=normalization_method,
                clip_quantile=clip_quantile,
                window_override=window_override,
                algorithm_params=algorithm_params,
                threshold_std_multiplier=threshold_std_multiplier,
            )

            display(configured_demo["comparison"].round(3))
            display(archived_dataset_metrics(results, dataset_name).round(3))
            """
        ),
        md(
            """
            The table above is the bridge between the **benchmark archive** and your **current tweakable run**.

            - `live_*` columns are produced from the current control-cell settings.
            - the archived columns are the historical benchmark values already stored in the repo.
            """
        ),
        code(
            """
            # PARAMETER SWEEP CONTROL CELL
            # This is the clearest place in the notebook to show argument sensitivity.
            # Change the algorithm, parameter name, and list of values, then rerun.
            # SAND sweeps are supported too, but they are slower.

            sweep_algorithm = "LOF"                  # IForest, LOF, or SAND
            sweep_parameter_name = "n_neighbors"     # examples: contamination, n_estimators, k, alpha, batch_size
            sweep_values = [5, 10, 20, 40, 60]
            """
        ),
        code(
            """
            sweep_df = sweep_parameter(
                dataset_name=dataset_name,
                algorithm=sweep_algorithm,
                parameter_name=sweep_parameter_name,
                parameter_values=sweep_values,
                normalization_method=normalization_method,
                clip_quantile=clip_quantile,
                window_override=window_override,
                algorithm_params=algorithm_params,
                threshold_std_multiplier=threshold_std_multiplier,
            )

            display(sweep_df.round(3))
            _ = plot_parameter_sweep(
                sweep_df,
                FIGURE_OUTPUT_DIR / f"03_sweep_{sweep_algorithm}_{sweep_parameter_name}.png",
            )
            """
        ),
        md(
            """
            The middle stage is complete once you are satisfied with the configuration. At that point the notebook has made the pipeline explicit:

            - where the datasets are,
            - where each algorithm source file is,
            - which arguments are active,
            - and how those arguments affect behaviour.
            """
        ),
        md(
            """
            ## End

            The final stage answers: **given the current prepared dataset and current algorithm settings, where do the anomalies appear and how do the three methods behave on the signal?**
            """
        ),
        code(
            """
            _ = plot_live_demo_suite(
                configured_demo,
                FIGURE_OUTPUT_DIR / "04_end_detection_suite.png",
            )
            """
        ),
        code(
            """
            _ = plot_dataset_landscape(dataset_catalog, FIGURE_OUTPUT_DIR / "05_benchmark_dataset_landscape.png")
            _ = plot_median_metric_heatmap(algorithm_summary, FIGURE_OUTPUT_DIR / "06_benchmark_median_heatmap.png")
            _ = plot_runtime_analysis(results, FIGURE_OUTPUT_DIR / "07_benchmark_runtime.png")
            _ = plot_win_heatmap(results, FIGURE_OUTPUT_DIR / "08_benchmark_wins.png")
            """
        ),
        md(
            """
            ## Practical Use During the Presentation

            If you want to demonstrate the notebook live in front of your professor:

            1. Change the **Beginning control cell** to switch dataset and normalization.
            2. Use the **Deep Dive Dataset Analysis** cells to explain the selected dataset before switching to model behaviour.
            3. Change the **Middle control cell** to tweak `IForest`, `LOF`, or `SAND` arguments.
            4. Optionally rerun the **Parameter Sweep** cells to show how one argument changes AUC, F1, recall, and runtime.
            5. Rerun the **End** plot cell to show the updated anomaly graphs.

            All exported figures are saved under `python/Results/presentation_figures/`.
            """
        ),
    ]

    notebook = nbf.v4.new_notebook()
    notebook.cells = cells
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.13",
        },
    }
    return notebook


def main() -> None:
    notebook = build_notebook()
    with NOTEBOOK_PATH.open("w", encoding="utf-8") as handle:
        nbf.write(notebook, handle)
    print(f"Wrote notebook: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
