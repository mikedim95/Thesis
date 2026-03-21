from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


NOTEBOOK_PATH = Path(__file__).resolve().parent / "run_anomaly_detection.ipynb"


def lines(text: str) -> list[str]:
    return dedent(text).strip().splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines(text)}


def build_notebook() -> dict:
    cells = [
        markdown(
            """
            # Simple Anomaly Detection Analysis

            This notebook starts with a visible control panel. The general run options stay at the top, and each algorithm has its own knob tab:

            - Isolation Forest
            - Local Outlier Factor
            - SAND
            - Matrix Profile
            - DAMP
            - HBOS
            - OCSVM
            - PCA

            Workflow:

            1. Change the knobs in the widget panel.
            2. Rerun the configuration cell.
            3. Rerun the benchmark cell.
            4. Inspect the algorithm sections independently.

            Notes:

            - `SAND`, `DAMP`, `OCSVM`, and `PCA` start disabled by default so the notebook does not become unusably slow on the full dataset collection.
            - The benchmark now runs shorter prepared datasets first, so you get earlier feedback and partial results faster.
            - Notebook outputs are organized into `results/tables/`, `results/figures/`, and `results/scores/` instead of being saved flat under `results/`.
            """
        ),
        markdown(
            """
            ## Control Reference

            This reference is visible before you run anything. The interactive widget panel appears after you run the setup cell below.

            **General**
            - `Dataset limit`: how many prepared datasets to benchmark. `0` means all datasets.
            - `Normalize`: preprocessing applied before any algorithm runs.
            - `Clip q`: optional quantile clipping before normalization.
            - `Window override`: force one window size for every dataset. `0` uses auto-estimation.
            - `Threshold sigma`: converts scores into anomaly flags for precision, recall, and F1.
            - `Deep dive`: chooses which dataset gets the detailed plots.
            - `Rebuild normalized datasets`: regenerate cached normalized files.
            - `Save per-dataset scores`: save raw score traces to `results/scores/`.
            - Shorter datasets are benchmarked first so the run starts yielding results sooner.

            **Isolation Forest**
            - Use `+` inside the algorithm tab to duplicate the current argument set and create another subtab.
            - `Trees`: number of trees.
            - `Contam.`: expected anomaly fraction.
            - `Max samples`: training windows per tree.
            - `Max feat.`: fraction of window features per tree.
            - `Bootstrap`: sample with replacement.
            - `Seed`: reproducible random state.

            **Local Outlier Factor**
            - Use `+` inside the algorithm tab to duplicate the current argument set and create another subtab.
            - `Neighbors`: local neighborhood size.
            - `Contam.`: expected anomaly fraction.
            - `Search`: sklearn neighbor-search backend.
            - `Leaf size`: tree-search tuning knob.
            - `Metric`: distance function between windows.
            - `p`: Minkowski distance power.

            **SAND**
            - Use `+` inside the algorithm tab to duplicate the current argument set and create another subtab.
            - `Alpha`: adaptation weight for newer data.
            - `Init length`: initialization span before online updates.
            - `Batch size`: online processing chunk size.
            - `k`: number of nearest subsequences, with `0` meaning auto.
            - `Subseq x`: multiplier from window size to subsequence length.
            - `Overlap`: subsequence step size, with `0` meaning auto.
            - This one starts disabled by default because it is slow on long series.

            **Matrix Profile**
            - Use `+` inside the algorithm tab to duplicate the current argument set and create another subtab.
            - `Subseq x`: multiplier from window size to the subsequence length used by the discord search.

            **DAMP**
            - Use `+` inside the algorithm tab to duplicate the current argument set and create another subtab.
            - `Start x`: multiplier controlling how far into the series DAMP begins its streaming-style discord search.
            - `x_lag x`: optional multiplier for how far back DAMP searches. `0` uses the internal default.
            - This one also starts disabled by default because it is slow on long series.

            **HBOS**
            - Use `+` inside the algorithm tab to duplicate the current argument set and create another subtab.
            - `Bins`: histogram granularity.
            - `Alpha`: density regularizer.
            - `Tol`: tolerance for values just outside histogram edges.
            - `Contam.`: expected anomaly fraction for the experiment settings.

            **OCSVM**
            - Use `+` inside the algorithm tab to duplicate the current argument set and create another subtab.
            - `Kernel`: One-Class SVM kernel.
            - `Nu`: support-vector / training-error bound.
            - `Gamma`: kernel coefficient.
            - `Train frac`: fraction of earliest windows used for fitting before scoring the full series.

            **PCA**
            - Use `+` inside the algorithm tab to duplicate the current argument set and create another subtab.
            - `Components`: total retained PCA components. Blank means default behavior.
            - `Score comps`: number of low-variance components used for anomaly scoring. `0` means all retained components.
            - `Whiten`, `Weighted`, and `Standardize`: control how PCA is fitted and how the score is computed.

            **TSB-UAD Research Metrics**
            - The benchmark now adds range-aware and affiliation-aware metrics to the main result tables.
            - A later cell computes heavier `Range AUC` and `VUS` metrics only on the selected deep-dive dataset, so the full benchmark does not become too slow.
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

            This next cell finds the `simple_anomaly_detection` folder, reloads the support module, loads dataset names, and renders the control panel widgets.
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
            PROJECT_ROOT = next((root for root in candidate_roots if (root / "notebook_support.py").exists()), None)
            if PROJECT_ROOT is None:
                raise FileNotFoundError("Could not locate the simple_anomaly_detection project root.")

            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            import notebook_support as ns
            ns = importlib.reload(ns)

            dataset_name_source = ns.RAW_DATASET_DIR if any(ns.RAW_DATASET_DIR.glob("*.txt")) else ns.LEGACY_VIRGIN_DIR
            DATASET_NAMES = [path.stem for path in sorted(dataset_name_source.glob("*.txt"))]

            panel_bundle = ns.build_control_panel(DATASET_NAMES)
            controls = panel_bundle["controls"]

            display(panel_bundle["panel"])
            display(ns.build_results_layout_frame())
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
            ## On Run: Read The Current Widget Values, Prepare Raw Datasets, And Build Normalized Benchmark Inputs

            This next cell reads the current control-panel settings, ensures the raw datasets exist in the workspace, generates normalized labeled datasets if needed, and shows the run configuration summary.
            """
        ),
        code(
            """
            CONFIG = ns.get_run_config(controls)
            context = ns.prepare_run_context(CONFIG)

            display(context["run_config_frame"])
            display(context["preparation_summary"])
            display(pd.DataFrame({"sample_raw_dataset_file": [path.name for path in context["raw_dataset_paths"][:10]]}))
            print(f"Prepared datasets will be loaded from: {context['prepared_dataset_dir']}")
            print(f"Benchmark dataset count: {len(context['benchmark_dataset_paths'])}")
            """
        ),
        markdown(
            """
            ## On Run: Execute Every Selected Algorithm Configuration Across The Prepared Datasets

            This next cell rereads the current widget values, refreshes the prepared dataset selection, runs every selected algorithm variant over the prepared datasets, writes the benchmark CSV files into `results/tables/`, and shows live progress with dataset, configuration, elapsed time, and ETA. The saved benchmark tables now include extra time-series-aware metrics such as range precision, range recall, range F1, and affiliation scores.
            """
        ),
        code(
            """
            CONFIG = ns.get_run_config(controls)
            context = ns.prepare_run_context(CONFIG)

            benchmark = ns.run_benchmark(CONFIG, context["prepared_dataset_dir"], context["benchmark_dataset_paths"])

            results = benchmark["results"]
            dataset_catalog = benchmark["dataset_catalog"]
            algorithm_summary = benchmark["algorithm_summary"]
            family_summary = benchmark["family_summary"]
            best_by_f1 = benchmark["best_by_f1"]
            best_by_auc = benchmark["best_by_auc"]
            errors = benchmark["errors"]
            overview = benchmark["overview"]
            deep_dive_payload = benchmark["deep_dive_payload"]

            display(overview)
            display(algorithm_summary)
            display(dataset_catalog.head(12))
            if not errors.empty:
                display(errors[["dataset_name", "algorithm_display", "error"]])

            print(f"Saved benchmark outputs to: {ns.RESULT_TABLES_DIR}")
            """
        ),
        markdown(
            """
            ## On Run: Compute TSB-UAD Range-AUC And VUS Metrics For The Selected Deep-Dive Dataset

            This next cell takes the best-performing selected variant of each enabled algorithm on the chosen deep-dive dataset, computes the heavier TSB-UAD `Range AUC` and `VUS` metrics, and saves the resulting comparison table into `results/tables/`.
            """
        ),
        code(
            """
            research_metrics = ns.build_deep_dive_research_table(results, deep_dive_payload, CONFIG["selected_algorithms"])
            if research_metrics.empty:
                print("No deep-dive research metrics available for the current run.")
            else:
                display(research_metrics)
                research_metrics_path = ns.result_table_path("deep_dive_research_metrics.csv")
                research_metrics.to_csv(research_metrics_path, index=False)
                print(f"Saved: {research_metrics_path}")
            """
        ),
        markdown(
            """
            ## On Run: Show Isolation Forest Variant Settings, Summary Tables, Comparison Graph, And Best Deep-Dive Plot

            This next cell shows the configured Isolation Forest subtabs, summarizes their benchmark results, saves the Isolation Forest comparison graph, and renders the best deep-dive plot for the chosen dataset.
            """
        ),
        code(
            """
            if "isolation_forest" not in CONFIG["selected_algorithms"]:
                print("Isolation Forest is disabled in the control panel.")
            else:
                display(ns.build_variant_config_table(CONFIG, "isolation_forest"))
                section_summary, section_top = ns.build_algorithm_section_tables(results, "isolation_forest")
                display(section_summary)
                display(section_top)

                fig = ns.plot_algorithm_benchmark_panel(results, "isolation_forest", ns.result_algorithm_panel_path("isolation_forest"))
                if fig is not None:
                    import matplotlib.pyplot as plt
                    plt.show()

                metric_row, score_values = ns.select_deep_dive_variant(results, deep_dive_payload, "isolation_forest")
                if metric_row is not None and score_values is not None:
                    raw_values = ns.load_raw_values_from_file(ns.RAW_DATASET_DIR / f"{deep_dive_payload['dataset']['dataset_name']}.txt")
                    print(f"Deep dive variant: {metric_row['algorithm_display']}")
                    fig = ns.plot_algorithm_deep_dive(
                        raw_values,
                        deep_dive_payload["dataset"]["values"],
                        deep_dive_payload["dataset"],
                        score_values,
                        metric_row,
                        "isolation_forest",
                        context_points=1200,
                        save_path=ns.result_deep_dive_path(metric_row["algorithm_run_id"], deep_dive_payload["dataset"]["dataset_name"]),
                    )
                    plt.show()
            """
        ),
        markdown(
            """
            ## On Run: Show Local Outlier Factor Variant Settings, Summary Tables, Comparison Graph, And Best Deep-Dive Plot

            This next cell shows the configured Local Outlier Factor subtabs, summarizes their benchmark results, saves the LOF comparison graph, and renders the best deep-dive plot for the chosen dataset.
            """
        ),
        code(
            """
            if "local_outlier_factor" not in CONFIG["selected_algorithms"]:
                print("Local Outlier Factor is disabled in the control panel.")
            else:
                display(ns.build_variant_config_table(CONFIG, "local_outlier_factor"))
                section_summary, section_top = ns.build_algorithm_section_tables(results, "local_outlier_factor")
                display(section_summary)
                display(section_top)

                fig = ns.plot_algorithm_benchmark_panel(results, "local_outlier_factor", ns.result_algorithm_panel_path("local_outlier_factor"))
                if fig is not None:
                    import matplotlib.pyplot as plt
                    plt.show()

                metric_row, score_values = ns.select_deep_dive_variant(results, deep_dive_payload, "local_outlier_factor")
                if metric_row is not None and score_values is not None:
                    raw_values = ns.load_raw_values_from_file(ns.RAW_DATASET_DIR / f"{deep_dive_payload['dataset']['dataset_name']}.txt")
                    print(f"Deep dive variant: {metric_row['algorithm_display']}")
                    fig = ns.plot_algorithm_deep_dive(
                        raw_values,
                        deep_dive_payload["dataset"]["values"],
                        deep_dive_payload["dataset"],
                        score_values,
                        metric_row,
                        "local_outlier_factor",
                        context_points=1200,
                        save_path=ns.result_deep_dive_path(metric_row["algorithm_run_id"], deep_dive_payload["dataset"]["dataset_name"]),
                    )
                    plt.show()
            """
        ),
        markdown(
            """
            ## On Run: Show SAND Variant Settings, Summary Tables, Comparison Graph, And Best Deep-Dive Plot

            This next cell shows the configured SAND subtabs, summarizes their benchmark results, saves the SAND comparison graph, and renders the best deep-dive plot for the chosen dataset.
            """
        ),
        code(
            """
            if "sand" not in CONFIG["selected_algorithms"]:
                print("SAND is disabled in the control panel.")
            else:
                display(ns.build_variant_config_table(CONFIG, "sand"))
                section_summary, section_top = ns.build_algorithm_section_tables(results, "sand")
                display(section_summary)
                display(section_top)

                fig = ns.plot_algorithm_benchmark_panel(results, "sand", ns.result_algorithm_panel_path("sand"))
                if fig is not None:
                    import matplotlib.pyplot as plt
                    plt.show()

                metric_row, score_values = ns.select_deep_dive_variant(results, deep_dive_payload, "sand")
                if metric_row is not None and score_values is not None:
                    raw_values = ns.load_raw_values_from_file(ns.RAW_DATASET_DIR / f"{deep_dive_payload['dataset']['dataset_name']}.txt")
                    print(f"Deep dive variant: {metric_row['algorithm_display']}")
                    fig = ns.plot_algorithm_deep_dive(
                        raw_values,
                        deep_dive_payload["dataset"]["values"],
                        deep_dive_payload["dataset"],
                        score_values,
                        metric_row,
                        "sand",
                        context_points=1200,
                        save_path=ns.result_deep_dive_path(metric_row["algorithm_run_id"], deep_dive_payload["dataset"]["dataset_name"]),
                    )
                    plt.show()
            """
        ),
        markdown(
            """
            ## On Run: Show Matrix Profile Variant Settings, Summary Tables, Comparison Graph, And Best Deep-Dive Plot

            This next cell shows the configured Matrix Profile subtabs, summarizes their benchmark results, saves the Matrix Profile comparison graph, and renders the best deep-dive plot for the chosen dataset.
            """
        ),
        code(
            """
            if "matrix_profile" not in CONFIG["selected_algorithms"]:
                print("Matrix Profile is disabled in the control panel.")
            else:
                display(ns.build_variant_config_table(CONFIG, "matrix_profile"))
                section_summary, section_top = ns.build_algorithm_section_tables(results, "matrix_profile")
                display(section_summary)
                display(section_top)

                fig = ns.plot_algorithm_benchmark_panel(results, "matrix_profile", ns.result_algorithm_panel_path("matrix_profile"))
                if fig is not None:
                    import matplotlib.pyplot as plt
                    plt.show()

                metric_row, score_values = ns.select_deep_dive_variant(results, deep_dive_payload, "matrix_profile")
                if metric_row is not None and score_values is not None:
                    raw_values = ns.load_raw_values_from_file(ns.RAW_DATASET_DIR / f"{deep_dive_payload['dataset']['dataset_name']}.txt")
                    print(f"Deep dive variant: {metric_row['algorithm_display']}")
                    fig = ns.plot_algorithm_deep_dive(
                        raw_values,
                        deep_dive_payload["dataset"]["values"],
                        deep_dive_payload["dataset"],
                        score_values,
                        metric_row,
                        "matrix_profile",
                        context_points=1200,
                        save_path=ns.result_deep_dive_path(metric_row["algorithm_run_id"], deep_dive_payload["dataset"]["dataset_name"]),
                    )
                    plt.show()
            """
        ),
        markdown(
            """
            ## On Run: Show DAMP Variant Settings, Summary Tables, Comparison Graph, And Best Deep-Dive Plot

            This next cell shows the configured DAMP subtabs, summarizes their benchmark results, saves the DAMP comparison graph, and renders the best deep-dive plot for the chosen dataset.
            """
        ),
        code(
            """
            if "damp" not in CONFIG["selected_algorithms"]:
                print("DAMP is disabled in the control panel.")
            else:
                display(ns.build_variant_config_table(CONFIG, "damp"))
                section_summary, section_top = ns.build_algorithm_section_tables(results, "damp")
                display(section_summary)
                display(section_top)

                fig = ns.plot_algorithm_benchmark_panel(results, "damp", ns.result_algorithm_panel_path("damp"))
                if fig is not None:
                    import matplotlib.pyplot as plt
                    plt.show()

                metric_row, score_values = ns.select_deep_dive_variant(results, deep_dive_payload, "damp")
                if metric_row is not None and score_values is not None:
                    raw_values = ns.load_raw_values_from_file(ns.RAW_DATASET_DIR / f"{deep_dive_payload['dataset']['dataset_name']}.txt")
                    print(f"Deep dive variant: {metric_row['algorithm_display']}")
                    fig = ns.plot_algorithm_deep_dive(
                        raw_values,
                        deep_dive_payload["dataset"]["values"],
                        deep_dive_payload["dataset"],
                        score_values,
                        metric_row,
                        "damp",
                        context_points=1200,
                        save_path=ns.result_deep_dive_path(metric_row["algorithm_run_id"], deep_dive_payload["dataset"]["dataset_name"]),
                    )
                    plt.show()
            """
        ),
        markdown(
            """
            ## On Run: Show HBOS Variant Settings, Summary Tables, Comparison Graph, And Best Deep-Dive Plot

            This next cell shows the configured HBOS subtabs, summarizes their benchmark results, saves the HBOS comparison graph, and renders the best deep-dive plot for the chosen dataset.
            """
        ),
        code(
            """
            if "hbos" not in CONFIG["selected_algorithms"]:
                print("HBOS is disabled in the control panel.")
            else:
                display(ns.build_variant_config_table(CONFIG, "hbos"))
                section_summary, section_top = ns.build_algorithm_section_tables(results, "hbos")
                display(section_summary)
                display(section_top)

                fig = ns.plot_algorithm_benchmark_panel(results, "hbos", ns.result_algorithm_panel_path("hbos"))
                if fig is not None:
                    import matplotlib.pyplot as plt
                    plt.show()

                metric_row, score_values = ns.select_deep_dive_variant(results, deep_dive_payload, "hbos")
                if metric_row is not None and score_values is not None:
                    raw_values = ns.load_raw_values_from_file(ns.RAW_DATASET_DIR / f"{deep_dive_payload['dataset']['dataset_name']}.txt")
                    print(f"Deep dive variant: {metric_row['algorithm_display']}")
                    fig = ns.plot_algorithm_deep_dive(
                        raw_values,
                        deep_dive_payload["dataset"]["values"],
                        deep_dive_payload["dataset"],
                        score_values,
                        metric_row,
                        "hbos",
                        context_points=1200,
                        save_path=ns.result_deep_dive_path(metric_row["algorithm_run_id"], deep_dive_payload["dataset"]["dataset_name"]),
                    )
                    plt.show()
            """
        ),
        markdown(
            """
            ## On Run: Show OCSVM Variant Settings, Summary Tables, Comparison Graph, And Best Deep-Dive Plot

            This next cell shows the configured OCSVM subtabs, summarizes their benchmark results, saves the OCSVM comparison graph, and renders the best deep-dive plot for the chosen dataset.
            """
        ),
        code(
            """
            if "ocsvm" not in CONFIG["selected_algorithms"]:
                print("OCSVM is disabled in the control panel.")
            else:
                display(ns.build_variant_config_table(CONFIG, "ocsvm"))
                section_summary, section_top = ns.build_algorithm_section_tables(results, "ocsvm")
                display(section_summary)
                display(section_top)

                fig = ns.plot_algorithm_benchmark_panel(results, "ocsvm", ns.result_algorithm_panel_path("ocsvm"))
                if fig is not None:
                    import matplotlib.pyplot as plt
                    plt.show()

                metric_row, score_values = ns.select_deep_dive_variant(results, deep_dive_payload, "ocsvm")
                if metric_row is not None and score_values is not None:
                    raw_values = ns.load_raw_values_from_file(ns.RAW_DATASET_DIR / f"{deep_dive_payload['dataset']['dataset_name']}.txt")
                    print(f"Deep dive variant: {metric_row['algorithm_display']}")
                    fig = ns.plot_algorithm_deep_dive(
                        raw_values,
                        deep_dive_payload["dataset"]["values"],
                        deep_dive_payload["dataset"],
                        score_values,
                        metric_row,
                        "ocsvm",
                        context_points=1200,
                        save_path=ns.result_deep_dive_path(metric_row["algorithm_run_id"], deep_dive_payload["dataset"]["dataset_name"]),
                    )
                    plt.show()
            """
        ),
        markdown(
            """
            ## On Run: Show PCA Variant Settings, Summary Tables, Comparison Graph, And Best Deep-Dive Plot

            This next cell shows the configured PCA subtabs, summarizes their benchmark results, saves the PCA comparison graph, and renders the best deep-dive plot for the chosen dataset.
            """
        ),
        code(
            """
            if "pca" not in CONFIG["selected_algorithms"]:
                print("PCA is disabled in the control panel.")
            else:
                display(ns.build_variant_config_table(CONFIG, "pca"))
                section_summary, section_top = ns.build_algorithm_section_tables(results, "pca")
                display(section_summary)
                display(section_top)

                fig = ns.plot_algorithm_benchmark_panel(results, "pca", ns.result_algorithm_panel_path("pca"))
                if fig is not None:
                    import matplotlib.pyplot as plt
                    plt.show()

                metric_row, score_values = ns.select_deep_dive_variant(results, deep_dive_payload, "pca")
                if metric_row is not None and score_values is not None:
                    raw_values = ns.load_raw_values_from_file(ns.RAW_DATASET_DIR / f"{deep_dive_payload['dataset']['dataset_name']}.txt")
                    print(f"Deep dive variant: {metric_row['algorithm_display']}")
                    fig = ns.plot_algorithm_deep_dive(
                        raw_values,
                        deep_dive_payload["dataset"]["values"],
                        deep_dive_payload["dataset"],
                        score_values,
                        metric_row,
                        "pca",
                        context_points=1200,
                        save_path=ns.result_deep_dive_path(metric_row["algorithm_run_id"], deep_dive_payload["dataset"]["dataset_name"]),
                    )
                    plt.show()
            """
        ),
        markdown(
            """
            ## On Run: Build The Cross-Configuration Comparison Charts And Save The Final Overview Images

            This next cell builds the overall comparison charts across all selected configurations, saves the overview images into `results/figures/`, and displays the top winner tables.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np

            fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)

            axes[0, 0].hist(dataset_catalog["series_length"], bins=24, color="#4c78a8", edgecolor="white")
            axes[0, 0].set_title("Dataset Length Distribution")
            axes[0, 0].set_xlabel("Series length")
            axes[0, 0].set_ylabel("Dataset count")

            axes[0, 1].hist(dataset_catalog["anomaly_ratio"], bins=24, color="#72b7b2", edgecolor="white")
            axes[0, 1].set_title("Anomaly Ratio Distribution")
            axes[0, 1].set_xlabel("Anomaly ratio")
            axes[0, 1].set_ylabel("Dataset count")

            for algorithm_name, frame in results.groupby("algorithm_display"):
                axes[1, 0].scatter(frame["runtime_seconds"], frame["range_f1"], alpha=0.7, label=algorithm_name)
            axes[1, 0].set_title("Runtime vs Range F1")
            axes[1, 0].set_xlabel("Runtime (seconds)")
            axes[1, 0].set_ylabel("Range F1")
            axes[1, 0].legend()

            bar_positions = np.arange(len(algorithm_summary))
            bar_width = 0.38
            axes[1, 1].bar(bar_positions - bar_width / 2, algorithm_summary["mean_range_f1"], width=bar_width, label="Mean Range F1", color="#f58518")
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

            heatmap_frame = algorithm_summary.set_index("algorithm_display")[
                [
                    "mean_roc_auc",
                    "mean_average_precision",
                    "mean_f1",
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

            family_heatmap = family_summary.pivot(index="family", columns="algorithm_display", values="mean_range_f1").fillna(0.0)
            fig, ax = plt.subplots(figsize=(16, max(6, 0.35 * len(family_heatmap.index))), constrained_layout=True)
            image = ax.imshow(family_heatmap.to_numpy(), cmap="YlGnBu", aspect="auto")
            ax.set_xticks(range(len(family_heatmap.columns)))
            ax.set_xticklabels(family_heatmap.columns, rotation=30, ha="right")
            ax.set_yticks(range(len(family_heatmap.index)))
            ax.set_yticklabels(family_heatmap.index)
            ax.set_title("Mean Range F1 by Dataset Family and Configuration")
            fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
            family_heatmap_path = ns.result_figure_path("family_range_f1_heatmap.png")
            fig.savefig(family_heatmap_path, dpi=160)
            plt.show()
            print(f"Saved: {family_heatmap_path}")

            f1_wins = best_by_f1["best_algorithm_display"].value_counts().reindex(algorithm_summary["algorithm_display"]).fillna(0)
            auc_wins = best_by_auc["best_algorithm_display"].value_counts().reindex(algorithm_summary["algorithm_display"]).fillna(0)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
            axes[0].bar(f1_wins.index, f1_wins.values, color="#4c78a8")
            axes[0].set_title("Configuration Wins by F1")
            axes[0].set_ylabel("Dataset wins")
            axes[0].tick_params(axis="x", rotation=25)

            axes[1].bar(auc_wins.index, auc_wins.values, color="#e45756")
            axes[1].set_title("Configuration Wins by ROC AUC")
            axes[1].set_ylabel("Dataset wins")
            axes[1].tick_params(axis="x", rotation=25)

            wins_path = ns.result_figure_path("algorithm_wins.png")
            fig.savefig(wins_path, dpi=160)
            plt.show()

            display(best_by_f1.head(15))
            display(best_by_auc.head(15))
            display(family_summary.head(30))
            print(f"Saved: {wins_path}")
            """
        ),
    ]

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
