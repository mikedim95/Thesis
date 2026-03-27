# High-ROI Paper Notes

This framework now supports paper-facing sweeps rather than single-baseline runs.

## What changed

- The notebook saves the exact run configuration in `run_configuration.csv`.
- The notebook saves every enabled variant and its exact arguments in `selected_run_parameters.csv`.
- Per-algorithm reports now save parameter-effect tables plus regime-aware summaries by dataset variant, length regime, and anomaly-ratio regime.
- The generated notebook keeps cross-cell state inside `NOTEBOOK_STATE` instead of spreading many globals across cells.

## Paper methodology guidance

- Use `paper_high_roi` for the main body when you want strong methods with manageable runtime.
- Use `paper_full_suite` for appendix material or family-specific reruns.
- Interpret results with `evaluation_f1`, not only raw point F1. In `range` mode this reflects interval overlap, which is usually the more defensible metric for UCR-style anomaly spans.
- Keep threshold settings fixed while discussing algorithm sensitivity. The presets now vary score-driving model parameters, not backend-only knobs.

## Score-driving parameters emphasized by the notebook

- Isolation Forest: `n_estimators`, `max_samples`, `max_features`, `bootstrap`
- Local Outlier Factor: `n_neighbors`, `metric`, `p`
- Matrix Profile: `subsequence_multiplier`
- OCSVM: `kernel`, `nu`, `gamma`, `train_fraction`
- HBOS: `n_bins`, `alpha`, `tol`
- DAMP: `start_index_multiplier`, `x_lag_multiplier`
- SAND: `alpha`, `init_length`, `batch_size`, `k`, `subsequence_multiplier`, `overlap`
- PCA: `n_components`, `n_selected_components`, `whiten`, `weighted`, `standardization`

## Important interpretation note

Some library parameters can affect internal decision thresholds without changing the continuous anomaly scores used by this notebook. Those are intentionally not the focus of the paper presets. The notebook evaluates algorithms from their score traces first, then applies one shared thresholding strategy so parameter comparisons stay methodologically clean.
