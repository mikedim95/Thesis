# High-ROI Anomaly Detection Methods

This note summarizes how the current notebook run detects anomalies, which algorithms are the highest ROI in the saved results, and whether their current arguments are reasonable.

## 1. Current Run Context

The saved benchmark configuration in `results/tables/run_configuration.csv` shows:

- `dataset_limit = 30`
- `normalization_method = zscore`
- `clip_quantile = None`
- `window_override = None`, so the window is estimated automatically per dataset
- `threshold_std_multiplier = 3.0`
- one `Baseline` variant per algorithm

Important limitation: the current result CSVs do not preserve the exact per-variant parameter dictionaries used at run time. Because the saved run contains only one `Baseline` tab per algorithm, the parameter values below are reconstructed from the baseline widget defaults in `notebook_support.py`. This is almost certainly correct for the current run, but it is not retroactively provable from the old CSVs alone.

I patched `notebook_support.py` so future runs also save `results/tables/selected_run_parameters.csv` with the exact variant parameters.

## 2. Shared Pipeline Used By All Methods

All algorithms in `run_anomaly_detection.ipynb` follow the same outer pipeline:

1. The raw series is normalized first. In the saved run, this is `zscore`.
2. A window size `W` is chosen per dataset. If no override is given, the notebook estimates `W` from the strongest local maximum of the autocorrelation curve, with a fallback capped around 125.
3. Each algorithm returns a continuous anomaly score, not a final binary anomaly label.
4. For window-based methods, the window score is aligned back to the time axis by assigning it to the center of the window and padding the edges.
5. The notebook then converts scores to anomaly labels with one common rule:

`prediction = 1 if score >= mean(score) + 3 * std(score)`

This point is critical. In this notebook, the final anomaly count is controlled mainly by the score distribution and the `threshold_std_multiplier`, not by each model's own built-in `predict()` threshold.

## 3. High-ROI Shortlist

I define ROI here as a simple performance/time proxy:

`ROI = mean_range_f1 / mean_runtime_seconds`

based on `results/tables/algorithm_summary.csv`.

| Algorithm | Mean Range F1 | Mean Runtime (s) | ROI Proxy | Keep? |
| --- | ---: | ---: | ---: | --- |
| Matrix Profile | 0.373 | 0.084 | 4.43 | Yes |
| OCSVM | 0.222 | 0.100 | 2.22 | Yes |
| LOF | 0.226 | 0.215 | 1.05 | Yes |
| Isolation Forest | 0.207 | 0.705 | 0.29 | Optional but worth explaining |
| SAND | 0.394 | 5.326 | 0.07 | No for ROI |
| DAMP | 0.221 | 5.585 | 0.04 | No for ROI |
| HBOS | 0.116 | 0.455 | 0.26 | No |
| PCA | 0.100 | 0.127 | 0.79 | No |

If you want only three methods in the main discussion, the strongest shortlist is:

- Matrix Profile
- OCSVM
- LOF

I would still include a short Isolation Forest note because it is familiar, it appears in the results, and it raises the contamination question explicitly.

## 4. Matrix Profile

### Baseline arguments

- `subsequence_multiplier = 1`
- effective subsequence length `m = min(series_length - 1, max(4, window_size * subsequence_multiplier))`

### How it finds anomalies

This method computes the matrix profile of the series. For each subsequence of length `m`, it finds the distance to its nearest other subsequence. A subsequence with a large nearest-neighbor distance is a discord, so it is treated as anomalous.

In the code, this is:

- `profile = stumpy.stump(values, m=subsequence_length)[:, 0]`
- the profile is min-max normalized
- the score is aligned back to the center of each subsequence
- final anomaly labels come from the common `mean + 3*std` threshold

### Are the arguments reasonable?

Yes. This is the cleanest high-ROI baseline in the current notebook.

- `subsequence_multiplier = 1` is a good default because it ties the discord length directly to the estimated base window.
- The most important parameter is the estimated `window_size`, not the multiplier.
- If you want one extra variant, test `subsequence_multiplier = 2`. That is enough for a serious discussion.

## 5. OCSVM

### Baseline arguments

- `kernel = "rbf"`
- `nu = 0.05`
- `gamma = "scale"`
- `train_fraction = 0.10`

### How it finds anomalies

The series is converted into sliding windows of length `W`.

Then the code applies an additional row-wise min-max normalization to each window before fitting the model. This means OCSVM is working more on window shape than on raw amplitude.

The model is trained only on the earliest `10%` of windows, with a minimum of 8 training windows and a cap at the total number of windows:

- `training_windows = normalized_windows[:train_count]`

Then every window is scored with:

- `window_scores = -model.decision_function(normalized_windows)`

For One-Class SVM, `decision_function` is positive for inliers and negative for outliers, so the minus sign makes larger values mean "more anomalous".

### Are the arguments reasonable?

Mostly yes, with one caveat.

- `kernel = "rbf"` and `gamma = "scale"` are standard, defensible defaults.
- `nu = 0.05` is reasonable for sparse anomalies.
- `train_fraction = 0.10` is only sensible if the first 10% of the series is mostly normal. That matches many UCR-style datasets, but it is an assumption.

What I would improve:

- Keep this algorithm in the high-ROI set.
- Add one more `nu` value, for example `0.10`.
- Add one more `train_fraction`, for example `0.20`, to test sensitivity to the warm-up assumption.

## 6. Local Outlier Factor

### Baseline arguments

- `n_neighbors = 20`
- `contamination = 0.10`
- `algorithm = "auto"`
- `leaf_size = 30`
- `metric = "minkowski"`
- `p = 2`, so this is effectively Euclidean distance

### How it finds anomalies

Again, the series is converted into sliding windows of length `W`.

LOF compares the local density of each window to the local density of its neighbors. If a window lies in a much lower-density region than its neighbors, its LOF value is high and it is treated as an outlier.

In the code:

- `model.fit_predict(windows)`
- `window_scores = -model.negative_outlier_factor_`

Also, the actual neighbor count is clipped so it cannot exceed the number of available windows minus one.

`negative_outlier_factor_` is more negative for more abnormal windows, so the minus sign again turns larger values into more anomalous scores.

### Are the arguments reasonable?

Mostly yes.

- `n_neighbors = 20` is a solid baseline.
- Euclidean distance is a sensible first choice.
- `algorithm` and `leaf_size` are performance knobs more than theory knobs.

Important notebook-specific note:

- In standard LOF, `contamination` is used to set the internal threshold for binary labels.
- In this notebook, those binary labels are not used in the benchmark.
- The notebook uses `negative_outlier_factor_` and then applies its own global `mean + 3*std` threshold.

So here, `contamination = 0.10` does not mean "find 10% anomalies" in the final figures. In this pipeline, `n_neighbors`, `metric`, and the common thresholding rule matter more than `contamination`.

## 7. Isolation Forest

### Baseline arguments

- `n_estimators = 200`
- `contamination = 0.10`
- `max_samples = "auto"`
- `max_features = 1.0`
- `bootstrap = False`
- `random_state = 42`

### How it finds anomalies

The series is converted into sliding windows of length `W`, and each window becomes a feature vector.

Isolation Forest builds many random trees. Windows that get isolated after only a few random splits have short path lengths, so they are considered more abnormal.

In the code:

- `model.fit(windows)`
- `window_scores = -model.score_samples(windows)`

For Isolation Forest, lower `score_samples` values are more abnormal, so the minus sign turns them into higher anomaly scores.

### Does contamination force the number of anomalies here?

No, not in this notebook.

In scikit-learn, `contamination` is used to define the threshold for the model's own decision function / binary labeling. But this notebook does not use `predict()` or `decision_function()` for the benchmark. It uses raw `score_samples()`, then applies the notebook's own `mean + 3*std` threshold afterward.

So the practical answer for the thesis is:

- in vanilla Isolation Forest, `contamination` is a threshold-calibration parameter
- in this notebook's pipeline, it does **not** directly tell the method how many anomalies to output in the final figures
- the final anomaly count is driven more by the post-processing threshold than by `contamination`

### Are the arguments reasonable?

Yes, but this is not one of the best ROI methods in the current run.

- `n_estimators = 200` is a good stable default.
- `max_samples = "auto"` is also reasonable; in scikit-learn this means `min(256, n_samples)`.
- `max_features = 1.0` is fine for a baseline.
- `random_state = 42` is correct for reproducibility.

What I would change in the write-up:

- Do not present `contamination = 0.10` as "we asked the model to find 10% anomalies". That would be misleading for this notebook.
- Present Isolation Forest as a window-scoring method whose final labels are produced by the notebook's external threshold.

## 8. Bottom Line

If you want the clearest and most defensible thesis discussion with high ROI only:

- Main methods: Matrix Profile, OCSVM, LOF
- Short extra note: Isolation Forest, mainly to clarify the contamination issue

If you want one sentence that captures the whole notebook:

"Each method produces a continuous anomaly score over sliding windows, that score is aligned back to the time axis, and the final anomalies shown in the figures are the time points whose score exceeds `mean(score) + 3 * std(score)`."
