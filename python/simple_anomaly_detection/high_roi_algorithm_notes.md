# High-ROI Algorithm Notes

This file is generated from `notebook_support.py` and the detector implementations in `python/simple_anomaly_detection/algorithms/`.

## Where the algorithm logic is stated

- Shared run orchestration, thresholding, and benchmark aggregation live in `python/simple_anomaly_detection/notebook_support.py`.
- The detector scoring logic itself lives in the per-algorithm modules listed below.
- The notebook UI now mirrors the same explanation in each algorithm tab under the new process/knob accordion.

## Shared scoring pipeline

1. The raw time series is optionally clipped by `Clip q`, then normalized by `Normalize` before any detector sees it.
1. Each detector receives the normalized 1D series together with a base `window_size` and `window_stride` from the general controls.
1. Every detector returns a continuous anomaly score trace, and the notebook compares detectors on that score trace first.
1. Each algorithm module rescales its own score trace to `[0, 1]` before returning it to the notebook.
1. `Threshold method` and `Threshold value` are applied after scoring to turn the continuous trace into binary detections.
1. `Evaluation mode` changes only the metric calculation, not the detector score itself.

## General controls visible in the notebook

- `Run name`: Binds the current setup to a saved session folder under results/run_sessions so you can reload the exact controls later and continue from the last successful checkpoint.
- `Argument mode`: Selects whether the run uses the visible subtabs exactly (`manual`) or replaces them with curated multi-variant sweeps from `paper_high_roi`, `paper_full_suite`, or `auto_ablation` at run time.
- `Dataset limit`: Filters how many prepared datasets are benchmarked. It does not change anomaly scores on any individual dataset.
- `Batch size`: Caps how many of the selected datasets are processed in the current notebook run. With resume enabled, it defines the size of each resumable batch.
- `Resume from existing`: Reuses successful benchmark rows already saved under the current run name, skips completed dataset/configuration pairs, and continues from the next incomplete work for the same benchmark setup.
- `Normalize`: Changes the numeric scale seen by every detector before windowing. This can materially change distance-based, density-based, and boundary-based scores.
- `Clip q`: Clamps extreme tails before normalization. This can suppress large spikes, which may reduce false positives or weaken genuine anomaly contrast.
- `Window size`: Sets the base temporal context used for sliding-window embedding. Matrix Profile, DAMP, and SAND also derive their subsequence lengths or warm-up positions from this base length.
- `Window stride`: Controls how densely windows or subsequences are sampled. Larger strides reduce overlap and runtime; most detectors then interpolate the lower-frequency window scores back onto the original time axis.
- `Threshold method`: Chooses the post-scoring rule that converts the normalized score trace into binary anomalies. It does not alter the detector's raw scoring process.
- `Threshold value`: Provides the numeric cutoff for the selected threshold rule. It changes which high-score regions are finally marked anomalous, not how the score trace is produced.
- `Evaluation mode`: Chooses whether metrics are computed as interval overlap (`range`) or exact point hits (`point`). It affects reported scores only.
- `Rebuild normalized datasets`: Forces regeneration of cached normalized CSV files. This is a data-preparation control and does not change detector math by itself.
- `Save per-dataset scores`: Writes the returned score traces to `results/scores/`. It does not change scoring or thresholding.
- `Algorithm checkboxes`: Choose which detectors are included in the benchmark. In auto modes they also act as a filter over the preset sweeps.

## Automatic sweep mode

- `manual`: use the visible subtabs exactly as edited.
- `paper_high_roi`: automatically benchmark the curated high-return variants from `PAPER_PRESET_DEFINITIONS` for the enabled high-ROI algorithms.
- `paper_full_suite`: automatically benchmark the broader appendix-style variants from `PAPER_PRESET_DEFINITIONS` for every enabled algorithm that has a configured sweep.
- `auto_ablation`: automatically benchmark one baseline plus one-knob-at-a-time ablations so parameter claims are paired against a fixed reference.
- In auto modes, the algorithm checkboxes still filter what runs, but the current subtab values are ignored at run time.

## Algorithm-by-algorithm reference

### Isolation Forest

- Implementation: `algorithms/isolation_forest.py`
- Summary: Fits an Isolation Forest on sliding windows and uses the negated `score_samples` output as the anomaly score.

How the score is produced:
1. Build overlapping rolling windows from the normalized series with length `window_size` and step `window_stride`.
2. Fit `sklearn.ensemble.IsolationForest` on those windows.
3. Compute `window_scores = -model.score_samples(windows)`, so windows isolated more easily by the trees receive higher anomaly scores.
4. Min-max normalize the window scores to `[0, 1]`.
5. Align the window scores back to the original time axis by center padding when stride is `1`, otherwise by interpolation across window centers.

Visible controls and exact effect:
- `Trees` (`n_estimators`): Adds more trees to the forest. More trees usually stabilize the average isolation score, but increase runtime and memory.
- `Max samples` (`max_samples`): Controls how many windows each tree is fit on. Smaller samples make each tree more local and cheaper; larger samples expose more global structure. `'auto'` delegates the sample count to sklearn's default rule.
- `Max feat.` (`max_features`): Controls what fraction of the window positions each tree can split on. Lower values add stronger feature subsampling; higher values let each tree use more of the full temporal context.
- `Bootstrap` (`bootstrap`): Switches tree training from sampling without replacement to sampling with replacement, which changes how much repeated windows can influence each tree.
- `Seed` (`random_state`): Fixes the random draws used by the forest so score traces are reproducible across runs.

Auto sweep variants:
- `paper_high_roi` -> `Baseline`: Stable tree baseline for comparison. (`n_estimators=200, max_samples=256, max_features=1, bootstrap=false, random_state=42`)
- `paper_high_roi` -> `Wide Sample`: More trees and larger sampling for a smoother global isolation score. (`n_estimators=400, max_samples=auto, max_features=1, bootstrap=false, random_state=42`)
- `paper_high_roi` -> `Feat 0.6`: Tests stronger random feature subsampling inside the forest. (`n_estimators=400, max_samples=256, max_features=0.6, bootstrap=false, random_state=42`)
- `paper_full_suite` -> `Baseline`: Stable tree baseline for comparison. (`n_estimators=200, max_samples=256, max_features=1, bootstrap=false, random_state=42`)
- `paper_full_suite` -> `Wide Sample`: More trees and larger sampling for a smoother global isolation score. (`n_estimators=400, max_samples=auto, max_features=1, bootstrap=false, random_state=42`)
- `paper_full_suite` -> `Feat 0.6`: Tests stronger random feature subsampling inside the forest. (`n_estimators=400, max_samples=256, max_features=0.6, bootstrap=false, random_state=42`)
- `auto_ablation` -> `Baseline`: Stable tree baseline for comparison. (`n_estimators=200, max_samples=256, max_features=1, bootstrap=false, random_state=42`)
- `auto_ablation` -> `Trees 400`: Measures whether a larger forest improves score stability enough to justify the runtime. (`n_estimators=400, max_samples=256, max_features=1, bootstrap=false, random_state=42`)
- `auto_ablation` -> `Max samples auto`: Measures how a broader training sample per tree changes the global isolation pattern. (`n_estimators=200, max_samples=auto, max_features=1, bootstrap=false, random_state=42`)
- `auto_ablation` -> `Max feat 0.6`: Measures the effect of stronger random feature subsampling inside the forest. (`n_estimators=200, max_samples=256, max_features=0.6, bootstrap=false, random_state=42`)
- `auto_ablation` -> `Bootstrap on`: Measures whether sampling windows with replacement changes the tree ensemble enough to alter detections. (`n_estimators=200, max_samples=256, max_features=1, bootstrap=true, random_state=42`)
- `auto_ablation` -> `Seed 7`: Measures sensitivity to stochastic initialization rather than score geometry. (`n_estimators=200, max_samples=256, max_features=1, bootstrap=false, random_state=7`)

### Local Outlier Factor

- Implementation: `algorithms/local_outlier_factor.py`
- Summary: Computes LOF on sliding windows and uses the negated `negative_outlier_factor_` as the anomaly score.

How the score is produced:
1. Build rolling windows from the normalized series.
2. Clamp `n_neighbors` into the valid range with `effective_neighbors = max(2, min(n_neighbors, len(windows) - 1))`.
3. Fit `sklearn.neighbors.LocalOutlierFactor` on the windows and call `fit_predict` to populate the local density ratios.
4. Compute `window_scores = -model.negative_outlier_factor_`, so windows with much lower local density than their neighbors score higher.
5. Min-max normalize and align the window scores back onto the original time axis.

Visible controls and exact effect:
- `Neighbors` (`n_neighbors`): Changes the size of the neighborhood used to estimate local density. Smaller values make the score more local and sensitive; larger values make it smoother and more global.
- `Search` (`algorithm`): Changes only the sklearn nearest-neighbor backend (`auto`, `ball_tree`, `kd_tree`, `brute`). It mainly affects runtime, not the density formula itself.
- `Leaf size` (`leaf_size`): Tunes the search-tree backend used by LOF. This is a performance knob rather than a scoring-logic knob.
- `Metric` (`metric`): Changes the distance function used between windows, which directly changes who counts as a neighbor and therefore the local density ratio.
- `p` (`p`): Changes the exponent of the Minkowski distance. It only has scoring impact when `Metric = minkowski`; `p=1` is Manhattan and `p=2` is Euclidean.

Auto sweep variants:
- `paper_high_roi` -> `Baseline`: Balanced neighborhood baseline. (`n_neighbors=20, algorithm=auto, leaf_size=30, metric=minkowski, p=2`)
- `paper_high_roi` -> `Local k10`: More sensitive to local shape changes and short anomalies. (`n_neighbors=10, algorithm=auto, leaf_size=30, metric=minkowski, p=2`)
- `paper_high_roi` -> `Global L1`: Broader neighborhood with Manhattan distance for more global structure. (`n_neighbors=50, algorithm=auto, leaf_size=30, metric=manhattan, p=1`)
- `paper_full_suite` -> `Baseline`: Balanced neighborhood baseline. (`n_neighbors=20, algorithm=auto, leaf_size=30, metric=minkowski, p=2`)
- `paper_full_suite` -> `Local k10`: More sensitive to local shape changes and short anomalies. (`n_neighbors=10, algorithm=auto, leaf_size=30, metric=minkowski, p=2`)
- `paper_full_suite` -> `Global L1`: Broader neighborhood with Manhattan distance for more global structure. (`n_neighbors=50, algorithm=auto, leaf_size=30, metric=manhattan, p=1`)
- `auto_ablation` -> `Baseline`: Balanced neighborhood baseline. (`n_neighbors=20, algorithm=auto, leaf_size=30, metric=minkowski, p=2`)
- `auto_ablation` -> `Neighbors 10`: Makes LOF more local so short or sharp anomalies have more leverage. (`n_neighbors=10, algorithm=auto, leaf_size=30, metric=minkowski, p=2`)
- `auto_ablation` -> `Search brute`: Measures the neighbor-search backend while leaving the density formula unchanged. (`n_neighbors=20, algorithm=brute, leaf_size=30, metric=minkowski, p=2`)
- `auto_ablation` -> `Leaf size 60`: Measures the tree-search backend tuning rather than a scoring-theory change. (`n_neighbors=20, algorithm=auto, leaf_size=60, metric=minkowski, p=2`)
- `auto_ablation` -> `Metric manhattan`: Measures how redefining window similarity changes local density estimates. (`n_neighbors=20, algorithm=auto, leaf_size=30, metric=manhattan, p=2`)
- `auto_ablation` -> `p = 1`: Measures the Minkowski exponent directly while keeping the metric family the same. (`n_neighbors=20, algorithm=auto, leaf_size=30, metric=minkowski, p=1`)

### SAND

- Implementation: `algorithms/sand.py`
- Summary: Runs the legacy SAND online detector on a subsequence representation derived from the base window size and uses `decision_scores_` as the anomaly trace.

How the score is produced:
1. Set `pattern_length = window_size` and compute `subsequence_length = max(subsequence_multiplier * window_size, window_size + 1)`, capped by the series length.
2. Resolve the subsequence step size from `Overlap`; when the UI leaves `Overlap = 0`, the code uses `window_size` when stride is `1` and uses `window_stride` otherwise.
3. Clamp `Init length` and `Batch size` so both are at least long enough for one subsequence.
4. Clamp `k` to the smallest number of subsequences available across the initialization block and all online batches. When the UI leaves `k = 0`, the call omits `k` and the implementation falls back to SAND's default of `6`, then clamps it.
5. Fit `SAND(...).fit(..., online=True, alpha=..., init_length=..., batch_size=..., overlaping_rate=overlap)` on the normalized series.
6. Use `model.decision_scores_`, min-max normalize them, and resize the result back to the original series length.

Visible controls and exact effect:
- `Alpha` (`alpha`): Controls how strongly new batches influence the online update. Higher values adapt faster to recent behavior; lower values keep more inertia from earlier batches.
- `Init length` (`init_length`): Controls how much initial history is used before the online updates continue. Larger values give SAND a longer starting reference set.
- `Batch size` (`batch_size`): Controls how much data SAND ingests per online update. Larger batches reduce update frequency but make each update coarser and heavier.
- `k` (`k`): Sets how many neighboring subsequences SAND compares. Smaller values keep the score local; larger values smooth it. `0` means use the implementation default and then clamp it to what the data can support.
- `Subseq x` (`subsequence_multiplier`): Scales the subsequence length relative to `window_size`. Larger values make SAND compare longer contexts.
- `Overlap` (`overlap`): Sets the subsequence step used by SAND's online fit. Smaller steps create denser comparisons; larger steps reduce overlap and runtime. `0` means infer the step from the shared window settings.

Auto sweep variants:
- `paper_full_suite` -> `Baseline`: Reference online clustering configuration. (`alpha=0.5, init_length=5000, batch_size=2000, k=0, subsequence_multiplier=4, overlap=0`)
- `paper_full_suite` -> `Adaptive`: Faster adaptation with shorter context and smaller batches. (`alpha=0.7, init_length=3000, batch_size=1000, k=0, subsequence_multiplier=2, overlap=0`)
- `auto_ablation` -> `Baseline`: Reference online clustering configuration. (`alpha=0.5, init_length=5000, batch_size=2000, k=0, subsequence_multiplier=4, overlap=0`)
- `auto_ablation` -> `Alpha 0.7`: Measures faster adaptation to recent behavior in the online updates. (`alpha=0.7, init_length=5000, batch_size=2000, k=0, subsequence_multiplier=4, overlap=0`)
- `auto_ablation` -> `Init 3000`: Measures a shorter initialization phase before online updates take over. (`alpha=0.5, init_length=3000, batch_size=2000, k=0, subsequence_multiplier=4, overlap=0`)
- `auto_ablation` -> `Batch 1000`: Measures finer-grained online updates at the cost of more update steps. (`alpha=0.5, init_length=5000, batch_size=1000, k=0, subsequence_multiplier=4, overlap=0`)
- `auto_ablation` -> `k = 3`: Measures a more local nearest-subsequence comparison. (`alpha=0.5, init_length=5000, batch_size=2000, k=3, subsequence_multiplier=4, overlap=0`)
- `auto_ablation` -> `Subseq x2`: Measures a shorter subsequence context while keeping the rest of SAND fixed. (`alpha=0.5, init_length=5000, batch_size=2000, k=0, subsequence_multiplier=2, overlap=0`)
- `auto_ablation` -> `Overlap 64`: Measures a coarser explicit subsequence step rather than the auto overlap heuristic. (`alpha=0.5, init_length=5000, batch_size=2000, k=0, subsequence_multiplier=4, overlap=64`)

### Matrix Profile

- Implementation: `algorithms/matrix_profile.py`
- Summary: Uses the first column of `stumpy.stump` as a discord score, so windows whose nearest neighbor is far away score highly.

How the score is produced:
1. Compute `subsequence_length = max(4, window_size * subsequence_multiplier)`, capped below the series length.
2. Run `stumpy.stump(values, m=subsequence_length)` and keep the first column of the returned matrix profile.
3. Subsample the profile by `window_stride` when stride is greater than `1`.
4. Normalize the profile to `[0, 1]`, replacing non-finite values before scaling.
5. Align the subsequence scores back to the original time axis around each subsequence center.

Visible controls and exact effect:
- `Subseq x` (`subsequence_multiplier`): Scales the discord subsequence length relative to `window_size`. Larger values look for broader anomalous contexts; smaller values focus on shorter local discords.

Auto sweep variants:
- `paper_high_roi` -> `Context x1`: Shortest context, strongest local discord detection. (`subsequence_multiplier=1`)
- `paper_high_roi` -> `Context x2`: Intermediate subsequence context for medium-length anomalies. (`subsequence_multiplier=2`)
- `paper_high_roi` -> `Context x4`: Longer context for broad discord patterns. (`subsequence_multiplier=4`)
- `paper_full_suite` -> `Context x1`: Shortest context, strongest local discord detection. (`subsequence_multiplier=1`)
- `paper_full_suite` -> `Context x2`: Intermediate subsequence context for medium-length anomalies. (`subsequence_multiplier=2`)
- `paper_full_suite` -> `Context x4`: Longer context for broad discord patterns. (`subsequence_multiplier=4`)
- `auto_ablation` -> `Baseline`: Balanced discord context for one-factor ablation. (`subsequence_multiplier=2`)
- `auto_ablation` -> `Subseq x1`: Measures a shorter discord context focused on local deviations. (`subsequence_multiplier=1`)
- `auto_ablation` -> `Subseq x4`: Measures a broader discord context for long anomalous structure. (`subsequence_multiplier=4`)

### DAMP

- Implementation: `algorithms/damp.py`
- Summary: Runs the DAMP streaming-discord search and uses backward nearest-neighbor distances as the anomaly score.

How the score is produced:
1. Set the DAMP start position `sp_index = max(window_size + 1, round(window_size * start_index_multiplier) + 1)`.
2. Resolve `x_lag`; when the UI leaves it at `0`, DAMP falls back to its internal heuristic `2^ceil(log2(8 * window_size))`.
3. From `sp_index` onward, compare each current window against historical reference windows using repeated MASS nearest-neighbor searches. The returned nearest-neighbor distance is the discord score.
4. Use DAMP's forward-pruning pass to skip future windows that already have a close enough match.
5. Subsample by `window_stride`, normalize the resulting profile, and align it back onto the original time axis.

Visible controls and exact effect:
- `Start x` (`start_index_multiplier`): Moves the first scored window later in time by a multiple of `window_size`. Larger values delay scoring and give the method more history before the backward search starts.
- `x_lag x` (`x_lag_multiplier`): Sets how far back the backward search can look, as a multiple of `window_size`. Larger values widen the historical search horizon; `0` means use DAMP's internal heuristic instead.

Auto sweep variants:
- `paper_full_suite` -> `Baseline`: Reference streaming-discord configuration. (`start_index_multiplier=1, x_lag_multiplier=0`)
- `paper_full_suite` -> `Delayed Start`: Longer historical reference before streaming detection starts. (`start_index_multiplier=2, x_lag_multiplier=0`)
- `paper_full_suite` -> `Long Lag`: Searches further back in the stream for similar windows. (`start_index_multiplier=1, x_lag_multiplier=8`)
- `auto_ablation` -> `Baseline`: Reference streaming-discord configuration. (`start_index_multiplier=1, x_lag_multiplier=0`)
- `auto_ablation` -> `Start x2`: Measures the effect of waiting longer before the backward search begins. (`start_index_multiplier=2, x_lag_multiplier=0`)
- `auto_ablation` -> `x_lag x8`: Measures a much deeper historical search horizon during backward matching. (`start_index_multiplier=1, x_lag_multiplier=8`)

### HBOS

- Implementation: `algorithms/hbos.py`
- Summary: Builds one histogram per window position and scores each window by the negative sum of log histogram densities.

How the score is produced:
1. Build rolling windows from the normalized series.
2. For each feature position inside the window, build a histogram with `n_bins` bins over that column across all windows.
3. For each window value, look up the corresponding histogram-bin density and compute `log2(hist + alpha)`.
4. If a value falls outside the learned histogram range, use `Tol` to decide whether to borrow the nearest edge-bin density or assign the minimum-density penalty.
5. Sum the negative log densities across all window positions to get the window anomaly score, then normalize and align it back to the time axis.

Visible controls and exact effect:
- `Bins` (`n_bins`): Changes the histogram resolution at each window position. More bins capture finer structure; fewer bins smooth the density estimate.
- `Alpha` (`alpha`): Adds smoothing inside `log2(hist + alpha)`, preventing empty bins from producing undefined scores and softening sparse-bin penalties.
- `Tol` (`tol`): Controls how far outside a histogram edge a value can land before it receives the harsh minimum-density penalty instead of the nearest edge-bin density.

Auto sweep variants:
- `paper_full_suite` -> `Baseline`: Lightweight histogram baseline. (`n_bins=10, alpha=0.1, tol=0.5`)
- `paper_full_suite` -> `Fine Bins`: Finer density structure through more bins and milder smoothing. (`n_bins=20, alpha=0.05, tol=0.5`)
- `paper_full_suite` -> `Strict Tol`: Stricter edge handling for stronger outlier penalties. (`n_bins=10, alpha=0.1, tol=0.2`)
- `auto_ablation` -> `Baseline`: Lightweight histogram baseline. (`n_bins=10, alpha=0.1, tol=0.5`)
- `auto_ablation` -> `Bins 20`: Measures finer histogram resolution at every window position. (`n_bins=20, alpha=0.1, tol=0.5`)
- `auto_ablation` -> `Alpha 0.05`: Measures milder smoothing inside the log-density score. (`n_bins=10, alpha=0.05, tol=0.5`)
- `auto_ablation` -> `Tol 0.2`: Measures stricter out-of-range penalties near histogram edges. (`n_bins=10, alpha=0.1, tol=0.2`)

### OCSVM

- Implementation: `algorithms/ocsvm.py`
- Summary: Fits One-Class SVM on the earliest normalized windows and scores all windows with the negated `decision_function`.

How the score is produced:
1. Build rolling windows from the normalized series.
2. Apply row-wise min-max scaling so every window is independently mapped into `[0, 1]` before model fitting.
3. Choose the training prefix from the earliest windows using `train_fraction`, clamped so at least `8` windows are used when available.
4. Fit `sklearn.svm.OneClassSVM` on that prefix only.
5. Compute `window_scores = -model.decision_function(all_windows)` so windows farther outside the learned boundary score higher.
6. Normalize and align the score trace back to the original time axis.

Visible controls and exact effect:
- `Kernel` (`kernel`): Changes the geometry of the decision boundary in window space. `linear` uses a flat boundary, while `rbf`, `poly`, and `sigmoid` introduce nonlinear boundaries.
- `Nu` (`nu`): Sets the One-Class SVM regularization level that bounds training errors and support-vector fraction. Larger values usually make the model more willing to score windows as abnormal.
- `Gamma` (`gamma`): Controls the locality of nonlinear kernels. Higher values make the boundary respond to finer local variation; lower values make it smoother. `linear` largely ignores this setting.
- `Train frac` (`train_fraction`): Sets how much of the earliest series prefix is assumed to be mostly normal and therefore used for fitting before scoring the full series.

Auto sweep variants:
- `paper_high_roi` -> `Baseline`: Standard RBF novelty boundary with short warm-up. (`kernel=rbf, nu=0.05, gamma=scale, train_fraction=0.1`)
- `paper_high_roi` -> `Nu 0.10`: Tests more permissive abnormal-boundary calibration. (`kernel=rbf, nu=0.1, gamma=scale, train_fraction=0.1`)
- `paper_high_roi` -> `Warmup 0.20`: Uses a longer mostly-normal prefix for model fitting. (`kernel=rbf, nu=0.05, gamma=scale, train_fraction=0.2`)
- `paper_full_suite` -> `Baseline`: Standard RBF novelty boundary with short warm-up. (`kernel=rbf, nu=0.05, gamma=scale, train_fraction=0.1`)
- `paper_full_suite` -> `Nu 0.10`: Tests more permissive abnormal-boundary calibration. (`kernel=rbf, nu=0.1, gamma=scale, train_fraction=0.1`)
- `paper_full_suite` -> `Warmup 0.20`: Uses a longer mostly-normal prefix for model fitting. (`kernel=rbf, nu=0.05, gamma=scale, train_fraction=0.2`)
- `paper_full_suite` -> `Linear`: Tests whether a simpler linear novelty boundary is sufficient. (`kernel=linear, nu=0.05, gamma=scale, train_fraction=0.1`)
- `auto_ablation` -> `Baseline`: Standard RBF novelty boundary with short warm-up. (`kernel=rbf, nu=0.05, gamma=scale, train_fraction=0.1`)
- `auto_ablation` -> `Kernel linear`: Measures whether a linear novelty boundary is sufficient in the embedded window space. (`kernel=linear, nu=0.05, gamma=scale, train_fraction=0.1`)
- `auto_ablation` -> `Nu 0.10`: Measures a more permissive novelty boundary. (`kernel=rbf, nu=0.1, gamma=scale, train_fraction=0.1`)
- `auto_ablation` -> `Gamma 0.1`: Measures a more local nonlinear boundary than the default `scale` heuristic. (`kernel=rbf, nu=0.05, gamma=0.1, train_fraction=0.1`)
- `auto_ablation` -> `Train frac 0.20`: Measures a longer mostly-normal warmup segment for fitting the boundary. (`kernel=rbf, nu=0.05, gamma=scale, train_fraction=0.2`)

### PCA

- Implementation: `algorithms/pca.py`
- Summary: Fits PCA on sliding windows and scores each window by its distance to the selected principal-component vectors used by the current implementation.

How the score is produced:
1. Build rolling windows from the normalized series.
2. If `Standardize` is enabled, standardize each window position across the full window matrix with `StandardScaler` before PCA.
3. Fit `sklearn.decomposition.PCA` with the requested `Components` and `Whiten` settings.
4. Select the trailing principal-component vectors with `model.components_[-effective_selected:, :]`, where `effective_selected` comes from `Score comps` and defaults to all retained components when `0` is entered.
5. Score each window with `sum(cdist(transformed_windows, selected_components) / selected_weights, axis=1)`. When `Weighted` is enabled, `selected_weights` comes from `explained_variance_ratio_`, so lower-variance components contribute more strongly because they divide by smaller values.
6. Normalize and align the resulting score trace back to the original time axis.

Visible controls and exact effect:
- `Components` (`n_components`): Sets how many principal components PCA retains. Blank means sklearn's default behavior; a float such as `0.95` uses explained variance; an integer keeps an exact count.
- `Score comps` (`n_selected_components`): Chooses how many trailing principal-component vectors are used in the score. Smaller values focus the score on the lowest-variance directions; `0` means score against all retained components.
- `Whiten` (`whiten`): Requests PCA whitening, which rescales the retained components before the distance calculation.
- `Weighted` (`weighted`): Divides each component distance by its explained-variance ratio. This gives low-variance components more leverage in the final anomaly score.
- `Standardize` (`standardization`): Applies featurewise standardization before PCA so each position inside the window contributes on a comparable scale.

Auto sweep variants:
- `paper_full_suite` -> `Baseline`: Weighted reconstruction-style baseline. (`n_components=, n_selected_components=0, whiten=false, weighted=true, standardization=true`)
- `paper_full_suite` -> `Residual 2`: Focuses scoring on a small set of low-variance components. (`n_components=0.95, n_selected_components=2, whiten=false, weighted=true, standardization=true`)
- `paper_full_suite` -> `Whitened`: Tests a whitened non-weighted PCA score. (`n_components=0.95, n_selected_components=0, whiten=true, weighted=false, standardization=true`)
- `auto_ablation` -> `Baseline`: Weighted reconstruction-style baseline. (`n_components=, n_selected_components=0, whiten=false, weighted=true, standardization=true`)
- `auto_ablation` -> `Components 0.95`: Measures explained-variance truncation rather than retaining the default PCA basis. (`n_components=0.95, n_selected_components=0, whiten=false, weighted=true, standardization=true`)
- `auto_ablation` -> `Score comps 2`: Measures scoring on a narrow set of trailing low-variance components. (`n_components=, n_selected_components=2, whiten=false, weighted=true, standardization=true`)
- `auto_ablation` -> `Whiten on`: Measures whitening of retained components before the PCA score is computed. (`n_components=, n_selected_components=0, whiten=true, weighted=true, standardization=true`)
- `auto_ablation` -> `Weighted off`: Measures the score without explained-variance weighting. (`n_components=, n_selected_components=0, whiten=false, weighted=false, standardization=true`)
- `auto_ablation` -> `Standardize off`: Measures PCA on raw normalized windows without per-feature standardization. (`n_components=, n_selected_components=0, whiten=false, weighted=true, standardization=false`)
