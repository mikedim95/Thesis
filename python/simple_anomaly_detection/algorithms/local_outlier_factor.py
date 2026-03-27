from __future__ import annotations

import math

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler


def _rolling_windows(values: np.ndarray, window_size: int, window_stride: int = 1) -> np.ndarray:
    if window_size >= values.size:
        raise ValueError("window_size must be smaller than the time-series length")
    return np.lib.stride_tricks.sliding_window_view(values, window_shape=window_size)[:: max(1, int(window_stride))]


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float).reshape(-1, 1)
    if scores.size == 0 or np.allclose(scores.min(), scores.max()):
        return np.zeros(scores.shape[0], dtype=float)
    return MinMaxScaler().fit_transform(scores).ravel()


def _align_scores(window_scores: np.ndarray, window_size: int, series_length: int, window_stride: int = 1) -> np.ndarray:
    if window_scores.size == 0:
        return np.zeros(series_length, dtype=float)
    if max(1, int(window_stride)) == 1:
        pad_left = math.ceil((window_size - 1) / 2)
        pad_right = (window_size - 1) // 2
        return np.pad(window_scores, (pad_left, pad_right), mode="edge")[:series_length]
    window_starts = np.arange(window_scores.size, dtype=float) * max(1, int(window_stride))
    centers = np.clip(window_starts + (window_size - 1) / 2.0, 0, series_length - 1)
    if window_scores.size == 1:
        return np.full(series_length, float(window_scores[0]), dtype=float)
    return np.interp(np.arange(series_length, dtype=float), centers, window_scores, left=window_scores[0], right=window_scores[-1])


def score_time_series(
    values: np.ndarray,
    window_size: int,
    window_stride: int = 1,
    *,
    n_neighbors: int = 20,
    contamination: float = 0.1,
    algorithm: str = "auto",
    leaf_size: int = 30,
    metric: str = "minkowski",
    p: int = 2,
    n_jobs: int = -1,
) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    window_size = int(window_size)

    windows = _rolling_windows(values, window_size, window_stride=window_stride)
    effective_neighbors = max(2, min(int(n_neighbors), len(windows) - 1))

    model = LocalOutlierFactor(
        n_neighbors=effective_neighbors,
        contamination=contamination,
        algorithm=algorithm,
        leaf_size=leaf_size,
        metric=metric,
        p=p,
        n_jobs=n_jobs,
    )
    model.fit_predict(windows)

    window_scores = -model.negative_outlier_factor_
    normalized_scores = _normalize_scores(window_scores)
    return _align_scores(normalized_scores, window_size, values.size, window_stride=window_stride)


__all__ = ["score_time_series"]
