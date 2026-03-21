from __future__ import annotations

import math

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler


def _rolling_windows(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size >= values.size:
        raise ValueError("window_size must be smaller than the time-series length")
    return np.lib.stride_tricks.sliding_window_view(values, window_shape=window_size)


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float).reshape(-1, 1)
    if scores.size == 0 or np.allclose(scores.min(), scores.max()):
        return np.zeros(scores.shape[0], dtype=float)
    return MinMaxScaler().fit_transform(scores).ravel()


def _align_scores(window_scores: np.ndarray, window_size: int, series_length: int) -> np.ndarray:
    pad_left = math.ceil((window_size - 1) / 2)
    pad_right = (window_size - 1) // 2
    return np.pad(window_scores, (pad_left, pad_right), mode="edge")[:series_length]


def score_time_series(
    values: np.ndarray,
    window_size: int,
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

    windows = _rolling_windows(values, window_size)
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
    return _align_scores(normalized_scores, window_size, values.size)


__all__ = ["score_time_series"]
