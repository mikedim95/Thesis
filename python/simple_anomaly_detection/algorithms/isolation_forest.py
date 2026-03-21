from __future__ import annotations

import math

import numpy as np
from sklearn.ensemble import IsolationForest
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
    n_estimators: int = 200,
    max_samples: str | int | float = "auto",
    contamination: float = 0.1,
    max_features: float = 1.0,
    bootstrap: bool = False,
    n_jobs: int = -1,
    random_state: int = 42,
) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    window_size = int(window_size)

    windows = _rolling_windows(values, window_size)
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        max_features=max_features,
        bootstrap=bootstrap,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    model.fit(windows)

    window_scores = -model.score_samples(windows)
    normalized_scores = _normalize_scores(window_scores)
    return _align_scores(normalized_scores, window_size, values.size)


__all__ = ["score_time_series"]
