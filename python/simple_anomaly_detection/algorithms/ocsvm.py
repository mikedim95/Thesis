from __future__ import annotations

import math

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM


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


def _rowwise_minmax(windows: np.ndarray) -> np.ndarray:
    row_min = windows.min(axis=1, keepdims=True)
    row_max = windows.max(axis=1, keepdims=True)
    scale = row_max - row_min
    scale[scale == 0.0] = 1.0
    return (windows - row_min) / scale


def _parse_gamma(gamma: str | float) -> str | float:
    if isinstance(gamma, str):
        text = gamma.strip().lower()
        if text in {"scale", "auto"}:
            return text
        return float(text)
    return float(gamma)


def score_time_series(
    values: np.ndarray,
    window_size: int,
    *,
    kernel: str = "rbf",
    nu: float = 0.05,
    gamma: str | float = "scale",
    degree: int = 3,
    coef0: float = 0.0,
    tol: float = 1e-3,
    shrinking: bool = True,
    train_fraction: float = 0.10,
    max_iter: int = -1,
) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    window_size = int(window_size)

    windows = _rolling_windows(values, window_size)
    normalized_windows = _rowwise_minmax(windows)
    effective_train_fraction = min(max(float(train_fraction), 0.01), 1.0)
    train_count = min(len(normalized_windows), max(8, int(round(len(normalized_windows) * effective_train_fraction))))
    training_windows = normalized_windows[:train_count]

    model = OneClassSVM(
        kernel=kernel,
        nu=float(nu),
        gamma=_parse_gamma(gamma),
        degree=int(degree),
        coef0=float(coef0),
        tol=float(tol),
        shrinking=bool(shrinking),
        max_iter=int(max_iter),
    )
    model.fit(training_windows)

    window_scores = -model.decision_function(normalized_windows)
    normalized_scores = _normalize_scores(window_scores)
    return _align_scores(normalized_scores, window_size, values.size)


__all__ = ["score_time_series"]
