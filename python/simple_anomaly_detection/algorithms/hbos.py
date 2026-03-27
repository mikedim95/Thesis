from __future__ import annotations

import math

import numpy as np
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


def _calculate_feature_scores(
    windows: np.ndarray,
    *,
    n_bins: int,
    alpha: float,
    tol: float,
) -> np.ndarray:
    n_samples, n_features = windows.shape
    hist = np.zeros((n_bins, n_features), dtype=float)
    bin_edges = np.zeros((n_bins + 1, n_features), dtype=float)

    for feature_index in range(n_features):
        hist[:, feature_index], bin_edges[:, feature_index] = np.histogram(
            windows[:, feature_index],
            bins=n_bins,
            density=True,
        )

    outlier_scores = np.zeros((n_samples, n_features), dtype=float)
    for feature_index in range(n_features):
        bin_indices = np.digitize(windows[:, feature_index], bin_edges[:, feature_index], right=True)
        density_scores = np.log2(hist[:, feature_index] + alpha)
        for sample_index, bin_index in enumerate(bin_indices):
            if bin_index == 0:
                dist = bin_edges[0, feature_index] - windows[sample_index, feature_index]
                bin_width = bin_edges[1, feature_index] - bin_edges[0, feature_index]
                outlier_scores[sample_index, feature_index] = (
                    density_scores[0] if dist <= bin_width * tol else np.min(density_scores)
                )
            elif bin_index == n_bins + 1:
                dist = windows[sample_index, feature_index] - bin_edges[-1, feature_index]
                bin_width = bin_edges[-1, feature_index] - bin_edges[-2, feature_index]
                outlier_scores[sample_index, feature_index] = (
                    density_scores[n_bins - 1] if dist <= bin_width * tol else np.min(density_scores)
                )
            else:
                outlier_scores[sample_index, feature_index] = density_scores[bin_index - 1]

    return -np.sum(outlier_scores, axis=1)


def score_time_series(
    values: np.ndarray,
    window_size: int,
    window_stride: int = 1,
    *,
    n_bins: int = 10,
    alpha: float = 0.1,
    tol: float = 0.5,
    contamination: float = 0.1,
) -> np.ndarray:
    del contamination

    values = np.asarray(values, dtype=float).ravel()
    window_size = int(window_size)
    windows = _rolling_windows(values, window_size, window_stride=window_stride)

    window_scores = _calculate_feature_scores(
        windows,
        n_bins=max(2, int(n_bins)),
        alpha=float(alpha),
        tol=float(tol),
    )
    normalized_scores = _normalize_scores(window_scores)
    return _align_scores(normalized_scores, window_size, values.size, window_stride=window_stride)


__all__ = ["score_time_series"]
