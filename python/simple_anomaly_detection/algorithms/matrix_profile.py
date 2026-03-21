from __future__ import annotations

import math

import numpy as np
import stumpy
from sklearn.preprocessing import MinMaxScaler


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float).reshape(-1, 1)
    finite_mask = np.isfinite(scores).ravel()
    if not finite_mask.any():
        return np.zeros(scores.shape[0], dtype=float)
    finite_scores = scores[finite_mask]
    if np.allclose(finite_scores.min(), finite_scores.max()):
        normalized = np.zeros(scores.shape[0], dtype=float)
        normalized[finite_mask] = 0.0
        return normalized
    cleaned = scores.copy()
    cleaned[~finite_mask] = finite_scores.max()
    return MinMaxScaler().fit_transform(cleaned).ravel()


def _align_scores(window_scores: np.ndarray, subsequence_length: int, series_length: int) -> np.ndarray:
    pad_left = math.ceil((subsequence_length - 1) / 2)
    pad_right = (subsequence_length - 1) // 2
    return np.pad(window_scores, (pad_left, pad_right), mode="edge")[:series_length]


def score_time_series(
    values: np.ndarray,
    window_size: int,
    *,
    subsequence_multiplier: int = 1,
) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    window_size = int(window_size)
    subsequence_multiplier = max(1, int(subsequence_multiplier))

    subsequence_length = min(values.size - 1, max(4, window_size * subsequence_multiplier))
    if subsequence_length >= values.size:
        raise ValueError("subsequence_length must be smaller than the time-series length")

    profile = stumpy.stump(values, m=subsequence_length)[:, 0]
    normalized_scores = _normalize_scores(profile)
    return _align_scores(normalized_scores, subsequence_length, values.size)


__all__ = ["score_time_series"]
