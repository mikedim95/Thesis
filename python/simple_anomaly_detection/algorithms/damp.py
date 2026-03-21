from __future__ import annotations

import math

import numpy as np
import stumpy as st
from sklearn.preprocessing import MinMaxScaler


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float).reshape(-1, 1)
    finite_mask = np.isfinite(scores).ravel()
    if not finite_mask.any():
        return np.zeros(scores.shape[0], dtype=float)
    finite_scores = scores[finite_mask]
    cleaned = scores.copy()
    cleaned[~finite_mask] = finite_scores.max()
    if np.allclose(cleaned.min(), cleaned.max()):
        return np.zeros(cleaned.shape[0], dtype=float)
    return MinMaxScaler().fit_transform(cleaned).ravel()


def _align_scores(window_scores: np.ndarray, window_size: int, series_length: int) -> np.ndarray:
    pad_left = math.ceil((window_size - 1) / 2)
    pad_right = (window_size - 1) // 2
    return np.pad(window_scores, (pad_left, pad_right), mode="edge")[:series_length]


class _DAMPModel:
    def __init__(self, window_size: int, sp_index: int, x_lag: int | None = None):
        self.window_size = int(window_size)
        self.sp_index = int(sp_index)
        self.x_lag = x_lag if x_lag is not None else 2 ** int(np.ceil(np.log2(8 * self.window_size)))
        self._bfs = 0.0

    def fit(self, values: np.ndarray) -> np.ndarray:
        pv = np.ones(len(values) - self.window_size + 1, dtype=int)
        profile = np.zeros_like(pv, dtype=float)

        for index in range(self.sp_index, len(values) - self.window_size + 1):
            if not pv[index]:
                profile[index] = profile[index - 1]
            else:
                profile[index] = self._backward_processing(values, index)
                self._forward_processing(values, index, pv)
        return profile

    def _backward_processing(self, values: np.ndarray, index: int) -> float:
        best = np.inf
        prefix = 2 ** int(np.ceil(np.log2(self.window_size)))
        max_lag = min(self.x_lag or index, index)
        reference = values[index - max_lag : index]
        first = True
        expansion_num = 0

        while best >= self._bfs:
            if prefix >= max_lag:
                best = float(np.min(st.core.mass(values[index : index + self.window_size], reference)))
                if best > self._bfs:
                    self._bfs = best
                break
            if first:
                first = False
                best = float(np.min(st.core.mass(values[index : index + self.window_size], reference[-prefix:])))
            else:
                start = index - max_lag + (expansion_num * self.window_size)
                end = int(index - (max_lag / 2) + (expansion_num * self.window_size))
                best = float(np.min(st.core.mass(values[index : index + self.window_size], values[start:end])))

            if best < self._bfs:
                break
            prefix = 2 * prefix
            expansion_num *= 1

        return best

    def _forward_processing(self, values: np.ndarray, index: int, pv: np.ndarray) -> None:
        start = index + self.window_size
        lookahead = 2 ** int(np.ceil(np.log2(self.window_size)))
        end = start + lookahead
        if end >= len(values):
            return
        distances = st.core.mass(values[index : index + self.window_size], values[start:end])
        indices = np.argwhere(distances < self._bfs).ravel()
        pv[indices + start] = 0


def score_time_series(
    values: np.ndarray,
    window_size: int,
    *,
    start_index_multiplier: float = 1.0,
    x_lag_multiplier: float | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    window_size = int(window_size)
    if values.size <= window_size + 2:
        raise ValueError("time series is too short for DAMP")

    sp_index = max(window_size + 1, int(round(window_size * float(start_index_multiplier))) + 1)
    x_lag = None if x_lag_multiplier is None or float(x_lag_multiplier) <= 0 else max(window_size, int(round(window_size * float(x_lag_multiplier))))
    model = _DAMPModel(window_size=window_size, sp_index=sp_index, x_lag=x_lag)
    window_scores = model.fit(values)
    normalized_scores = _normalize_scores(window_scores)
    return _align_scores(normalized_scores, window_size, values.size)


__all__ = ["score_time_series"]
