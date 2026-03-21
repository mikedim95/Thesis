from __future__ import annotations

import math

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


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


def _parse_component_value(value: str | int | float | None) -> str | int | float | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "auto", "none"}:
            return None
        if text == "mle":
            return "mle"
        if "." in text:
            return float(text)
        return int(text)
    return value


def score_time_series(
    values: np.ndarray,
    window_size: int,
    *,
    n_components: str | int | float | None = None,
    n_selected_components: int | None = None,
    whiten: bool = False,
    svd_solver: str = "auto",
    tol: float = 0.0,
    random_state: int | None = 42,
    weighted: bool = True,
    standardization: bool = True,
) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    window_size = int(window_size)

    windows = _rolling_windows(values, window_size)
    transformed_windows = windows
    if standardization:
        transformed_windows = StandardScaler().fit_transform(transformed_windows)

    parsed_components = _parse_component_value(n_components)
    model = PCA(
        n_components=parsed_components,
        whiten=bool(whiten),
        svd_solver=svd_solver,
        tol=float(tol),
        random_state=random_state,
    )
    model.fit(transformed_windows)

    component_count = model.components_.shape[0]
    effective_selected = component_count if not n_selected_components or n_selected_components <= 0 else min(int(n_selected_components), component_count)
    selected_components = model.components_[-effective_selected:, :]
    weights = np.ones(component_count, dtype=float)
    if weighted and hasattr(model, "explained_variance_ratio_"):
        weights = np.asarray(model.explained_variance_ratio_, dtype=float)
        weights[weights == 0.0] = np.finfo(float).eps
    selected_weights = weights[-effective_selected:]

    window_scores = np.sum(cdist(transformed_windows, selected_components) / selected_weights, axis=1).ravel()
    normalized_scores = _normalize_scores(window_scores)
    return _align_scores(normalized_scores, window_size, values.size)


__all__ = ["score_time_series"]
