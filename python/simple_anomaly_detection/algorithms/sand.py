from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def _load_sand_class():
    if not hasattr(np, "Inf"):
        np.Inf = np.inf

    legacy_root = None
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "SANDAnomalyDetection"
        if candidate.exists():
            legacy_root = candidate
            break
    if legacy_root is None:
        raise FileNotFoundError("Could not locate the SANDAnomalyDetection directory from the current project layout.")
    if str(legacy_root) not in sys.path:
        sys.path.insert(0, str(legacy_root))

    from Utils.sand import SAND

    return SAND


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float).reshape(-1, 1)
    if scores.size == 0 or np.allclose(scores.min(), scores.max()):
        return np.zeros(scores.shape[0], dtype=float)
    return MinMaxScaler().fit_transform(scores).ravel()


def _resize_scores(scores: np.ndarray, series_length: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=float).ravel()
    if scores.size == series_length:
        return scores
    if scores.size == 0:
        return np.zeros(series_length, dtype=float)
    if scores.size < series_length:
        return np.pad(scores, (0, series_length - scores.size), mode="edge")
    return scores[:series_length]


def _count_subsequences(
    start: int,
    span: int,
    series_length: int,
    subsequence_length: int,
    overlap: int,
) -> int:
    stop = min(start + span, series_length - subsequence_length)
    if stop <= start:
        return 0
    return len(range(start, stop, overlap))


def score_time_series(
    values: np.ndarray,
    window_size: int,
    window_stride: int = 1,
    *,
    alpha: float = 0.5,
    init_length: int = 5_000,
    batch_size: int = 2_000,
    k: int = 6,
    subsequence_multiplier: int = 4,
    overlap: int | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    window_size = int(window_size)

    if values.size <= window_size + 2:
        raise ValueError("time series is too short for SAND")

    subsequence_length = min(max(subsequence_multiplier, 1) * window_size, values.size - 2)
    if subsequence_length <= window_size:
        subsequence_length = window_size + 1

    window_stride = max(1, int(window_stride))
    overlap = max(1, window_size if overlap is None and window_stride <= 1 else (window_stride if overlap is None else int(overlap)))
    effective_init_length = min(max(init_length, subsequence_length + 1), values.size)
    effective_batch_size = max(batch_size, subsequence_length + 1)

    subsequence_counts = [
        _count_subsequences(0, effective_init_length, values.size, subsequence_length, overlap)
    ]
    batch_start = effective_init_length
    while batch_start < values.size - subsequence_length:
        count = _count_subsequences(
            batch_start,
            effective_batch_size,
            values.size,
            subsequence_length,
            overlap,
        )
        if count > 0:
            subsequence_counts.append(count)
        batch_start += effective_batch_size

    effective_k = max(1, min(k, min(subsequence_counts)))

    SAND = _load_sand_class()
    model = SAND(
        pattern_length=window_size,
        subsequence_length=subsequence_length,
        k=effective_k,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(
            values,
            online=True,
            alpha=alpha,
            init_length=effective_init_length,
            batch_size=effective_batch_size,
            overlaping_rate=overlap,
            verbose=False,
        )

    normalized_scores = _normalize_scores(model.decision_scores_)
    return _resize_scores(normalized_scores, values.size)


__all__ = ["score_time_series"]
