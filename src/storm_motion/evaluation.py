"""Phase 3 labels and metrics for validating the storm-motion baseline."""

from __future__ import annotations

import numpy as np

from .tracking import RadarFrame


def target_reflectivity_coverage(
    frame: RadarFrame,
    target_x_m: float,
    target_y_m: float,
    *,
    radius_m: float = 2_000,
    threshold_dbz: float = 35.0,
) -> bool:
    """Whether thresholded reflectivity actually reaches a small target area."""
    if radius_m <= 0:
        raise ValueError('radius_m must be positive')
    target_area = np.hypot(frame.x_m - target_x_m, frame.y_m - target_y_m) <= radius_m
    return bool(np.any(target_area & np.isfinite(frame.reflectivity_dbz) & (frame.reflectivity_dbz >= threshold_dbz)))


def probability_any(probabilities: list[float]) -> float:
    """Combine independent per-track point-coverage probabilities.

    This is an explicit approximation for baseline evaluation; Phase 3 will
    measure its calibration rather than treating it as meteorological truth.
    """
    values = np.asarray(probabilities, dtype=float)
    if np.any((values < 0) | (values > 1)):
        raise ValueError('probabilities must be between zero and one')
    return float(1 - np.prod(1 - values))


def brier_score(probabilities: list[float], outcomes: list[bool]) -> float:
    """Mean squared probability error; lower is better and zero is perfect."""
    if len(probabilities) != len(outcomes) or not probabilities:
        raise ValueError('probabilities and outcomes must be non-empty and the same length')
    return float(np.mean((np.asarray(probabilities, dtype=float) - np.asarray(outcomes, dtype=float)) ** 2))
