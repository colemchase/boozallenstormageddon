"""Simple time-nearest NEXRAD reflectivity mosaics on a local Cartesian grid."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .impact import local_xy_from_lat_lon
from .tracking import RadarFrame


def max_reflectivity_mosaic(
    frames: Sequence[RadarFrame],
    *,
    origin_latitude_deg: float,
    origin_longitude_deg: float,
    x_m: np.ndarray,
    y_m: np.ndarray,
) -> np.ndarray:
    """Bin several radar sweeps onto a common grid, retaining maximum valid dBZ.

    All input frames must carry their radar latitude/longitude. The local
    tangent-plane approximation is appropriate for this regional nowcasting
    view, not a national-scale product. Max dBZ is a transparent first fusion
    rule: it retains the strongest observed reflectivity where radar coverage
    overlaps, without claiming to correct beam-height differences.
    """
    if x_m.ndim != 1 or y_m.ndim != 1 or len(x_m) < 2 or len(y_m) < 2:
        raise ValueError('x_m and y_m must be one-dimensional grids with at least two values')
    x_step_m, y_step_m = np.diff(x_m), np.diff(y_m)
    if not np.allclose(x_step_m, x_step_m[0]) or not np.allclose(y_step_m, y_step_m[0]):
        raise ValueError('x_m and y_m must be evenly spaced')

    result = np.full((len(y_m), len(x_m)), -np.inf, dtype=float)
    for frame in frames:
        if frame.radar_latitude_deg is None or frame.radar_longitude_deg is None:
            raise ValueError('every RadarFrame needs radar_latitude_deg and radar_longitude_deg')
        radar_x_m, radar_y_m = local_xy_from_lat_lon(
            frame.radar_latitude_deg, frame.radar_longitude_deg,
            origin_latitude_deg, origin_longitude_deg,
        )
        grid_x = radar_x_m + frame.x_m
        grid_y = radar_y_m + frame.y_m
        column = np.rint((grid_x - x_m[0]) / x_step_m[0]).astype(int)
        row = np.rint((grid_y - y_m[0]) / y_step_m[0]).astype(int)
        valid = (
            np.isfinite(frame.reflectivity_dbz)
            & (column >= 0) & (column < len(x_m))
            & (row >= 0) & (row < len(y_m))
        )
        flat_result = result.ravel()
        flat_index = row[valid] * len(x_m) + column[valid]
        np.maximum.at(flat_result, flat_index, frame.reflectivity_dbz[valid])
    result[result == -np.inf] = np.nan
    return result
