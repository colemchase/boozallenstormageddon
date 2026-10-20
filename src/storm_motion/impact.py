"""Transparent point-impact estimates for an early storm-motion baseline.

This module deliberately reports the probability that a tracked reflectivity
cluster covers a point. It is not a probability of severe weather, damage, or
a warning. Those require historical labels and a separately evaluated model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .tracking import KalmanEstimate, StormCluster

EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class PointImpactForecast:
    """Probability that an uncertain circular storm footprint covers a target."""

    probability: float
    target_distance_m: float
    footprint_radius_m: float
    sample_count: int


def local_xy_from_lat_lon(
    latitude_deg: float,
    longitude_deg: float,
    radar_latitude_deg: float,
    radar_longitude_deg: float,
) -> tuple[float, float]:
    """Convert geographic coordinates to metres east/north of a radar.

    An equirectangular local approximation is accurate enough for the short
    (tens-of-kilometres) nowcasting area used by this baseline.
    """
    latitude = np.deg2rad(latitude_deg)
    longitude = np.deg2rad(longitude_deg)
    radar_latitude = np.deg2rad(radar_latitude_deg)
    radar_longitude = np.deg2rad(radar_longitude_deg)
    x_m = EARTH_RADIUS_M * (longitude - radar_longitude) * np.cos((latitude + radar_latitude) / 2)
    y_m = EARTH_RADIUS_M * (latitude - radar_latitude)
    return float(x_m), float(y_m)


def equivalent_cluster_radius_m(
    cluster: StormCluster,
    representative_gate_area_m2: float,
) -> float:
    """Return a same-area circular footprint radius for a thresholded cluster.

    Radar gates are polar, so the caller supplies an area representative of
    the tracked cluster's range. The assumption is visible in the notebook.
    """
    if representative_gate_area_m2 <= 0:
        raise ValueError('representative_gate_area_m2 must be positive')
    return float(np.sqrt(cluster.gate_count * representative_gate_area_m2 / np.pi))


def point_impact_probability(
    estimate: KalmanEstimate,
    target_x_m: float,
    target_y_m: float,
    footprint_radius_m: float,
    *,
    sample_count: int = 20_000,
    random_seed: int = 42,
) -> PointImpactForecast:
    """Monte-Carlo probability that the forecast footprint covers one target.

    Samples reflect only the Kalman position uncertainty. They do not model
    storm growth, decay, shape changes, or the chance that reflectivity is
    severe. Keep those limitations when communicating the result.
    """
    if footprint_radius_m <= 0:
        raise ValueError('footprint_radius_m must be positive')
    if sample_count <= 0:
        raise ValueError('sample_count must be positive')
    random = np.random.default_rng(random_seed)
    centers = random.normal(
        loc=np.array([estimate.x_m, estimate.y_m]),
        scale=estimate.position_std_m,
        size=(sample_count, 2),
    )
    distances = np.hypot(centers[:, 0] - target_x_m, centers[:, 1] - target_y_m)
    return PointImpactForecast(
        probability=float(np.mean(distances <= footprint_radius_m)),
        target_distance_m=float(np.hypot(estimate.x_m - target_x_m, estimate.y_m - target_y_m)),
        footprint_radius_m=footprint_radius_m,
        sample_count=sample_count,
    )


def shape_impact_probability(
    estimate: KalmanEstimate,
    cluster: StormCluster,
    target_x_m: float,
    target_y_m: float,
    *,
    target_radius_m: float = 2_000,
    gate_buffer_m: float = 500,
    sample_count: int = 2_000,
    random_seed: int = 42,
) -> PointImpactForecast:
    """Probability that an advected observed storm shape covers a target.

    The cluster's thresholded radar-gate geometry is held intact and translated
    for each Kalman position sample. This retains gaps, arms, and irregular
    boundaries that are lost when a cluster is replaced by an equivalent-area
    circle. ``gate_buffer_m`` treats a gate centre as a small areal sample.

    It is still a persistence forecast: it does not yet model growth, decay,
    or deformation. Those are the next optical-flow and ensemble stages.
    """
    if cluster.gate_x_m is None or cluster.gate_y_m is None or len(cluster.gate_x_m) == 0:
        raise ValueError('cluster has no gate geometry; detect it from a RadarFrame first')
    if target_radius_m <= 0 or gate_buffer_m < 0 or sample_count <= 0:
        raise ValueError('target_radius_m and sample_count must be positive; gate_buffer_m cannot be negative')

    offsets = np.column_stack((
        cluster.gate_x_m - cluster.centroid_x_m,
        cluster.gate_y_m - cluster.centroid_y_m,
    ))
    shape_tree = cKDTree(offsets)
    random = np.random.default_rng(random_seed)
    centers = random.normal(
        loc=np.array([estimate.x_m, estimate.y_m]),
        scale=estimate.position_std_m,
        size=(sample_count, 2),
    )
    # For each possible forecast centre, find the nearest point in the fixed
    # storm shape to the target expressed in shape-relative coordinates.
    target_in_shape_coordinates = np.array([target_x_m, target_y_m]) - centers
    nearest_gate_distance_m, _ = shape_tree.query(target_in_shape_coordinates)
    covered = nearest_gate_distance_m <= target_radius_m + gate_buffer_m
    max_shape_radius_m = float(np.max(np.hypot(offsets[:, 0], offsets[:, 1])))
    return PointImpactForecast(
        probability=float(np.mean(covered)),
        target_distance_m=float(np.hypot(estimate.x_m - target_x_m, estimate.y_m - target_y_m)),
        footprint_radius_m=max_shape_radius_m,
        sample_count=sample_count,
    )
