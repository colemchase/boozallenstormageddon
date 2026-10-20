from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class RadarFrame:
    """Reflectivity and Cartesian gate coordinates for one radar sweep."""

    observed_at: datetime
    reflectivity_dbz: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    radar_latitude_deg: float | None = None
    radar_longitude_deg: float | None = None


@dataclass(frozen=True)
class Cell:
    cell_id: int
    gate_count: int
    centroid_x_m: float
    centroid_y_m: float
    max_reflectivity_dbz: float
    gate_x_m: np.ndarray | None = None
    gate_y_m: np.ndarray | None = None


@dataclass(frozen=True)
class Match:
    previous_cell_id: int
    current_cell_id: int
    distance_m: float
    speed_mps: float
    bearing_degrees: float


@dataclass(frozen=True)
class StormCluster:
    """A nearby group of detected cells represented by one weighted centroid."""

    cluster_id: int
    member_cell_ids: tuple[int, ...]
    gate_count: int
    centroid_x_m: float
    centroid_y_m: float
    max_reflectivity_dbz: float
    gate_x_m: np.ndarray | None = None
    gate_y_m: np.ndarray | None = None


@dataclass(frozen=True)
class KalmanEstimate:
    """Position, velocity, and one-sigma position uncertainty of a tracked cluster."""

    observed_at: datetime
    x_m: float
    y_m: float
    velocity_x_mps: float
    velocity_y_mps: float
    position_std_m: float


@dataclass(frozen=True)
class ClusterTrack:
    """One active tracked storm cluster at a single scan time."""

    track_id: int
    estimate: KalmanEstimate
    observed_cluster: StormCluster | None
    age_scans: int
    missed_scans: int
    status: str
    last_observed_cluster: StormCluster | None


@dataclass(frozen=True)
class MultiTrackUpdate:
    """Association outcome for all clusters in one radar scan."""

    observed_at: datetime
    tracks: tuple[ClusterTrack, ...]
    retired_track_ids: tuple[int, ...]


@dataclass
class _ActiveTrack:
    tracker: 'ConstantVelocityKalmanTracker'
    age_scans: int
    last_observed_cluster: StormCluster
    missed_scans: int = 0


def detect_cells(frame: RadarFrame, threshold_dbz: float = 35.0, min_gates: int = 20) -> list[Cell]:
    """Label contiguous reflectivity gates and return centroid summaries."""
    if frame.reflectivity_dbz.shape != frame.x_m.shape or frame.reflectivity_dbz.shape != frame.y_m.shape:
        raise ValueError("reflectivity, x_m, and y_m must have identical shapes")
    mask = np.isfinite(frame.reflectivity_dbz) & (frame.reflectivity_dbz >= threshold_dbz)
    labels, label_count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    cells: list[Cell] = []
    for label in range(1, label_count + 1):
        gates = labels == label
        gate_count = int(gates.sum())
        if gate_count < min_gates:
            continue
        cells.append(Cell(
            cell_id=len(cells) + 1,
            gate_count=gate_count,
            centroid_x_m=float(np.mean(frame.x_m[gates])),
            centroid_y_m=float(np.mean(frame.y_m[gates])),
            max_reflectivity_dbz=float(np.max(frame.reflectivity_dbz[gates])),
            gate_x_m=frame.x_m[gates].astype(float, copy=True),
            gate_y_m=frame.y_m[gates].astype(float, copy=True),
        ))
    return cells


def match_cells(
    previous: list[Cell], current: list[Cell], elapsed_seconds: float, max_distance_m: float = 30_000,
) -> list[Match]:
    """Greedily assign the closest unassigned current cell to each old cell."""
    if elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be positive")
    candidates: list[tuple[float, Cell, Cell]] = []
    for old in previous:
        for new in current:
            dx, dy = new.centroid_x_m - old.centroid_x_m, new.centroid_y_m - old.centroid_y_m
            distance = float(np.hypot(dx, dy))
            if distance <= max_distance_m:
                candidates.append((distance, old, new))
    matched_old: set[int] = set()
    matched_new: set[int] = set()
    results: list[Match] = []
    for distance, old, new in sorted(candidates, key=lambda item: item[0]):
        if old.cell_id in matched_old or new.cell_id in matched_new:
            continue
        dx, dy = new.centroid_x_m - old.centroid_x_m, new.centroid_y_m - old.centroid_y_m
        # Bearing is clockwise from north; radar x points east and y points north.
        bearing = float((np.degrees(np.arctan2(dx, dy)) + 360) % 360)
        results.append(Match(old.cell_id, new.cell_id, distance, distance / elapsed_seconds, bearing))
        matched_old.add(old.cell_id)
        matched_new.add(new.cell_id)
    return results


def cluster_cells(cells: list[Cell], linkage_distance_m: float = 15_000) -> list[StormCluster]:
    """Group nearby detected cells into storm complexes using centroid linkage."""
    if linkage_distance_m <= 0:
        raise ValueError("linkage_distance_m must be positive")
    remaining = set(range(len(cells)))
    groups: list[list[int]] = []
    while remaining:
        group = {remaining.pop()}
        expanded = True
        while expanded:
            expanded = False
            for candidate in tuple(remaining):
                if any(np.hypot(
                    cells[candidate].centroid_x_m - cells[member].centroid_x_m,
                    cells[candidate].centroid_y_m - cells[member].centroid_y_m,
                ) <= linkage_distance_m for member in group):
                    remaining.remove(candidate)
                    group.add(candidate)
                    expanded = True
        groups.append(sorted(group))

    clusters: list[StormCluster] = []
    for members in sorted(groups, key=min):
        member_cells = [cells[index] for index in members]
        weights = np.array([cell.gate_count for cell in member_cells], dtype=float)
        has_gate_geometry = all(cell.gate_x_m is not None and cell.gate_y_m is not None for cell in member_cells)
        clusters.append(StormCluster(
            cluster_id=len(clusters) + 1,
            member_cell_ids=tuple(cell.cell_id for cell in member_cells),
            gate_count=int(weights.sum()),
            centroid_x_m=float(np.average([cell.centroid_x_m for cell in member_cells], weights=weights)),
            centroid_y_m=float(np.average([cell.centroid_y_m for cell in member_cells], weights=weights)),
            max_reflectivity_dbz=max(cell.max_reflectivity_dbz for cell in member_cells),
            gate_x_m=np.concatenate([cell.gate_x_m for cell in member_cells]) if has_gate_geometry else None,
            gate_y_m=np.concatenate([cell.gate_y_m for cell in member_cells]) if has_gate_geometry else None,
        ))
    return clusters


class ConstantVelocityKalmanTracker:
    """A 2-D constant-velocity Kalman filter for a storm-cluster centroid."""

    def __init__(self, measurement_std_m: float = 4_000, acceleration_std_mps2: float = 0.01) -> None:
        if measurement_std_m <= 0 or acceleration_std_mps2 <= 0:
            raise ValueError("measurement_std_m and acceleration_std_mps2 must be positive")
        self.measurement_std_m = measurement_std_m
        self.acceleration_std_mps2 = acceleration_std_mps2
        self._state: np.ndarray | None = None
        self._covariance: np.ndarray | None = None
        self._timestamp: datetime | None = None

    @property
    def is_initialized(self) -> bool:
        return self._state is not None

    def fork(self) -> 'ConstantVelocityKalmanTracker':
        """Return an independent filter copy for a what-if forecast."""
        duplicate = ConstantVelocityKalmanTracker(self.measurement_std_m, self.acceleration_std_mps2)
        if self._state is not None:
            duplicate._state = self._state.copy()
            assert self._covariance is not None
            duplicate._covariance = self._covariance.copy()
            duplicate._timestamp = self._timestamp
        return duplicate

    def update(self, observed_at: datetime, x_m: float, y_m: float) -> KalmanEstimate:
        """Predict to `observed_at`, then incorporate a cluster-centroid observation."""
        if self._state is None:
            self._state = np.array([x_m, y_m, 0.0, 0.0], dtype=float)
            self._covariance = np.diag([self.measurement_std_m ** 2, self.measurement_std_m ** 2, 25.0 ** 2, 25.0 ** 2])
            self._timestamp = observed_at
            return self.estimate
        self.predict(observed_at)
        assert self._covariance is not None
        observation = np.array([x_m, y_m], dtype=float)
        observation_matrix = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        measurement_covariance = np.eye(2) * self.measurement_std_m ** 2
        innovation = observation - observation_matrix @ self._state
        innovation_covariance = observation_matrix @ self._covariance @ observation_matrix.T + measurement_covariance
        gain = self._covariance @ observation_matrix.T @ np.linalg.inv(innovation_covariance)
        self._state = self._state + gain @ innovation
        self._covariance = (np.eye(4) - gain @ observation_matrix) @ self._covariance
        return self.estimate

    def predict(self, target_time: datetime) -> KalmanEstimate:
        """Advance the filter without a new observation."""
        if self._state is None or self._covariance is None or self._timestamp is None:
            raise RuntimeError("Update the tracker with an observation before predicting")
        elapsed_seconds = (target_time - self._timestamp).total_seconds()
        if elapsed_seconds < 0:
            raise ValueError("target_time cannot precede the current tracker time")
        if elapsed_seconds:
            transition = np.array([
                [1.0, 0.0, elapsed_seconds, 0.0],
                [0.0, 1.0, 0.0, elapsed_seconds],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
            q = self.acceleration_std_mps2 ** 2
            process_noise = q * np.array([
                [elapsed_seconds ** 4 / 4, 0.0, elapsed_seconds ** 3 / 2, 0.0],
                [0.0, elapsed_seconds ** 4 / 4, 0.0, elapsed_seconds ** 3 / 2],
                [elapsed_seconds ** 3 / 2, 0.0, elapsed_seconds ** 2, 0.0],
                [0.0, elapsed_seconds ** 3 / 2, 0.0, elapsed_seconds ** 2],
            ])
            self._state = transition @ self._state
            self._covariance = transition @ self._covariance @ transition.T + process_noise
            self._timestamp = target_time
        return self.estimate

    @property
    def estimate(self) -> KalmanEstimate:
        if self._state is None or self._covariance is None or self._timestamp is None:
            raise RuntimeError("Tracker has not received an observation")
        return KalmanEstimate(
            observed_at=self._timestamp,
            x_m=float(self._state[0]),
            y_m=float(self._state[1]),
            velocity_x_mps=float(self._state[2]),
            velocity_y_mps=float(self._state[3]),
            position_std_m=float(np.sqrt(max(self._covariance[0, 0], self._covariance[1, 1]))),
        )


class MultiClusterTracker:
    """Associate all detected clusters with independent Kalman tracks.

    This baseline uses globally optimal nearest-neighbour assignment inside a
    distance gate. It handles new clusters and temporary missed detections;
    split/merge lineage remains a later enhancement, not an implied fact.
    """

    def __init__(
        self,
        measurement_std_m: float = 4_000,
        acceleration_std_mps2: float = 0.01,
        association_distance_m: float = 30_000,
        max_missed_scans: int = 1,
    ) -> None:
        if association_distance_m <= 0:
            raise ValueError('association_distance_m must be positive')
        if max_missed_scans < 0:
            raise ValueError('max_missed_scans cannot be negative')
        self.measurement_std_m = measurement_std_m
        self.acceleration_std_mps2 = acceleration_std_mps2
        self.association_distance_m = association_distance_m
        self.max_missed_scans = max_missed_scans
        self._active: dict[int, _ActiveTrack] = {}
        self._next_track_id = 1

    def update(self, observed_at: datetime, clusters: list[StormCluster]) -> MultiTrackUpdate:
        """Advance all tracks and associate this scan's clusters to them."""
        track_ids = sorted(self._active)
        predictions = [self._active[track_id].tracker.predict(observed_at) for track_id in track_ids]
        assignments: dict[int, int] = {}
        if track_ids and clusters:
            distances = np.array([
                [np.hypot(prediction.x_m - cluster.centroid_x_m, prediction.y_m - cluster.centroid_y_m)
                 for cluster in clusters]
                for prediction in predictions
            ])
            rows, columns = linear_sum_assignment(distances)
            for row, column in zip(rows, columns):
                if distances[row, column] <= self.association_distance_m:
                    assignments[track_ids[row]] = int(column)

        observed_cluster_by_track: dict[int, StormCluster | None] = {}
        status_by_track: dict[int, str] = {}
        retired: list[int] = []
        for track_id in track_ids:
            active = self._active[track_id]
            cluster_index = assignments.get(track_id)
            if cluster_index is not None:
                cluster = clusters[cluster_index]
                active.tracker.update(observed_at, cluster.centroid_x_m, cluster.centroid_y_m)
                active.age_scans += 1
                active.missed_scans = 0
                active.last_observed_cluster = cluster
                observed_cluster_by_track[track_id] = cluster
                status_by_track[track_id] = 'matched'
            else:
                active.age_scans += 1
                active.missed_scans += 1
                observed_cluster_by_track[track_id] = None
                status_by_track[track_id] = 'coasting'
                if active.missed_scans > self.max_missed_scans:
                    retired.append(track_id)

        for track_id in retired:
            del self._active[track_id]
            observed_cluster_by_track.pop(track_id, None)
            status_by_track.pop(track_id, None)

        assigned_clusters = set(assignments.values())
        for cluster_index, cluster in enumerate(clusters):
            if cluster_index in assigned_clusters:
                continue
            tracker = ConstantVelocityKalmanTracker(self.measurement_std_m, self.acceleration_std_mps2)
            tracker.update(observed_at, cluster.centroid_x_m, cluster.centroid_y_m)
            track_id = self._next_track_id
            self._next_track_id += 1
            self._active[track_id] = _ActiveTrack(
                tracker=tracker, age_scans=1, last_observed_cluster=cluster,
            )
            observed_cluster_by_track[track_id] = cluster
            status_by_track[track_id] = 'new'

        tracks = tuple(
            ClusterTrack(
                track_id=track_id,
                estimate=active.tracker.estimate,
                observed_cluster=observed_cluster_by_track.get(track_id),
                age_scans=active.age_scans,
                missed_scans=active.missed_scans,
                status=status_by_track[track_id],
                last_observed_cluster=active.last_observed_cluster,
            )
            for track_id, active in sorted(self._active.items())
        )
        return MultiTrackUpdate(observed_at=observed_at, tracks=tracks, retired_track_ids=tuple(retired))

    def predict(self, target_time: datetime) -> tuple[ClusterTrack, ...]:
        """Advance all active tracks to a future time without observations."""
        return tuple(
            ClusterTrack(
                track_id=track_id,
                estimate=active.tracker.predict(target_time),
                observed_cluster=None,
                age_scans=active.age_scans,
                missed_scans=active.missed_scans,
                status='forecast',
                last_observed_cluster=active.last_observed_cluster,
            )
            for track_id, active in sorted(self._active.items())
        )

    def forecast(self, target_time: datetime) -> tuple[ClusterTrack, ...]:
        """Forecast active tracks without mutating the live tracker state.

        Unlike :meth:`predict`, this is safe for scoring multiple forecast
        horizons while continuing to assimilate later real radar scans.
        """
        return tuple(
            ClusterTrack(
                track_id=track_id,
                estimate=active.tracker.fork().predict(target_time),
                observed_cluster=None,
                age_scans=active.age_scans,
                missed_scans=active.missed_scans,
                status='forecast',
                # Reuse immutable radar-gate geometry instead of deep-copying
                # it for every forecast horizon.
                last_observed_cluster=active.last_observed_cluster,
            )
            for track_id, active in sorted(self._active.items())
        )
