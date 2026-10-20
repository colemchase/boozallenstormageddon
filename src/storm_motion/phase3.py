"""Repeatable historical evaluation for the multi-cluster tracking baseline.

This is deliberately a *baseline evaluator*, not a trained ML model.  It
answers a concrete question for each archived scan: did the probability that
the tracker assigned to McLean agree with what a later radar scan showed?
"""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Callable, Sequence

import numpy as np

from .evaluation import probability_any, target_reflectivity_coverage
from .impact import local_xy_from_lat_lon, shape_impact_probability
from .tracking import MultiClusterTracker, RadarFrame, cluster_cells, detect_cells


@dataclass(frozen=True)
class EvaluationRow:
    """One forecast/observation pair for the McLean target."""

    event_id: str
    scan_time: datetime
    forecast_time: datetime
    horizon_minutes: int
    active_track_count: int
    per_track_probabilities: tuple[float, ...]
    combined_probability: float
    observed_35dbz_coverage: bool

    def as_dict(self) -> dict[str, object]:
        row = asdict(self)
        row['scan_time'] = self.scan_time.isoformat()
        row['forecast_time'] = self.forecast_time.isoformat()
        row['per_track_probabilities'] = list(self.per_track_probabilities)
        return row


def _closest_frame_index(
    timestamps: Sequence[datetime], target: datetime, tolerance: timedelta,
) -> int | None:
    candidates = [
        (abs(observed_at - target), index)
        for index, observed_at in enumerate(timestamps)
        if abs(observed_at - target) <= tolerance
    ]
    return min(candidates)[1] if candidates else None


def _crop_frame_around_target(
    frame: RadarFrame,
    target_latitude_deg: float,
    target_longitude_deg: float,
    radius_m: float,
) -> tuple[RadarFrame, float, float]:
    """Copy the smallest rectangular sweep subset enclosing a target radius.

    Level-II volumes are large polar arrays. Cropping immediately after load is
    important in the 4 GB notebook environment: later tracking never needs
    gates beyond the office-specific forecast radius.
    """
    target_x_m, target_y_m = local_xy_from_lat_lon(
        target_latitude_deg, target_longitude_deg,
        frame.radar_latitude_deg, frame.radar_longitude_deg,
    )
    in_radius = np.hypot(frame.x_m - target_x_m, frame.y_m - target_y_m) <= radius_m
    row_indices, column_indices = np.where(in_radius)
    if len(row_indices) == 0:
        raise ValueError('target is outside this radar frame')
    row_slice = slice(int(row_indices.min()), int(row_indices.max()) + 1)
    column_slice = slice(int(column_indices.min()), int(column_indices.max()) + 1)
    return (
        RadarFrame(
            observed_at=frame.observed_at,
            reflectivity_dbz=np.array(frame.reflectivity_dbz[row_slice, column_slice], dtype=float, copy=True),
            x_m=np.array(frame.x_m[row_slice, column_slice], dtype=float, copy=True),
            y_m=np.array(frame.y_m[row_slice, column_slice], dtype=float, copy=True),
            radar_latitude_deg=frame.radar_latitude_deg,
            radar_longitude_deg=frame.radar_longitude_deg,
        ),
        target_x_m,
        target_y_m,
    )


def evaluate_frames(
    *,
    event_id: str,
    frames: Sequence[RadarFrame],
    target_latitude_deg: float,
    target_longitude_deg: float,
    horizons_minutes: Sequence[int] = (30, 60),
    label_radius_m: float = 2_000,
    threshold_dbz: float = 35.0,
    min_gates: int = 20,
    cluster_linkage_m: float = 15_000,
    tracking_radius_m: float = 120_000,
    label_time_tolerance: timedelta = timedelta(minutes=7),
) -> list[EvaluationRow]:
    """Evaluate every eligible scan against the closest future observed scan.

    ``frames`` must come from one radar station in chronological order.  A
    single station keeps the coordinate system and observed label unambiguous;
    the product mosaic remains a display layer.  The forecast uses all active
    clusters, so a one-cell event and a multi-cell event use identical logic.

    Detection is limited to ``tracking_radius_m`` around the target. This keeps
    distant cells and small radar artefacts from inflating an office-specific
    forecast.
    """
    ordered = sorted(frames, key=lambda frame: frame.observed_at)
    return _evaluate_sources(
        event_id=event_id,
        sources=ordered,
        timestamps=[frame.observed_at for frame in ordered],
        load_frame=lambda frame: frame,
        target_latitude_deg=target_latitude_deg,
        target_longitude_deg=target_longitude_deg,
        horizons_minutes=horizons_minutes,
        label_radius_m=label_radius_m,
        threshold_dbz=threshold_dbz,
        min_gates=min_gates,
        cluster_linkage_m=cluster_linkage_m,
        tracking_radius_m=tracking_radius_m,
        label_time_tolerance=label_time_tolerance,
    )


def _evaluate_sources(
    *,
    event_id: str,
    sources: Sequence[object],
    timestamps: Sequence[datetime],
    load_frame: Callable[[object], RadarFrame],
    target_latitude_deg: float,
    target_longitude_deg: float,
    horizons_minutes: Sequence[int] = (30, 60),
    label_radius_m: float = 2_000,
    threshold_dbz: float = 35.0,
    min_gates: int = 20,
    cluster_linkage_m: float = 15_000,
    tracking_radius_m: float = 120_000,
    label_time_tolerance: timedelta = timedelta(minutes=7),
) -> list[EvaluationRow]:
    """Evaluate scan sources without retaining every full radar volume."""
    if len(sources) < 2:
        return []
    if any(horizon <= 0 for horizon in horizons_minutes):
        raise ValueError('horizons_minutes must be positive')
    if tracking_radius_m <= 0:
        raise ValueError('tracking_radius_m must be positive')

    tracker = MultiClusterTracker()
    rows: list[EvaluationRow] = []

    for index, source in enumerate(sources):
        raw_frame = load_frame(source)
        frame, target_x_m, target_y_m = _crop_frame_around_target(
            raw_frame, target_latitude_deg, target_longitude_deg, tracking_radius_m,
        )
        del raw_frame
        gc.collect()
        in_tracking_area = np.hypot(
            frame.x_m - target_x_m, frame.y_m - target_y_m,
        ) <= tracking_radius_m
        tracking_frame = RadarFrame(
            observed_at=frame.observed_at,
            reflectivity_dbz=np.where(in_tracking_area, frame.reflectivity_dbz, np.nan),
            x_m=frame.x_m,
            y_m=frame.y_m,
            radar_latitude_deg=frame.radar_latitude_deg,
            radar_longitude_deg=frame.radar_longitude_deg,
        )
        clusters = cluster_cells(
            detect_cells(tracking_frame, threshold_dbz=threshold_dbz, min_gates=min_gates),
            linkage_distance_m=cluster_linkage_m,
        )
        update = tracker.update(frame.observed_at, clusters)
        # One observation has no storm velocity yet, so it is not a useful
        # nowcast origin.
        if index == 0 or not update.tracks:
            del tracking_frame, frame
            continue

        for horizon_minutes in horizons_minutes:
            expected_time = frame.observed_at + timedelta(minutes=horizon_minutes)
            future_index = _closest_frame_index(timestamps, expected_time, label_time_tolerance)
            if future_index is None:
                continue
            raw_future_frame = load_frame(sources[future_index])
            future_frame, future_target_x_m, future_target_y_m = _crop_frame_around_target(
                raw_future_frame, target_latitude_deg, target_longitude_deg, tracking_radius_m,
            )
            del raw_future_frame
            gc.collect()
            # Each horizon must be an independent what-if forecast. forecast()
            # copies only tiny Kalman state arrays, not the radar-gate shape.
            forecast_tracks = tracker.forecast(future_frame.observed_at)
            per_track = []
            for track in forecast_tracks:
                per_track.append(shape_impact_probability(
                    track.estimate, track.last_observed_cluster, target_x_m, target_y_m,
                    target_radius_m=label_radius_m,
                    sample_count=2_000,
                    random_seed=track.track_id + horizon_minutes,
                ).probability)
            rows.append(EvaluationRow(
                event_id=event_id,
                scan_time=frame.observed_at,
                forecast_time=future_frame.observed_at,
                horizon_minutes=int(horizon_minutes),
                active_track_count=len(forecast_tracks),
                per_track_probabilities=tuple(per_track),
                combined_probability=probability_any(per_track),
                observed_35dbz_coverage=target_reflectivity_coverage(
                    future_frame, future_target_x_m, future_target_y_m,
                    radius_m=label_radius_m, threshold_dbz=threshold_dbz,
                ),
            ))
            del future_frame
            gc.collect()
        del tracking_frame, frame
        gc.collect()
    return rows


def evaluate_paths(
    *,
    event_id: str,
    paths: Sequence[object],
    load_frame: Callable[[object], RadarFrame],
    **kwargs: object,
) -> list[EvaluationRow]:
    """Evaluate archived paths while retaining only one full source scan.

    The first pass reads metadata then releases each volume. The evaluation
    pass streams each origin scan and temporarily loads only its verification
    scan. This is intentionally disk-heavy but stays inside the 4 GB notebook
    environment and behaves the same in an EMR worker.
    """
    timestamps: list[datetime] = []
    for path in paths:
        frame = load_frame(path)
        timestamps.append(frame.observed_at)
        del frame
        gc.collect()
    ordered = sorted(zip(timestamps, paths), key=lambda item: item[0])
    ordered_timestamps, ordered_paths = zip(*ordered) if ordered else ((), ())
    return _evaluate_sources(
        event_id=event_id,
        sources=ordered_paths,
        timestamps=ordered_timestamps,
        load_frame=load_frame,
        **kwargs,
    )
