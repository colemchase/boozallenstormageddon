from datetime import datetime, timedelta, timezone

import numpy as np

from storm_motion.tracking import (
    Cell,
    ConstantVelocityKalmanTracker,
    MultiClusterTracker,
    RadarFrame,
    StormCluster,
    cluster_cells,
    detect_cells,
    match_cells,
)
from storm_motion.impact import (
    equivalent_cluster_radius_m,
    local_xy_from_lat_lon,
    point_impact_probability,
    shape_impact_probability,
)
from storm_motion.mosaic import max_reflectivity_mosaic
from storm_motion.evaluation import brier_score, probability_any, target_reflectivity_coverage
from storm_motion.phase3 import evaluate_frames, evaluate_paths


def frame_at(minutes: int, hot_slice: tuple[slice, slice]) -> RadarFrame:
    reflectivity = np.zeros((20, 20), dtype=float)
    reflectivity[hot_slice] = 42.0
    x_m, y_m = np.meshgrid(np.arange(20) * 1_000, np.arange(20) * 1_000)
    return RadarFrame(datetime(2026, 10, 5, tzinfo=timezone.utc) + timedelta(minutes=minutes), reflectivity, x_m, y_m)


def test_detects_and_tracks_a_moving_cell() -> None:
    earlier = frame_at(0, (slice(4, 9), slice(4, 9)))
    later = frame_at(5, (slice(4, 9), slice(6, 11)))
    old_cells = detect_cells(earlier, threshold_dbz=35, min_gates=20)
    new_cells = detect_cells(later, threshold_dbz=35, min_gates=20)

    assert len(old_cells) == len(new_cells) == 1
    matches = match_cells(old_cells, new_cells, elapsed_seconds=300)

    assert len(matches) == 1
    assert matches[0].distance_m == 2_000
    assert matches[0].speed_mps == 2_000 / 300
    assert matches[0].bearing_degrees == 90


def test_groups_nearby_cells_into_one_cluster() -> None:
    cells = [
        Cell(1, 10, 0, 0, 45),
        Cell(2, 30, 5_000, 0, 50),
        Cell(3, 20, 50_000, 0, 40),
    ]
    clusters = cluster_cells(cells, linkage_distance_m=10_000)

    assert len(clusters) == 2
    assert clusters[0].member_cell_ids == (1, 2)
    assert clusters[0].gate_count == 40
    assert clusters[0].centroid_x_m == 3_750


def test_kalman_tracker_forecasts_constant_velocity_motion() -> None:
    start = datetime(2026, 10, 5, tzinfo=timezone.utc)
    tracker = ConstantVelocityKalmanTracker(measurement_std_m=100, acceleration_std_mps2=0.01)
    tracker.update(start, 0, 0)
    tracker.update(start + timedelta(seconds=60), 600, 0)
    tracker.update(start + timedelta(seconds=120), 1_200, 0)
    forecast = tracker.predict(start + timedelta(seconds=180))

    assert abs(forecast.x_m - 1_800) < 50
    assert abs(forecast.y_m) < 1
    assert abs(forecast.velocity_x_mps - 10) < 1
    assert forecast.position_std_m > 0


def test_local_coordinates_are_zero_at_the_radar() -> None:
    x_m, y_m = local_xy_from_lat_lon(39.0, -77.5, 39.0, -77.5)
    assert x_m == 0
    assert y_m == 0


def test_point_impact_probability_distinguishes_near_and_far_targets() -> None:
    tracker = ConstantVelocityKalmanTracker(measurement_std_m=100, acceleration_std_mps2=0.01)
    estimate = tracker.update(datetime(2026, 10, 5, tzinfo=timezone.utc), 0, 0)
    near = point_impact_probability(estimate, 0, 0, footprint_radius_m=1_000)
    far = point_impact_probability(estimate, 10_000, 0, footprint_radius_m=1_000)

    assert near.probability > 0.99
    assert far.probability == 0


def test_equivalent_cluster_radius_uses_cluster_area() -> None:
    cluster = cluster_cells([Cell(1, 100, 0, 0, 45)])[0]
    assert equivalent_cluster_radius_m(cluster, representative_gate_area_m2=100) == np.sqrt(10_000 / np.pi)


def test_advected_shape_keeps_a_gap_that_a_circle_would_hide() -> None:
    tracker = ConstantVelocityKalmanTracker(measurement_std_m=100, acceleration_std_mps2=0.01)
    estimate = tracker.update(datetime(2026, 10, 5, tzinfo=timezone.utc), 0, 0)
    # Two radar-gate groups lie on opposite sides of the office. Their shared
    # centroid is at the office, but the observed storm shape has a clear gap.
    cluster = StormCluster(
        1, (1,), 2, 0, 0, 45,
        gate_x_m=np.array([-5_000.0, 5_000.0]),
        gate_y_m=np.array([0.0, 0.0]),
    )
    forecast = shape_impact_probability(
        estimate, cluster, 0, 0, target_radius_m=500, gate_buffer_m=0,
        sample_count=1_000,
    )

    assert forecast.probability == 0


def test_mosaic_keeps_strongest_valid_reflectivity() -> None:
    timestamp = datetime(2026, 10, 5, tzinfo=timezone.utc)
    x_m, y_m = np.meshgrid(np.array([0.0, 1_000.0]), np.array([0.0, 1_000.0]))
    first = RadarFrame(timestamp, np.array([[20.0, np.nan], [30.0, 40.0]]), x_m, y_m, 39.0, -77.5)
    second = RadarFrame(timestamp, np.array([[35.0, 25.0], [np.nan, 10.0]]), x_m, y_m, 39.0, -77.5)
    mosaic = max_reflectivity_mosaic(
        [first, second], origin_latitude_deg=39.0, origin_longitude_deg=-77.5,
        x_m=np.array([0.0, 1_000.0]), y_m=np.array([0.0, 1_000.0]),
    )
    np.testing.assert_array_equal(mosaic, np.array([[35.0, 25.0], [30.0, 40.0]]))


def test_phase3_target_label_and_probability_metrics() -> None:
    frame = frame_at(0, (slice(4, 9), slice(4, 9)))
    assert target_reflectivity_coverage(frame, 6_000, 6_000, radius_m=500)
    assert not target_reflectivity_coverage(frame, 15_000, 15_000, radius_m=500)
    assert probability_any([0.2, 0.5]) == 0.6
    assert brier_score([0.0, 1.0], [False, True]) == 0.0


def test_multi_cluster_tracker_keeps_two_tracks_when_cluster_order_changes() -> None:
    start = datetime(2026, 10, 5, tzinfo=timezone.utc)
    initial = [
        cluster_cells([Cell(1, 20, 0, 0, 45)])[0],
        cluster_cells([Cell(2, 20, 20_000, 0, 45)])[0],
    ]
    later = [
        cluster_cells([Cell(1, 20, 21_000, 0, 45)])[0],
        cluster_cells([Cell(2, 20, 1_000, 0, 45)])[0],
    ]
    tracker = MultiClusterTracker(measurement_std_m=100, association_distance_m=5_000)

    first_update = tracker.update(start, initial)
    second_update = tracker.update(start + timedelta(minutes=5), later)

    assert [track.status for track in first_update.tracks] == ['new', 'new']
    assert [track.status for track in second_update.tracks] == ['matched', 'matched']
    assert [track.observed_cluster.centroid_x_m for track in second_update.tracks] == [1_000, 21_000]
    assert [track.last_observed_cluster.centroid_x_m for track in second_update.tracks] == [1_000, 21_000]
    forecast = tracker.predict(start + timedelta(minutes=10))
    assert [track.status for track in forecast] == ['forecast', 'forecast']
    assert all(track.observed_cluster is None for track in forecast)


def test_multi_cluster_forecast_does_not_mutate_live_tracker() -> None:
    start = datetime(2026, 10, 5, tzinfo=timezone.utc)
    cluster = cluster_cells([Cell(1, 20, 0, 0, 45)])[0]
    tracker = MultiClusterTracker()
    tracker.update(start, [cluster])
    forecast = tracker.forecast(start + timedelta(minutes=30))
    update = tracker.update(start + timedelta(minutes=5), [cluster])

    assert forecast[0].estimate.observed_at == start + timedelta(minutes=30)
    assert update.tracks[0].estimate.observed_at == start + timedelta(minutes=5)


def test_multi_cluster_tracker_retires_after_configured_missed_scans() -> None:
    start = datetime(2026, 10, 5, tzinfo=timezone.utc)
    cluster = cluster_cells([Cell(1, 20, 0, 0, 45)])[0]
    tracker = MultiClusterTracker(max_missed_scans=1)
    tracker.update(start, [cluster])
    coasting = tracker.update(start + timedelta(minutes=5), [])
    retired = tracker.update(start + timedelta(minutes=10), [])

    assert coasting.tracks[0].status == 'coasting'
    assert retired.tracks == ()
    assert retired.retired_track_ids == (1,)


def test_phase3_evaluates_all_active_clusters_against_future_scan() -> None:
    timestamp = datetime(2026, 10, 5, tzinfo=timezone.utc)
    x_m, y_m = np.meshgrid(np.arange(40) * 1_000, np.arange(40) * 1_000)

    def moving_frame(minutes: int, first_x: int, second_x: int) -> RadarFrame:
        reflectivity = np.zeros((40, 40), dtype=float)
        reflectivity[10:15, first_x:first_x + 5] = 45
        reflectivity[25:30, second_x:second_x + 5] = 45
        return RadarFrame(
            timestamp + timedelta(minutes=minutes), reflectivity, x_m, y_m,
            radar_latitude_deg=0.0, radar_longitude_deg=0.0,
        )

    # Two clusters move east.  The target at (0.1°, 0.1°) is inside the first
    # cluster in the actual +30 minute scan, so the evaluator creates one row.
    frames = [
        moving_frame(0, 2, 25),
        moving_frame(5, 5, 28),
        moving_frame(35, 20, 35),
    ]
    rows = evaluate_frames(
        event_id='synthetic', frames=frames,
        target_latitude_deg=0.1, target_longitude_deg=0.1,
        horizons_minutes=(30,), label_radius_m=2_000,
        label_time_tolerance=timedelta(minutes=1),
    )

    assert len(rows) == 1
    assert rows[0].active_track_count == 2
    assert len(rows[0].per_track_probabilities) == 2
    assert rows[0].forecast_time == frames[-1].observed_at

    streamed_rows = evaluate_paths(
        event_id='synthetic', paths=frames, load_frame=lambda frame: frame,
        target_latitude_deg=0.1, target_longitude_deg=0.1,
        horizons_minutes=(30,), label_radius_m=2_000,
        label_time_tolerance=timedelta(minutes=1),
    )
    assert len(streamed_rows) == 1
