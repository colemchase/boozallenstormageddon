from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .tracking import RadarFrame


def load_lowest_reflectivity_sweep(path: str | Path) -> RadarFrame:
    """Load the lowest NEXRAD reflectivity sweep into a Cartesian gate frame."""
    try:
        import pyart
    except ImportError as error:  # pragma: no cover - dependency-specific path
        raise RuntimeError("Install the nexrad extra: pip install -e '.[nexrad]'") from error

    radar = pyart.io.read_nexrad_archive(str(path))
    # Prefer the actual reflectivity field. A broad substring match can select
    # differential_reflectivity (ZDR), whose values are not dBZ and therefore
    # make a 25–35 dBZ storm-cell threshold appear empty.
    field_name = next((name for name in radar.fields if name.lower() == "reflectivity"), None)
    if field_name is None:
        field_name = next((
            name for name in radar.fields
            if "reflectivity" in name.lower() and "differential" not in name.lower()
        ), None)
    if field_name is None:
        raise ValueError("NEXRAD file has no reflectivity field")
    reflectivity = np.ma.filled(radar.get_field(0, field_name), np.nan).astype(float)
    azimuth = np.deg2rad(radar.get_azimuth(0))[:, np.newaxis]
    ranges_m = radar.range["data"][np.newaxis, :]
    x_m = np.sin(azimuth) * ranges_m
    y_m = np.cos(azimuth) * ranges_m
    start = radar.time["units"].replace("seconds since ", "")
    start_time = datetime.fromisoformat(start.replace("Z", "+00:00"))
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    observed_at = start_time + timedelta(seconds=float(radar.time["data"][0]))
    return RadarFrame(
        observed_at,
        reflectivity,
        x_m,
        y_m,
        radar_latitude_deg=float(radar.latitude['data'][0]),
        radar_longitude_deg=float(radar.longitude['data'][0]),
    )
