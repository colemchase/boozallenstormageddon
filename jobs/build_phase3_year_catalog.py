"""Build a reproducible, report-anchored annual catalog for Phase 3.

The catalog intentionally selects weather episodes from NOAA NCEI Storm Events
reports near McLean, then evaluates only the KLWX Level-II radar windows around
those episodes.  This keeps a one-year Kalman backtest focused on storms rather
than paying to decode an entire year of clear-air radar scans.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import gzip
import io
import json
from math import asin, cos, radians, sin, sqrt
import re
from typing import Any, Callable, Iterable
from urllib.request import urlopen
from zoneinfo import ZoneInfo


NCEI_DIRECTORY = 'https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/'
DMV_STATES = frozenset({'VIRGINIA', 'MARYLAND', 'DISTRICT OF COLUMBIA'})
EVENT_TYPES = frozenset({
    'Flash Flood', 'Funnel Cloud', 'Hail', 'Heavy Rain', 'Lightning',
    'Thunderstorm Wind', 'Tornado',
})


def haversine_km(latitude_a: float, longitude_a: float, latitude_b: float, longitude_b: float) -> float:
    """Great-circle distance for selecting report locations near McLean."""
    latitude_delta = radians(latitude_b - latitude_a)
    longitude_delta = radians(longitude_b - longitude_a)
    value = sin(latitude_delta / 2) ** 2 + cos(radians(latitude_a)) * cos(radians(latitude_b)) * sin(longitude_delta / 2) ** 2
    return 6_371.0088 * 2 * asin(sqrt(value))


def discover_details_url(year: int, opener: Callable[..., Any] = urlopen) -> str:
    """Find the current NCEI bulk-file version for ``year`` from its index."""
    with opener(NCEI_DIRECTORY, timeout=60) as response:
        listing = response.read().decode('utf-8')
    matches = re.findall(rf'StormEvents_details-ftp_v1\.0_d{year}_c\d+\.csv\.gz', listing)
    if not matches:
        raise RuntimeError(f'NCEI does not list a Storm Events details CSV for {year}.')
    return NCEI_DIRECTORY + sorted(set(matches))[-1]


def download_details_rows(year: int, opener: Callable[..., Any] = urlopen) -> tuple[str, list[dict[str, str]]]:
    """Download NCEI's annual details table as decoded CSV rows."""
    source_url = discover_details_url(year, opener)
    with opener(source_url, timeout=120) as response:
        payload = response.read()
    with gzip.GzipFile(fileobj=io.BytesIO(payload)) as archive:
        text = io.TextIOWrapper(archive, encoding='latin-1', newline='')
        return source_url, list(csv.DictReader(text))


def report_timestamp_utc(row: dict[str, str]) -> datetime:
    """Interpret NCEI DMV report times using the IANA Eastern time zone.

    The bulk CSV's ``CZ_TIMEZONE`` is commonly ``EST-5`` even for daylight
    saving dates.  Because this catalog only admits DMV reports, America/New_York
    gives the correct historical daylight-saving conversion.
    """
    local = datetime.strptime(row['BEGIN_DATE_TIME'], '%d-%b-%y %H:%M:%S')
    return local.replace(tzinfo=ZoneInfo('America/New_York')).astimezone(timezone.utc)


def relevant_reports(
    rows: Iterable[dict[str, str]], *, latitude_deg: float, longitude_deg: float, radius_km: float,
) -> list[dict[str, Any]]:
    """Keep weather reports with a usable point location near the product target."""
    reports: list[dict[str, Any]] = []
    for row in rows:
        if row.get('STATE') not in DMV_STATES or row.get('EVENT_TYPE') not in EVENT_TYPES:
            continue
        try:
            report_latitude = float(row['BEGIN_LAT'])
            report_longitude = float(row['BEGIN_LON'])
        except (KeyError, TypeError, ValueError):
            continue
        distance_km = haversine_km(latitude_deg, longitude_deg, report_latitude, report_longitude)
        if distance_km <= radius_km:
            reports.append({
                'observed_at': report_timestamp_utc(row),
                'event_type': row['EVENT_TYPE'],
                'distance_km': distance_km,
            })
    return sorted(reports, key=lambda report: report['observed_at'])


def group_episode_reports(reports: Iterable[dict[str, Any]], gap_hours: int = 3) -> list[list[dict[str, Any]]]:
    """Group nearby-in-time reports into independent radar evaluation episodes."""
    episodes: list[list[dict[str, Any]]] = []
    for report in sorted(reports, key=lambda candidate: candidate['observed_at']):
        if not episodes or report['observed_at'] - episodes[-1][-1]['observed_at'] > timedelta(hours=gap_hours):
            episodes.append([report])
        else:
            episodes[-1].append(report)
    return episodes


def build_catalog_from_rows(
    year: int,
    rows: Iterable[dict[str, str]],
    *,
    source_url: str,
    latitude_deg: float = 38.922403,
    longitude_deg: float = -77.232125,
    radius_km: float = 150,
    pre_window_minutes: int = 90,
    post_window_minutes: int = 90,
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
) -> dict[str, Any]:
    """Return the immutable Kalman-only catalog uploaded with an EMR run."""
    reports = relevant_reports(rows, latitude_deg=latitude_deg, longitude_deg=longitude_deg, radius_km=radius_km)
    if start_utc is not None:
        reports = [report for report in reports if report['observed_at'] >= start_utc]
    if end_utc is not None:
        reports = [report for report in reports if report['observed_at'] < end_utc]
    episodes = group_episode_reports(reports)
    events = []
    for index, episode in enumerate(episodes, start=1):
        start = episode[0]['observed_at'] - timedelta(minutes=pre_window_minutes)
        end = episode[-1]['observed_at'] + timedelta(minutes=post_window_minutes)
        event_types = sorted({report['event_type'] for report in episode})
        events.append({
            'event_id': f'klwx-{year}-episode-{index:03d}',
            'radars': ['KLWX'],
            'start_utc': start.isoformat().replace('+00:00', 'Z'),
            'end_utc': end.isoformat().replace('+00:00', 'Z'),
            'purpose': f'NCEI report-anchored episode: {", ".join(event_types)}',
            'report_count': len(episode),
        })
    return {
        'schema_version': 2,
        'catalog_kind': 'report_anchored_kalman_backtest',
        'source': {
            'name': 'NOAA NCEI Storm Events details CSV',
            'url': source_url,
            'year': year,
            'selection': f'{sorted(EVENT_TYPES)} reports in DMV states within {radius_km:g} km of McLean; grouped by 3-hour gaps',
        },
        'target': {
            'name': 'Booz Allen McLean office',
            'latitude_deg': latitude_deg,
            'longitude_deg': longitude_deg,
            'label_radius_m': 2_000,
            'threshold_dbz': 35,
        },
        'horizons_minutes': [30, 60],
        'scan_cadence_minutes': 10,
        'events': events,
    }


def build_year_catalog(
    year: int, *, start_utc: datetime | None = None, end_utc: datetime | None = None,
) -> dict[str, Any]:
    source_url, rows = download_details_rows(year)
    catalog = build_catalog_from_rows(year, rows, source_url=source_url, start_utc=start_utc, end_utc=end_utc)
    if not catalog['events']:
        raise RuntimeError(f'No report-anchored KLWX episodes were selected for the requested window in {year}.')
    if start_utc is not None or end_utc is not None:
        catalog['source']['selection'] += (
            f'; report time from {start_utc.isoformat() if start_utc else "the beginning"}'
            f' to {end_utc.isoformat() if end_utc else "the end"} (end exclusive)'
        )
    return catalog


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description='Build a NCEI report-anchored annual Storm Phase 3 catalog.')
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--output', help='Optional local JSON path; otherwise prints the catalog.')
    args = parser.parse_args()
    catalog = build_year_catalog(args.year)
    rendered = json.dumps(catalog, indent=2) + '\n'
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(rendered)
    else:
        print(rendered)


if __name__ == '__main__':
    main()
