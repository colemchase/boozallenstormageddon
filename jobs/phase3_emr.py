"""Spark entry point for memory-isolated Phase 3 historical evaluation.

Each Spark task processes one historical event. A task gets its own Python
process and EMR worker memory, which keeps Py-ART Level-II decoding out of the
4 GB SageMaker notebook kernel.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import tempfile
from typing import Any

import boto3

from storm_motion.nexrad import load_lowest_reflectivity_sweep
from storm_motion.phase3 import evaluate_paths


NOAA_BUCKET = 'unidata-nexrad-level2'


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith('s3://'):
        raise ValueError(f'Expected s3:// URI, got {uri}')
    bucket_and_key = uri[5:].split('/', 1)
    if len(bucket_and_key) != 2:
        raise ValueError(f'Expected bucket and key in {uri}')
    return bucket_and_key[0], bucket_and_key[1]


def select_regular_scans(event: dict[str, Any], client: Any, cadence_minutes: int = 5) -> list[tuple[datetime, str]]:
    """Select nearest distinct KLWX volumes on a regular evaluation cadence."""
    station = 'KLWX'
    start = datetime.fromisoformat(event['start_utc'].replace('Z', '+00:00'))
    end = datetime.fromisoformat(event['end_utc'].replace('Z', '+00:00'))
    pattern = re.compile(rf'{station}(\d{{8}})_(\d{{6}})_V06$')
    available: list[tuple[datetime, str]] = []
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while day <= end:
        prefix = f'{day:%Y/%m/%d}/{station}/'
        paginator = client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=NOAA_BUCKET, Prefix=prefix):
            for item in page.get('Contents', []):
                match = pattern.search(item['Key'])
                if not match:
                    continue
                observed_at = datetime.strptime(''.join(match.groups()), '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)
                if start <= observed_at <= end:
                    available.append((observed_at, item['Key']))
        day += timedelta(days=1)
    if not available:
        raise RuntimeError(f"No {station} scans found for {event['event_id']}")
    requested, current = [], start
    while current <= end:
        requested.append(current)
        current += timedelta(minutes=cadence_minutes)
    selected: list[tuple[datetime, str]] = []
    used: set[str] = set()
    for requested_time in requested:
        observed_at, key = min(available, key=lambda item: abs(item[0] - requested_time))
        if key not in used:
            selected.append((observed_at, key))
            used.add(key)
    return selected


def record_event_progress(output_uri: str, event_id: str, outcome: dict[str, object]) -> None:
    """Write one immutable completion marker so the submitter can show progress.

    Each Spark task owns a different event ID, so workers never contend for the
    same S3 key. Progress is deliberately best-effort: a display issue must
    not turn an otherwise useful backtest result into a failed event.
    """
    try:
        bucket, prefix = parse_s3_uri(output_uri.rstrip('/') + '/')
        boto3.client('s3').put_object(
            Bucket=bucket,
            Key=f"{prefix.rstrip('/')}/progress/{event_id}.json",
            Body=json.dumps({
                'event_id': event_id,
                'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                'row_count': len(outcome['rows']),
                'error': outcome['error'],
            }, sort_keys=True).encode(),
            ContentType='application/json',
        )
    except Exception:
        pass


def evaluate_event(payload: tuple[dict[str, Any], dict[str, Any], list[int], int, str]) -> dict[str, object]:
    """Spark worker function; returns rows or a per-episode failure record."""
    event, target, horizons_minutes, cadence_minutes, output_uri = payload
    # The EMR instance profile is permitted to read this public bucket. Using
    # normal signed requests keeps this compatible with current botocore
    # endpoint rules on the EMR nodes.
    s3 = boto3.client('s3', region_name='us-east-1')
    try:
        selected = select_regular_scans(event, s3, cadence_minutes=cadence_minutes)
        with tempfile.TemporaryDirectory(prefix='storm-phase3-') as directory:
            paths: list[Path] = []
            for _, key in selected:
                path = Path(directory) / Path(key).name
                s3.download_file(NOAA_BUCKET, key, str(path))
                paths.append(path)
            rows = evaluate_paths(
                event_id=event['event_id'], paths=paths, load_frame=load_lowest_reflectivity_sweep,
                target_latitude_deg=target['latitude_deg'], target_longitude_deg=target['longitude_deg'],
                horizons_minutes=horizons_minutes, label_radius_m=target['label_radius_m'],
                threshold_dbz=target['threshold_dbz'],
            )
        outcome = {'event_id': event['event_id'], 'rows': [row.as_dict() for row in rows], 'error': None}
    except Exception as error:  # Preserve a one-year batch when a source episode is malformed.
        outcome = {'event_id': event['event_id'], 'rows': [], 'error': f'{type(error).__name__}: {error}'}
    record_event_progress(output_uri, event['event_id'], outcome)
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description='Run Storm Phase 3 evaluation as one Spark task per event.')
    parser.add_argument('--catalog-uri', required=True)
    parser.add_argument('--output-uri', required=True, help='S3 prefix for rows.jsonl and manifest.json')
    args = parser.parse_args()

    catalog_bucket, catalog_key = parse_s3_uri(args.catalog_uri)
    catalog = json.loads(boto3.client('s3').get_object(Bucket=catalog_bucket, Key=catalog_key)['Body'].read())

    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName('storm-phase3-historical-evaluation').getOrCreate()
    cadence_minutes = int(catalog.get('scan_cadence_minutes', 5))
    payloads = [
        (event, catalog['target'], catalog['horizons_minutes'], cadence_minutes, args.output_uri)
        for event in catalog['events']
    ]
    outcomes = spark.sparkContext.parallelize(payloads, min(len(payloads), 24)).map(evaluate_event).collect()
    rows = [row for outcome in outcomes for row in outcome['rows']]
    failures = [
        {'event_id': outcome['event_id'], 'error': outcome['error']}
        for outcome in outcomes if outcome['error'] is not None
    ]

    output_bucket, output_prefix = parse_s3_uri(args.output_uri.rstrip('/') + '/')
    output_prefix = output_prefix.rstrip('/')
    output = boto3.client('s3')
    output.put_object(
        Bucket=output_bucket,
        Key=f'{output_prefix}/rows.jsonl',
        Body=''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows).encode(),
        ContentType='application/x-ndjson',
    )
    output.put_object(
        Bucket=output_bucket,
        Key=f'{output_prefix}/failed_events.jsonl',
        Body=''.join(json.dumps(failure, sort_keys=True) + '\n' for failure in failures).encode(),
        ContentType='application/x-ndjson',
    )
    output.put_object(
        Bucket=output_bucket,
        Key=f'{output_prefix}/manifest.json',
        Body=json.dumps({
            'event_count': len(payloads), 'successful_event_count': len(payloads) - len(failures),
            'failed_event_count': len(failures), 'row_count': len(rows), 'scan_cadence_minutes': cadence_minutes,
            'catalog_uri': args.catalog_uri, 'method': 'shape-aware-kalman-persistence',
        }, indent=2).encode(),
        ContentType='application/json',
    )
    print(json.dumps({'output_uri': args.output_uri, 'row_count': len(rows)}))
    spark.stop()


if __name__ == '__main__':
    main()
