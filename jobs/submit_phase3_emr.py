"""Package and submit a Phase 3 Kalman catalog to a transient EMR cluster.

Run this from the cloned SageMaker repository after the infra stack has been
deployed. It starts a fresh cluster, runs the Spark step, and terminates it
after the one-shot batch is complete.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import boto3


REGION = 'us-west-2'
ENVIRONMENT = 'dev'


def s3_uri(bucket: str, key: str) -> str:
    return f's3://{bucket}/{key}'


def package_source(repository: Path) -> bytes:
    """Build a small importable zip containing only the storm_motion package."""
    output = io.BytesIO()
    package_root = repository / 'src' / 'storm_motion'
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for source in package_root.rglob('*.py'):
            archive.write(source, source.relative_to(repository / 'src'))
    return output.getvalue()


def first_step_id(emr, cluster_id: str, timeout_seconds: int = 120) -> str:
    """Return the initial EMR step after RunJobFlow has registered it."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        steps = emr.list_steps(ClusterId=cluster_id)["Steps"]
        if steps:
            return steps[0]["Id"]
        time.sleep(5)
    raise RuntimeError(f"Timed out waiting for the Phase 3 step on {cluster_id}.")


def download_results(s3, bucket: str, repository: Path, run_id: str, local_results_name: str) -> Path:
    """Download results from a completed EMR run without submitting another one."""
    local_results = repository / 'data' / local_results_name
    local_results.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, f'phase3/runs/{run_id}/rows.jsonl', str(local_results))
    return local_results


def completed_event_count(s3, bucket: str, run_id: str) -> int:
    """Count worker-written completion markers for a live EMR run."""
    prefix = f'phase3/runs/{run_id}/progress/'
    paginator = s3.get_paginator('list_objects_v2')
    return sum(
        1
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix)
        for item in page.get('Contents', [])
        if item['Key'].endswith('.json')
    )


def show_progress(completed: int, total: int, elapsed_seconds: float) -> None:
    width = 28
    ratio = min(completed / total, 1.0) if total else 0.0
    filled = round(width * ratio)
    bar = '#' * filled + '-' * (width - filled)
    elapsed_minutes = int(elapsed_seconds // 60)
    print(f'\rPhase 3: [{bar}] {completed}/{total} episodes complete; {elapsed_minutes}m elapsed', end='', flush=True)


def wait_for_step_with_progress(emr, s3, bucket: str, cluster_id: str, step_id: str, run_id: str, event_count: int, timeout_seconds: int) -> None:
    """Wait for EMR and render actual completed-event progress in the terminal."""
    started = time.monotonic()
    deadline = started + timeout_seconds
    while time.monotonic() < deadline:
        completed = completed_event_count(s3, bucket, run_id)
        show_progress(completed, event_count, time.monotonic() - started)
        state = emr.describe_step(ClusterId=cluster_id, StepId=step_id)['Step']['Status']['State']
        if state == 'COMPLETED':
            show_progress(event_count, event_count, time.monotonic() - started)
            print()
            return
        if state in {'CANCELLED', 'FAILED', 'INTERRUPTED'}:
            print(file=sys.stderr)
            raise RuntimeError(f'EMR step {step_id} ended with state {state}. Check the EMR logs for details.')
        time.sleep(30)
    print(file=sys.stderr)
    raise TimeoutError(f'EMR step {step_id} is still running after {timeout_seconds // 3600} hours; it was not resubmitted or cancelled.')


def main() -> None:
    parser = argparse.ArgumentParser(description='Submit a Storm Phase 3 Kalman evaluation to EMR.')
    parser.add_argument('--wait', action='store_true', help='Wait for the EMR step to finish.')
    parser.add_argument('--year', type=int, help='Build and run the NCEI report-anchored KLWX catalog for this calendar year.')
    parser.add_argument('--start-date', help='Limit evaluation to a local Eastern start date (YYYY-MM-DD); use with --days.')
    parser.add_argument('--days', type=int, help='Number of local Eastern calendar days to evaluate; use with --start-date.')
    parser.add_argument('--fetch-run', metavar='RUN_ID', help='Download a completed run from S3; does not submit EMR work.')
    args = parser.parse_args()
    if bool(args.start_date) != bool(args.days):
        parser.error('--start-date and --days must be used together.')
    if args.days is not None and args.days <= 0:
        parser.error('--days must be positive.')
    window_start_utc = window_end_utc = None
    if args.start_date:
        try:
            window_start_date = date.fromisoformat(args.start_date)
        except ValueError as error:
            parser.error(f'--start-date must be YYYY-MM-DD: {error}')
        if args.year is not None and args.year != window_start_date.year:
            parser.error('--year must match the year in --start-date.')
        # NCEI reports use DMV local time. Preserve Eastern calendar days
        # across daylight-saving transitions before converting to UTC.
        from zoneinfo import ZoneInfo
        eastern = ZoneInfo('America/New_York')
        window_start_utc = datetime.combine(window_start_date, datetime.min.time(), eastern).astimezone(timezone.utc)
        window_end_utc = datetime.combine(window_start_date + timedelta(days=args.days), datetime.min.time(), eastern).astimezone(timezone.utc)
        args.year = window_start_date.year
    repository = Path(__file__).resolve().parents[1]
    session = boto3.session.Session(region_name=REGION)
    ssm = session.client('ssm')
    launch = json.loads(ssm.get_parameter(
        Name=f'/storm/{ENVIRONMENT}/emr/launch-configuration',
    )['Parameter']['Value'])
    derived_bucket = ssm.get_parameter(
        Name=f'/storm/{ENVIRONMENT}/derived-data-bucket-name',
    )['Parameter']['Value']

    if args.start_date:
        catalog_name = f'phase3_{args.start_date}_{args.days}d_kalman_catalog'
        local_results_name = 'phase3_latest_kalman_results.jsonl'
        run_label = f'window-{args.start_date}-{args.days}d'
    elif args.year is not None:
        catalog_name = f'phase3_{args.year}_kalman_catalog'
        local_results_name = f'phase3_{args.year}_kalman_results.jsonl'
        run_label = f'year-{args.year}'
    else:
        catalog_name = 'phase3_seed_catalog'
        local_results_name = 'phase3_smoke_results.jsonl'
        run_label = 'smoke'

    s3 = session.client('s3')
    if args.fetch_run:
        local_results = download_results(s3, derived_bucket, repository, args.fetch_run, local_results_name)
        print('Downloaded completed EMR results:', local_results)
        return

    if args.year is not None:
        from build_phase3_year_catalog import build_year_catalog
        catalog = build_year_catalog(args.year, start_utc=window_start_utc, end_utc=window_end_utc)
    else:
        catalog = json.loads((repository / 'config' / 'phase3_seed_catalog.json').read_text())

    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    artifact_prefix = f'emr/phase3/{run_id}'
    package_key = f'{artifact_prefix}/storm_motion.zip'
    catalog_key = f'{artifact_prefix}/{catalog_name}.json'
    job_key = f'{artifact_prefix}/phase3_emr.py'
    s3.put_object(Bucket=derived_bucket, Key=package_key, Body=package_source(repository))
    s3.put_object(
        Bucket=derived_bucket,
        Key=catalog_key,
        Body=(json.dumps(catalog, sort_keys=True, indent=2) + '\n').encode(),
        ContentType='application/json',
    )
    s3.upload_file(str(repository / 'jobs' / 'phase3_emr.py'), derived_bucket, job_key)

    output_uri = s3_uri(derived_bucket, f'phase3/runs/{run_id}')
    step_args = [
        'spark-submit', '--deploy-mode', 'cluster',
        '--conf', 'spark.pyspark.python=/usr/bin/python3.11',
        '--conf', 'spark.pyspark.driver.python=/usr/bin/python3.11',
        '--conf', 'spark.executorEnv.PYSPARK_PYTHON=/usr/bin/python3.11',
        '--conf', 'spark.yarn.appMasterEnv.PYSPARK_PYTHON=/usr/bin/python3.11',
        '--conf', 'spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON=/usr/bin/python3.11',
        '--conf', 'spark.executor.instances=3',
        '--conf', 'spark.executor.cores=2',
        '--conf', 'spark.executor.memory=8g',
        '--conf', 'spark.executor.memoryOverhead=4g',
        '--conf', 'spark.driver.memory=8g',
        '--py-files', s3_uri(derived_bucket, package_key),
        s3_uri(derived_bucket, job_key),
        '--catalog-uri', s3_uri(derived_bucket, catalog_key),
        '--output-uri', output_uri,
    ]
    emr = session.client('emr')
    response = emr.run_job_flow(
        Name=f'storm-{ENVIRONMENT}-phase3-{run_id}',
        ReleaseLabel=launch['release_label'],
        Applications=[{'Name': 'Spark'}, {'Name': 'Hive'}, {'Name': 'Presto'}],
        ServiceRole=launch['service_role_name'],
        JobFlowRole=launch['instance_profile_name'],
        VisibleToAllUsers=True,
        LogUri=launch['log_uri'],
        AutoTerminationPolicy={'IdleTimeout': launch['idle_timeout_seconds']},
        BootstrapActions=[{
            'Name': 'install-storm-python-dependencies',
            'ScriptBootstrapAction': {'Path': launch['bootstrap_uri']},
        }],
        Instances={
            'Ec2SubnetId': launch['subnet_id'],
            # This is a one-shot batch evaluation, not an interactive EMR
            # workspace. Terminate after its only step so it cannot consume
            # the account's limited m5.xlarge quota while idle.
            'KeepJobFlowAliveWhenNoSteps': False,
            'AdditionalMasterSecurityGroups': [launch['additional_security_group_id']],
            'AdditionalSlaveSecurityGroups': [launch['additional_security_group_id']],
            'InstanceGroups': [
                {'Name': 'Primary', 'InstanceRole': 'MASTER', 'InstanceType': 'm5.xlarge', 'InstanceCount': 1, 'Market': 'ON_DEMAND'},
                {'Name': 'Core', 'InstanceRole': 'CORE', 'InstanceType': 'm5.xlarge', 'InstanceCount': 1, 'Market': 'ON_DEMAND'},
                {'Name': 'Tasks', 'InstanceRole': 'TASK', 'InstanceType': 'm5.xlarge', 'InstanceCount': 2, 'Market': 'ON_DEMAND'},
            ],
        },
        Steps=[{
            'Name': f'storm-phase3-kalman-{run_label}',
            'ActionOnFailure': 'TERMINATE_CLUSTER',
            'HadoopJarStep': {'Jar': 'command-runner.jar', 'Args': step_args},
        }],
    )
    cluster_id = response['JobFlowId']
    print(json.dumps({
        'cluster_id': cluster_id,
        'output_uri': output_uri,
        'log_uri': launch['log_uri'],
        'command': f'aws emr describe-cluster --region {REGION} --cluster-id {cluster_id}',
    }, indent=2))
    if args.wait:
        step_id = first_step_id(emr, cluster_id)
        wait_for_step_with_progress(
            emr, s3, derived_bucket, cluster_id, step_id, run_id, len(catalog['events']),
            # A one-year radar backtest is expected to take hours. The old
            # AWS default of 30 minutes is appropriate for small steps only.
            timeout_seconds=6 * 60 * 60 if args.year and not args.start_date else 2 * 60 * 60,
        )
        local_results = download_results(s3, derived_bucket, repository, run_id, local_results_name)
        print('EMR step completed. Results:', output_uri)
        print('Downloaded results for the matching review notebook:', local_results)


if __name__ == '__main__':
    main()
