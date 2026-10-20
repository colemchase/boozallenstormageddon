# Phase 3: historical baseline evaluation

## Smoke run on EMR

Notebook 07 is useful as a small control panel, but Level-II radar decoding is
too memory-intensive for the 4 GB shared JupyterLab space once the evaluator
preserves each storm's gate geometry. The Phase 3 smoke catalog therefore runs
as one Spark task per event on a transient EMR cluster.

After the `storm-infra` stack has published the EMR launch configuration, run
this from the cloned SageMaker repository terminal:

```bash
python jobs/submit_phase3_emr.py --wait
```

The submitter packages `src/storm_motion`, uploads the immutable catalog and
Spark entry point to the Storm derived-data bucket, starts the configured EMR
7.12 cluster, and writes the result to:

```text
s3://<storm-derived-bucket>/phase3/runs/<UTC-run-id>/rows.jsonl
```

It uses one primary, one core, and two task `m5.xlarge` nodes. The bootstrap
action installs pinned Py-ART dependencies with EMR 7.12's Python 3.11 on
every node; Spark then uses that interpreter for both its driver and workers.
This is a one-shot batch cluster: it terminates automatically after the
evaluation step completes (or fails), so it cannot hold the limited
`m5.xlarge` quota between runs.

With `--wait`, the submitter also downloads `rows.jsonl` into
`data/phase3_smoke_results.jsonl`, which is the file consumed by Notebook 08
for visual review.

The Phase 3 unit of data is one target, event, scan time, and forecast horizon.
Each derived row records the event ID, radar inputs, active track IDs, modelled
per-track/combined McLean coverage probability, and the later observed 35 dBZ
coverage label within a 2 km office buffer. Store immutable JSONL/Parquet rows
under `s3://<derived-bucket>/phase3/` with the catalog version and code commit.

Start with the two smoke cases in `config/phase3_seed_catalog.json`. After the
schema is proven, expand to tens of report-anchored storm cases, process them
in Spark on EMR, and report Brier score, reliability, hit rate, miss rate,
false-alarm rate, track continuity, and position error by 30- and 60-minute
horizon.

## One-year Kalman-only backtest

The next step is a 2024 KLWX backtest with **no ML model**. The submitter
downloads the official NOAA NCEI Storm Events details CSV, selects DMV storm
reports within 150 km of McLean, groups reports separated by less than three
hours into radar episodes, then evaluates KLWX Level-II scans at a 10-minute
cadence. This is a storm-event sample, not a random sample of all 2024 weather;
that distinction stays in the output catalog and later evaluation report.

Run it from the SageMaker terminal:

```bash
python jobs/submit_phase3_emr.py --year 2024 --wait
```

The terminal reports real completed episodes as the workers finish them (for
example, `18/70 episodes complete`), rather than guessing from elapsed time.

For a quicker, known-storm 30-day pilot, use May 1–30, 2024. This includes the
May 27 multi-cluster case and writes the result used by Notebook 09:

```bash
python jobs/submit_phase3_emr.py --start-date 2024-05-01 --days 30 --wait
```

The annual result is downloaded to:

```text
data/phase3_2024_kalman_results.jsonl
```

If the terminal's wait expires or disconnects, do **not** submit a second
annual job.  Once the EMR step is complete, download its already-written S3
results by using the run ID printed at submission:

```bash
python jobs/submit_phase3_emr.py --year 2024 --fetch-run 20261020T003148Z
```

The S3 run folder also contains `failed_events.jsonl` so a malformed or missing
source episode does not discard the rest of the one-year batch. The same
transient four-node EMR shape is used and terminates after the step completes.

The current label is **radar reflectivity coverage**, not severe weather.
Verified severe-wind/hail/tornado reports become a separate target only after
their spatial and temporal matching policy is reviewed.
