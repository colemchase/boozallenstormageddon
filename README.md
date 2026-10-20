# storm-tracker-app

Baseline storm-cell detection and motion tracking for the 2026 Stormageddon
Datathon. The application deliberately starts with a small, inspectable
algorithm: threshold reflectivity, label connected regions, and match cells
between two consecutive scans.

It is independent of the CDK repository. It reads NOAA NEXRAD Level II files
locally through Py-ART and does not need to copy the public source data into
our AWS account.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[nexrad,dev]'
```

## Run a first comparison

Download two nearby scans from one radar station, for example `KLWX`, then:

```bash
storm-track first-scan second-scan --threshold-dbz 35 --output tracks.json
```

The command emits a JSON document containing detected cells and matched motion
vectors. `speed_mps` is the centroid-to-centroid distance divided by the scan
timestamp interval. Review the cells on a radar plot before treating a match as
meteorological truth: this is a baseline, not an operational warning system.

## Current assumptions

- The two files are consecutive scans from the same station.
- Detection uses the lowest elevation sweep with a reflectivity field.
- A storm cell is at least 20 contiguous polar-grid gates at or above the
  configured dBZ threshold.
- Matching is one-to-one nearest neighbour within 30 km by default.

## Checks

```bash
pytest
```

Next milestones are scan visualization, split/merge classification, and
short-horizon extrapolated position forecasts.
jupyter notebook app for tracking storms
