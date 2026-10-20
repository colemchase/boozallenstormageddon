from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .nexrad import load_lowest_reflectivity_sweep
from .tracking import detect_cells, match_cells


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect and match storm cells in two NEXRAD scans.")
    parser.add_argument("previous_scan")
    parser.add_argument("current_scan")
    parser.add_argument("--threshold-dbz", type=float, default=35.0)
    parser.add_argument("--min-gates", type=int, default=20)
    parser.add_argument("--max-distance-m", type=float, default=30_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    previous = load_lowest_reflectivity_sweep(args.previous_scan)
    current = load_lowest_reflectivity_sweep(args.current_scan)
    previous_cells = detect_cells(previous, args.threshold_dbz, args.min_gates)
    current_cells = detect_cells(current, args.threshold_dbz, args.min_gates)
    elapsed_seconds = (current.observed_at - previous.observed_at).total_seconds()
    matches = match_cells(previous_cells, current_cells, elapsed_seconds, args.max_distance_m)
    result = {
        "previous_observed_at": previous.observed_at.isoformat(),
        "current_observed_at": current.observed_at.isoformat(),
        "previous_cells": [asdict(cell) for cell in previous_cells],
        "current_cells": [asdict(cell) for cell in current_cells],
        "matches": [asdict(match) for match in matches],
    }
    serialized = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
