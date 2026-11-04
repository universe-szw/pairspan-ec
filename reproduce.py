"""Offline numerical reproduction with independent frozen reference checks."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def verify_files():
    manifest = ROOT / "MANIFEST.sha256"
    declared = set()
    for line in manifest.read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.resolve().is_relative_to(ROOT.resolve()) or relative in declared:
            raise ValueError("Invalid or duplicate manifest path")
        declared.add(relative)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Package integrity mismatch: {relative}")
    return len(declared)


def compare_csv(actual, reference):
    with actual.open(newline="", encoding="utf-8") as handle:
        actual_rows = list(csv.DictReader(handle))
    with reference.open(newline="", encoding="utf-8") as handle:
        reference_rows = list(csv.DictReader(handle))
    if len(actual_rows) != len(reference_rows):
        raise ValueError(f"Row-count mismatch: {actual.name}")
    compared = 0
    for index, (left, right) in enumerate(zip(actual_rows, reference_rows)):
        if left.keys() != right.keys():
            raise ValueError(f"Column mismatch: {actual.name}")
        for field in left:
            a, b = left[field], right[field]
            if a != b:
                try:
                    agrees = math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)
                except ValueError:
                    agrees = False
                if not agrees:
                    raise ValueError(f"Mismatch: {actual.name} row {index} field {field}: {a} != {b}")
            compared += 1
    return {"rows": len(actual_rows), "compared_cells": compared}


def run(output):
    checked = verify_files()
    from fit_thresholds import run_fit, read_json, prepare_output, write_json
    from evaluate_operating_points import run_evaluation
    output = prepare_output(output)
    thresholds = run_fit(ROOT / "data/calibration.jsonl.gz", output / "fit")
    entries = read_json(thresholds)["entries"]
    if entries != read_json(ROOT / "reference/threshold_entries.json"):
        raise ValueError("Refitted threshold entries differ from the frozen paper values")
    evaluated = run_evaluation(thresholds, ROOT / "data/scores.jsonl.gz",
                               ROOT / "data/default_anchors.json", output / "evaluation")
    comparisons = {name: compare_csv(evaluated / name, ROOT / "reference" / name)
                   for name in ("operating_points.csv", "per_length_calibrated.csv",
                                "base_positive_filtering.csv", "paired_cluster_deltas.csv")}
    from summarize_timing import summarize
    timing = summarize(ROOT, output)
    report = dict(status="passed", files_hash_verified=checked,
                  threshold_entries_verified=len(entries), evaluation=comparisons,
                  timing=timing,
                  scope="Recomputation from original numeric measurements; no fresh model inference.")
    write_json(output / "verification.json", report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    arguments = parser.parse_args()
    run(arguments.output)
