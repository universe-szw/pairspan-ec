"""Public numeric validation; no private files, models, network, or GPU."""

import copy
import csv
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import fit_thresholds as fit
import evaluate_operating_points as evaluate


class NumericTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = fit.load_rows(ROOT / "data/scores.jsonl.gz", set(fit.SIZES))
        cls.cells, cls.keys = fit.validate_grid(cls.rows, set(fit.SIZES))

    def test_complete_grid_and_denominators(self):
        self.assertEqual(len(self.rows), 33935)
        self.assertEqual(len(self.keys), 11)
        attack = self.cells[("harmbench2385", self.keys[0])]
        self.assertEqual(sum(not r["base_harmful"] for r in attack), 2380)
        self.assertEqual(len({r["base_group_id"] for r in attack}), 159)

    def test_calibration_subset_identity(self):
        calibration = fit.load_rows(ROOT / "data/calibration.jsonl.gz", {"calibration200"})
        self.assertEqual(calibration, [r for r in self.rows if r["dataset_id"] == "calibration200"])

    def test_fitted_thresholds_match_reference(self):
        calibration = [r for r in self.rows if r["dataset_id"] == "calibration200"]
        document, _ = fit.fit_document(calibration, fit.ALPHAS)
        self.assertEqual(document["entries"], fit.read_json(ROOT / "reference/threshold_entries.json"))

    def test_calibration_entry_point_rejects_test_data(self):
        with self.assertRaisesRegex(ValueError, "unexpected dataset"):
            fit.load_rows(ROOT / "data/scores.jsonl.gz", {"calibration200"})

    def test_missing_and_duplicate_rows_rejected(self):
        for rows in (self.rows[:-1], self.rows + [self.rows[-1]]):
            with self.assertRaises(ValueError):
                fit.validate_grid(rows, set(fit.SIZES))

    def test_identity_mismatch_rejected(self):
        rows = copy.deepcopy(self.rows)
        index = next(i for i, r in enumerate(rows) if r["method_id"] == "pairspan_b384")
        rows[index]["request_id"] = "wrong-id"
        with self.assertRaises(ValueError):
            fit.validate_grid(rows, set(fit.SIZES))

    def test_loader_rejects_nonfinite_caps_and_extra_fields(self):
        source = next(r for r in self.rows if r["method_id"] == "pairspan_b384")
        for field, value in (("ec_score", float("nan")), ("ec_score", float("inf")),
                             ("ec_score", -0.1), ("classifier_checks", 385),
                             ("base_harmful", 0), ("prompt_text", "not allowed")):
            row = dict(source, **{field: value})
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "bad.jsonl"
                path.write_text(json.dumps(row) + "\n")
                with self.assertRaises(ValueError):
                    fit.load_rows(path, set(fit.SIZES))

    def test_ties_are_retained_no_jitter(self):
        rows = [{"default_harmful": True, "ec_score": x} for x in (0.8, 0.8, 1.0)]
        steps = fit.threshold_steps(rows)
        self.assertEqual(steps[1], (0.8, 3))
        self.assertEqual(steps[-1], (math.nextafter(1.0, math.inf), 0))

    def test_paper_default_anchors(self):
        evaluate.validate_anchors(ROOT / "data/default_anchors.json", self.cells, self.keys)

    def test_scoped_schema_contains_only_public_fields(self):
        allowed = fit.REQUIRED
        self.assertTrue(all(set(r) == allowed for r in self.rows))
        self.assertEqual(len(allowed), 12)

    def test_wilson_extremes(self):
        self.assertAlmostEqual(evaluate.wilson(0, 500)[0], 0)
        self.assertAlmostEqual(evaluate.wilson(500, 500)[1], 1)

    def test_threshold_round_trip_without_sidecars(self):
        calibration = [r for r in self.rows if r["dataset_id"] == "calibration200"]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fit"
            path = fit.run_fit(ROOT / "data/calibration.jsonl.gz", output)
            document = evaluate.load_thresholds(path, calibration)
            self.assertEqual(document["entries"], fit.read_json(ROOT / "reference/threshold_entries.json"))
            self.assertEqual({p.name for p in output.iterdir()},
                             {"thresholds.json", "calibration_steps.csv"})
            self.assertEqual(set(document), {"schema_version", "comparison", "threshold_grid",
                             "selection", "budget_rule", "alphas", "configurations", "entries"})

    def test_modified_thresholds_rejected_by_refitting(self):
        calibration = [r for r in self.rows if r["dataset_id"] == "calibration200"]
        document, _ = fit.fit_document(calibration, fit.ALPHAS)
        for field in ("entries", "comparison"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                changed = copy.deepcopy(document)
                if field == "entries":
                    changed[field][0]["theta"] = -1.0
                    changed[field][0]["theta_hex"] = (-1.0).hex()
                else:
                    changed[field] = "ec_score > theta"
                path = Path(directory) / "thresholds.json"
                fit.write_json(path, changed)
                with self.assertRaisesRegex(ValueError, "thresholds differ from calibration"):
                    evaluate.load_thresholds(path, calibration)

    def test_result_map_covers_all_aggregate_exports(self):
        text = (ROOT / "RESULT_MAP.md").read_text(encoding="utf-8")
        paths = re.findall(r"\]\((data/aggregate/[^)]+\.csv)\)", text)
        actual = {p.relative_to(ROOT).as_posix()
                  for p in (ROOT / "data/aggregate").glob("*.csv")}
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(set(paths), actual)
        for relative in paths:
            path = ROOT / relative
            with path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            self.assertTrue(reader.fieldnames)
            self.assertEqual(len(reader.fieldnames), len(set(reader.fieldnames)))
            self.assertTrue(rows)
            self.assertTrue(all(None not in row and None not in row.values() for row in rows))

    def test_jailbreakbench_counts_and_intervals(self):
        with (ROOT / "data/aggregate/jailbreakbench.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 21)
        subsets = {row["subset"] for row in rows}
        self.assertEqual(len(subsets), 7)
        self.assertEqual({(row["subset"], row["method"]) for row in rows},
                         {(subset, method) for subset in subsets
                          for method in ("base", "greedy20", "pairspan_b384")})
        for row in rows:
            k, n = int(row["detections"]), int(row["n"])
            self.assertTrue(0 <= k <= n)
            self.assertAlmostEqual(float(row["detection_rate"]), 100 * k / n)
            low, high = evaluate.wilson(k, n)
            self.assertAlmostEqual(float(row["wilson_ci_low"]), 100 * low)
            self.assertAlmostEqual(float(row["wilson_ci_high"]), 100 * high)
            if row["method"] == "pairspan_b384":
                self.assertLessEqual(int(row["max_queries"]), 384)
        manual = {row["method"]: row for row in rows
                  if row["subset"] == "jbc_manual_vicuna_13b_v1_5"}
        self.assertEqual(int(manual["pairspan_b384"]["detections"]), 1)
        self.assertEqual(int(manual["greedy20"]["detections"]), 100)
        self.assertEqual(float(manual["greedy20"]["mean_queries"]), 4701.34)


if __name__ == "__main__":
    unittest.main()
