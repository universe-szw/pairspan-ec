"""Public numeric validation; no private files, models, network, or GPU."""

import copy
import ast
import gzip
import hashlib
import json
import math
from pathlib import Path
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

    def test_scoped_schema_contains_no_text_or_hashes(self):
        allowed = fit.REQUIRED
        self.assertTrue(all(set(r) == allowed for r in self.rows))
        self.assertEqual(len(allowed), 12)

    def test_wilson_extremes(self):
        self.assertAlmostEqual(evaluate.wilson(0, 500)[0], 0)
        self.assertAlmostEqual(evaluate.wilson(500, 500)[1], 1)

    def test_exported_function_identities(self):
        identity = json.loads((ROOT / "provenance/code_identity.json").read_text())
        source = (ROOT / "src/pairspan.py").read_text(encoding="utf-8")
        functions = {node.name: ast.get_source_segment(source, node)
                     for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
        self.assertEqual(set(functions), set(identity["function_sha256"]))
        for name, expected in identity["function_sha256"].items():
            self.assertEqual(hashlib.sha256(functions[name].encode()).hexdigest(), expected, name)


if __name__ == "__main__":
    unittest.main()
