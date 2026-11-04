"""Optional inference tests use harmless synthetic fixtures, not paper inputs."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
AVAILABLE = importlib.util.find_spec("torch") is not None
if AVAILABLE:
    import pairspan as policy


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(int(i)) for i in ids)


class Classifier:
    def __init__(self, original, mode):
        self.original = original
        self.mode = mode
        self.checked = []

    def __call__(self, texts):
        self.checked.extend(texts)
        outputs = []
        for text in texts:
            p = 0.8 if self.mode == "base" or (text != self.original and self.mode == "hit") else 0.2
            outputs.append({"label": "LABEL_0" if p >= 0.5 else "LABEL_1",
                            "score": p if p >= 0.5 else 1 - p})
        return outputs


@unittest.skipUnless(AVAILABLE, "Install requirements-inference.txt for core tests")
class CoreTests(unittest.TestCase):
    def test_base_positive_stops_at_one(self):
        text = "red green blue white"
        classifier = Classifier(text, "base")
        result = policy.pairspan_ec_trace(text, classifier, Tokenizer(), None)
        self.assertTrue(result["harmful"])
        self.assertEqual(result["classifier_query_count"], 1)

    def test_hard_cap_all_negative(self):
        for cap in (1, 2, 16, 64, 128, 384):
            text = "abcdefghijklmnopqrstuvwxyz0123456789"
            classifier = Classifier(text, "none")
            result = policy.pairspan_ec_trace(text, classifier, Tokenizer(), None,
                                              total_query_cap=cap, use_saliency=False)
            self.assertFalse(result["harmful"])
            self.assertLessEqual(result["classifier_query_count"], cap)
            self.assertEqual(result["classifier_query_count"], len(classifier.checked))

    def test_stage_stopping_not_first_candidate_rewrite(self):
        text = "abcdefghijklmnopqrstuvwxyz0123456789"
        classifier = Classifier(text, "hit")
        result = policy.pairspan_ec_trace(text, classifier, Tokenizer(), None,
                                          total_query_cap=64, use_saliency=False)
        self.assertTrue(result["harmful"])
        self.assertGreater(result["classifier_query_count"], 2)
        self.assertEqual(result["classifier_query_count"], len(classifier.checked))

    def test_margin_and_harmful_label_both_required(self):
        self.assertEqual(policy._margin_hit_count([True], [0.51], 0.49), 0)
        self.assertEqual(policy._margin_hit_count([True], [0.53], 0.49), 1)
        self.assertEqual(policy._margin_hit_count([False], [0.9], 0.1), 0)

    def test_no_overlap_and_dedup_interleaving(self):
        self.assertTrue(policy._non_overlapping((0, 3), (3, 5)))
        self.assertFalse(policy._non_overlapping((0, 3), (2, 5)))
        prompts, _ = policy._interleave_pair_families(
            [("a", []), ("b", [])], [("a", []), ("c", [])], 4)
        self.assertEqual(prompts, ["a", "b", "c"])

    def test_geometric_and_random_count_all_executed(self):
        text = "abcdefghijklmnopqrstuvwxyz0123456789"
        for runner in (policy.geometric_prefix_ec_trace, policy.random_pair_grid_ec_trace):
            classifier = Classifier(text, "hit")
            result = runner(text, classifier, Tokenizer(), candidate_budget=32)
            self.assertEqual(result["classifier_query_count"], len(classifier.checked))
            self.assertLessEqual(len(classifier.checked), 32)

    def test_saliency_proposals_counted_separately(self):
        import numpy as np
        text = "abcdefghijklmnopqrstuvwxyz0123456789"
        classifier = Classifier(text, "none")
        with patch.object(policy, "get_risk_heatmap", return_value=(np.arange(len(text)), list(text))):
            result = policy.pairspan_ec_trace(text, classifier, Tokenizer(), None, total_query_cap=64)
        self.assertEqual(result["gradient_pass_count"], 1)
        self.assertEqual(result["classifier_query_count"], len(classifier.checked))

    def test_boolean_api_reports_saliency_failure(self):
        text = "abcdefghijklmnopqrstuvwxyz0123456789"
        with patch.object(policy, "get_risk_heatmap", side_effect=RuntimeError("test failure")):
            result = policy.pairspan_ec_trace(text, Classifier(text, "none"), Tokenizer(), None,
                                              total_query_cap=64)
            self.assertEqual(result["saliency_error_type"], "RuntimeError")
            with self.assertRaisesRegex(RuntimeError, "Saliency failed: RuntimeError"):
                policy.pairspan_ec(text, Classifier(text, "none"), Tokenizer(), None,
                                   total_query_cap=64)

    def test_boolean_api_allows_explicit_no_saliency(self):
        text = "abcdefghijklmnopqrstuvwxyz0123456789"
        with patch.object(policy, "get_risk_heatmap", side_effect=AssertionError("not requested")):
            result = policy.pairspan_ec(text, Classifier(text, "none"), Tokenizer(), None,
                                       total_query_cap=64, use_saliency=False)
        self.assertFalse(result)

    def test_boolean_api_matches_successful_trace(self):
        import numpy as np
        text = "abcdefghijklmnopqrstuvwxyz0123456789"
        with patch.object(policy, "get_risk_heatmap", return_value=(np.arange(len(text)), list(text))):
            for mode in ("base", "none", "hit"):
                trace = policy.pairspan_ec_trace(text, Classifier(text, mode), Tokenizer(), None,
                                                 total_query_cap=64)
                result = policy.pairspan_ec(text, Classifier(text, mode), Tokenizer(), None,
                                           total_query_cap=64)
                self.assertEqual(result, trace["harmful"])

    def test_actual_gradient_saliency_with_tiny_local_model(self):
        import torch
        from types import SimpleNamespace
        from saliency import get_risk_heatmap

        class GradientTokenizer(Tokenizer):
            def __call__(self, text, **kwargs):
                ids = [1] + self.encode(text)[:kwargs["max_length"] - 2] + [2]
                return {"input_ids": torch.tensor([ids]),
                        "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)}

            def convert_ids_to_tokens(self, ids):
                return [str(int(i)) for i in ids]

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(128, 4)

            @property
            def device(self):
                return self.embedding.weight.device

            def get_input_embeddings(self):
                return self.embedding

            def forward(self, inputs_embeds, attention_mask):
                weights = torch.arange(1, inputs_embeds.shape[1] + 1, dtype=inputs_embeds.dtype)
                value = (inputs_embeds * weights[None, :, None]).sum()
                return SimpleNamespace(logits=torch.stack([value, -value])[None, :])

        model = TinyModel().eval()
        scores, tokens = get_risk_heatmap("red green", SimpleNamespace(model=model),
                                         GradientTokenizer(), max_length=8)
        self.assertEqual(len(scores), 6)
        self.assertEqual(len(tokens), 6)
        self.assertAlmostEqual(float(scores.min()), 0.0)
        self.assertAlmostEqual(float(scores.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
