"""Optional end-to-end CLI smoke test with a tiny random local classifier."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
AVAILABLE = all(importlib.util.find_spec(name) is not None for name in (
    "torch", "transformers", "safetensors"))


@unittest.skipUnless(AVAILABLE, "Install requirements-inference.txt for runner test")
class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("USE_TF", "0")
        import torch
        from transformers import DistilBertTokenizerFast, DistilBertConfig, DistilBertForSequenceClassification
        torch.manual_seed(42)
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        cls.model_dir = cls.root / "model"
        cls.model_dir.mkdir()
        cls.words = "red green blue yellow white black silver gold pink gray cyan violet".split()
        vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *cls.words]
        vocab = cls.root / "vocab.txt"
        vocab.write_text("\n".join(vocabulary) + "\n")
        tokenizer = DistilBertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)
        tokenizer.save_pretrained(cls.model_dir)
        config = DistilBertConfig(vocab_size=len(vocabulary), n_layers=1, n_heads=2,
                                 dim=16, hidden_dim=32, num_labels=2,
                                 max_position_embeddings=512)
        model = DistilBertForSequenceClassification(config)
        with torch.no_grad():
            model.classifier.bias[:] = torch.tensor([-2.0, 2.0])
        model.save_pretrained(cls.model_dir, safe_serialization=True)
        cls.boundary_model_dir = cls.root / "boundary_model"
        tokenizer.save_pretrained(cls.boundary_model_dir)
        with torch.no_grad():
            model.classifier.bias[:] = torch.tensor([2.0, -2.0])
        model.save_pretrained(cls.boundary_model_dir, safe_serialization=True)

    def run_cli(self, root, text, model_dir=None):
        inputs = root / "prompts.txt"
        inputs.write_text(text + "\n")
        output = root / "decisions.jsonl"
        environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                           PYTHONDONTWRITEBYTECODE="1", USE_TF="0",
                           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        completed = subprocess.run([sys.executable, "-B", str(ROOT / "run_inference.py"),
                                    "--model-dir", str(model_dir or self.model_dir), "--input", str(inputs),
                                    "--output", str(output)], env=environment,
                                   capture_output=True, text=True, timeout=120)
        return completed, output

    def test_offline_local_safetensors_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            completed, output = self.run_cli(Path(directory), " ".join(self.words))
            self.assertEqual(completed.returncode, 0, completed.stderr)
            results = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(results), 1)
            self.assertFalse(results[0]["harmful"])
            self.assertEqual(results[0]["gradient_pass_count"], 1)
            self.assertLessEqual(results[0]["classifier_query_count"], 384)
            self.assertFalse(any(key in results[0] for key in ("prompt", "text", "token_ids")))

    def test_classifier_context_boundary_includes_special_tokens(self):
        for words, accepted in ((510, True), (511, False)):
            with self.subTest(tokens_including_special=words + 2):
                with tempfile.TemporaryDirectory() as directory:
                    completed, output = self.run_cli(Path(directory), " ".join(["red"] * words),
                                                     self.boundary_model_dir)
                    if accepted:
                        self.assertEqual(completed.returncode, 0, completed.stderr)
                        self.assertEqual(len(output.read_text().splitlines()), 1)
                    else:
                        self.assertNotEqual(completed.returncode, 0)
                        self.assertIn("513 tokens including special tokens", completed.stderr)
                        self.assertIn("Inputs are not truncated", completed.stderr)
                        self.assertEqual(output.read_text(), "")


if __name__ == "__main__":
    unittest.main()
