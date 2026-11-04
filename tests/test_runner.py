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
    def test_offline_local_safetensors_runner(self):
        os.environ.setdefault("USE_TF", "0")
        import torch
        from transformers import DistilBertTokenizerFast, DistilBertConfig, DistilBertForSequenceClassification
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "model"
            model_dir.mkdir()
            words = "red green blue yellow white black silver gold pink gray cyan violet".split()
            vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]
            vocab = root / "vocab.txt"
            vocab.write_text("\n".join(vocabulary) + "\n")
            tokenizer = DistilBertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)
            tokenizer.save_pretrained(model_dir)
            config = DistilBertConfig(vocab_size=len(vocabulary), n_layers=1, n_heads=2,
                                     dim=16, hidden_dim=32, num_labels=2)
            model = DistilBertForSequenceClassification(config)
            with torch.no_grad():
                model.classifier.bias[:] = torch.tensor([-2.0, 2.0])
            model.save_pretrained(model_dir, safe_serialization=True)
            inputs = root / "prompts.txt"
            inputs.write_text(" ".join(words) + "\n")
            output = root / "decisions.jsonl"
            environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                               PYTHONDONTWRITEBYTECODE="1", USE_TF="0")
            completed = subprocess.run([sys.executable, "-B", str(ROOT / "run_inference.py"),
                                        "--model-dir", str(model_dir), "--input", str(inputs),
                                        "--output", str(output)], env=environment,
                                       capture_output=True, text=True, timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            results = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(results), 1)
            self.assertFalse(results[0]["harmful"])
            self.assertEqual(results[0]["gradient_pass_count"], 1)
            self.assertLessEqual(results[0]["classifier_query_count"], 384)
            self.assertFalse(any(key in results[0] for key in ("prompt", "text", "token_ids")))


if __name__ == "__main__":
    unittest.main()
