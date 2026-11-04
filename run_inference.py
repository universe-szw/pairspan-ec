"""Run the paper policy on local inputs; never download models or record text."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.model_dir.is_dir():
        raise ValueError("--model-dir must be an existing local directory")
    if args.output.exists() or args.output.resolve() == args.input.resolve():
        raise ValueError("Output must be a new file, distinct from the input")
    os.environ.setdefault("USE_TF", "0")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    from pairspan import pairspan_ec_trace
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True,
                                              trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False,
        use_safetensors=True).to(args.device).eval()
    if model.config.num_labels != 2 or model.config.id2label != {0: "LABEL_0", 1: "LABEL_1"}:
        raise ValueError("Expected two labels: LABEL_0 harmful and LABEL_1 benign")
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer,
                          device=torch.device(args.device), truncation=True, max_length=512)
    with args.input.open(encoding="utf-8") as source, args.output.open("x", encoding="utf-8") as target:
        for index, line in enumerate(source):
            prompt = line.strip()
            if not prompt:
                continue
            trace = pairspan_ec_trace(prompt, classifier, tokenizer, model)
            if "saliency_error_type" in trace:
                raise RuntimeError(f'Saliency failed: {trace["saliency_error_type"]}')
            result = {key: trace.get(key) for key in (
                "harmful", "base_harmful", "base_prob", "ec_score",
                "classifier_query_count", "gradient_pass_count", "stage_query_counts")}
            result["row_id"] = index
            target.write(json.dumps(result, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
