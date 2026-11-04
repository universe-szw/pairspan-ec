# PairSpan-EC: WSDM 2027 Verification Package

Code and numeric evidence for **Finite-Budget Pair-Span Search for
Classifier-Based Prompt Screening**.

**Accepted at WSDM 2027, Main Conference; to appear.** Final publication
metadata and the paper link will be added when available.

This package corresponds to the paper's camera-ready content revision,
not the subsequent decision-layer system.

## Quick Start: Recompute Reported Results

Python 3.10 or newer and NumPy are sufficient. Once NumPy is installed,
reproduction is entirely offline: no GPU, API key, model download, original
repository, or private data is used.

```sh
python -m pip install -r requirements.txt
python reproduce.py --output results
python -B -m unittest discover -s tests -v
```

`results` must be new or empty. The command verifies the package checksums,
refits thresholds using **only** the 200 calibration requests, evaluates the
fixed 500-request benign set and 2,385 attack records, and recomputes the
10,000-replicate base-group bootstrap. It checks every generated evaluation
cell against the paper's frozen numerical reference, including all five
random orders, their within-request mean, paired differences, and per-length
counts. A mismatch is an error, not a warning.

The calibration set was filter-selected and used for threshold selection;
the separate ShareGPT-500 set was author-confirmed human-validated as benign.
Conditional attack recovery uses 2,380 original base-classifier bypasses.
The five original base positives are reported separately, never added to that
denominator. Retrospective score filtering does not save classifier queries.

## What Can Be Reproduced

| Component | Included verification |
| --- | --- |
| Primary default-rule results | Recomputed from per-request decisions, scores and costs |
| Cross-method calibration | Thresholds refitted; test operating points recomputed |
| Statistical intervals | Shared, seeded group bootstrap and benign Wilson intervals |
| Timing summaries | Recomputed from all five archived timing repetitions |
| Other published evaluations | Sanitized aggregate source tables for inspection; not new inference |
| PairSpan-EC and EC controls | Original function bodies and saliency implementation |

The included records are **measurements from the paper's original runs**, not
synthetic benchmarks. Recomputing their statistics is distinct from repeating
classifier inference. Original prompts, adversarial token strings, model
weights, training data, and attack-generation code are not distributed.
Consequently, this package does not enable an independent end-to-end rerun of
every experiment, nor does it reproduce hardware latency by running statistics.
Core implementation tests use explicitly synthetic, harmless fixtures.

## Run PairSpan-EC on Your Own Local Inputs

Optional dependencies are separate from the lightweight reproduction path:

```sh
python -m pip install -r requirements-inference.txt
python run_inference.py --model-dir /path/to/local/classifier --input prompts.txt --output decisions.jsonl
```

Use a local Hugging Face sequence-classification directory with safetensors
weights and tokenizer files. No downloads, API calls, remote model code, or
pickle model loading are enabled. Class 0 must mean harmful and class 1 benign.
The paper's default is DistilBERT, 512-token truncation, two insertion spans,
maximum span length 64, and a cap of 384 classifier evaluations. Saliency is
one separate gradient pass. Stage/prefix stopping is retained; this is not the
later chunk-16/early-accept decision system.

The paper's DistilBERT state-dict SHA-256 is recorded in
`configs/paper_settings.json` for identification, not as a bundled/downloadable
checkpoint. Your own checkpoint or inputs need not reproduce published rates.
The runner writes numeric decisions only, not your input or deletion text.

## Contents

- `src/`: core paper algorithms and numeric analysis.
- `data/`: de-identified numeric records and aggregate source tables.
- `reference/`: published numerical targets, separate from computation inputs.
- `configs/`: default paper settings.
- `provenance/`: code identity, export definitions, and source hashes.
- `tests/`: focused tests; no raw prompt fixtures or model downloads.
- `MANIFEST.sha256`: integrity checks for distributable files.

`REPRODUCTION_REPORT.json` records the tested scope and actual verification
outcome for this release. See `DATA_SCOPE.md` and `THIRD_PARTY_NOTICES.md` for
data boundaries and attribution, and `RESULT_MAP.md` for the table-by-table
verification scope. No full reproducibility or artifact badge is implied by
the existence of this package.
