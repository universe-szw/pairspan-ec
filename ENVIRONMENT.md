# Verification Environment

The paper's experiment environment is recorded in
`configs/paper_settings.json`. The release verification environment is separate:
its tests check the exported implementation and numeric recomputation, not a new
measurement of paper recovery or hardware latency.

## Pinned CPU Installation

Use Python 3.11 in a new virtual environment:

```text
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` on Windows or
`source .venv/bin/activate` on Linux/macOS, then run:

```text
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-inference-verified.txt
python -B -m unittest discover -s tests -v
python -B reproduce.py --output results
```

These are direct dependency pins, not a lockfile for all transitive dependencies.
The CPU wheel command targets Windows/Linux. Other platforms should select a
compatible PyTorch wheel. `REPRODUCTION_REPORT.json` records the environments
actually tested; configured GitHub Actions jobs are not evidence of a hosted run
until those jobs have executed successfully.

## Supported Scope

`requirements.txt` installs only NumPy for numerical recomputation.
`requirements-inference.txt` specifies broader inference compatibility ranges.
The pinned file provides a concrete verification target within those ranges.
Inference smoke tests build a tiny random local classifier and use harmless
synthetic inputs. They do not require the paper checkpoint, benchmark text,
network access, a GPU, or an API key.

The CLI does not truncate classifier inputs. It checks input length including
special tokens and rejects overlength requests. The 512-position limit in the
paper settings belongs to the saliency gradient pass, not classifier inference.
