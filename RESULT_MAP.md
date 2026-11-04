# Reported Results and Verification Levels

The main and supplementary content revisions are identified by PDF hashes in
`configs/paper_settings.json`. The files below contain no manuscript source,
publisher template, original input text, or model weights.

| Paper evidence | Public input and recomputation |
| --- | --- |
| Main Table 1, B384 policies, RandEC and GreedyEC-20 recovery/checks | `data/scores.jsonl.gz` -> `results/evaluation/operating_points.csv`, default point |
| Main Table 1, other rows and protocol-safe counts | `data/aggregate/fresh.csv`, `protocol.csv`, `no_saliency_csv.csv`; aggregate inspection |
| Main Table 2, latency/checks/gradient passes | `data/timing/*.csv` -> `results/timing_summaries.json`; all five repetitions retained |
| Main Table 3, ShareGPT-500 except no-saliency | Numeric records -> operating points, default point |
| Main Table 3, protocol-safe/XSTest-safe and no-saliency | `data/aggregate/benign_full.csv`, `no_saliency_csv.csv`; Wilson intervals can be calculated from counts |
| Main Table 4, cross-method calibration | Calibration subset -> `results/fit/`; numeric test records -> `results/evaluation/`; all thresholds and evaluation cells checked |
| Main Table 5, second/cross classifier | `data/aggregate/roberta.csv`, `roberta_random.csv`, `cross.csv`, `cross_deltas.csv`, `cross_per_length.csv`; aggregate inspection |
| Main Figure 2, common-cap budget results | `data/aggregate/fresh.csv`, `random.csv`; aggregate inspection |
| Main Figure 3, sensitivity | `data/aggregate/development_full.csv`, `development_full_deltas.csv`; aggregate inspection |
| Supplementary boundaries/negative findings | `data/aggregate/`; full source list in `provenance/aggregate_sources.json` |

The exact observed shared-FPR comparison is present in the recomputed
`point=0.05` rows: PairSpan-EC 29.03% recovery and GeometricPrefix 32.10%,
both with 31/500 positives (6.20%) on ShareGPT-500. It is an observed shared
operating point, not a fitted optimal ROC frontier.

Historical aggregate intervals are preserved as original evidence. Where the
paper reuses the canonical B384 interval, use the recomputed operating-point
file rather than the historical aggregate interval. Counts and point estimates
are unchanged. None of these operations reruns the original attack optimizer,
training, classification of restricted inputs, or downstream generation.
