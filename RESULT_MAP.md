# Paper-to-Result Map

The package contains numerical records for the main paper and supplementary
material. Metric definitions and units are in [data/README.md](data/README.md).

## Recomputed Results

| Paper evidence | Input and output |
| --- | --- |
| Main Tables 1 and 3, B384 defaults and reference-schedule RandEC/GreedyEC-20 on HarmBench and ShareGPT-500 | `data/scores.jsonl.gz` -> `results/evaluation/operating_points.csv`, default point |
| Main Table 2, latency, checks, and gradient passes | `data/timing/*.csv` -> `results/timing_summaries.json`; all five repetitions |
| Main Table 4 and supplementary calibration analysis | `data/calibration.jsonl.gz` -> `results/fit/thresholds.json`; numeric test records -> `results/evaluation/` |
| Supplementary calibrated per-length results and paired contrasts | `results/evaluation/per_length_calibrated.csv` and `paired_cluster_deltas.csv` |

`reference/operating_points.csv` is the publication reference for repeated
B384 default and calibrated intervals. It is regenerated from the numeric
records. In the `point=0.05` rows, PairSpan-EC recovers 29.03% and
GeometricPrefix 32.10%, both at 31/500 ShareGPT positives (6.20%). This is an
observed shared operating point, not a fitted optimal ROC frontier.

## Aggregate Tables

These tables cover the additional experiments and supporting statistics.
All 25 aggregate CSVs are listed below. Their counts and point estimates are
the recorded results; historical intervals for repeated B384 endpoints defer
to the publication reference above. Aggregate-only intervals cannot all be
reconstructed from the released per-request records.

| Paper evidence or supporting statistic | Aggregate files |
| --- | --- |
| Main Table 1 and Figure 2, default recovery and budget curves | [fresh.csv](data/aggregate/fresh.csv), [random.csv](data/aggregate/random.csv) |
| HarmBench paired method differences and per-length bypass denominators | [fresh_deltas.csv](data/aggregate/fresh_deltas.csv), [fresh_denominators.csv](data/aggregate/fresh_denominators.csv) |
| Main length analysis and supplementary per-length curves | [fresh_per_length.csv](data/aggregate/fresh_per_length.csv) |
| Main Tables 1 and 3, protocol-safe evaluation and randomized controls | [protocol.csv](data/aggregate/protocol.csv), [protocol_random.csv](data/aggregate/protocol_random.csv) |
| Main Table 3 and supplementary benign-set matrix | [benign_full.csv](data/aggregate/benign_full.csv) |
| Additional benign cost statistics: P95 checks, gradient passes, and raw run times | [benign.csv](data/aggregate/benign.csv); recorded run times are not the isolated timing comparison |
| Held-out no-saliency results | [no_saliency_csv.csv](data/aggregate/no_saliency_csv.csv) |
| Main Table 5(a), second-classifier evaluation | [roberta.csv](data/aggregate/roberta.csv), [roberta_random.csv](data/aggregate/roberta_random.csv) |
| Main Table 5(b), cross-classifier evaluation and supporting breakdowns | [cross.csv](data/aggregate/cross.csv), [cross_deltas.csv](data/aggregate/cross_deltas.csv), [cross_per_length.csv](data/aggregate/cross_per_length.csv) |
| Main Figure 3 and supplementary development ablations | [development_full.csv](data/aggregate/development_full.csv), [development_full_deltas.csv](data/aggregate/development_full_deltas.csv) |
| Mode matrix and all-configuration fallback | [mismatch.csv](data/aggregate/mismatch.csv), [fallback_harmful.csv](data/aggregate/fallback_harmful.csv), [fallback_safe.csv](data/aggregate/fallback_safe.csv) |
| Fixed L1/L2 development stress sets | [adaptive.csv](data/aggregate/adaptive.csv) |
| Main JBC-manual result and supplementary JailbreakBench table | [jailbreakbench.csv](data/aggregate/jailbreakbench.csv); all seven subsets and three methods |
| Two-hit confirmation and first-positive stages | [confirmation_fpr.csv](data/aggregate/confirmation_fpr.csv), [confirmation_recovery.csv](data/aggregate/confirmation_recovery.csv), [benign_stage.csv](data/aggregate/benign_stage.csv) |

Numerical recomputation does not rerun attack generation, classifier training,
inference on the original inputs, or downstream generation.
