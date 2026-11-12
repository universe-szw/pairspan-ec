# Measurement Definitions

## Publication Results and Units

`reference/operating_points.csv` is the publication reference for B384 default
and calibrated endpoints. `reproduce.py` regenerates it from the numeric
measurements. Rates and interval endpoints in this reference file are
fractions in [0, 1]; multiply by 100 for percentages. A fractional
`positive_k` in a RandomPairGrid mean averages five orders rather than pooling
five independent datasets. The JSONL score files contain harmful probabilities
in [0, 1], binary decisions, and per-request costs.

The aggregate CSVs generally express rates and interval endpoints in percent,
including legacy fields named `conditional_tpr`, `fpr`, and `detection_rate`.
They preserve the original experiment summaries. Their historical bootstrap
intervals can differ slightly from the shared-draw publication intervals;
use the publication reference for repeated B384 endpoints. Other budgets and
aggregate-only experiments use the aggregate files mapped in `RESULT_MAP.md`.
The exception is `aggregate/no_saliency_csv.csv`: `conditional_recovery`,
`ci_low`, `ci_high`, `rate`, `wilson_low`, and `wilson_high` are fractions in
[0, 1]. Blank cells are not applicable, not zero measurements.

Checks count classified texts, including the original input, not model batches.
Gradient passes are separate. `raw_mean_seconds` and mode-grid `mean_latency`
are recorded run times, not the isolated timing comparison. For that comparison,
use `data/timing/*.csv` and `reference/timing_summaries.json`: each prompt's
median over five repetitions is computed before summarizing across prompts.

## Mode Matrix and Fallback

`aggregate/mismatch.csv` compares complete classifier--search configurations.
`source_mode` identifies the classifier used to generate the attacks;
`branch_mode` identifies the target configuration, which uses its own weights
and deletion schedule. The legacy columns mean:

| Column | Definition |
| --- | --- |
| `successful_attacks` | Source-classifier bypass count; the row's detection denominator. |
| `recovered_attacks` | Target detections among those bypasses, including detection of the original input without deletion. |
| `conditional_tpr`, `ci_low`, `ci_high` | Detection percentage and base-request clustered 95% interval on source-classifier bypasses. |
| `mean_queries`, `mean_checked_candidates`, `mean_gradient_passes`, `mean_latency` | Means over all generated inputs, including t=0: 1,120 for suffix, 1,120 for insertion, and 560 for infusion. |

On matching-mode cells, detection coincides with conditional recovery under
that configuration's decision rule. Off-diagonal differences do not isolate
deletion geometry because the classifier also changes.

`aggregate/fallback_harmful.csv` uses the same source-bypass detection
denominators and all-input cost populations. Its costs sum three independent
configuration traces, including three original-input checks; no single B384
cap applies. `fallback_safe.csv` reports positives and summed costs on the
120 protocol-safe prompts.

## Fixed Stress Sets and Confirmation

`aggregate/adaptive.csv` contains the fixed N=70 development stress sets.
L1 targets classifier-safe predictions for the input and fixed proxy erasures.
L2 reuses previously generated attacks; it was not regenerated against the
screening policies compared in this table. Recovery uses 350 L1 and 177 L2
pre-EC bypasses. `mean_queries_all` and `mean_gradient_passes` use all 350
inputs per level; `mean_queries_conditional` uses the level's bypasses.
These are development diagnostics, not a new held-out adaptive confirmation.

The confirmation tables count qualifying deletion evaluations, which may
include the same decoded text across stages. They do not count independent
or necessarily distinct evidence. Stage counts in `benign_stage.csv` use
the default rule.

## JailbreakBench Artifacts

`aggregate/jailbreakbench.csv` retains all seven evaluated prompt-artifact
subsets and all three methods. `n` is the full subset size and `detections`
counts harmful decisions, including those from the original classifier.
`detection_rate`, `wilson_ci_low`, and `wilson_ci_high` are percentages.
`mean_queries` and `max_queries` use the full subset.

PairSpan-EC uses B384; GreedyEC-20 uses its reference schedule. For the
JBC-manual / Vicuna-13B subset, the respective counts are 1/100 and 100/100,
with 384.00 and 4,701.34 mean checks. These are screening outcomes on prompt
artifacts, not conditional recovery, equal-budget comparisons, or downstream
generation attack-success rates. No source prompts or adversarial strings
are included.
