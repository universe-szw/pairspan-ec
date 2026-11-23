# Release 1.0.0

Code and numerical records for **Finite-Budget Pair-Span Search for
Classifier-Based Prompt Screening**, accepted for poster presentation at WSDM 2027.
The [paper](https://doi.org/10.1145/3862322.3863422) and its
[supplementary material](https://github.com/universe-szw/pairspan-ec/blob/main/supplementary.pdf)
are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Included

- PairSpan-EC and EC comparison policies, with local-classifier inference.
- De-identified scores, calibration data, aggregate results, and five timing
  repetitions, including the reported ablations and external-attack results.
- Offline recomputation of operating points, confidence intervals, paired
  comparisons, and timing summaries, checked against reference results.
- Method settings, metric definitions, paper-to-result mappings, and examples.
- Numerical and CPU inference tests, CI configuration, and citation metadata.

Original prompts, model weights, and attack-generation code are not included.

See [ENVIRONMENT.md](ENVIRONMENT.md) for installation and testing,
[DATA_SCOPE.md](DATA_SCOPE.md) for measurement coverage, and
[RESULT_MAP.md](RESULT_MAP.md) for the result files.
