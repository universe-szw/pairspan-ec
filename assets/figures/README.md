# Budget-Recovery Figure

`budget_recovery.png` is the authors' budget-recovery artwork used in the
accepted WSDM 2027 paper, **Finite-Budget Pair-Span Search for Classifier-Based
Prompt Screening**. It is copied without changing curves, intervals, colors,
or layout from the paper's content revision.

The figure compares default-rule conditional recovery at maximum classifier
evaluation budgets 128, 256, 384, and 768. The denominator is 2,380 original
base-classifier bypass attacks, with 159 base-request clusters. RandomPairGrid
uses the mean of five fixed orders. Error bars are clustered 95% bootstrap
intervals; the vertical axis is restricted to 40-100%.

Numerical sources are `data/aggregate/fresh.csv` and
`data/aggregate/random.csv`. The default B384 canonical intervals are also
available in `reference/operating_points.csv`.

The figure contains aggregate values, not source prompts or token sequences.
It is author-created artwork accompanying an accepted paper, not a publisher
PDF or a separate claim of formal publication. Its file hash is included in
the root `MANIFEST.sha256`.
