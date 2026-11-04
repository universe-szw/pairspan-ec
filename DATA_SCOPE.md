# Data and Verification Scope

`data/scores.jsonl.gz` contains 33,935 measurements: 11 configurations on
200 calibration requests, 500 benign evaluation requests, and 2,385 attack
requests. Methods are Base, PairSpan-EC B384, GeometricPrefix B384, MCE B384,
five fixed RandomPairGrid B384 orders, RandEC reference schedule, and
GreedyEC-20 reference schedule. RandEC and GreedyEC are not relabeled as B384.

Only dataset-scoped numeric row IDs, insertion lengths, ranked base-group IDs,
boolean decisions, scalar scores, classifier counts and gradient counts are
exported. Raw text, token IDs, original prompt hashes, source paths, hosts,
conversation IDs and annotator identities are excluded. Ranked group IDs
preserve the original lexicographic ordering used by the canonical bootstrap;
they are not behavioral category labels. Rows remain aligned across methods.

`data/calibration.jsonl.gz` is exactly the calibration subset of that file.
No test records are accepted by the threshold-fitting entry point. Refitting
recovers the frozen thresholds; it is not a fresh validation exercise.

Aggregate CSVs preserve the paper's extra experiments, including negative
findings and all preset random orders. They are not per-request evidence from
which every reported interval can be independently reconstructed. Repeated
default B384 intervals use the canonical per-request analysis; historical
aggregate exports can retain their originally computed intervals.

The content revision is identified by SHA-256 hashes of the manuscript PDFs
in `configs/paper_settings.json`; PDFs themselves are not redistributed here.
Code integrity and agreement with recorded results do not establish label
ground truth or generalization beyond the reported experimental population.
