# Release Candidate 1.0.0-rc.1

This version includes the PairSpan-EC search policy, EC controls, numerical
recomputation scripts, and de-identified measurements for the WSDM 2027 paper.

## Highlights

- Classifier inference follows the paper's no-truncation protocol. The CLI
  rejects inputs exceeding the model context, including special tokens.
- The boolean API reports saliency failures explicitly. Canonical trace/search
  functions and numerical references are unchanged;
  `provenance/code_identity.json` records the API wrapper change.
- CPU tests cover query caps, stage stopping, saliency, failure reporting, and
  input-length boundaries. CI also checks numerical recomputation.
- Pinned verification dependencies, harmless example inputs, and
  machine-readable citation metadata are included.

See [ENVIRONMENT.md](ENVIRONMENT.md) for installation and testing,
[DATA_SCOPE.md](DATA_SCOPE.md) for measurement coverage, and
[RESULT_MAP.md](RESULT_MAP.md) for paper-to-result mappings.
