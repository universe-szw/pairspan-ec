# Release Candidate 1.0.0-rc.1

This is a locally prepared release candidate. No public release, hosted CI run,
archive DOI, or remote tag is implied by this file.

## Changes

- Match the paper's classifier inference protocol: no automatic truncation;
  reject inputs exceeding the model context.
- Make the boolean API report saliency failures instead of silently returning
  a decision from a reduced candidate family.
- Preserve the canonical trace/search functions, recorded measurements, and
  statistical references. Document the boolean wrapper change in
  `provenance/code_identity.json`.
- Add CPU CI, direct dependency pins, input-boundary and failure tests,
  citation metadata, and harmless example inputs.

## Before Publication

1. Run the commands in `ENVIRONMENT.md` on the final package. Regenerate
   `MANIFEST.sha256` after any tracked-file change; never update the numerical
   references merely to make a failing check pass.
2. Confirm the author list, the permitted data/code scope, and the intended
   GitHub repository contents. Do not include model weights, raw prompts, or
   internal project logs.
3. After an authorized push, inspect the hosted CI results and create a tag for
   the exact tested commit. Use `v1.0.0-rc.1` for this candidate; update the
   version here and in `CITATION.cff` before a final `v1.0.0` release.
4. Change repository visibility only with the owner's approval. Update the
   paper's availability statement only after the cited version is accessible.
   Add a DOI only if an archive actually assigns one.

The package supports numerical recomputation and core-policy use on compatible
user-provided classifiers. It does not provide a from-scratch reproduction of
every paper experiment. Exact paper-model inference additionally requires the
original fine-tuned checkpoint and input data, neither of which is distributed.
