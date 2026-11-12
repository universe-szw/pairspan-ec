# Attribution and License Scope

The underlying erase-and-check code derives from **Certifying LLM Safety
against Adversarial Prompting**, Aounon Kumar et al., arXiv:2309.02705.
Upstream project: https://github.com/aounon/certified-llm-safety
and paper: https://arxiv.org/abs/2309.02705.
The upstream MIT notice is retained without removing its copyright holder in
`LICENSE`. PairSpan-EC extensions and release-specific analysis utilities are
distributed under the same MIT terms by the PairSpan-EC contributors.

PyTorch, NumPy, Transformers and safetensors are dependencies installed
separately under their respective licenses. No dependency source or model
weights are vendored. Public benchmark names identify evaluation sources;
this is not a redistribution or relicensing of their source datasets.

The numerical measurement records and aggregate tables in this package were
produced by the paper's experiments and are supplied under the MIT terms.
The source conversations, benchmark requests, adversarial strings, human
annotation notes, model parameters, paper PDF and publisher template are not
included in this release and are not licensed by this notice.
