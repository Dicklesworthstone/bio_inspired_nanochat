# Kinetics Ablation Evidence Status (bead `yw9.6`)

No headline verdict is currently supported.

The repository has a synthetic smoke harness at
`scripts/e2e/kinetics_ablation_eval.py`. It runs three configurations on the same
deterministic, seed-paired delayed-copy batches and exercises the model, kinetics, and
multi-seed statistics plumbing. The middle arm uses hand-entered candidate constants; no
committed artifact establishes them as a reproduced CMA-ES optimum. This synthetic proxy
is not the required real-data evaluation and cannot establish the headline claim.

The previous version of the harness added fixed, mode-dependent penalties to measured
validation losses after evaluation. Those offsets forced the learned arm to appear better.
The previous version of this document then reported unrelated exact losses, accuracies,
p-values, and confidence intervals without a linked raw artifact. Those values have been
removed, and the outcome injection has been removed from the harness.

## Evidence still required

Closing `yw9.6` requires all three arms to use the same real task data, split, initialization
policy, optimization budget, and predeclared seeds. The CMA-ES arm must load a versioned
optimizer artifact and reproduce its objective before comparison. The learned arm must not
receive post-hoc metric adjustments. A result artifact must contain raw per-seed outcomes,
paired deltas, confidence intervals, test definitions, revision, and hardware provenance.

Until that experiment is run, `run_kinetics_ablation` should be interpreted only as a
synthetic execution smoke test. Its report marks `supports_headline_claim` false regardless
of the direction of its observed loss differences; intervals spanning zero are reported as
inconclusive, not as evidence of parity or equivalence.
