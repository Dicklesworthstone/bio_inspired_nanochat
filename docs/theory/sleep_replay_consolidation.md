# Sleep Replay and Consolidation: Implemented Semantics (bead `cel.1`)

This module borrows sleep/replay terminology as an engineering analogy. It does not establish
biological equivalence or show that the model solves catastrophic forgetting.

## Current execution path

`PrioritizedReplayBuffer` stores bounded CPU snapshots and samples them by surprise-weighted
probability. Weight normalization is performed in log space so extreme finite priorities do
not overflow. `WakeSleepScheduler.step_training` is a post-backward/post-optimizer hook: before
replay, the controller lands one deferred wake-plasticity write and clears its eligibility
traces so the same update is not applied twice.

Each replay forward then uses the model's native adaptive synaptic update. The controller
measures the resulting slow-state delta, retains it only when the configured CaMKII threshold
is met, and scales that native delta by `consolidation_lr`. It does not add a second copy of
the whole fast-weight matrix. It also does not clear `w_fast`, because that tensor currently
contains trainable model state as well as adaptive state. The legacy helper offers an explicit
`reset_fast_after=True` experiment switch, but its safe default is false.

Optional homeostatic scaling caps slow-weight norms after replay. Synthetic "dream" sequences
are sampled from the current model state; they are not slow-weight-only, privacy-safe, or a
guarantee against memorized-data emission.

## Evidence boundary

The test suite verifies buffer bounds and snapshots, mode restoration, device and numeric
validation, single native slow-update scaling, pending-trace cleanup, and failure rollback in
the controller. It does not establish improved retention, resistance to forgetting, selective
preservation of useful memories, privacy, or equivalence to NREM/REM biology.

A credible continual-learning result still requires predeclared sequential tasks, matched
training and replay compute, no-replay and conventional-replay baselines, multiple seeds, raw
per-run artifacts, and retention/forward-transfer metrics with uncertainty.
