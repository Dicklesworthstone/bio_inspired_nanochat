# Synaptic Serving Engine: Implemented Guard Semantics

This note describes the current implementation in
`bio_inspired_nanochat/synaptic_serving_engine.py`. It is not a certification report.

## Request controls

`ServingKnobs` currently exposes deliberation depth, an abstract ATP budget, a minimum
top-token probability, self-correction sharpening, and an opt-in adaptive-serving mode.
`SLARequirement` adds a latency limit and, for strict requests, another confidence floor.
Inputs, token ranges, context length, and numeric bounds are validated before generation.

The latency pre-check uses a fixed analytical estimate scaled by batch size and the growing
sequence length. Runtime is also checked against the configured limit after each model
forward. ATP is a bookkeeping unit derived from deliberation depth and the same request-work
factors; neither value is a calibrated hardware or energy measurement. A configurable batch
limit prevents a single request from carrying an unbounded number of rows.

The scheduler partitions queued requests into zero-deliberation and deliberative groups,
then executes the fast tier before the deliberative tier. It restores caller order only in
the returned response list; execution order is tiered. It does not perform a single batched
model forward or automatically reduce deliberation depth.

## Confidence guard

At each generated position, the engine computes the largest softmax probability for each
batch row and gates the whole request on the least-confident row. Falling below the active
threshold produces `CONFIDENCE_ABSTENTION`; meeting it records
`confidence_floor_met`. A zero-token request records that confidence was not evaluated,
and non-finite logits or probabilities abstain explicitly. These are heuristic decisions.

The repository does not currently provide a calibration dataset, nonconformity score,
quantile calculation, exchangeability check, or empirical coverage artifact for this
engine. Therefore the top-probability threshold is not a conformal predictor and must not
be described as a statistical certificate or coverage guarantee. A future conformal API
would need those missing elements plus tests that measure achieved coverage on held-out
data.

## State behavior

Serving temporarily places modules in evaluation mode and restores each module's prior
mode afterward. Ordinary requests pass `train_mode=False`, so they do not persist online
plasticity. `adaptive_serving=True` explicitly opts into model adaptation and should only
be used where cross-request state mutation is intended.
