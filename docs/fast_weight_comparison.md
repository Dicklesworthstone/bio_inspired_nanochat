# Fast-Weight-Programmer Framing (bead sax.5)

This note describes the intended architectural relationship between this project's
synaptic fast weights and other memory mechanisms. It is not a benchmark report.

## Implementation mapping

| Mechanism | State updated during execution | Implemented project behavior |
|:---|:---|:---|
| Conventional transformer attention | KV cache | Reuses prior keys and values during incremental decoding. |
| Synaptic fast weights | `SynapticLinear.w_fast` and eligibility buffers | Applies optional local Hebbian updates with configured decay and norm bounds. |
| Postsynaptic consolidation | CaMKII/PP1/BDNF state and slow weights | Gates transfer between fast and slow state according to configured kinetics. |
| Presynaptic dynamics | Calcium, vesicle, release, and energy state | Modulates attention using state carried across token steps. |

These mechanisms add mutable state and bookkeeping. Whether they improve retrieval,
adaptation, stability, or quality is an empirical question; their presence in the code is
not evidence of an advantage.

## Evidence boundary

The repository's unit tests exercise implementation properties such as tensor shapes,
state updates, reset behavior, norm bounds, and checkpoint round trips. They do not
establish comparative retrieval accuracy, superiority to DeltaNet or Hopfield models,
long-context retention across thousands of tokens, a zero-divergence guarantee, or a
throughput delta against another architecture.

Any future comparative result should name the executable harness, exact revision,
dataset, model configurations, seeds, hardware, raw artifact, and statistical method.
Until such an artifact exists, proposed benefits are hypotheses rather than findings.
