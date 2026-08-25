# Synaptic Granularity Comparison Status (bead `vap.2`)

This document describes implementation scales to compare. It is not a completed benchmark.

## Architectural granularity spectrum

| Level | Granularity mode | State scaling | Implemented path | Fidelity evidence | Benchmark status |
|:---|:---|:---|:---|:---|:---|
| **Fine (L1)** | Per-attention-edge state | Grows with batch, heads, queries, and cached keys | Presynaptic modulation in `SynapticAttention` | Not established | Not run |
| **Medium (L2)** | Per-neuron rank-$R$ traces | Low-rank buffers and projections | Eligibility state in `SynapticLinear` | Not established | Not run |
| **Coarse (L3)** | Per-expert scalar state | Grows with expert count | MoE metabolic routing state | Not established | Not run |

The labels “fine,” “medium,” and “coarse” describe state granularity, not measured
biological fidelity or quality.

## Evidence boundary

No raw artifact or executable apples-to-apples harness is linked for the exact VRAM,
bits-per-byte, or throughput numbers previously shown here. Those values and the resulting
production recommendation have been removed. Unit tests for individual mechanisms do not
establish the relative quality/cost frontier.

A valid comparison needs one configuration switch that changes only granularity, identical
model/data/training budgets, peak-memory measurement on named hardware, synchronized token
throughput timing, held-out quality metrics, multiple seeds, and archived raw results. Until
then, the best granularity and any selective-use recommendation remain open hypotheses.
