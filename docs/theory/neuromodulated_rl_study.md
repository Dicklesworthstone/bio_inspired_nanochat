# Neuromodulated RL Integration (bead `hy8.3`) — what is actually measured

_Neuromodulation & Homeostatic Control (`hy8`). Rewritten 2026-09-01._

> The previous version of this note reported a 1.72× sample-efficiency gain (18.2 ± 2.1 vs
> 31.4 ± 3.8 steps to reward ≥ 0.85, p = 0.0012) and a final-reward advantage with
> p = 0.0028 for neuromodulated three-factor RL over vanilla GRPO. Those numbers appear in no
> script, test, or results artifact in this repository, and no vanilla-GRPO comparison arm
> exists in the code. They have been removed and bead `hy8.3` was reopened.

## What exists

- `bio_inspired_nanochat/neuromod.py` — the `NeuromodulatoryBus`: dopamine (DA) from loss
  improvement / reward-prediction error, acetylcholine (ACh) from predictive entropy,
  norepinephrine (NE) from loss surprise; EMA-smoothed and broadcast as multiplicative gains onto
  Hebbian consolidation (DA), the stochastic-release fraction and input gain (ACh), and the
  synaptic output gain plus working-memory flush (NE). Default-neutral when
  `neuromod_enabled=0`.
- `scripts/e2e/neuromod_rl.py` — a 2-layer, 64-dim, vocab-64 CPU micro-run with 35
  synthetic-reward RL steps that turns the bus on and writes the per-step DA/ACh/NE levels and
  gains to an events log.
- `tests/test_e2e_neuromod_rl.py` — asserts the invariants that run establishes: DA > 0 and
  ACh > 0 after rewarded steps, plasticity gain > 1 under positive DA, at least one broadcast,
  and that the events trace exists. It measures no sample efficiency and has no baseline arm.

## What is not known

Whether dopamine-gated (three-factor) plasticity improves sample efficiency or stability over
plain GRPO on any task. Answering that needs a baseline arm with the bus off at matched
compute, at least three seeds, a fixed reward threshold, and paired statistics via
`bio_inspired_nanochat/eval_stats.py`, at a scale where GRPO learns anything (the current
micro-run does not). That is the open acceptance criterion of `hy8.3`; it runs after the GPU
baseline (`hwxb.3`) exists.

## Routing architecture

```text
                  ┌─────────────────────────────────┐
                  │       REWARD & LOSS STREAM      │
                  │  R_t, Loss_t, Entropy H(P)      │
                  └────────────────┬────────────────┘
                                   │
                                   ▼
                  ┌─────────────────────────────────┐
                  │     Neuromodulatory Bus         │
                  │  • DA  = EMA_RPE(R_t - V_t)     │
                  │  • ACh = Uncertainty(H(P))      │
                  │  • NE  = Novelty(Surprise)      │
                  └────┬───────────┼───────────┬────┘
                       │           │           │
           ┌───────────┘           │           └───────────┐
           ▼                       ▼                       ▼
   ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
   │ SynapticLin  │        │ SynapticPres │        │  Fast-Weight │
   │ Plasticity   │        │ Release Var  │        │ Consolidation│
   │ Gain (DA)    │        │ Gain (ACh)   │        │ Gate (DA*NE) │
   └──────────────┘        └──────────────┘        └──────────────┘
```

## Integration guidance

Keep `neuromod_enabled` off by default. Enable it for RL fine-tuning experiments only
alongside the baseline arm described above, and log the per-step levels the bus already
exposes so the comparison is auditable.
