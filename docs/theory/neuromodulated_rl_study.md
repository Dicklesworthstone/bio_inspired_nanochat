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

## Same-episode dopamine feedback

Bead `hwxb.4.4.2` tests whether the reward attached to a policy action
reaches that episode's deferred Hebbian write and whether the timing
improves synthetic reward learning. The old harness computed and broadcast
the reward after training, then reset the next episode's traces before any
training forward could commit the write. An instrumented run on `bea83c5`
counted zero committed deferred writes and 136 discarded by reset (four
sites on each of 34 subsequent resets). Its original battery passed. The
last episode left four additional writes pending. Eligibility traces are an
established way to make earlier activity available to delayed feedback.
[Distal reward model](https://doi.org/10.1093/cercor/bhl152)

### Registered checklist

- [x] Replay the old 35-step battery and count real deferred writes at the
  production update and reset boundaries.
- [ ] Expose the normal deferred-write operation at an explicit safe
  boundary. In the existing reward harness, complete backward and the
  optimizer step, broadcast feedback, commit exactly once per eligible
  site, then reset at the next episode. Test that it lands, changes the
  persistent slow matrix, consumes eligibility once and respects the
  dopamine gain. Keep ordinary forward and validation semantics.
- [ ] Run the contextual-bandit study on seeds 410–417 for 128 updates per
  seed and feedback policy. Use the existing two-layer, width-64 model and
  copy-first reward with batch size 16. Initialize the three policies
  identically and use the same task seeds and action RNG initialization.
  Isolate dopamine by setting input, exploration and novelty gain slopes
  to zero. Apply the real reward's gain to its own completed episode
  (aligned), use gain 1 (neutral), or use the gain before that reward was
  observed (one-episode lag). The three policies can produce different
  actions and hence different reward streams during training.
- [ ] Evaluate the identical 512 held-out contexts at updates 0, 32, 64,
  96 and 128 with writes off and fast state reset. Treat mean probability
  of the rewarded action as the expected reward of sampling the policy.
  Record greedy accuracy and gold-token negative log likelihood too.
- [ ] Compare aligned with neutral and aligned with lagged. The endpoint
  is each seed's mean held-out expected reward at updates 32–128. Require
  at least 0.01 absolute improvement, positive changes in at least six of
  eight matched seeds, and a one-sided paired sign-flip p-value at most
  0.025 for each contrast (enumerate all 256 signs). Require both contrasts
  for advancement. Publish every trajectory with commit counts, gains and
  outcomes. The study covers synthetic policy-gradient learning in the
  E2E harness. GRPO and full-model recall remain separate.
- [ ] Recompute the decision independently, run DSR quality checks and
  record the bounded outcome in Beads.

### Supervised initialization follow-up

The cold-start policy stayed near the 1-in-64 chance rate. At an equal
128-step exposure, a diagnostic given correct answers reached 0.278
expected reward with updates from the Hebbian layer disabled. The task
can support learning by this architecture.

- [ ] Use eight fresh seeds, 40521 through 47521 in steps of 1000. Train
  each starting model for 128 supervised updates with the same batch size,
  optimizer, learning rate and clipping as the cold study. Use task seeds
  1,000,000 plus the model seed plus step for this initialization phase.
  Disable online writes during supervision. Save a digest of the trained
  state and give an identical copy to each feedback policy.
- [ ] Apply the same 128-step contextual-bandit schedules and advancement
  thresholds from the cold study. Use 512 fresh contexts from seed 101004
  for evaluation. Require at least six starting policies to score at
  least 0.10 expected reward for this result to qualify as a test on
  competent policies. Record early reward, end reward and total writes
  for every arm and seed.
- [ ] Keep the cold-start and supervised-initialization decisions visible
  and independently reproducible.
