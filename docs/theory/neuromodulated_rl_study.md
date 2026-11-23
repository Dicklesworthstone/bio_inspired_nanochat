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
- [x] Expose the normal deferred-write operation at an explicit safe
  boundary. In the existing reward harness, complete backward and the
  optimizer step, broadcast feedback, commit exactly once per eligible
  site, then reset at the next episode. Test that it lands, changes the
  persistent slow matrix, consumes eligibility once and respects the
  dopamine gain. Keep ordinary forward and validation semantics.
- [x] Run the contextual-bandit study on seeds 410–417 for 128 updates per
  seed and feedback policy. Use the existing two-layer, width-64 model and
  copy-first reward with batch size 16. Initialize the three policies
  identically and use the same task seeds and action RNG initialization.
  Isolate dopamine by setting input, exploration and novelty gain slopes
  to zero. Apply the real reward's gain to its own completed episode
  (aligned), use gain 1 (neutral), or use the gain before that reward was
  observed (one-episode lag). The three policies can produce different
  actions and hence different reward streams during training.
- [x] Evaluate the identical 512 held-out contexts at updates 0, 32, 64,
  96 and 128 with writes off and fast state reset. Treat mean probability
  of the rewarded action as the expected reward of sampling the policy.
  Record greedy accuracy and gold-token negative log likelihood too.
- [x] Compare aligned with neutral and aligned with lagged. The endpoint
  is each seed's mean held-out expected reward at updates 32–128. Require
  at least 0.01 absolute improvement, positive changes in at least six of
  eight matched seeds, and a one-sided paired sign-flip p-value at most
  0.025 for each contrast (enumerate all 256 signs). Require both contrasts
  for advancement. Publish every trajectory with commit counts, gains and
  outcomes. The study covers synthetic policy-gradient learning in the
  E2E harness. GRPO and full-model recall remain separate.
- [x] Recompute the decision independently, run DSR quality checks and
  record the bounded outcome in Beads.

### Supervised initialization follow-up

The cold-start policy stayed near the 1-in-64 chance rate. At an equal
128-step exposure, a diagnostic given correct answers reached 0.278
expected reward with updates from the Hebbian layer disabled. The task
can support learning by this architecture.

- [x] Use eight fresh seeds, 40521 through 47521 in steps of 1000. Train
  each starting model for 128 supervised updates with the same batch size,
  optimizer, learning rate and clipping as the cold study. Use task seeds
  1,000,000 plus the model seed plus step for this initialization phase.
  Disable online writes during supervision. Save a digest of the trained
  state and give an identical copy to each feedback policy.
- [x] Apply the same 128-step contextual-bandit schedules and advancement
  thresholds from the cold study. Use 512 fresh contexts from seed 101004
  for evaluation. Require at least six starting policies to score at
  least 0.10 expected reward for this result to qualify as a test on
  competent policies. Record early reward, end reward and total writes
  for every arm and seed.
- [x] Keep the cold-start and supervised-initialization decisions visible
  and independently reproducible.

### Observed timing and reward

The corrected battery committed **140 of 140** deferred writes in the
same episode as their reward. It finished with zero discarded and zero
pending writes. The original battery committed zero, discarded 136 and
finished with four pending. In a regression test, changing only the
dopamine gain changed the norm of each persistent matrix update by the
registered factor. A second commit left the matrices exactly equal.

All 48 policy runs completed. An independent implementation reconstructed
6,144 feedback gains and 24,576 commit counts from both sets of traces.
All eight supervised starts exceeded the 0.10 competence floor.

| Starting policy | Comparison with same-episode feedback | Mean expected reward gain, percentage points | Positive seeds | One-sided paired p | Passed |
| --- | --- | ---: | ---: | ---: | :---: |
| Cold | Neutral | +0.001949 | 6/8 | 0.144531 | No |
| Cold | Previous episode | +0.000530 | 6/8 | 0.027344 | No |
| Supervised | Neutral | +0.358337 | 7/8 | 0.019531 | No |
| Supervised | Previous episode | +0.246096 | 6/8 | 0.039063 | No |

The minimum effect was **1 percentage point**. The cold-start policy's
held-out expected reward averaged 1.56% before training and 1.54% after
aligned training. In the supervised follow-up, aligned feedback moved
from 34.58% to 60.69%. Neutral and lagged feedback finished at 60.10% and
60.21%. Those endpoints are descriptive. The registered comparisons
average four checkpoints and both studies declined advancement. The
supervised follow-up showed that competent policies improved under all
three schedules and the timing advantage fell below the registered
threshold across these seeds. It measures the timing of the
dopamine multiplier on the Hebbian matrix writes in this copy-first
task. Larger models, delayed rewards across multiple episodes, and GRPO
remain open. [Full numerical results and DSR receipts](../../results/delayed_reward_2026-11-23.json) · [Training and holdout traces](../../results/delayed_reward_2026-11-23.json.gz)

### Replay

The synthetic CLI accepts `--timing-study --run-dir "$NEW_DIRECTORY"` in
the normal DSR check running `uv run --no-sync python -m scripts.e2e.neuromod_rl`.
The summary JSON contains the supervised-initialization driver and the
independent verifier. To replay the archived curves, decompress the trace
archive, extract `cold` and `supervised` into distinct JSON files, and
pass the corresponding file and a fresh result path to `verify_reward.py`
in a DSR quality check. Append the positional argument `warm` for
the supervised trace.

### Qualification

An unchanged 12-step ordinary training replay matched every output,
gradient and state hash. Evaluation interleaved with training preserved
model state. Both timing studies reproduced byte for byte from an
isolated checkout after the CUDA reward-device repair. The final Linux
CPU suite passed 2,082 tests with 11 skips. The native CUDA battery and
the compiled axiom checks for all 19 mapped Lean theorems passed.
Ruff and ty passed on the locked Linux environment. The receipts retain
the stale-hash rejection, the isolated RAM scratch quota failure, and
the separate throughput reruns. [Executable evidence](../../results/delayed_reward_2026-11-23.json)

## Attribution within a group of rollouts

The chat trainer uses centered rewards and a token-normalized on-policy
objective. Each group consists of completions for one question. The
original GRPO formulation includes additional normalization and a
reference policy. [DeepSeekMath](https://arxiv.org/abs/2402.03300)

For rollout `i`, form the two ordinary co-activity trace factors at input
positions that consumed tokens sampled by the policy. Keep the final
sampled token in the forward. The separate loss mask scores each action
at the preceding prediction position and ignores the added terminal
position. Prediction states before the first action are the same across
a question group, so their identical eligibility would cancel under
centered reward. Compute the unit Frobenius direction `Dᵢ` before
any results are combined. For each slow matrix, add
`α Σᵢ (rᵢ − mean_group(r)) Dᵢ / N` after the optimizer step, with `N`
the number of rollouts processed across all groups and ranks at that
step. Trace directions are computed on the same forward that produced
the policy gradient. The normalized product of pooled factors contains
cross-rollout terms. Computing each direction before the reduction removes
those terms and allows a zero-advantage group to produce exactly zero
reward-driven update.

### Registered work

- [ ] Stage an opt-in chat trainer path with signed group-centered credit.
  Match the gradient path in the `off`, `aligned` and `shuffled` modes
  by disabling ordinary online updates in these three modes. Keep the
  existing default. Shuffle advantages within each full question group
  before slicing microbatches. Apply one finite, globally normalized update
  after the optimizer. Preserve the action mask for padding, prompts and
  forced tool tokens.
- [ ] Check individually computed trace products, cancellation, zero
  advantages, permutation and microbatch remainder, post-backward mutation,
  and two-rank reduction. Verify that large CUDA batches retain all
  policy log probabilities. Use the actual GSM8K runner with archived
  official rows. Explicitly record if the starting checkpoint produces
  only one reward value in each group.
- [ ] On the synthetic copy-first task, initialize eight fresh models
  (seeds 70521–77521, spaced by 1000) with 128 supervised updates,
  batch size 16, on task seeds `2000000 + model_seed + update`. Use
  the existing depth-2, width-64 model and AdamW at `0.002`, clip 1,
  weight decay `0.0001`. Clone each trained state across all three
  policy arms. Train for 96 policy updates with four questions and four
  sampled one-token rollouts each. Use the fresh question stream
  `3000000 + model_seed + update`, a common seeded sampler, and
  `α=0.03` for the aligned and shuffled arms. Turn off stochastic release
  in this comparison so that sampler and training forwards share the
  same biological setting.
- [ ] Evaluate expected reward on the same 512 fresh contexts from seed
  102004 at updates 0, 32, 64 and 96. Require at least six starting
  checkpoints above 0.10 and nonconstant reward groups in every seed.
  Test aligned versus off and aligned versus shuffled on the mean of
  the three post-baseline checkpoints. Require at least one percentage
  point in the mean, positive changes in at least six of eight seeds,
  and paired one-sided exact sign-flip `p ≤ 0.025` in both contrasts.
  Archive missing, null and failed runs. The controlled bandit and
  official GSM8K runner smoke have distinct scopes.
- [ ] Independently verify the comparisons and publish the executed
  source, test and DSR evidence. Keep full GSM8K effectiveness open
  unless the staged starting policy earns diverse rewards in a
  matched evaluation at the appropriate scale.
