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

- [x] Stage an opt-in chat trainer path with signed group-centered credit.
  Match the gradient path in the `off`, `aligned` and `shuffled` modes
  by disabling ordinary online updates in these three modes. Keep the
  existing default. Shuffle advantages within each full question group
  before slicing microbatches. Apply one finite, globally normalized update
  after the optimizer. Preserve the action mask for padding, prompts and
  forced tool tokens.
- [x] Check individually computed trace products, cancellation, zero
  advantages, permutation and microbatch remainder, post-backward mutation,
  and two-rank reduction. Verify that large CUDA batches retain all
  policy log probabilities. Use the actual GSM8K runner with archived
  official rows. Explicitly record if the starting checkpoint produces
  only one reward value in each group.
- [x] On the synthetic copy-first task, initialize eight fresh models
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
- [x] Evaluate expected reward on the same 512 fresh contexts from seed
  102004 at updates 0, 32, 64 and 96. Require at least six starting
  checkpoints above 0.10 and nonconstant reward groups in every seed.
  Test aligned versus off and aligned versus shuffled on the mean of
  the three post-baseline checkpoints. Require at least one percentage
  point in the mean, positive changes in at least six of eight seeds,
  and paired one-sided exact sign-flip `p ≤ 0.025` in both contrasts.
  Archive missing, null and failed runs. The controlled bandit and
  official GSM8K runner smoke have distinct scopes.
- [x] Independently verify the comparisons and publish the executed
  source, test and DSR evidence. Keep full GSM8K effectiveness open
  unless the staged starting policy earns diverse rewards in a
  matched evaluation at the appropriate scale.

### Registered local direction diagnostic

Measure whether the fixed 0.03 rule points toward better reward before
either arm takes an optimizer step. Use the eight supervised starts
registered above. Freeze each policy, and draw four completions for
each of 128 new questions from seed `6000000 + model_seed`. Use a
separate sampler seeded with `7000000 + model_seed` and a within-group
shuffle seeded with `8000000 + 128 * model_seed + group_index`. Use the
same questions and actions for both writes. Keep both directions in
their native layerwise scale and make one write with the usual 512
rollout denominator.

Use 512 further questions from seed 104552. Measure the dot product
between each jointly applied slow-matrix perturbation and the gradient
of exact expected reward on these questions. Cross-check by a central
difference of the same objective at plus and minus one write. Match
each nonzero aligned layer's write norm in an ascent-direction positive
control. Report every seed and retain the original trial outcome.
The diagnostic favors action-specific credit only if the mean aligned
derivative is positive and exceeds the shuffled derivative in at least
six seeds with a one-sided exact sign-flip `p ≤ 0.025`.

### Projection sign symmetry

Let `C = XᵀY` be a rollout's masked co-activity matrix. The raw
production direction is proportional to `C Pout Pinᵀ C`. Flipping
`Pout` flips the whole write and leaves its norm unchanged. With
online writes disabled, the frozen forward uses neither projection,
so the sampled actions, rewards, and objective gradient are identical
under the flip. The first conditional mean update and its directional
derivative are zero under a sign-symmetric projection draw independent
of the frozen policy. This argument applies to any fixed sequence of
masked activations and to the sum across a rollout group. Subsequent
adaptation can make the activations depend on the projection. Feedback
alignment also relies on the evolution of forward weights during
learning. [Lillicrap et al., 2016](https://www.nature.com/articles/ncomms13276)

### Registered projection-independent comparison

- [x] Use new supervised starts at seeds 80521 through 87521, spaced
  by 1000, and holdout seed 107112. Use the original initialization,
  rollout, optimizer and evaluation budgets. Match arms within seed
  and measure competence at initialization.
- [x] Compare four policies: off, projected aligned, direct aligned,
  and direct shuffled. The direct rule replaces the unit direction
  alone with `C / ‖C‖` for nonzero masked co-activity. It retains the
  0.03 step size, reward coefficients, action mask, and rollout
  denominator. For a single consumed token the projected rule uses
  the same outer product, multiplied by the sign of
  `yᵀ Pout Pinᵀ x`. Use the production collector with the direct
  direction substituted only in the experimental driver.
- [x] Evaluate the mean of checkpoints 32, 64 and 96. Require all
  three comparisons with direct aligned to have a mean gain of at
  least one percentage point, at least six positive seeds, and a
  one-sided exact sign-flip `p ≤ 0.0125`. Require at least six of
  eight starts above 0.10 and mixed reward groups in every seed.
  Publish the result even if the comparison declines advancement.

### Results

All 16 supervised starts qualified and all 56 training runs had mixed
reward groups. The original 24 runs reproduced every recorded value in
a second execution that retained each holdout row. The 32 runs of the
projection-independent comparison reproduced byte for byte. Independent scoring
reconstructed 114,688 holdout scores and 86,016 sampled action rewards
across the two comparisons. Gains are absolute percentage points of
expected reward, averaged over the three registered policy checkpoints.

| Treatment | Comparator | Gain (percentage points) | Positive seeds | One-sided exact p |
| --- | --- | ---: | ---: | ---: |
| Projected aligned | Off | −0.335194 | 3/8 | 0.824219 |
| Projected aligned | Projected shuffled | −0.538595 | 2/8 | 0.984375 |
| Direct aligned | Off | +0.025953 | 3/8 | 0.464844 |
| Direct aligned | Projected aligned | +0.094841 | 5/8 | 0.339844 |
| Direct aligned | Direct shuffled | +0.273553 | 5/8 | 0.136719 |

Both trials declined advancement. Removing the random sign left small
differences that fell short of the one-percentage-point threshold
against every control. The direct direction remains experimental.

The frozen-policy diagnostic measured a mean aligned derivative of
−5.20 × 10⁻⁷, compared with −5.80 × 10⁻⁸ for the shuffle and
+5.58 × 10⁻⁵ for the matched-norm ascent control. Aligned beat the
shuffle in three of eight seeds with p = 0.824219. The sign reflection
preserved all eight frozen holdout objectives exactly and reversed the
slow-weight writes, with maximum cancellation error 3.73 × 10⁻⁹.
Two-scale finite differences reproduced the analytic derivatives
within 1.04 × 10⁻⁷ absolute error. The archive retains the earlier
diagnostic attempts and the switch to a frozen evaluation forward.

The official GSM8K Parquet splits matched all 7,473 train and 1,319
test rows from OpenAI's original JSONL in order. Six sampled training
completions and four held-out completions all received zero reward
from the archived FineWeb starting policy. The chat trainer processed
four policy microbatches, reported zero credited rollouts, and exported
a checkpoint that the normal loader recovered. The two-GPU trainer
processed the same total number of rollouts and wrote a byte-identical
checkpoint. The GSM8K
sample-efficiency study remains open (`hy8.3`).
[OpenAI GSM8K data](https://huggingface.co/datasets/openai/gsm8k/tree/740312add88f781978c0658806c59bc2815b9866) · [Original GSM8K release](https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data)

Two-rank Gloo and NCCL tests matched serial updates with mixed reward
on both ranks, an empty rank and zero credit on every rank. The native
RTX 4090 test matched a full-head reference at 2,052 token positions
and across 26 gradient tensors. Editing the terminal token changed
none of the earlier log probabilities. Twelve legacy-mode updates
matched the earlier path bit for bit across token log probabilities,
gradients, and resulting model state.

Ruff, ty, scoped UBS and all eight targeted tests passed. The final
Linux CPU suite had 2,087 passes, 11 skips and one failure. The live
synaptic training benchmark was 2.11× slower than the standard
transformer against a 2.00× limit. A single DSR invocation tested the
baseline, patch, patch and baseline in that order. The slowdowns were
2.39×, 2.30×, 2.35× and 2.30×. The benchmark and its threshold stayed
intact. The standing DSR qualification is tracked in `u0fl.1`.
[Results and replay drivers](../../results/rollout_credit_2026-11-23.json) · [Rollout and holdout traces](../../results/rollout_credit_2026-11-23.json.gz)

For a replay, write each `study_sources` entry's `content` to its
named file in a fresh scratch directory. In DSR, run the drivers
with `PYTHONPATH=.` from the recorded implementation checkout.
`credit_bandit.py` and `credit_direct_trial.py` each take an output
JSON path. The gzip archive contains their full `projected` and
`direct` rows. Pass the direct rows and a fresh result path to
`verify_direct_trial.py`. The projected verifier takes the original
trace without the two per-example holdout arrays, the replay trace
with those arrays, and an output path. It compares all retained
values and recomputes the registered statistics.

## Prediction-time score traces

REINFORCE supplies an action-specific eligibility through the score
`∇ log π(a|s)`. Eligibility propagation also separates a local trace
from a later learning signal. Here the target is the final dense
synaptic projection of the existing transformer. The question is how
much of its score update survives compression when the trace is stored
at the decision that produced the action. [Williams, 1992](https://doi.org/10.1007/BF00992696) · [Bellec et al., 2020](https://doi.org/10.1038/s41467-020-17236-y)

For a single prediction state, let `x` be the input to the final
synaptic projection, `B` its postsynaptic linear map and `J` the
final RMS normalization Jacobian. Let `u_a` be the unembedding row
for action `a`, multiplied by its logit-softcap derivative. For a
group with exactly centered coefficients `αᵢ`, the common softmax
expectation cancels. The group update for the last slow matrix is
`x ⊗ B Jᵀ (Σᵢ αᵢ u_{aᵢ})`, with feedback vectors written as columns.
This factorization depends on a shared prediction state and sampler
normalization. Different prefixes can have different normalizers.
The hypothesis is that a fixed rank-16 orthogonal projection of the
feedback retains enough of this signal to support learning after the
reward arrives.

### Registered work

- [x] Independently verify exact factorization against the full
  autograd score with repeated actions, zero rewards, RMS normalization
  on and off, and saturated and uncapped logits. Verify storage and
  reward arrival order, and exhibit an unequal-prefix counterexample
  to canceling the normalizer. Check the abstract cancellation and
  projection sign identities in Lean.
- [x] Reuse the copy-first initialization procedure for eight new
  seeds, 90521 through 97521 spaced by 1000. Draw four actions for each
  of 128 questions seeded by `11000000 + model_seed` with sampler seed
  `12000000 + model_seed`. Use an action-independent random orthogonal
  rank-16 basis from seed `13000000 + model_seed`. Shuffle only the
  reward coefficients within each question using seed
  `14000000 + 128 * model_seed + question_index`. Keep the fixed
  prediction states, actions and rewards across all directions.
- [x] Measure directional derivatives of expected reward over an
  independent set of 512 questions from seed 108059. Normalize exact,
  rank-16 and shuffled matrix directions to 0.003 Frobenius norm.
  Compare with a central finite difference at ±one write, checking
  a half step as well. Require six or more starting policies above
  0.10 expected reward, a mixed reward group for every seed, positive
  exact derivatives in at least seven seeds, a positive mean rank-16
  derivative at least one quarter of the exact mean, and rank-16
  aligned beating shuffled in at least six seeds with a one-sided
  exact sign-flip `p ≤ 0.025`.
- [x] If the frozen test qualifies, execute four matched arms on new
  seeds 100521 through 107521 spaced by 1000. Use the same 128
  supervised updates and 96 on-policy updates as the previous
  studies, with 512 new holdout questions from seed 109073. Compare
  ordinary policy gradient with no extra write, the exact score write,
  the rank-16 score write, and the rank-16 write with within-question
  shuffled coefficients. Use the same rank and write norm as the
  frozen test. The rank-16 arm must beat both off and shuffled by
  at least one percentage point on the mean of checkpoints 32, 64
  and 96. Require six positive seeds and a one-sided exact sign-flip
  `p ≤ 0.025` against each. Record the gap to the exact-score arm
  and require the mean to be within one percentage point of that arm.
- [x] Screen the archived FineWeb synaptic start, both 50-step
  synaptic scale pilots, and the 150-step vanilla smoke checkpoint.
  Run 16 fixed questions from the official shuffled training split
  on each candidate, four samples per question, temperature 1,
  top-k 50 and a 64-token completion limit. Require four mixed groups,
  six correct completions and an enabled dense Hebbian checkpoint
  before using this screen to justify a larger matched GSM8K run.
  Archive both checkpoint and dataset identities and leave the
  official test split out of selection.

### Registered multi-token check

Partition scored actions by their exact prediction prefix. At a
prefix `q`, sum the coefficient of each action edge to obtain
`w(q,a)` and sum all active coefficients to obtain the prefix mass
`m(q)`. The score contribution before the local Jacobians is
`Σₐ w(q,a) uₐ − m(q) Σᵥ π(v|q) uᵥ`. Centering across completions
can make the root mass zero while leaving nonzero masses farther
down the tree. Forced tokens define the next prefix and carry no
score coefficient.

- [x] Use the production KV cache and a frozen dense synaptic model
  with presynaptic recurrence disabled. Capture prediction traces
  before revealing fixed rewards. Include a shared prompt, a
  branch shared after a forced token, repeated actions, divergent
  branches, and unequal endings.
- [x] Reconstruct the aggregate and compare it with full masked
  autograd. Run the comparison with and without output capping.
  Check that omitting every prefix normalizer fails, and that
  reducing the corrected matrix to the fixed rank-16 basis agrees
  with projecting the full gradient. Use the same valid-token
  denominator for both implementations.

### Registered delayed-update cost check

Compare three ways to compute the same final-matrix update after
reward arrives. Keep all other weights frozen. Use the archived
paired FineWeb model with eight fixed 128-token contexts, four
samples per context, and explicit FP32 on one RTX 4090. Use fixed
signed coefficients to test update equivalence. This check
measures score computation with a common input and objective.

- [x] Compare a retained autograd graph, a fresh autograd replay
  after reward, and the exact decision-time trace. Use three warmups
  and 12 timed rounds per method in one invocation, rotating order.
  Require sampled actions and the final matrix gradient to agree
  for every timed result.
- [x] Measure synchronized end-to-end and post-reward wall time and
  peak and retained PyTorch tensor allocations. Report the comparison
  against both autograd methods. The workload computes an update
  only for the final matrix.

### Observed score and reward

On the eight frozen policies, exact, rank-16 and shuffled credit had
mean directional derivatives of +2.823 × 10⁻⁵, +1.519 × 10⁻⁵ and
−2.528 × 10⁻⁶ respectively. Rank-16 beat the shuffle in all eight
seeds, with a one-sided exact p = 0.00390625. It retained 53.8% of the
exact derivative. A four-action trace needs 320 stored scalars in
the sketched representation and 512 in the exact factorized
representation, excluding the fixed shared basis and bookkeeping.

The frozen result qualified the fresh training cohort. The
training result fell short of the one-percentage-point threshold.
All 32 policies trained with mixed rewards. The full measurements
and traces from both phases reproduced byte for byte. Independent
scoring recovered 53,248 sampled action rewards and 65,536 training
holdout scores.

| Rank-16 aligned compared with | Mean expected reward gain (percentage points) | Positive seeds | One-sided exact p |
| --- | ---: | ---: | ---: |
| Off | +0.191388 | 5/8 | 0.277344 |
| Rank-16 shuffled | +0.187482 | 5/8 | 0.210938 |
| Exact local score | +0.154712 | 6/8 | 0.273438 |

The exact local formula matched autograd in all six float64
RMS and softcap fixtures within 8.68 × 10⁻¹⁹ absolute error.
Lean checked shared-normalizer cancellation, the counterexample
with unequal normalizers, sketch sign invariance and
nonnegativity. The proofs address the finite-sum identities.

On the four-completion fixture, the root mass was zero and the
shared prefix after a forced token had mass 0.5. The corrected cache
accumulator matched full masked autograd within 5.69 × 10⁻¹³
absolute error. Omitting all normalizers produced about
8.90 × 10⁻⁴ Frobenius error. With presynaptic recurrence enabled,
the accumulator matched independent cached autograd within
5.43 × 10⁻¹⁹. Full replay differed by up to 7.65 × 10⁻¹² there.
For a fixed top-8 support at temperature 0.7, full and truncated
softmax produced identical weighted gradients on shared prefixes.

### Native cost and GSM8K checkpoint screen

The live RTX 4090 comparison matched sampled actions in every
round. The exact trace's matrix update differed from autograd by
at most 3.73 × 10⁻⁹. The table gives medians of 12 timed rounds in
one invocation. Retained memory is the extra PyTorch tensor memory
still allocated when feedback arrives.

| Method | End to end (ms) | After reward (ms) | Retained tensor bytes |
| --- | ---: | ---: | ---: |
| Retained autograd | 319.68 | 14.47 | 8,132,096 |
| Autograd replay | 527.32 | 267.10 | 512 |
| Exact trace | 243.46 | 4.38 | 49,664 |

The trace took less end-to-end time in 8 of 12 rounds versus a
retained graph and 11 of 12 versus replay. It preserved more
pre-reward state than replay while cutting post-reward latency.
This establishes the cost tradeoff for the frozen, final-matrix
workload.

Each of the four staged checkpoints produced zero correct
completions in 64 samples and zero mixed groups in 16 questions.
The screen used only the official training split. The paired
FineWeb checkpoint was the sole archive with eligible dense
Hebbian layers enabled. The larger GSM8K sample-efficiency test
remains open in `hy8.3` and needs a compatible checkpoint that
earns mixed rewards.
[Protocol, sources and receipts](../../results/prediction_credit_2026-11-23.json) · [Replay rows](../../results/prediction_credit_2026-11-23.json.gz)

For replay, extract each `sources` entry's `content` to its named
file in a fresh directory. Run the Python drivers through DSR with
`PYTHONPATH=.` from the recorded implementation checkout.
`prediction_credit.py` takes an output JSON path and
`train_prediction_credit.py` takes the frozen result and an output
path. Split the gzip archive's `frozen` and `training` entries
into separate files before passing them, in that order, to
`verify_prediction_credit.py` with a fresh result path.

## Qualifying a GSM8K starting policy

The archive screen found no mixed rewards. This pilot trains two
supervised starting policies on the official GSM8K training split
and tests whether the correct answers depend on the question.
[Cobbe et al., 2021](https://arxiv.org/abs/2110.14168)

- [x] Load the same archived two-layer, width-192 FineWeb checkpoint
  for both arms and verify its weights and tokenizer. One arm learns
  worked solutions with the production token masks for calculator
  outputs. The other learns just the official final answer in
  `#### number` format. Keep the question text and start state matched.
- [x] Use official revision `740312add88f781978c0658806c59bc2815b9866`.
  Take selection questions from shuffled training indices [128, 192)
  and confirmation questions from [192, 256). Train on indices
  [256, 7473), excluding question text duplicates of either panel
  and worked solutions longer than 512 rendered tokens. Apply the
  same exclusions to both arms. Retain the official test split
  for later evaluation.
- [x] Train for 384 updates with 32 questions per update, microbatches
  of eight and fixed shuffled epochs. Use AdamW with peak learning
  rate 0.001, 16 warmup steps, cosine decay to 0.0001, weight decay
  0.0001, gradient norm cap 1 and BF16 autocast. Supervise assistant
  and sampled calculator tokens and exclude prompt and forced
  calculator output tokens. Disable ordinary synaptic writes in
  both arms. Save final checkpoints with the production writer
  and reload using the production reader.
- [x] Sample four unforced completions per question with the production
  engine, temperature 1, top-k 50 and a 160-token cap. On each panel,
  require at least 24 correct completions out of 256, at least 16 mixed
  reward groups out of 64, and correct answers of at least four distinct
  values. Require more correct completions than the most frequent
  training answer and at least eight more than scoring the very same
  outputs against gold labels rotated by 17 questions. Test the paired
  differences across questions with a one-sided exact sign-flip
  `p ≤ 0.025`. A checkpoint qualifies only if it passes both panels.
  If both qualify, prioritize the higher selection score. As a
  descriptive check, also score 4,095 whole-panel label shuffles
  from seed 620033 plus the identity assignment. These shuffles
  preserve the full answer histogram.
- [x] Independently reconstruct every reward and qualification
  decision. If any arm supplies mixed reward groups, replay an actual
  group through the exact final-matrix score and compare with full
  masked autograd. Keep task qualification and gradient parity as
  separate results. Publish the complete denominators and checkpoint
  identities.

### Compatibility check for a deeper starting model

- [x] Stage the public depth-20 NanoChat SFT artifact at revision
  `53b7aa58db3e1c512d121a53f747f9649c15bc55`. Verify the model and
  tokenizer hashes, use the repository's restricted readers, and
  check its token mapping against the local tokenizer.
- [x] On fixed token inputs, add an initially zero synaptic residual
  to the final MLP projection of the loaded vanilla model. Compare
  logits and the final-matrix score gradient with the original
  projection. Use this as a compatibility check. The public starting
  model's training data would need a separate validation fold.
  [Public checkpoint and model card](https://huggingface.co/gaviego/nanochat/tree/53b7aa58db3e1c512d121a53f747f9649c15bc55)

### Observed checkpoint qualification

The independent data audit matched all 7,473 training rows to the
original OpenAI JSONL and rebuilt the shuffle. All 7,217 eligible
training questions fit the filter. Neither held-out panel had a
question in training with the same text after digit normalization.
Both arms saw 12,288 question exposures. Answer-only training
supervised 66,041 tokens and worked-solution training supervised
1,241,316. Both final checkpoints passed GPU and local CPU reload
with identical weights and tied embeddings. In the masked-loss
fixture, both objectives matched exactly and the worked-solution
gradients agreed within 5.97 × 10⁻⁸ across 49 tensors. Its 15 forced
calculator tokens were excluded.

| Arm | Held-out panel | Parseable / 256 | Correct / 256 | Mixed groups / 64 | Rotated-label matches | Constant-answer matches | Paired p |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Answer only | Selection | 256 | 2 | 2 | 1 | 8 | 0.500 |
| Answer only | Confirmation | 255 | 2 | 2 | 5 | 4 | 0.875 |
| Worked solution | Selection | 215 | 1 | 1 | 0 | 8 | 0.500 |
| Worked solution | Confirmation | 210 | 1 | 1 | 2 | 4 | 0.875 |

Neither checkpoint qualified. Every panel scored below the
constant-answer control. The histogram-preserving label shuffles
gave p-values from 0.751 to 0.945. The answer-only policy learned to
emit the answer format, and both policies supplied occasional real
mixed rewards. The official test split remains reserved.

On the first mixed confirmation group from each arm, the cached
score matched masked autograd within 1.71 × 10⁻¹³ across 17 scored
tokens for answer-only and 1.94 × 10⁻¹² across 197 for worked
solutions. Both groups had rewards `[0, 1, 0, 0]`. The longer group
needed the normalization correction at 193 of its 194 scored
prefixes. The replay used the trainer's untruncated softmax
objective in FP64 and the samples recorded from the BF16 top-k
engine. The identical macOS CPU replay finished first for the
longer group, and the redundant Linux attempt was stopped.

### Deeper checkpoint compatibility

The public depth-20 SFT checkpoint loaded through the native
weights-only reader. Its 65,527 BPE ranks, nine special tokens and
tokenizer pattern matched the local mapping exactly. On two fixed
32-token contexts, the zero-initialized final projection residual
preserved every measured logit and the final-matrix score gradient
exactly in FP32 and BF16 on an RTX 4090. The first probe changed the
required rotary-buffer dtype when converting the weights to FP32.
The rerun retained the checkpoint's native BF16 rotary tables.

The verified weights are staged for a larger-model follow-up.
Validation on prompts independent of its training still needs to
establish mixed rewards and performance above the label controls.
[Pinned public weights](https://huggingface.co/gaviego/nanochat/tree/53b7aa58db3e1c512d121a53f747f9649c15bc55)

Both checkpoint qualification and the full RL study stay open
under `hy8.3.3` and `hy8.3`. To replay the pilot, unpack the
`sources` entries to their named files. The gzip archive contains
the `train` and `eval` objects for each arm. Write them as
`answer-train.json`, `answer-evaluation.json`,
`rationale-train.json`, and `rationale-evaluation.json` under a
fresh scratch directory. Pass that directory and a result path to
`audit_gsm_sft.py` in a DSR check.
[Qualification, checkpoint locations and DSR receipts](../../results/gsm8k_sft_2026-11-23.json) · [Training and scoring traces](../../results/gsm8k_sft_2026-11-23.json.gz)

## Fresh arithmetic qualification of the public D20 checkpoint

The pinned public weights predate these generated questions. Each
pair changes one quantity and the correct answer. The four-choice
version keeps the answer options and their order fixed across both
variants. Comparing true labels with labels exchanged within each
pair checks whether the change in the question affects the answer
distribution. The official GSM8K test split stays reserved.

- [x] Generate and hash four disjoint 32-question panels before
  model execution. Each panel has 16 pairs, with four pairs each
  of addition, subtraction, equal rows and fair shares. Keep the
  two different correct answers out of the displayed quantities.
  In each choice panel, each correct letter appears eight times.
  Reserve the panels for screening, confirmation, adaptation and
  transfer, respectively.
- [x] On the screen panel, run four free-form completions per
  question with the native engine, temperature 1, top-k 50 and
  a 96-token limit. Prompt for the final integer after `####`.
  Require 12 of 128 correct, eight of 32 mixed groups and four
  distinct correct values. Require strictly more successes than
  the best constant integer on the panel, at least four more
  than against exchanged pair labels and a one-sided exact
  pair-sign-flip p-value at most 0.025. If it passes, confirm
  on the separate panel under the same rule and test exact
  cached credit on its first genuine mixed group.
- [x] Score the screen and confirmation panels separately with
  a policy restricted to ` A`, ` B`, ` C`, and ` D`, each verified
  as one token. Use the temperature-1 softmax over those four
  native logits and draw four samples per question. On both
  panels require mean exact expected accuracy at least 0.35,
  an expected true-minus-exchanged accuracy gap at least 0.08,
  at least eight mixed groups, four distinct correct integers
  in the samples and a one-sided exact pair-sign-flip
  `p ≤ 0.025` for the probability gap.
- [x] Use a genuinely sampled mixed four-choice group to compare
  the exact trace against masked autograd. If both choice
  panels qualify, use the adaptation panel for eight updates,
  four questions and four sampled choices each. Freeze the
  entire D20 model except the zero-initialized last-projection
  adapter. Compare exact late credit with the same centered
  policy-gradient update from autograd replay, no extra update,
  and within-question shuffled credit. Use a fixed 0.02
  Frobenius-norm write for each nonzero score. Keep the sampling
  and question seeds matched and report state parity between
  exact and replay. On the untouched transfer panel, report
  expected accuracy before and after. Any advancement requires
  a mean gain of at least two percentage points over both off
  and shuffled with a one-sided exact pair-sign-flip
  `p ≤ 0.025` for each.
- [x] Independently check prompts, rewards, paired contrasts,
  scope and DSR receipts. Track arithmetic-panel qualification
  separately from the original GSM8K checkpoint and RL gates.

### External reference policy

The D20 four-choice pilot came in near chance. To distinguish a
weak starting checkpoint from an unusable arithmetic panel, run
the sealed questions on Qwen2.5-Math-1.5B-Instruct at revision
`aafeb0fc6f22cbf0eaeed126eff8be45b0360a35`. This is a diagnostic
reference with a different tokenizer, training history and
parameter count. Record it separately from the native NanoChat run.
[Pinned Qwen checkpoint](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct/tree/aafeb0fc6f22cbf0eaeed126eff8be45b0360a35)

- [x] Verify the pinned safetensors checksum and use the installed
  Transformers implementation with local files only. Score the
  same free-form instruction with the model's own chat template.
  Use four samples per question, temperature 1, top-k 50 and
  a 96-token limit. Run both held-out panels and keep the D20
  qualification thresholds. Compare parseable answers, correct
  answers, mixed groups and exchanged-label controls for both.
- [x] Report the external result with its own receipts. Use the
  reserved adaptation and transfer panels only on a policy that
  qualifies in the registered four-choice comparison.
- [x] Apply the gate for the external reference score check. On
  a qualified pair of panels, check a
  zero-initialized final-projection synaptic residual on the first
  mixed confirmation group. Require identical starting logits and
  agreement between the exact detached trace and full autograd for
  its recorded tokens. Use FP32 and the untruncated softmax in the
  comparison, with coefficients derived from the recorded on-policy
  rewards. Keep this numerical transfer check separate from a
  training trial.

The external screen logged zero answers under the required
`####` marker before its raw completions were inspected. The
Qwen evaluation recipe asks for a LaTeX boxed answer. Preserve
the original result and add a diagnostic on the same archived
samples. Parse the last completed `\boxed{…}` whose entire
content is an integer or an integer-valued decimal, and apply
the same thresholds. If both panels clear those thresholds, use
the first mixed confirmation group for the numerical transfer
check above with boxed rewards. Keep the `####` and boxed
results separate. [Qwen evaluation instructions](https://github.com/QwenLM/Qwen2.5-Math#evaluation)

The archived samples using the original prompt cleared neither
panel under the boxed-answer diagnostic. Run one follow-up on
the same two panels using Qwen's published chain-of-thought
system message and a 256-token cap. Keep the seeds, sampling
temperature, top-k and denominators. Use the boxed parser and
the same qualification gates. If both panels qualify, run the
same numerical transfer check with the first mixed confirmation
group from this follow-up. This tests the model and response
format together, with no new training.
[Published prompt](https://github.com/QwenLM/Qwen2.5-Math/blob/main/evaluation/utils.py)

### D20 observations

The native free-form model answered 18 of 128 correctly, compared
with zero when labels were exchanged within pairs. It produced
16 mixed groups across 32 questions and got ten distinct correct
values. The paired exact test gave `p = 1/2048`. The best constant
answer on this panel scored 24 of 128, so free-form confirmation
did not advance.

| Four-choice panel | Expected accuracy | True minus exchanged | Correct / 128 | Mixed / 32 |
| --- | ---: | ---: | ---: | ---: |
| Screen | 23.87% | +0.386 percentage points | 29 | 5 |
| Confirmation | 25.29% | −0.331 percentage points | 30 | 6 |

Neither four-choice panel qualified. The eight-step adaptation
trial was gated off. With the model frozen, the first recorded
mixed group from each panel gave a nonzero delayed score.
The exact last-matrix trace agreed with autograd within
4.48 × 10⁻⁸ maximum absolute error on both groups.

A separate free-form replay used the first mixed screen group,
with recorded rewards `[0, 1, 0, 0]`. The trace matched the FP32
full-softmax gradient on 131 scored tokens. Maximum absolute
error was 1.41 × 10⁻⁷ and relative Frobenius error was 1.02 × 10⁻⁵.
The original samples came from the BF16 top-k policy. This replay
compares two implementations of the same FP32 objective.

### External reference observations

The original Qwen run reached the 96-token cap in 235 of 256
completions and produced zero parseable `####` answers. The
boxed-answer diagnostic on those same samples found 10 correct
on the screen and 15 on confirmation, with 9 and 13 mixed
groups respectively. There were zero matches to exchanged
labels. Both boxed counts fell below the 24-correct constant
control. This was the same sampled text under two score rules.

Under Qwen's published prompt and the 256-token cap, the model
answered all 256 completions correctly. No recorded completion
matched the exchanged labels. Each paired exact test gave
`p = 1/65536`. Both panels had zero mixed groups,
so the centered reward was zero on every group. The two
configurations changed both the instruction and the token budget.
The recorded counts measure their combined effect. The external
score check and training trial did not advance, and neither
model used the adaptation or transfer panels.

The D20 study closes `hy8.3.4`. Native GSM8K qualification and
RL effectiveness stay open under `hy8.3.3` and `hy8.3`.
For replay, extract the source entries to files and the archived
panel and prediction objects to separate JSON files. The DSR
commands ran from the recorded implementation checkout. The
three `audit_*.py` drivers reconstruct every reported decision
from the archived rows and the frozen panel.
[Results, replay source and DSR receipts](../../results/d20_fresh_arithmetic_2026-11-23.json) · [Panel and generation archive](../../results/d20_fresh_arithmetic_2026-11-23.json.gz)

## Calibrating arithmetic difficulty for mixed rewards

The [pinned Qwen policy](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct/tree/aafeb0fc6f22cbf0eaeed126eff8be45b0360a35)
solved all the single-operation controls. This study varies computation
depth and number size while retaining exact answer checks. Its aim is to
find a setting where reward responds to the question and has meaningful
variation within a group of samples.

- [x] Seal screening, confirmation, adaptation and transfer panels before
  model inference. Each has eight pairs per difficulty, split evenly
  between an ordered register and a packing calculation. Each pair
  changes only the starting quantity. Verify both correct answers with
  an independent integer calculation and exchange labels within each pair
  for the control. Set the generator seed to 811023. The three levels
  use 1, 2 and 3 rounds of four operations with starting magnitudes
  100–999, 1,000–9,999 and 10,000–99,999 respectively. Each round
  adds, multiplies, subtracts and floor-divides in that order. Discard
  pairs whose answers coincide with each other or a printed quantity.
- [x] Use the same public checkpoint and published chain-of-thought
  system prompt, with four independent BF16 samples per question,
  temperature 1, top-k 50, top-p 1 and a 512-token limit. Parse the
  last complete integer-valued boxed answer. Use batches of eight
  questions and derive batch seeds from panel and level, independent
  of which level is selected.
- [x] On screening, a level qualifies with at least 48 parseable and
  16 correct out of 64 samples, at least 4 mixed groups out of 16,
  correct answers for at least 4 distinct integers, a strict lead over
  the best constant integer, and at least eight more correct than the
  exchanged-label control with exact pair-sign-flip `p ≤ 0.025`.
  Select the lowest qualifying level using screening alone. Run
  confirmation at that level. If no level qualifies on screening,
  keep confirmation and all adaptation panels sealed.
- [x] Freeze the selected level and checkpoint for later adaptation
  only if confirmation passes the same gates. If confirmation has
  any genuine mixed group, use its first one for a numerical check
  on the exact delayed score of Qwen's final MLP projection. Compare
  a detached local trace with independent autograd for the same FP32
  full-softmax objective and recorded completions. Record the sampled
  actions, reward denominators and gradient error. Keep the official
  GSM8K test split and the adaptation and transfer panels untouched.
- [x] Independently reconstruct panel math, rewards, the screen
  selection, confirmation and the archived evidence. Preserve all
  levels and publish the gate result, including any lack of reward
  variation or a format/length failure.

### Ladder results

An independent integer evaluator verified all 192 questions and the
exchanged-label construction before inference. Each evaluated stratum has
16 questions and 64 responses.

| Level | Parseable | Correct | Mixed groups | Distinct correct targets | Best constant | Exchanged labels | Hit limit | Paired exact p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Screen L1 | 64 | 53 | 6 | 15 | 8 | 0 | 0 | 1/256 |
| Screen L2 | 47 | 23 | 7 | 10 | 4 | 0 | 18 | 1/128 |
| Screen L3 | 1 | 0 | 0 | 0 | 4 | 0 | 63 | 1 |
| Confirmation L1 | 64 | 53 | 4 | 15 | 4 | 0 | 0 | 1/256 |

L1 was the only level to clear screening. L2 missed the parseability
cutoff by one response. The 512-token limit was binding at L3, so its
observed accuracy combines the effects of difficulty and response length.

L1 also cleared the independent confirmation gate, with exactly the four
mixed groups required. Qwen answered all 32 register samples correctly
on each L1 panel. It answered 21 of 32 packing samples correctly on
each, and packing supplied all ten mixed groups. An exploratory check
found that six of the 22 packing errors equal the answer obtained by
counting one extra copy. This pattern points to a possible wording
effect. The frozen setting retains both formulations and the original
sampling budget.

The first mixed confirmation group had rewards `[1, 0, 1, 1]`
across four responses. Its 1,262 scored tokens occupied 1,252 unique
prefixes. The centered coefficients summed to zero at the common
prompt and had nonzero mass at 1,251 prefixes farther along the
completions. The zero-initialized residual preserved the logits
bit for bit. The detached update for Qwen's 8,960 × 1,536
final MLP projection agreed with autograd to a relative Frobenius error
of `1.20 × 10⁻⁵` and maximum absolute error of `1.70 × 10⁻⁷`.
The comparison used the FP32 full-softmax score on the recorded
BF16 top-k responses.

The independent checks reconstructed every gate on both macOS and
Linux and recovered all 256 response texts from the pinned tokenizer
and saved token IDs. The next experiment, `hy8.3.6`, will compare
the score trace with autograd under the sampling support actually
used at generation time before testing compression and adaptation.
The native `hy8.3.3` GSM8K checkpoint and the `hy8.3` learning
study remain open. The adaptation and transfer panels and the
official GSM8K test split were unused.

[Protocol, source and DSR receipts](../../results/qwen_reward_ladder_2026-11-23.json) ·
[Panels, samples and exact numerators](../../results/qwen_reward_ladder_2026-11-23.json.gz)

To replay the gates, extract `replay_ladder.py` and `ladder_audit.py`
from the report's `sources` mapping. Run the replay script through DSR
with the report, gzip archive and a fresh working directory as arguments.

## Delayed credit under Qwen's actual sampling support

The full-softmax result uses more candidates than the policy allowed
when sampling. Repeat the fixed L1 confirmation batch that contains the
previously selected first mixed group. Use the identical seed, eight
questions, four BF16 samples per question and 512-token limit.

- [x] Require all 32 continuations in that batch, up to their archived
  stop tokens, to match the previous record. Capture the final MLP
  input and the residual before final RMSNorm at generation time.
- [x] Read Transformers' returned post-processor scores for each
  scored action. Match every finite candidate and score against the
  original logit at or above its fiftieth-largest value. Account for
  all boundary ties and require restricted softmax probabilities to
  agree with the sampler's full masked probabilities within `10⁻⁶`.
  Verify that intentionally dropping a tied candidate fails.
- [x] On all 1,262 scored tokens in the predetermined group, compare
  the detached analytical update with FP64 autograd. At the zero
  write, the differentiable score takes the captured sampler logits
  as its values. Its tangent adds the change produced by a final
  projection perturbation through real-valued RMSNorm and the
  unembedding. Hold the sampled actions and support fixed. Require
  relative Frobenius error at most `10⁻⁴` and maximum absolute error
  at most `6 × 10⁻⁷`.
- [x] Archive all support IDs, logits, cutoffs and selected actions.
  Independently verify the archive against the prior episode and
  publish the DSR receipts. The preregistered sampler script SHA-256
  is `07763d5c3e215448f8f012d9d02131de8082fbc81c9800e9cb623f5c192a0122`.
- [x] Proceed to a separate, held-out compression comparison only
  after sampler and gradient agreement. Register projection ranks,
  calibration and evaluation groups, retained bytes and stopping
  thresholds before fitting a basis. Preserve the reserved
  adaptation and transfer panels for the subsequent learning study.

All 32 completions reproduced token for token before EOS or padding.
The first mixed group still had rewards `[1, 0, 1, 1]`. Across its
1,262 scored actions, 456 sampling distributions had more than 50
candidates because of boundary ties. The largest support had 58.
Capturing the ties preserved 661 candidate occurrences beyond a
hard 50 per position. The largest probability discrepancy between
the restricted and full masked softmax was `3.58 × 10⁻⁷`.

The detached score matched FP64 autograd at relative Frobenius error
`4.83 × 10⁻⁷` and maximum absolute error `3.33 × 10⁻⁹` for the
8,960 × 1,536 matrix. An independent CPU audit recovered every
action and the objective from the archived candidates and logits.
This measures the fixed-support tangent anchored at logits actually
used by BF16 generation. The initial launch failed during imports.
Staging the missing archived helper allowed the preregistered
script to run byte for byte on retry.

[Source and DSR receipts](../../results/qwen_sampler_score_2026-11-23.json) ·
[Per-action sampler support](../../results/qwen_sampler_score_2026-11-23.json.gz)

### Registered held-out compression screen

- [x] Replay the eight packing questions from L1 confirmation with
  the same batch seed. Score the actual retained support for all
  four completions per question. Fit separate rank-32 input and
  feedback bases without reward weights on both sides of packing
  pairs 0 and 1. Use uncentered randomized PCA with seed 811027,
  48 sketch vectors and two power iterations. Freeze both bases
  before projecting either side of pairs 2 and 3.
- [x] On each held-out mixed group, verify that outer products of
  the projected inputs and feedback agree with projecting the
  resulting dense matrix (relative Frobenius error at most
  `10⁻⁴`). Require at least two mixed groups and at least 25%
  retained squared gradient norm in each. Require alignment with
  the exact update to beat a random orthogonal basis of the same
  rank (seed 811028), the fitted basis with rewards rotated by one
  within the question, and the zero update.
- [x] Record both rank-32 factors and their fixed storage costs.
  Compare a dense update, a compressed coordinate update and a
  compressed update expanded to the full matrix. Use two warmups
  and five synchronized rounds per method with rotating order in
  the same invocation. These timings start with stored activations
  and feedback.
- [x] Require the registered direction gate before preparing any
  adaptation or transfer rollouts. The preregistered compression
  script SHA-256 is `e2371546f72172827af5b54f8f5c0408102dd938701257df9284e8f852c508d3`.
- [x] Re-orthogonalize the frozen spans in FP64 and rerun the
  independent `10⁻⁵` Gram check.
- [x] Compare against whole-model post-reward replay on the
  same trajectories. Register update sizes, matched sampling
  budgets and the smallest meaningful accuracy gain before
  releasing the adaptation and transfer panels.

The unweighted fit used 4,662 calibration tokens. The four
held-out questions contained 4,610 tokens, including 2,336 from
two mixed groups. The signed direction for each mixed group beat
its controls and cleared the 25% threshold.

| Held-out packing question | Squared gradient norm retained | Same-size random basis | Alignment with rotated rewards |
| --- | ---: | ---: | ---: |
| Pair 2, first question | 36.28% | 0.00738% | −36.28% |
| Pair 3, first question | 34.76% | 0.00693% | 16.24% |

The projected outer products matched the coordinates formed from
the exact matrix with at most `5.22 × 10⁻⁷` relative error.
The FP32 trace vectors used 256 bytes per token with the two
rank-32 projections and 41,984 bytes per token at full width.
The frozen bases occupy 1,343,488 bytes, excluding the file header.

The feedback basis missed the independent Gram-matrix tolerance
of `10⁻⁵` with a maximum entry error of `1.37 × 10⁻⁵`.
The failed audit remains archived. An independent calculation
bounded retention under the orthogonal projections onto the same
spans by dividing the measured squared norms by the products of the
extreme Gram eigenvalues. The lower bounds were 36.277% and
34.755%, both above the registered 25% floor. The current artifact
contains the original FP32 factors used in the measurement.
FP64 QR with a positive diagonal then normalized the same spans
without refitting. After FP32 storage, the normalized factors
passed the original check with maximum Gram error
`1.25 × 10⁻⁸` and relative span residual `2.72 × 10⁻⁸`.
The report contains the normalization source and output digest.

| Held-out packing question | Dense update (ms) | Compressed coordinates (ms) | Compressed to dense (ms) |
| --- | ---: | ---: | ---: |
| Pair 2, first question | 0.971 | 0.287 | 0.474 |
| Pair 3, first question | 0.905 | 0.233 | 0.500 |

Each time is the median of five synchronized updates on RTX 4090
with the decision-time vectors in memory. The timing starts after
reward and excludes generation, projection of incoming vectors and
storage allocation for the surrounding model. The whole-model replay
comparison follows below. Learning gains are still pending in `hy8.3.6`.

[Compression measurements, source and receipts](../../results/qwen_sampler_score_2026-11-23_compression.json) ·
[Frozen FP32 bases](../../results/qwen_sampler_score_2026-11-23_compression.safetensors)

### Registered whole-model replay comparison

- [x] Replay the published packing batch with the original seed.
  Require the same 32 completions and use the first held-out mixed
  group, packing pair 2, side 0, with 1,172 scored tokens. Capture
  its decision-time vectors and actual sampler support. Use the
  normalized rank-32 spans from the compression study without
  refitting.
- [x] Check parallel teacher-forced replay on the four relevant
  sequences first. If it fails, check single-token cached replay
  with the original 32-sequence batch geometry. Replay runs the
  whole frozen model over fixed token IDs and reconstructs the
  delayed gradient. Require identical candidate IDs and retained
  logits at every scored token, relative gradient error at most
  `10⁻⁴`, and maximum absolute error at most `6 × 10⁻⁷`.
  Record failed routes and withhold the matched comparison if
  neither qualifies.
- [x] Use the first eligible route in three timed rounds. Synchronize
  PyTorch at each method boundary and rotate the order of
  replay, the exact stored vectors and the compressed stored
  vectors expanded to a dense write. The qualification replay
  serves as a warmup and each retained method has one untimed
  update. Check every timed update against its own reference
  matrix. Report logical retained bytes, median elapsed
  post-reward time and incremental peak tensor bytes.
- [x] Preserve the adaptation and transfer panels for this comparison. The registered
  benchmark source SHA-256 is `0c592655daf77bb10b629b6f90a45db0525657081bc7ee6c7620f9f7dbbbc27c`.

The four-sequence parallel replay changed candidate IDs at 762 of
1,172 scored positions, changed retained scores at all 1,172 and
had relative gradient error of 3.26%. The difference combines
parallel teacher forcing with the change in batch geometry.
Cached replay in the original 32-sequence geometry recovered
every sampler support, retained logit and matrix entry exactly.
All nine timed updates agreed with their own references.

| RTX 4090 method | Median post-reward time | Logical retained state | Median incremental peak bytes |
| --- | ---: | ---: | ---: |
| Exact stored trace | 1.278 ms | 49,205,248 bytes | 110,100,480 |
| Compressed stored trace expanded to dense | 1.161 ms | 300,032 bytes plus 1,343,488 shared basis bytes | 56,885,248 |
| Cached replay in the original geometry | 67.445 s | 142,848 bytes of token IDs and prompt masks | 1,189,081,088 |

The timings measure three post-reward updates per method with
the model resident. The retained modes start from prepared vectors.
Replay includes the frozen-model forward passes and reconstructs the
same analytical update. The measurement is specific to the original
batch geometry required by the two replay routes tried here. The
normalized compressed trace retained 36.28% of squared gradient norm
on this group.

[Replay benchmark, source and DSR receipts](../../results/qwen_whole_model_replay_2026-11-23.json)

### Registered single-write adaptation trial

- [x] Use the reserved L1 adaptation questions, including both
  wordings. Draw four BF16 continuations per question with the
  frozen sampler and 512-token limit. Derive each batch seed with
  `stable_seed_u64(811030, salt="qwen.single_write.adaptation.L1.<offset>")`
  reduced modulo `2**63`. Reapply the registered
  L1 qualification gates to the initial samples. Stop before any
  transfer rollout if the adaptation panel misses a gate.
- [x] From the same decision-time states and actions, form one
  final-projection write per arm. Use the exact trace, cached
  post-reward replay, the normalized rank-32 trace, the same rank
  with rewards rotated by one within each question, and zero
  write. Use total scored tokens as the denominator, center
  rewards in each group and scale each nonzero write to 0.5
  Frobenius norm. Freeze every other parameter. Stop if a
  nonzero arm has no usable direction or if replay fails the
  exact-support and gradient checks.
- [x] Freeze the writes before evaluating the reserved L1
  transfer questions. Use four samples per question per arm,
  the same 512-token cap, and batch seeds generated with
  `stable_seed_u64(811031, salt="qwen.single_write.transfer.L1.<offset>")`
  reduced modulo `2**63`. Merge the replay and exact evaluations
  only when their deployed writes match bit for bit. Record
  all actual scored lengths, truncation and parseability by
  arm and wording. Apply the frozen write as a FP32 input-matrix
  product added to the down-projection output in its BF16 dtype.
  Check zero-write logits bit for bit before sampling transfer.
- [x] Require the compressed arm to gain at least 4 of 64
  correct samples over both the rotated-reward and zero-write
  arms. In both comparisons require a one-sided exact pair
  sign-flip `p ≤ 0.025`. Report the exact-arm contrast and
  whether compressed finishes within two correct samples of
  it. Publish every outcome, including a failed or inconclusive
  pilot, and keep the official GSM8K test split sealed.
- [x] Run compressed, rotated and zero on the same GPU for
  the primary contrasts. Run exact and an independent zero on
  the second RTX 4090, along with replay if its write differs.
  Require the two zero runs to agree token for token before
  interpreting the cross-device contrast with exact.

The paired gate needs at least six positive pairs. If there are
`k` positive pairs, all sign assignments that keep those pairs
positive contribute to the one-sided tail, so `p ≥ 2⁻ᵏ`.
A perfect zero-write score on the register wording would leave
only four packing pairs in which to gain. In that case the
registered comparison can estimate a gain but cannot certify one
at `p ≤ 0.025`.

| Registered source | SHA-256 |
| --- | --- |
| Adaptation qualification | `04c9a674b87d7c19a7955c88133253110325212a6524070cc0c5826cdb80da40` |
| Cached adaptation replay | `5847074a2b53d3a1a0d4af0b97208eaff13889ce0a6ae7cf676696473d5b3512` |
| Transfer evaluation | `d2121bfe719b84ce0e44d665558924ab532cc13368d20b2801810fde56101dc6` |

Adaptation passed at 49/64 correct, 64/64 parseable and 4/16
mixed reward groups. The best constant scored 4/64, the exchanged
labels scored 0/64 and the paired exact `p` was 1/256.
The three candidate writes used 17,453 scored tokens. The
compressed coordinates matched the projected exact matrix at
relative error `4.19 × 10⁻⁷` and retained 29.55% of squared
gradient norm on this adaptation batch.

The initial FP32 norm report was low. An independent FP64 sum
over the archived matrix values confirmed all three Frobenius
norms were within `4 × 10⁻⁸` of the registered 0.5. The candidate
matrices were preserved byte for byte.

[Adaptation gate, write digests and receipts](../../results/qwen_single_write_2026-11-23_adaptation.json) ·
[Saved adaptation trajectories](../../results/qwen_single_write_2026-11-23_adaptation.json.gz)

Cached replay recovered the archived support and logits on
all 17,453 adaptation tokens across both batches. Its normalized
dense write matched the exact decision-time-trace write bit for bit.
The two arms therefore share one transfer evaluation.

### Single-write transfer results

| Transfer arm | Correct / 64 | Parseable / 64 | Hit 512-token limit | Median scored tokens |
| --- | ---: | ---: | ---: | ---: |
| Zero | 55 | 63 | 1 | 250.5 |
| Exact and matched replay | 47 | 55 | 9 | 255.5 |
| Rank-32 rotated reward | 37 | 41 | 23 | 272.5 |
| Rank-32 compressed | 34 | 38 | 26 | 299.0 |

Compressed had 21 fewer correct samples than zero and three fewer
than rotated reward. The registered one-sided exact paired p-values
were 1 and 0.875 respectively. Against zero, the deficit was
19 packing answers and two register answers. Every unparseable
response reached the token limit in this pilot, and every response
that reached it was unparseable. The two devices produced identical
zero-write continuations. An independent audit scored all 320 draws,
including the duplicated zero arm.

This completes the matched single-write pilot in `hy8.3.6`.
The measured storage and replay tradeoff comes from reconstructing
the same gradient. The 0.5-norm write failed the transfer gain gate.
Native GSM8K qualification remains open under `hy8.3.3` with the
official test split still sealed.

The saved adaptation trajectories contained 64 sampled EOS actions
just beyond their scoring masks. Sixteen came from mixed-reward
groups and would have had nonzero centered coefficients. This
identifies an omitted term in the tested objective. The measured
length shifts alone leave its effect unresolved. The follow-up,
`hy8.3.7`, will distinguish EOS credit from write size before
sampling a new, adequately powered transfer panel.

[Transfer scores, source and DSR receipts](../../results/qwen_single_write_2026-11-23_transfer.json) ·
[All transfer continuations](../../results/qwen_single_write_2026-11-23_transfer.json.gz)

### Registered fixed-prefix termination diagnostic

- [x] Replay the adaptation tokens through the frozen checkpoint.
  Match both content support digests. Capture the final MLP input,
  down-projection output and pre-MLP residual at every sampled
  EOS and every 32nd scored content action. Require the unchanged
  residual reconstruction to yield identical logits.
- [x] At each captured prefix, measure the sum of the two EOS
  probabilities under the actual top-k policy. Compare the
  original exact, compressed and rotated writes at norms
  0.5 and 0.125. Measure whether the retained set changes and
  compute KL after renormalizing both distributions on the
  original support. This keeps the KL finite when the actual
  selected sets differ.
- [x] Add the exact score contribution of sampled EOS with
  group-centered reward to the original dense numerator.
  Normalize this direction at the same two radii. On correct
  EOS prefixes in mixed groups, require the 0.5-norm EOS-aware
  write to improve mean termination probability over the
  original exact write by at least one percentage point.
  Require its mean content KL to stay within 10% of the
  original exact write. Record the quarter-radius comparisons
  and every matched prefix even if this directional gate fails.
- [x] Preserve the failed transfer panel for audit. The probe
  uses fixed adaptation prefixes and executes no new generation.
  The registered diagnostic SHA-256 is `00f774bc64bba78448d035138d512d3bc04c42ea20243f86e1b03bf3d5c593f9`.

The diagnostic matched both support digests and the unchanged
logits on 518 content and 64 EOS prefixes. The sampled EOS
contribution was 0.0693% of the content numerator's Frobenius
norm. Adding it gave cosine 0.99999976 with the original direction.

At the nine correct mixed-group EOS prefixes, the original
0.5-norm exact write assigned mean stop probability 99.8042%.
Adding EOS credit raised it by 0.0069 percentage points, with
mean content KL 99.70% of the original write's KL. The registered
one-point improvement gate failed. The selected EOS prefixes had
only 0.1958 points of headroom, so that threshold could never
pass on these prefixes. This limits inference about termination
earlier in a response.

| Direction | Content KL at norm 0.5 | Content KL at norm 0.125 | Changed supports at 0.5 / 0.125, out of 518 |
| --- | ---: | ---: | ---: |
| Exact | 0.04893 | 0.003086 | 500 / 391 |
| Compressed | 0.12531 | 0.006240 | 507 / 454 |
| Rotated | 0.10125 | 0.005724 | 518 / 476 |
| Exact including EOS | 0.04879 | 0.003083 | 499 / 391 |

Quarter-radius writes reduced fixed-support KL, even though most
actual top-k sets still changed. The next experiment in `hy8.3.7`
will calibrate write size on independent prefixes and test actual
completion lengths on fresh questions. The failed one-point
threshold and its source remain archived.

[Fixed-prefix EOS results, source and DSR receipts](../../results/qwen_eos_credit_2026-11-23.json) ·
[Per-prefix probabilities and divergence](../../results/qwen_eos_credit_2026-11-23.json.gz)

### Registered write-size calibration

- [x] Seal 56 new L1 questions with generator seed 811040.
  Calibration and verification each use three packing pairs
  and one register pair. The reserved transfer panel uses
  sixteen packing pairs and four register pairs. Audit all
  rational quotients and check disjointness against each other
  and the original ladder. The sealed panel SHA-256 is
  `c6ae7d997fe857ff4f0fdf271af73a1782ed5b1d377ce8a0d33879cc83c73b38`.
- [x] Keep the exact, compressed and rotated directions from
  the earlier adaptation run. Draw four baseline BF16 samples
  per calibration question with the same top-k sampler and cap.
  Use batch seeds from `stable_seed_u64(811041, salt="qwen.kl.<panel>.L1")`
  modulo `2**63`. Measure every 32nd scored content prefix
  and each sampled EOS. Require the unchanged reconstruction
  to match the actual retained sampler logits at every measured
  prefix.
- [x] Evaluate each direction at fractions 1/32, 1/16, 1/8, 1/4,
  1/2 and 1 of its original 0.5 norm. In both wording strata,
  require mean KL on the original support at most 0.005 nats
  and mean Jensen–Shannon divergence on the two actual masked
  policies at most 0.00125 nats. The latter compares the
  union of the supports and bounds mean total variation at
  0.05 by Pinsker's inequality and Jensen's inequality. Record
  actual total variation and changes in support separately.
  For each arm choose the largest consecutive passing fraction
  starting from the smallest tested fraction.
- [x] Freeze the chosen sizes and repeat the drift checks on the
  independent verification panel at those sizes only. Require
  all three arms to pass both budgets for each wording.
  Preserve the reserved transfer panel if any arm fails and
  keep the original selections in the record.
- [x] After verification passes, qualify the zero-write policy
  on the fresh transfer panel before comparing writes. Use the
  same sampler with four responses per question, and require
  at least 96/128 parseable packing responses and 8/32 mixed
  packing groups. For the primary transfer comparison, expect
  compressed to gain at least 8/128 packing answers over both
  zero and rotated with a one-sided exact pair-sign-flip
  `p ≤ 0.025` on both contrasts. Report exact-write accuracy,
  capped responses, actual lengths and the register controls.
  Use `stable_seed_u64(811042, salt="qwen.kl.transfer.L1.<offset>")`
  modulo `2**63` for each eight-question batch. Run zero, compressed
  and rotated on the same GPU. Run exact with a duplicate zero on
  the second RTX 4090 and check token agreement across the zero arms
  before interpreting the exact-arm comparison.

The registered calibration source SHA-256 is
`7794436112d6cbb4a91abaf492126cf5af5fe972417541ecbb2f289d1bd5567a`.
The registered transfer source SHA-256 is
`8d6dc2e452c16d1328183210e156ee89e9b1322ebf4d62beb39bd6afd194fe32`.

The panel and result audits checked all 56 questions and reproduced
the selection from 320 calibration prefixes. All three frozen sizes
cleared both budgets on 316 new verification prefixes. Packing used
249 verification prefixes and register used 67.

| Direction | Selected Frobenius norm | Verification packing KL | Verification packing JS | Dynamic total variation | Packing supports changed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact | 0.0625 | 0.001658 | 0.000435 | 0.00824 | 162/249 |
| Compressed | 0.03125 | 0.000889 | 0.000252 | 0.00620 | 152/249 |
| Rotated | 0.0625 | 0.002331 | 0.000606 | 0.01016 | 190/249 |

Register mean KL ranged from 0.000071 to 0.000196 and mean JS
from 0.000018 to 0.000050. Budget qualification establishes drift
control at sampled baseline prefixes. The fresh transfer run
tests the resulting accuracy and termination behavior.

Packing exceeded both budgets at the next larger grid point for
each direction. At twice the selected compressed norm, pooled
fixed-support KL was 0.00476, while packing alone reached 0.00586.
Pooled JS was 0.001259, just above its budget. These pooled
calculations are descriptive and did not change the frozen choices.

[Calibration results and receipts](../../results/qwen_kl_calibration_2026-11-23.json) ·
[Sealed panels and per-prefix drift](../../results/qwen_kl_calibration_2026-11-23.json.gz)

The fresh zero-write packing panel produced 60/128 correct
responses, 128/128 parseable responses and 28/32 mixed groups.
The register check scored 28/32 correct. None of the 160 zero-write
responses reached the token cap. The independent audit recovered
both packing qualification counts from the raw completions.

### Calibrated transfer results

| Transfer arm | Packing correct / 128 | Packing parseable / 128 | Packing hit limit | Median packing tokens | Register correct / 32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero | 60 | 128 | 0 | 312 | 28 |
| Exact | 65 | 128 | 0 | 308.5 | 28 |
| Rotated | 62 | 127 | 1 | 312 | 28 |
| Compressed | 61 | 127 | 1 | 320 | 28 |

Compressed gained one correct packing response over zero and
finished one behind rotated. The registered one-sided exact
paired p-values were 0.500 and 0.683 respectively. The primary
gain gate failed on both contrasts. The duplicate zero runs
matched token for token, and the audit verified all 800 sampled
responses including the duplicate. Independently recomputed
byte digests tied the three deployed matrices to their frozen
directions and chosen scales.

The calibrated writes produced at most one capped response per
arm on this fresh panel. The earlier pilot used different
questions and a 0.5 norm for each write. The new comparison used
the same drift ceilings for each direction with different
Frobenius norms. These experiments qualify the calibration
procedure and complete the registered transfer trial in
`hy8.3.7`. Native GSM8K qualification remains open.

An exploratory audit of the four mixed adaptation groups found
ten shared prefixes with different next actions and different
observed mean rewards across those actions. Thirty of 4,939
actions occurred at those prefixes, including sampled EOS in the
denominator. This measures observed sibling coverage in the
archived batch and motivates deliberately sampling continuations
from both actions at the same state in `hy8.3.8`.

[Transfer comparison and DSR receipts](../../results/qwen_kl_calibration_2026-11-23_transfer.json) ·
[All sampled transfer responses](../../results/qwen_kl_calibration_2026-11-23_transfer.json.gz)

### Registered sibling-continuation study

- [x] Seal 16 new paired L1 packing questions with seed 811050.
  Independently check 32 answers and zero overlap in wording
  or arithmetic recipes with the original ladder or KL panels.
  The panel SHA-256 is `5fec0aee39eaf223c620f0b6585c50e321ed1d5e52cb402f8b129451cdc59d4b`.
- [x] Use side zero of each pair for the probe pool. Roll in for
  at most 96 tokens with the original BF16 top-k sampler.
  Use base seed 811051 and salt `qwen.sibling.guide`. Consider
  positions 16 through 95, starting at zero. Prefer positions
  where the guide sampled a decimal digit and there are two
  distinct legal digits. Then consider any two legal nonterminal
  actions. In each class, rank by the smaller of the top two
  eligible probabilities, with earlier positions breaking ties.
  Try at most four positions per class. Require both actions
  to have policy mass at least 0.01 in the guide and in a
  canonical prefill with 16 identical rows. Freeze up to 12
  eligible probes in panel order before scoring any suffix.
  Stop if fewer than eight probes qualify.
- [ ] At each probe, draw eight suffixes from each sibling action
  in each of two blocks. Use base seeds 811052 for discovery and
  811053 for confirmation, salted with `qwen.sibling.<question ID>`.
  Check the packed support and retained logits against the frozen
  hash before forcing the first action. Let the usual sampler
  generate every subsequent token. The 512-token response cap
  includes the roll-in and forced action. Decode the whole
  response and compare the last boxed integer with the sealed
  answer.
- [ ] Orient each probe by the discovery difference in correct
  counts, and include it only if the absolute difference is at
  least two of eight. Require at least six such probes. On that
  fixed set, require at least six positive confirmation gaps,
  mean oriented confirmation difference of at least one correct
  response per eight, and one-sided exact sign-flip `p ≤ 0.025`.
  Report every probe, both blocks and capped responses.
  Require confirmation before any write experiment.

The registered sibling driver SHA-256 is
`dca7b9d70146239a0e08da68aed0281b6783042eb245f138be0fba9d676b00d9`.

The frozen manifest contains 12 probes, five digit pairs and
seven other action pairs. The weaker action's baseline mass
ranged from 0.0373 to 0.4685. The independent audit reproduced
the packed support hash for every probe before either suffix
block started. The frozen manifest SHA-256 is
`594852e00175781dcb345529436a7904a2356316ce85f553b0c3a26afcb49089`.
