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

- [ ] Load the same archived two-layer, width-192 FineWeb checkpoint
  for both arms and verify its weights and tokenizer. One arm learns
  worked solutions with the production token masks for calculator
  outputs. The other learns just the official final answer in
  `#### number` format. Keep the question text and start state matched.
- [ ] Use official revision `740312add88f781978c0658806c59bc2815b9866`.
  Take selection questions from shuffled training indices [128, 192)
  and confirmation questions from [192, 256). Train on indices
  [256, 7473), excluding question text duplicates of either panel
  and worked solutions longer than 512 rendered tokens. Apply the
  same exclusions to both arms. Retain the official test split
  for later evaluation.
- [ ] Train for 384 updates with 32 questions per update, microbatches
  of eight and fixed shuffled epochs. Use AdamW with peak learning
  rate 0.001, 16 warmup steps, cosine decay to 0.0001, weight decay
  0.0001, gradient norm cap 1 and BF16 autocast. Supervise assistant
  and sampled calculator tokens and exclude prompt and forced
  calculator output tokens. Disable ordinary synaptic writes in
  both arms. Save final checkpoints with the production writer
  and reload using the production reader.
- [ ] Sample four unforced completions per question with the production
  engine, temperature 1, top-k 50 and a 160-token cap. On each panel,
  require at least 24 correct completions out of 256, at least 16 mixed
  reward groups out of 64, and correct answers of at least four distinct
  values. Require more correct completions than the most frequent
  training answer and at least eight more than scoring the very same
  outputs against gold labels rotated by 17 questions. Test the paired
  differences across questions with a one-sided exact sign-flip
  `p ≤ 0.025`. A checkpoint qualifies only if it passes both panels.
  If both qualify, prioritize the higher selection score.
- [ ] Independently reconstruct every reward and qualification
  decision. If any arm supplies mixed reward groups, replay an actual
  group through the exact final-matrix score and compare with full
  masked autograd. Keep task qualification and gradient parity as
  separate results. Publish the complete denominators and checkpoint
  identities.
