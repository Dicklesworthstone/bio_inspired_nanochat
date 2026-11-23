# Online fast-weight learning — status & characterization (bead sax.1)

_Last updated: 2026-10-07 (pre-registered decision below; measurement-regime update 2026-09-01; original note 2026-06-11, OrangeMill)._

## Decision (2026-10-07): not demonstrated; online Hebbian plasticity is now opt-in

The pre-registered deciding experiment ran to completion (`scripts/e2e/hebbian_chunked_regime.py
--budget preregistered --jobs 4`, protocol `hebbian-chunked-regime-v1-2026-09-02`, artifact
`results/hebbian_chunked_regime_2026-10-07_preregistered.json`): 2L/64d, associative recall with 2–16
pairs, 5 seeds × {Hebbian on, off} × {chunked (chunk 8), full} training, 2,000 AdamW steps each.

| Contrast (recall at 16 pairs, chance 0.010) | ON | OFF | ON − OFF |
|---|---:|---:|---:|
| **Pre-registered:** chunked train, chunked read, confirmation seeds 2–4 | 0.031 | 0.021 | **+0.010** (MDE 0.016) |
| Chunked train, chunked read, discovery seeds 0–1 | 0.039 | 0.063 | −0.023 |
| Full train, full read, confirmation seeds | 0.912 | 0.838 | +0.073 |
| Full train, full read, all 5 seeds (paired) | | | +0.034 ± 0.101 (exploratory, n.s.) |

Controls: OFF models read chunked = read full to 1.1e-5 (tolerance 1e-4); ON models' writes move the
logits by ≥ 0.67 (witness threshold 0.01); the OFF/full attention baseline reaches 0.995 at 2 pairs.
Countermetric: chunked training 0.132 s/step vs 0.122 full, final loss **3.24 vs 0.35**.

By the pre-registered rule the claim is **not demonstrated**: README demotes "infinite local context"
to a hypothesis and `enable_hebbian` moved from bio_all's default stack to the add-one-in set
(`SynapticConfig.enable_hebbian=False`; the D1 matrix tests it as `add_hebbian` and BDNF, which acts
only through it, as `add_bdnf`).

What the run teaches about the design: chunked truncated-BPTT training at chunk 8 never learned the
task in either arm (loss stayed near 3.2 while full-forward training reached 0.35), because each
chunk's loss cannot reach the keys and values of earlier chunks once the cache is detached. The
regime therefore could not have shown a fast-weight benefit even if one exists. A fair test needs a
chunked regime that back-propagates through the earlier chunks' attention (or a task whose retrieval
span fits inside one chunk), not a bigger budget of the same design.


This note records what the online Hebbian fast-weight ("fast-weight programmer") mechanism
**does** and **does not** do today, with measured numbers, so downstream beads build on facts
rather than the README's aspirational framing.

## The thesis

> "The model learns context online without a weight update": within a sequence, eligibility
> traces accumulate and consolidate into `W_fast`, so the model adapts to repeated/bound content
> as it reads — no SGD step.

## What is wired and working

- **Plasticity runs during training, autograd-safe** (vg9.2): eligibility traces + CaMKII/PP1/BDNF
  update immediately; the four `W_fast/W_slow/post.fast/post.slow` writes are deferred to the top
  of the next forward so backward never sees an in-place-mutated saved tensor.
- **Genuine rank-R eligibility traces** (vg9.9).
- **Per-sequence reset** (vg9.4): `GPTSynaptic.reset_sequence_state()` clears the per-sequence
  fast/eligibility state across all `SynapticLinear` layers (verified at the model level).

## The gap (measured)

On a tiny untrained `GPTSynaptic` (CPU), feeding a fixed token pattern repeatedly:

| Condition | Observation |
|---|---|
| Legacy control (`fast_weight_normalized=False`), `|Δw_fast|` over 8 passes | **~1e-7** — the raw rank-R Hebbian delta is `O(trace²)` and numerically negligible. |
| `y_fast` contribution to the output | **≈ 0** — gated, and added mid-network where the pre-`lm_head` norm suppresses it. Forcing `‖w_fast‖→1.0` still moves logits by only `~4e-3`. |
| Naive `post_fast_lr` boost (×3–×5) | **NaN** — positive feedback (`w_fast`→`y_fast`→activations→traces→`w_fast`), worsened by the un-decayed `w_slow` online drift. |
| Adapt on pattern P, then loss on P vs novel Q | `ΔlossP ≈ −0.0017` (slightly **worse**), `ΔlossQ ≈ +0.0003`. **No predictive specificity.** |

**Conclusion:** unsupervised Hebbian auto-association amplifies the layer's own (untrained, ~random)
response. It does **not** by itself improve next-token prediction of repeated content — the
"improves prediction" half of sax.1's acceptance is **not** met on an untrained model, and is not
reachable by hyperparameter tuning alone.

## What this bead delivered (the foundational fix)

`SynapticConfig.fast_weight_normalized` defaults **on** after the raw write caused the reproducible
all-bio Synaptic-MoE divergence tracked by **jpqc**. The matrix writes use the unit-norm Hebbian
direction, with step sizes `fast_weight_eta` and `post_slow_lr`. The fast matrix is capped at
`fast_weight_max_norm` and stayed within that cap over the original 200 repeated passes.
The molecular fast and slow gains subsequently exposed another feedback path during training.
In the original six-seed continual-learning comparison, 11 of 12 biological fits failed during
their third task. The corrected normalized rule passes $UV/\max(1,\lVert UV\rVert_F)$ to the molecular
updates. Small eligibility signals retain their amplitude, and large signals retain their direction.
The raw-update control remains available with `fast_weight_normalized=False`. [Original failures and numerical diagnosis](theory/gauge_consolidation.md)

## What's needed for the behavioral claim (downstream)

1. **A learning signal that shapes the fast write toward _correct_ predictions** — three-factor
   (reward-/error-modulated) Hebbian, bead **hy8.2**. Unsupervised correlation is direction-blind;
   a third factor turns "amplify whatever I output" into "amplify what reduces loss".
2. **A chunked-sequence training/eval regime** so fast-weights carry _within_ a sequence across
   forwards (single-forward-per-batch gives attention, not fast-weights, the cross-context job).
   _2026-09-01: the evaluation half exists (`retrieval_accuracy(chunk_len=…)`, see below).
   2026-09-02: the training half exists too — `GPTSynaptic.chunked_train_step` behind
   `--hebb_chunk_len` in `base_train`/`eval_matrix` (bead hwxb.8); the deciding ON-vs-OFF
   experiment under it is bead hwxb.9._
3. **Trained-model e2e validation** on the working-memory suite — bead **eqyk.9**.

## Ordering note for the team

sax.1's behavioral acceptance effectively depends on hy8.2, but the graph currently has hy8.2
depending on sax.1. sax.1's _mechanism_ pieces (this note + the normalized write + reset) are the
right substrate for hy8.2; the "improves next-token prediction" validation should land with hy8.2 /
eqyk.9, not be claimed here.

## 2026-09-01 update — the measurement was blind, and noisy

Two defects in how the mechanism was *measured* were found while wiring the chunked probe. Neither
changes the mechanism; both change what the numbers above and in `hwxb.4.4` mean.

**1. Every probe read the sequence in one forward, so it could not see the writes.** In a single
teacher-forced forward the Hebbian writes are deferred (training) or applied after the matmuls
(inference), so nothing written while the key/value pairs are read can influence the query position
of the same forward. `retrieval_accuracy`, the working-memory suite, `fast_weight_comparison_bench`
and the held-out-loss comparison in `hwxb.4.4` all ran in this regime. Their ON = OFF result is
therefore not evidence about the mechanism. `synthetic_tasks.retrieval_accuracy(..., chunk_len=k)`
and the suite now read the sequence through a KV cache `k` tokens at a time, the regime in which
generation actually runs; `tests/test_working_memory_chunked_eval.py` shows the write routine runs
once per chunk per synaptic linear and changes later logits.

**2. Every probe ran with stochastic vesicle sampling on.** `GPTSynaptic.forward(train_mode=True)`
was the default and also enabled stochastic release; two identical eval-mode forwards differed by up
to 0.067 in logits. The default now follows the module's training flag, and the probes read with
`train_mode=False, update_mem=True` (deterministic, plasticity live).

**First numbers under the fixed regime** (3 seeds each, CPU, 2 layers × 64 dims, associative
recall with 2/4/8 pairs, batch 64 per pair count, mean over pair counts; read-only probe, no
chunked training; script kept out of the repo until the G2 training regime exists):

| Model (seeds 0/1/2) | full-forward read | `chunk_len=1` read |
|---|---|---|
| untrained, Hebbian on | 0.005 / 0.005 / 0.021 (chance ≈ 0.010) | identical to full |
| untrained, Hebbian off | 0.010 / 0.016 / 0.016 | identical to full |
| trained 300 steps under full forwards, Hebbian on | 0.224 / 0.188 / 0.208 | 0.219 / 0.193 / 0.208 |
| trained 300 steps under full forwards, Hebbian off | 0.198 / 0.276 / 0.161 | identical to full |

Hebbian off is bit-identical between the two read regimes, as it must be; Hebbian on differs by at
most 0.005, inside seed noise (±0.04); ON and OFF are indistinguishable (means 0.207 vs 0.212).

## 2026-09-02 — pilot of the chunked TRAINING regime (bead hwxb.9; no decision)

`scripts/e2e/hebbian_chunked_regime.py --budget pilot` (2 discovery seeds, 300 steps, pairs 2–8,
chunk 4; `results/hebbian_chunked_regime_2026-09-02_pilot.json`). Recall accuracy at 8 pairs,
chance ≈ 0.010:

| Training | Read | Hebbian on (s0 / s1) | Hebbian off (s0 / s1) |
|---|---|---|---|
| chunked (chunk 4) | chunked | 0.000 / 0.125 | 0.016 / 0.047 |
| chunked (chunk 4) | full | 0.000 / 0.125 | 0.016 / 0.047 |
| full | full | 0.188 / 0.172 | 0.188 / 0.063 |
| full | chunked | 0.172 / 0.156 | 0.188 / 0.063 |

Effect at 8 pairs under chunked training and reading: +0.031 (ON − OFF), against a 3σ minimum
detectable effect of 0.047 from the OFF arm's seed spread — not detectable at this budget. Chunked
training costs 2.1× the step time and +0.29 final loss (truncated back-propagation at chunk
boundaries). Controls: the planted witness is visible (ON |Δlogit| ≥ 0.80), the OFF arm's read
regimes agree to ≤ 4e-6 (the pilot predates the 1e-4 tolerance now in the script; its JSON flag
reflects the old `== 0` rule), and the attention baseline reads 0.45 at 2 pairs. The pilot triggers
no decision by design; the pre-registered run (5 seeds, 2,000 steps, pairs to 16, chunk 8) was
started 2026-09-02 14:25 UTC and its artifact will carry the decision.

A model trained with full forwards never sees its own writes during training, so its slow weights
have no reason to exploit them at read time; the chunked read changes nothing. That is the expected
result of this probe and it is **not** the experiment that decides the claim. The deciding experiment
(bridge plan G2) trains under the chunked regime — the loss is accumulated over `k`-token chunks with
the deferred writes landing between them — and compares Hebbian ON vs OFF under chunked reading, with
the throughput cost beside the effect. Until that runs, the "infinite local context" claim is
**unmeasured**, not null.
## Matched intervention on trained Hebbian memory

The previous five-seed toy study measured the entire memory mechanism on
versus off. This experiment asks whether ordinary language-model training
learns to use the **actual assignment of activations** in the online write.
Both models have the same parameters and write capacity. The control
shuffles only the association between pre- and postsynaptic rows, using one
permutation for both trace factors in a block. The original normalized
production write and all of its downstream dynamics execute in both arms.
This is the behavioral follow-up to the raw-product identity in
[the permutation study](episodic_write_policy.md#a-matched-control-for-nanochats-raw-hebbian-write).
Related architectures learn to use fast weights and test-time updates through
different training rules. [Fast-weight programmers](https://proceedings.mlr.press/v139/schlag21a.html) · [Test-time training layers](https://arxiv.org/abs/2407.04620)

### Registered experiment

- [ ] Pin the implementation, tokenizer, staged train and validation shards,
  GPU identity, torch build and both random streams. Seeds are 6311–6314.
  Within each seed, initialize both training arms identically. Use a
  depth-two, width-192, three-head, tied-head GPTSynaptic with a 1,024-token
  context, Hebbian memory enabled, STDP, presynaptic release and metabolism
  disabled, and zero online slow consolidation. Retain the normalized
  transient write, its default learning and decay rates, and the postsynaptic
  modulation. Reset fast state between independent training sequences.
  Initialize the tied embedding at a standard deviation of 0.02. The
  training-only apparatus run with the ordinary unit-scale tied embedding
  started above 22 nats per token. A shared small initialization avoids
  starting either AdamW arm in widespread softcap saturation.
- [ ] Exercise the real chunked-backpropagation path on a time-boxed
  apparatus run. Use one sequence per step and 128-token chunks. Verify that
  the fresh model's first chunk is identical before an intervention can take
  effect, both active read arms perform the same number of writes, and a
  later chunk registers a change. Verify lane-local marginal preservation
  and restoration of the original method.
- [ ] Train two models per seed for 768 AdamW updates at `3e-4`, no weight
  decay, global gradient clipping at 1, with an identical 786,432-token
  training exposure per arm. Check loss, gradients, checkpoints, write
  activity, memory and wall time. Use independently seeded uniform
  permutations for the training control, shared across all write sites
  within a chunk. Use two cards as independent single-process workers.
  Before seeing held-out results, measure steps 5–16 in each training arm.
  Admit the registered run if the slower arm's mean time times 3,072
  fits 2,400 seconds per card. If it exceeds that budget, report the shortfall
  and test the same contrast using the largest admitted exposure in the
  fixed sequence 128 then 64 updates (131,072 or 65,536 tokens per arm).
  Apply the same time rule to the shorter exposures. Select the exposure
  from training time alone, before reading any held-out score. Conclusions
  from a shortened run apply to that early stage of training.
- [ ] Score every checkpoint on the same 128 complete or prefix-truncated
  documents drawn in order from the midpoint validation row group and
  subsequent groups. Use the first at most 512 tokens of each document,
  require at least 256 tokens, reject duplicate document hashes, and reset
  state for each article. Start with BOS and score only positions after the
  first complete 128-token chunk. Both trained arms and the common initial
  model get an identity read, four independent shuffled reads and a frozen
  read. Average the four shuffled probabilities before taking the coding
  loss. Record the average shuffled loss too, to expose the Jensen gap.
  Also score the existing associative-recall task at 4 and 8 pairs using
  64 held-out examples apiece. These probe exact recall using randomized
  symbols and serve as descriptive checks on models trained on text. Pad with
  BOS between the examples and query to put the query in the next chunk.
- [ ] Publish complete per-document bits, bytes, first-block invariance,
  checkpoint and input digests, probe counts, per-arm throughput and all
  statistical decisions. Advance a claim of learned pairing only if the
  paired-trained identity read saves at least 0.005 BPB against **both**
  the paired-trained shuffle ensemble and its frozen read, and the
  pairing benefit grows by at least 0.005 BPB over the common initialization
  and over the paired-vs-shuffled read gap of the shuffle-trained model.
  Require positive savings on all four seeds for each contrast and a
  positive one-sided 98.75% lower bound from 32,768 paired document
  bootstrap draws over the seed-averaged loss differences. The document
  bounds condition on these four seeds. Report other read contrasts as
  descriptive.
- [ ] Recompute every decision from the raw held-out scores, perform static
  and runtime qualification through DSR, and close only the verified scope
  in Beads. D1 scale and reward-gated training retain distinct acceptance.

### Running the experiment

Use a locked environment and a fresh output directory. Set
`NANOCHAT_BASE_DIR` to the cache with the archived 28 shards and project
tokenizer. Run the following commands as DSR quality checks.

```sh
uv run --no-sync python -m scripts.e2e.trained_hebbian_binding probe \
  --seeds 6311 --out "$PROBE"

CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m scripts.e2e.trained_hebbian_binding train \
  --seeds 6311 6312 --steps "$ADMITTED_STEPS" --out "$STUDY"
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -m scripts.e2e.trained_hebbian_binding train \
  --seeds 6313 6314 --steps "$ADMITTED_STEPS" --out "$STUDY"

CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m scripts.e2e.trained_hebbian_binding score \
  --seeds 6311 6312 --out "$STUDY"
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -m scripts.e2e.trained_hebbian_binding score \
  --seeds 6313 6314 --out "$STUDY"
uv run --no-sync python -m scripts.e2e.trained_hebbian_binding decision --out "$STUDY"
```

The paired and shuffled checkpoints must share the same initialization and
training-stream hashes within each seed. Scoring checks each checkpoint's file
and tensor digests before reading it. Each shuffled score averages four
draws of the normalized update, including each draw's effect on later
activations. This estimates a finite ensemble, which differs from the
analytic mean of a raw single write conditional on fixed activations.

