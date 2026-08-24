# Online fast-weight learning — status & characterization (bead sax.1)

_Last updated: 2026-06-11 (OrangeMill)._

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
all-bio Synaptic-MoE divergence tracked by **jpqc**. Both online Hebbian writes step along the **unit-norm** Hebbian direction
(`fast_weight_eta` for `w_fast`, `post_slow_lr` for `w_slow`) and `‖w_fast‖` is capped by
`fast_weight_max_norm`. This makes the update **impactful** (`|Δw_fast| ~ O(eta)`, not 1e-7) **and
stable** (finite & bounded over 200 repeated passes — where the naive boost NaNs). It is the
prerequisite for any consolidation signal to actually move the fast weights; it does **not** on its
own make the adaptation predictive. Setting the flag false retains the historical raw path only as
a negative-control ablation. Tests: `tests/test_online_fast_adaptation.py` and the multi-seed MoE
stability regression in `tests/test_e2e_train_bio.py`.

## What's needed for the behavioral claim (downstream)

1. **A learning signal that shapes the fast write toward _correct_ predictions** — three-factor
   (reward-/error-modulated) Hebbian, bead **hy8.2**. Unsupervised correlation is direction-blind;
   a third factor turns "amplify whatever I output" into "amplify what reduces loss".
2. **A chunked-sequence training/eval regime** so fast-weights carry _within_ a sequence across
   forwards (single-forward-per-batch gives attention, not fast-weights, the cross-context job).
3. **Trained-model e2e validation** on the working-memory suite — bead **eqyk.9**.

## Ordering note for the team

sax.1's behavioral acceptance effectively depends on hy8.2, but the graph currently has hy8.2
depending on sax.1. sax.1's _mechanism_ pieces (this note + the normalized write + reset) are the
right substrate for hy8.2; the "improves next-token prediction" validation should land with hy8.2 /
eqyk.9, not be claimed here.
