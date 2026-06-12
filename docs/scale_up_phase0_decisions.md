# Scale-Up Phase 0 — Committed Decisions

> **Epic:** `bio_inspired_nanochat-hwxb` — Scale-Up to a Real, Useful Model on dual RTX 4090s.
> **This document is the committed output of Phase 0** and closes beads `hwxb.1.1` (dataset + capability),
> `hwxb.1.2` (model scale & architecture), and `hwxb.1.3` (success criteria & decision rule).
> It is **pre-registered**: Phases 2–4 must follow the spec below; deviations require an explicit bead + rationale.

All numbers here are grounded in the actual code: `scripts/base_train.py` derives model dims from a single
`depth` knob (`model_dim = depth·64`, `head_dim = 128`, `num_heads = ceil(model_dim/128)`), `vocab_size`
comes from the trained tokenizer (`tokenizer.get_vocab_size()`, default **65,536**), and embeddings are
currently **untied** (`gpt.py`: separate `wte` and `lm_head`). The eval harness is `scripts/eval_matrix.py`
(presets from `ablation_registry.py`), the working-memory probes live in `synthetic_tasks.py`, and the stats
layer is `bio_inspired_nanochat/eval_stats.py`.

---

## 1. Dataset + the capability that counts as "useful" (`hwxb.1.1`)

### Decision

- **Training data:** **FineWeb-Edu** (the real-English LM stream the repo already loads via
  `dataloader.py` / `dataset.py`). This gives a genuine language-modeling signal rather than a toy.
- **Primary "useful" capability (the headline target):** the **working-memory suite** —
  `synthetic_tasks.working_memory_suite` = associative **recall** by #pairs, variable **binding** by
  #distractors, and **NIAH** retrieval by context length. This is where the bio thesis (fast-weights /
  presynaptic memory / structural plasticity = "infinite local context") is *specifically* supposed to win.
- **Secondary metrics:** validation **bits-per-byte** (`loss_eval.evaluate_bpb`) on FineWeb-Edu, plus a
  small **qualitative generation** rubric (coherence of open-ended samples).

### Rationale

A small model (tens of millions of params) trained on raw FineWeb will be only *marginally* coherent, so a
pure-loss headline would **under-sell** the bio thesis: lower val loss is not what these mechanisms are for.
The mechanisms are about **memory and adaptation**, so the working-memory suite is the battlefield where bio
should beat vanilla *if it beats it anywhere*. We keep val bpb as the "is it a real LM at all" sanity signal
and generation samples as a human-readable smell test, but the **decision** rides on working memory.

### Realism caveat (carried from the bead thread)

A 20–90M model on raw FineWeb is weakly coherent. Two consequences, both already reflected below:
(1) we headline the working-memory capability, not raw coherence; (2) we keep seq_len ≥ 1024 so the
binding/NIAH probes are actually interesting (they need ≥512–1024 context to be non-trivial).

---

## 2. Model scale & architecture for 2×4090 (`hwxb.1.2`)

24 GB per card, **PCIe, no NVLink** → DDP all-reduce crosses PCIe, so we keep the model modest enough that
compute/communication stays favorable, and we spend the parameter budget on **depth/width, not the vocab
table**.

### 2.1 The vocab problem (decided lever: **tie embeddings**)

With untied embeddings and the default 65,536-token vocab, the embedding tables **dominate** a small model:

| depth | model_dim | transformer params ≈ `12·L·d²` | untied emb `2·V·d` | tied emb `V·d` |
|------:|----------:|-------------------------------:|-------------------:|---------------:|
| 8     | 512       | 25.2M                          | 67.1M              | 33.6M          |
| 10    | 640       | 49.2M                          | 83.9M              | 41.9M          |
| 12    | 768       | 84.9M                          | 100.7M             | 50.3M          |

At depth 10, untied embeddings (84M) are **larger than the entire transformer** (49M) — the param budget would
be wasted on the lookup table. **Decision:** for all scale-up runs, **tie** `wte` and `lm_head`
(`weight_tying`). Tying is the standard choice at this scale (it is also a mild regularizer) and reclaims
~40–80M params for depth/width. This is a small code lever (one shared `nn.Parameter`) and is filed as a
Phase-1 prerequisite (see "Code prerequisites" below). We keep the existing 65,536 tokenizer to avoid a
tokenizer-retrain detour; a smaller (16–32k) vocab is an *optional* future optimization, not on the critical
path.

### 2.2 The dense-vs-MoE fork (decided: primary **dense**, optional **MoE** side-track)

`SynapticConfig` mechanisms split cleanly by which model they need:

- **Dense** carries the **core bio claims**: presynaptic kinetics, online Hebbian fast-weights, neuromodulatory
  bus, differentiable learnable kinetics. These are the headline ablation.
- **MoE** is required *only* for **structural plasticity** (`uta.3` function-preserving split/merge) and
  expert metabolism, because there are no experts to split/merge in a dense MLP.

**Decision:** run a **PRIMARY DENSE track** for the headline bio-vs-vanilla comparison, and a **SMALLER,
OPTIONAL MoE track** dedicated to structural plasticity. This keeps the headline ablation tractable on
2×4090 instead of doubling every run. Each Phase-3 mechanism is explicitly assigned to a track below.

### 2.3 Committed configs

All use `head_dim=128`, GQA disabled (`n_kv_head == n_head`), tied embeddings, bf16 autocast, AdamW+Muon
(the existing `setup_optimizers`).

| config id | role | `--depth` | model_dim | seq_len | MoE | params (tied, ~) | purpose |
|-----------|------|----------:|----------:|--------:|----:|-----------------:|---------|
| **`S0` smoke** | fast iteration / CI | 8 | 512 | 1024 | no | ~59M | smoke trains, infra tests, regression |
| **`D1` primary dense** | headline bio-vs-vanilla | 10 | 640 | 1024 | no | ~91M | the real comparison |
| **`M1` MoE side-track** | structural plasticity only | 8 | 512 | 1024 | 8 experts, top-2 | ~59M dense + experts | test split/merge (`uta.3`) |

**Why depth 10 / ~91M for the primary:** big enough to be a real (if small) LM and to give the working-memory
mechanisms something to work with, small enough that a single 4090 holds it comfortably and DDP across two
cards just **doubles the effective batch** (good for PCIe — gradients are ~91M·2 bytes ≈ 182 MB/all-reduce,
overlappable with backward). Memory budget for `D1` on one 4090 (bf16 params + fp32 AdamW m,v + grads):
~91M·(2+4+4+2) ≈ **1.1 GB** of optimizer/param/grad state, leaving the vast majority of 24 GB for activations,
the **extra synaptic state** (per-key calcium/RRP/energy ≈ `B·H·T_key` per layer, plus rank-R fast-weights),
and KV cache. This is the headroom the bead asked for.

### 2.4 Code prerequisites this decision creates (filed for Phase 1)

1. **Weight tying** — **implemented** in `hwxb.2.9` as the opt-in `tie_embeddings` flag
   (`GPTConfig`/`GPTSynapticConfig`; `--tie_embeddings=1` in `base_train`). Default **off** (keeps
   the legacy untied behaviour + separate embedding/unembedding LRs); scale-up runs turn it on, which
   makes `D1` ~91M params (vs ~133M untied). The shared weight is optimized exactly once at the
   `unembedding_lr`, and `tie_weights()` re-establishes the tie after `load_state_dict(assign=True)`
   on resume. Tested in `tests/test_scaleup_tying.py`.
2. `base_train.py` does **not** expose most `SynapticConfig` fields via CLI; the eval/ablation harness owns
   preset→config wiring (already true for `eval_matrix.py`). Scale-up runs go through the harness, not raw
   `base_train` flags, for any non-default mechanism.

---

## 3. Success criteria, metrics & the bio-vs-vanilla decision rule (`hwxb.1.3`)

**Pre-registered before any scale-up training** (anti-p-hacking). Phases 2 and 4 reference this section verbatim.

### 3.1 What "genuinely useful" means (absolute bar, evaluated on `D1`)

The model is "useful" iff **both** hold on the held-out eval:

1. **It is a real LM:** validation **bpb** on FineWeb-Edu is meaningfully below a unigram/bigram floor and
   strictly decreasing across training (no divergence). Concretely: final `val_bpb` < **1.2** (a small but
   real LM target on this tokenizer; recorded and revisited once the vanilla baseline in Phase 2 sets the bar).
2. **It has working memory:** `working_memory_suite` **overall accuracy is above chance by a clear margin** —
   recall and binding both **> chance + 0.20**, and NIAH retrieval **> 0.50** at the model's max context.

(Chance levels are task-defined in `synthetic_tasks.py`: e.g. 1/vocab-ish for recall, 1/num-distractors for
binding. The suite already reports per-difficulty curves.)

### 3.2 The bio-vs-vanilla rule (the headline)

- **Equal-compute discipline — equalize on TOKEN BUDGET**, not wall-clock. Bio runs are slower per step, so
  matching wall-clock would *handicap bio on data*. We hold **params, data (same token budget + same shards +
  same seed schedule), and optimizer** fixed; bio simply costs more wall-clock and we report that cost
  honestly as a separate efficiency number. (Stated explicitly because the bead flags this as subtle.)
- **Seeds:** **3 seeds** per config — `{1337, 1338, 1339}` (matches `eval_benchmark_matrix.md`).
- **Statistics:** paired test (paired t **and** Wilcoxon signed-rank) **+ bootstrap and Student-t 95% CIs**,
  via `bio_inspired_nanochat/eval_stats.py` (the `74f.3` stats layer), direction-aware multi-seed aggregation.
- **"Bio helps" is declared iff** the **95% CI excludes 0 in the favorable direction on the PRIMARY metric**
  (working-memory-suite overall accuracy), across the 3 seeds. A bpb win is reported but is **secondary** and
  cannot by itself declare success.
- **Per-mechanism attribution:** the ablation matrix (`hwxb.5.1`, presets in `ablation_registry.py`:
  `bio_all`, `bio_no_presyn`, `bio_no_hebbian`, `bio_no_metabolism`, …) decomposes *which* mechanism carries
  any win, using the same CI rule on the delta vs `bio_all`.

### 3.3 Eval protocol (fixed, reproducible)

- Fixed eval seeds; fixed FineWeb train/val split (`dataset.parquet_paths_for_split`: last shard = val).
- Invocation through the harness: `python -m scripts.eval_matrix matrix --presets vanilla,bio_all,... --seeds 1337,1338,1339 --train-tokens <budget>`.
- NIAH lengths sized to the model context: `--niah-lengths 64,256,1024` for `D1`/`S0`/`M1` (seq_len 1024),
  with a fixed `--seed` for reproducible needle placement.
- Token budgets (from `eval_benchmark_matrix.md`): **Smoke 10M** (regression), **Short 100M** (usable signal),
  **Medium 500M** (research-grade headline). The headline bio-vs-vanilla verdict uses **≥ Short (100M)**, and
  **Medium (500M)** if the Short result is ambiguous (CI straddles 0).

### 3.4 Honest-null commitment

A **null or negative** result (bio does not beat vanilla, or is worse per equal-token) **will be reported as
such**. The epic's value is a trustworthy verdict, not a positive one. "No mechanism cleared its CI" is a valid,
publishable Phase-4 outcome.

---

## 4. What this unblocks

- `hwxb.2.7` (training recipe) — now has concrete configs (`S0`/`D1`/`M1`) and the tying decision to size LR/batch.
- `hwxb.3.x` (vanilla baseline) — `D1` is the model; §3.1 sets the bar to record.
- `hwxb.4.x` (scale-harden mechanisms) — §2.2 assigns each mechanism to the dense or MoE track.
- `hwxb.5.x` (ablation) — §3.2/§3.3 *are* the ablation's success rule and protocol.
