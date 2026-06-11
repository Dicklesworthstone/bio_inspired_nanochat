# Scale-Up Training Recipe (validated)

> **Bead:** `bio_inspired_nanochat-hwxb.2.7`. The training recipe — LRs, schedule, batch,
> optimizers, grad-clip, horizon — adapted from the proven nanochat recipe to the Phase-0
> configs (`S0`/`D1`/`M1`, see `docs/scale_up_phase0_decisions.md`) and the 2×4090 effective
> batch. **The same recipe is used identically by the vanilla baseline and every bio run**, so
> the ablation comparison is fair. A bad recipe makes a small model useless regardless of bio
> mechanisms — this fixes it once.

## The recipe at a glance (config `D1`, the headline dense model)

| Knob | Value | Source / rationale |
|------|-------|--------------------|
| Optimizers | **AdamW** (embeddings, `lm_head`, all 1-D/0-D params) + **Muon** (2-D matrices) | the repo's proven split (`gpt.py`/`gpt_synaptic.py` `setup_optimizers`) |
| AdamW betas / eps | `(0.8, 0.95)` / `1e-10` | nanochat default |
| `embedding_lr` | `0.2` × `s` | base × d-model scale |
| `unembedding_lr` | `0.004` × `s` | base × d-model scale |
| `matrix_lr` (Muon) | `0.02` × `s` | base × d-model scale |
| d-model LR scale `s` | `(model_dim/768)**-0.5` = **1.095** for `D1` (model_dim 640) | `setup_optimizers` scales AdamW LRs ∝ 1/√(d/768) |
| Muon momentum | warms **0.85 → 0.95 over the first 300 steps** | `get_muon_momentum` |
| Weight decay | `0.0` | nanochat default (the bio params want freedom to move) |
| Grad clip | **1.0** | `clip_grad_norm_`; with the divergence guard below |
| LR schedule | **WSD (warmup-stable-decay), trapezoidal** | `get_lr_multiplier`: linear warmup → constant → linear warmdown |
| Warmup | `warmup_ratio = 0.1` (~95 steps at the `D1` horizon) | default is 0.0 (none); **add warmup at scale** so Muon/kinetics settle |
| Warmdown | `warmdown_ratio = 0.2`, `final_lr_frac = 0.0` | nanochat default (decay to 0 over the last 20%) |
| Precision | **bf16 autocast** on CUDA, `logit_softcap = 15.0` | bounds the unbounded `log(ε+release)` attention bias |
| Divergence guard | `vg9.7` on, with LR backoff on a loss spike | NaN/Inf detection + gentler step; protects long runs |

> Note the schedule is **WSD/trapezoidal, not cosine** — `get_lr_multiplier` does linear warmup
> to 1.0, holds, then linearly decays over the last `warmdown_ratio` of the horizon to
> `final_lr_frac`. That is the recipe the repo already ships; we keep it and only set warmup.

## Batch & gradient accumulation (2×4090)

| Config | `device_batch_size` | seq_len | world_size | grad_accum | `total_batch_size` (tokens) |
|--------|--------------------:|--------:|-----------:|-----------:|-----------------------------:|
| `S0` (smoke, d8) | 16 | 1024 | 1–2 | derived | 131072 (0.125M) |
| `D1` (headline, d10) | 16 | 1024 | 2 | **8** | **262144 (0.25M)** |
| `M1` (MoE, d8) | 8 | 1024 | 2 | derived | 131072 (0.125M) |

`base_train.py` derives `grad_accum_steps = total_batch_size / (device_batch_size × seq_len × world_size)`
and asserts divisibility. For `D1`: `262144 / (16 × 1024 × 2) = 8`. We use **0.25M tokens/step** for the
~91M `D1` model (smaller than nanochat's 0.5M default, which targets larger models) — a small model
trains better with a smaller batch at fixed token budget. The effective batch doubles for free with the
2nd GPU (DDP averages gradients; see `docs/scale_up_ddp.md`).

## Horizon — the equal-compute basis (FIXED HERE)

The ablation equalizes on **token budget** (Phase-0 §3.2). The headline `D1` comparison is fixed at:

- **Headline budget: 500M tokens** (~1900 steps at 0.25M tokens/step) — "Medium" in `eval_benchmark_matrix.md`.
- Fast-iteration: **100M tokens** ("Short", ~380 steps) for quick signal / regressions.
- Chinchilla-optimal for a 91M model is ~1.8B tokens; 500M is a deliberate, fixed, equal-compute
  basis (both vanilla and bio train for **exactly** this budget; bio simply costs more wall-clock,
  reported separately). Do **not** change it per-arm — that would break the comparison.

## Validation (e2e harness LR sweep)

Ran the reduced synaptic e2e (`bio_inspired_nanochat.e2e_harness`, 80 steps, tiny model — a proxy for
LR *magnitude*, single-AdamW) across learning rates; the invariant battery gives a clean trainable band
and a divergence boundary:

| proxy lr | final loss | max grad-norm | verdict |
|---------:|-----------:|--------------:|---------|
| 1e-3 | 2.94 | 5.8 | under-trains (loss high; online fast-weights drift) |
| **3e-3** | **1.27** | 7.1 | **trains cleanly — all invariants pass** |
| **1e-2** | **0.38** | 3.9 | **trains cleanly — all invariants pass** |
| 3e-2 | NaN | 670 | **diverges** (loss NaN, grad explosion) |

Takeaways encoded in the recipe: (1) there is a clean ~3×-wide LR band before divergence, so the
chosen `matrix_lr`/`embedding_lr` magnitudes (with the d-model scaling) sit safely inside it; (2)
divergence is abrupt (3e-2 → NaN), which is exactly why **grad-clip 1.0 + the `vg9.7` divergence guard
with LR backoff** are part of the recipe, not optional; (3) too-low LR under-trains a small model — the
risk for a small model is under-training, not just divergence.

## Launch (the committed recipe)

`D1` headline, dual-4090 (see `docs/scale_up_ddp.md` for NCCL/PCIe env):

```bash
export NCCL_P2P_LEVEL=PXB OMP_NUM_THREADS=8
.venv/bin/torchrun --standalone --nproc_per_node=2 -m scripts.base_train \
  --depth=10 --max_seq_len=1024 \
  --device_batch_size=16 --total_batch_size=262144 \
  --warmup_ratio=0.1 --warmdown_ratio=0.2 --final_lr_frac=0.0 \
  --grad_clip=1.0 \
  --embedding_lr=0.2 --unembedding_lr=0.004 --matrix_lr=0.02 --weight_decay=0.0 \
  --target_param_data_ratio=-1 --num_iterations=1900   # 1900*0.25M ≈ 500M tokens
```

Vanilla baseline (`hwxb.3.1`) and every bio arm (`hwxb.5.x`) use the **same** flags, toggling only
`--synapses` and the mechanism's `SynapticConfig` field via the eval/ablation harness. The recipe
config plumbing (LRs → both optimizers, with d-model scaling) is unit-tested in
`tests/test_scaleup_recipe.py`.
