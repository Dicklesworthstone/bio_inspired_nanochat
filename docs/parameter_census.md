# Parameter Census — `SynapticConfig`

> **Generated** by `scripts/param_census.py` (bead `bio_inspired_nanochat-8j9.6`). Do not hand-edit; re-run `uv run python -m scripts.param_census`. Machine-readable companion: [`parameter_census.json`](./parameter_census.json).

`SynapticConfig` has **108 fields** — **108 LIVE** (read by runtime code) and **0 DEAD** (declared, read by nothing). This is the ground truth behind the README's *“48-parameter genome”* framing, which conflated three different counts.

## What the counts actually are

- **The learned genome is 4-D, not 48.** The biological 'genome' is the learned per-expert Xi vector (xi_dim=4), expanded by one shared learned decoder to six bounded phenotype kinetics (fatigue/recovery rates, CaMKII/PP1 gains, calcium retention/influx). Set xi_dim=0 for learned shared kinetics without per-expert Xi. The genome is NOT the SynapticConfig hyperparameters; every field here is a fixed hyperparameter, not a learned weight.

- **The wired search space is 10 params**, not 48. CMA-ES Phase 1 (`TOP10_PARAM_SPECS` in `scripts/tune_bio_params.py`) tunes: `alpha_ca`, `complexin_bias`, `doc2_gain`, `lambda_loge`, `nsf_recover`, `prime_rate`, `syt_fast_kd`, `syt_slow_kd`, `tau_c`, `unprime_per_release`. The 48-/82-parameter figures are the *aspirational* two-phase plan, not shipping code.

- **The config surface is 108 hyperparameters**, every one of which is read by runtime code — `8j9.5` pruned the last dead fields (`enabled`, `camkii_down`, `router_sim_threshold`, `native_metrics`, `native_plasticity`); `jyb.2` later reintroduced `native_presyn` only after wiring it to live decode.


## Dead fields (read by nothing)

None — every `SynapticConfig` field is read on some runtime path (invariant enforced by `tests/test_param_census.py`).


## Full census by subsystem


### `general` (7/7 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `granularity` | `per_connection` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:213 |
| `rank_eligibility` | `8` | LIVE |  | bio_inspired_nanochat/synaptic.py:2407 |
| `attn_topk` | `32` | LIVE |  | bio_inspired_nanochat/synaptic.py:1046 |
| `stochastic_train_frac` | `0.12` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:217 |
| `stochastic_mode` | `normal_reparam` | LIVE |  | bio_inspired_nanochat/certificate_bundle.py:2699 |
| `stochastic_tau` | `1.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:2058 |
| `stochastic_count_cap` | `8` | LIVE |  | bio_inspired_nanochat/synaptic.py:1047 |

### `presynaptic` (13/13 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `tau_c` | `6.0` | LIVE | ✓ | bio_inspired_nanochat/ablation_registry.py:245 |
| `learnable_kinetics` | `False` | LIVE |  | bio_inspired_nanochat/synaptic.py:1191 |
| `differentiable_recurrence` | `False` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:266 |
| `recurrence_block_size` | `64` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:267 |
| `recurrence_chunk_len` | `0` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:271 |
| `recurrence_checkpoint_len` | `0` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:275 |
| `doc2_gain` | `0.08` | LIVE | ✓ | bio_inspired_nanochat/flex_synaptic.py:52 |
| `prime_rate` | `0.075` | LIVE | ✓ | bio_inspired_nanochat/synaptic.py:1060 |
| `unprime_per_release` | `0.05` | LIVE | ✓ | bio_inspired_nanochat/synaptic.py:1061 |
| `nsf_recover` | `0.08` | LIVE | ✓ | bio_inspired_nanochat/synaptic.py:1062 |
| `rec_rate` | `0.06` | LIVE |  | bio_inspired_nanochat/synaptic.py:1059 |
| `endo_delay` | `3` | LIVE |  | bio_inspired_nanochat/synaptic.py:2174 |
| `metriplectic_integrator` | `False` | LIVE |  | bio_inspired_nanochat/certificate_bundle.py:2845 |

### `initial_state` (6/6 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `init_rrp` | `6.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:3014 |
| `init_reserve` | `18.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:3015 |
| `init_snare` | `0.7` | LIVE |  | bio_inspired_nanochat/synaptic.py:3016 |
| `init_clamp` | `0.6` | LIVE |  | bio_inspired_nanochat/synaptic.py:3017 |
| `init_amp` | `1.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:3020 |
| `init_energy` | `0.85` | LIVE |  | bio_inspired_nanochat/synaptic.py:3018 |

### `energy` (3/3 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `energy_fill` | `0.02` | LIVE |  | bio_inspired_nanochat/hf_bio_adapter.py:165 |
| `energy_use` | `0.02` | LIVE |  | bio_inspired_nanochat/hf_bio_adapter.py:168 |
| `energy_max` | `1.0` | LIVE |  | bio_inspired_nanochat/hf_bio_adapter.py:166 |

### `attention` (4/4 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `lambda_loge` | `1.0` | LIVE | ✓ | bio_inspired_nanochat/synaptic.py:3352 |
| `barrier_strength` | `0.1` | LIVE |  | bio_inspired_nanochat/tropical_certificate.py:1478 |
| `epsilon` | `1e-06` | LIVE |  | bio_inspired_nanochat/synaptic.py:2341 |
| `loge_bias_clamp` | `10.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:3357 |

### `kernel_compat` (18/18 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `tau_buf` | `4.0` | LIVE |  | bio_inspired_nanochat/cusp_certificate.py:114 |
| `tau_prime` | `5.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:2277 |
| `tau_rrp` | `40.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:2278 |
| `tau_energy` | `50.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:2279 |
| `alpha_ca` | `0.55` | LIVE | ✓ | bio_inspired_nanochat/ablation_registry.py:247 |
| `alpha_buf_on` | `0.1` | LIVE |  | bio_inspired_nanochat/certificate_bundle.py:2718 |
| `alpha_buf_off` | `0.1` | LIVE |  | bio_inspired_nanochat/certificate_bundle.py:2720 |
| `alpha_prime` | `0.1` | LIVE |  | bio_inspired_nanochat/synaptic.py:2316 |
| `alpha_unprime` | `0.1` | LIVE |  | bio_inspired_nanochat/synaptic.py:2350 |
| `alpha_refill` | `0.1` | LIVE |  | bio_inspired_nanochat/synaptic.py:2317 |
| `energy_in` | `0.01` | LIVE |  | bio_inspired_nanochat/synaptic.py:2319 |
| `energy_cost_rel` | `0.015` | LIVE |  | bio_inspired_nanochat/synaptic.py:2353 |
| `energy_cost_pump` | `0.01` | LIVE |  | bio_inspired_nanochat/synaptic.py:2354 |
| `syt_fast_kd` | `0.4` | LIVE | ✓ | bio_inspired_nanochat/flex_synaptic.py:50 |
| `syt_slow_kd` | `1.0` | LIVE | ✓ | bio_inspired_nanochat/flex_synaptic.py:51 |
| `complexin_bias` | `0.0` | LIVE | ✓ | bio_inspired_nanochat/flex_synaptic.py:54 |
| `qmax` | `2.0` | LIVE |  | bio_inspired_nanochat/flex_synaptic.py:57 |
| `q_beta` | `1.0` | LIVE |  | bio_inspired_nanochat/flex_synaptic.py:57 |

### `postsynaptic` (17/17 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `post_fast_decay` | `0.95` | LIVE |  | bio_inspired_nanochat/synaptic.py:2590 |
| `post_fast_lr` | `0.0015` | LIVE |  | bio_inspired_nanochat/synaptic.py:2590 |
| `post_slow_lr` | `0.0005` | LIVE |  | bio_inspired_nanochat/synaptic.py:2578 |
| `post_trace_decay` | `0.96` | LIVE |  | bio_inspired_nanochat/certificate_bundle.py:2714 |
| `fast_weight_normalized` | `True` | LIVE |  | bio_inspired_nanochat/synaptic.py:2812 |
| `fast_weight_eta` | `0.5` | LIVE |  | bio_inspired_nanochat/synaptic.py:2824 |
| `fast_weight_max_norm` | `1.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:2826 |
| `camkii_up` | `0.05` | LIVE |  | bio_inspired_nanochat/synaptic.py:2490 |
| `pp1_tau` | `0.985` | LIVE |  | bio_inspired_nanochat/synaptic.py:2491 |
| `camkii_thr` | `1.0` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:256 |
| `pp1_thr` | `0.7` | LIVE |  | bio_inspired_nanochat/synaptic.py:2488 |
| `bdnf_tau` | `0.985` | LIVE |  | bio_inspired_nanochat/synaptic.py:2505 |
| `bdnf_scale` | `1.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:2570 |
| `bdnf_gamma` | `0.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:2570 |
| `bdnf_hebb_accumulate` | `True` | LIVE |  | bio_inspired_nanochat/synaptic.py:2502 |
| `bdnf_max` | `10.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:2511 |
| `plasticity_during_training` | `True` | LIVE |  | bio_inspired_nanochat/synaptic.py:2969 |

### `latch` (14/14 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `bistable_latch` | `False` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:249 |
| `latch_ltd_thr` | `0.5` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:256 |
| `latch_input_gain` | `12.0` | LIVE |  | bio_inspired_nanochat/certificate_bundle.py:2726 |
| `latch_alpha_ca` | `0.6` | LIVE |  | bio_inspired_nanochat/cusp_certificate.py:50 |
| `latch_beta_pp1` | `1.0` | LIVE |  | bio_inspired_nanochat/cusp_certificate.py:51 |
| `latch_gamma_auto` | `0.45` | LIVE |  | bio_inspired_nanochat/cusp_certificate.py:52 |
| `latch_hill_n` | `6.0` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:250 |
| `latch_hill_k` | `0.6` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:252 |
| `latch_alpha_pp1` | `0.5` | LIVE |  | bio_inspired_nanochat/cusp_certificate.py:274 |
| `latch_beta_camkii` | `0.3` | LIVE |  | bio_inspired_nanochat/cusp_certificate.py:275 |
| `latch_pp1_basal` | `0.3` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:254 |
| `latch_gate_beta` | `6.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:2560 |
| `cusp_latch` | `False` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:261 |
| `cusp_eps_max` | `0.98` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:261 |

### `structural` (7/7 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `structural_interval` | `50000` | LIVE |  | bio_inspired_nanochat/certificate_bundle.py:2716 |
| `structural_tau_util` | `0.2` | LIVE |  | bio_inspired_nanochat/synaptic.py:3842 |
| `structural_age_bias` | `1.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:3852 |
| `router_embed_dim` | `24` | LIVE |  | bio_inspired_nanochat/synaptic.py:3613 |
| `router_contrastive_lr` | `0.0001` | LIVE |  | bio_inspired_nanochat/synaptic.py:3810 |
| `router_contrastive_push` | `0.1` | LIVE |  | bio_inspired_nanochat/synaptic.py:3810 |
| `topological_nas` | `False` | LIVE |  | bio_inspired_nanochat/synaptic_splitmerge.py:1223 |

### `glial_homeostasis` (6/6 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `glial_homeostasis` | `False` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:227 |
| `glial_group_size` | `4` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:231 |
| `glial_ema_rate` | `0.05` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:233 |
| `glial_feedback_rate` | `0.05` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:235 |
| `glial_energy_weight` | `0.25` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:239 |
| `glial_bias_cap` | `4.0` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:243 |

### `genetics` (1/1 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `xi_dim` | `4` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:221 |

### `toggle` (5/5 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `enable_presyn` | `True` | LIVE |  | bio_inspired_nanochat/certificate_bundle.py:2843 |
| `enable_hebbian` | `True` | LIVE |  | bio_inspired_nanochat/certificate_bundle.py:2911 |
| `enable_metabolism` | `True` | LIVE |  | bio_inspired_nanochat/hf_bio_adapter.py:164 |
| `use_flex_attention` | `False` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:291 |
| `tropical_skeleton` | `False` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:223 |

### `native_toggle` (2/2 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `native_presyn` | `False` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:309 |
| `native_genetics` | `False` | LIVE |  | bio_inspired_nanochat/synaptic.py:3718 |

---
*Status = LIVE when read by a runtime module (`bio_inspired_nanochat/**` or the Rust kernel), DEAD otherwise. “Read at” shows the first runtime/Rust read site; full evidence is in the JSON.*
