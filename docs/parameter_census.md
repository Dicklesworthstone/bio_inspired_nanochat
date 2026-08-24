# Parameter Census — `SynapticConfig`

> **Generated** by `scripts/param_census.py` (bead `bio_inspired_nanochat-8j9.6`). Do not hand-edit; re-run `uv run python -m scripts.param_census`. Machine-readable companion: [`parameter_census.json`](./parameter_census.json).

`SynapticConfig` has **94 fields** — **94 LIVE** (read by runtime code) and **0 DEAD** (declared, read by nothing). This is the ground truth behind the README's *“48-parameter genome”* framing, which conflated three different counts.

## What the counts actually are

- **The learned genome is 4-D, not 48.** The biological 'genome' is the learned per-expert Xi vector (xi_dim=4), expanded by one shared learned decoder to six bounded phenotype kinetics (fatigue/recovery rates, CaMKII/PP1 gains, calcium retention/influx). Set xi_dim=0 for learned shared kinetics without per-expert Xi. The genome is NOT the SynapticConfig hyperparameters; every field here is a fixed hyperparameter, not a learned weight.

- **The wired search space is 10 params**, not 48. CMA-ES Phase 1 (`TOP10_PARAM_SPECS` in `scripts/tune_bio_params.py`) tunes: `alpha_ca`, `complexin_bias`, `doc2_gain`, `lambda_loge`, `nsf_recover`, `prime_rate`, `syt_fast_kd`, `syt_slow_kd`, `tau_c`, `unprime_per_release`. The 48-/82-parameter figures are the *aspirational* two-phase plan, not shipping code.

- **The config surface is 94 hyperparameters**, every one of which is read by runtime code — `8j9.5` pruned the last dead fields (`enabled`, `camkii_down`, `router_sim_threshold`, `native_presyn`, `native_metrics`, `native_plasticity`).


## Dead fields (read by nothing)

None — every `SynapticConfig` field is read on some runtime path (invariant enforced by `tests/test_param_census.py`).


## Full census by subsystem


### `general` (6/6 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `rank_eligibility` | `8` | LIVE |  | bio_inspired_nanochat/synaptic.py:1205 |
| `attn_topk` | `32` | LIVE |  | bio_inspired_nanochat/synaptic.py:1943 |
| `stochastic_train_frac` | `0.12` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:185 |
| `stochastic_mode` | `normal_reparam` | LIVE |  | bio_inspired_nanochat/synaptic.py:932 |
| `stochastic_tau` | `1.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:931 |
| `stochastic_count_cap` | `8` | LIVE |  | bio_inspired_nanochat/synaptic.py:928 |

### `presynaptic` (12/12 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `tau_c` | `6.0` | LIVE | ✓ | bio_inspired_nanochat/ablation_registry.py:195 |
| `learnable_kinetics` | `False` | LIVE |  | bio_inspired_nanochat/synaptic.py:695 |
| `differentiable_recurrence` | `False` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:216 |
| `recurrence_block_size` | `64` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:217 |
| `recurrence_chunk_len` | `0` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:221 |
| `doc2_gain` | `0.08` | LIVE | ✓ | bio_inspired_nanochat/flex_synaptic.py:52 |
| `prime_rate` | `0.075` | LIVE | ✓ | bio_inspired_nanochat/synaptic.py:1016 |
| `unprime_per_release` | `0.05` | LIVE | ✓ | bio_inspired_nanochat/synaptic.py:1021 |
| `nsf_recover` | `0.08` | LIVE | ✓ | bio_inspired_nanochat/synaptic.py:1022 |
| `rec_rate` | `0.06` | LIVE |  | bio_inspired_nanochat/synaptic.py:1011 |
| `endo_delay` | `3` | LIVE |  | bio_inspired_nanochat/synaptic.py:1009 |
| `metriplectic_integrator` | `False` | LIVE |  | bio_inspired_nanochat/synaptic.py:703 |

### `initial_state` (6/6 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `init_rrp` | `6.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1716 |
| `init_reserve` | `18.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1717 |
| `init_snare` | `0.7` | LIVE |  | bio_inspired_nanochat/synaptic.py:1718 |
| `init_clamp` | `0.6` | LIVE |  | bio_inspired_nanochat/synaptic.py:1719 |
| `init_amp` | `1.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1722 |
| `init_energy` | `0.85` | LIVE |  | bio_inspired_nanochat/synaptic.py:1720 |

### `energy` (3/3 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `energy_fill` | `0.02` | LIVE |  | bio_inspired_nanochat/synaptic.py:1030 |
| `energy_use` | `0.02` | LIVE |  | bio_inspired_nanochat/synaptic.py:1031 |
| `energy_max` | `1.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1030 |

### `attention` (4/4 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `lambda_loge` | `1.0` | LIVE | ✓ | bio_inspired_nanochat/synaptic.py:1988 |
| `barrier_strength` | `0.1` | LIVE |  | bio_inspired_nanochat/tropical_certificate.py:1478 |
| `epsilon` | `1e-06` | LIVE |  | bio_inspired_nanochat/synaptic.py:1148 |
| `loge_bias_clamp` | `10.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1991 |

### `kernel_compat` (18/18 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `tau_buf` | `4.0` | LIVE |  | bio_inspired_nanochat/cusp_certificate.py:114 |
| `tau_prime` | `5.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1084 |
| `tau_rrp` | `40.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1085 |
| `tau_energy` | `50.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1086 |
| `alpha_ca` | `0.55` | LIVE | ✓ | bio_inspired_nanochat/ablation_registry.py:197 |
| `alpha_buf_on` | `0.1` | LIVE |  | bio_inspired_nanochat/cusp_certificate.py:115 |
| `alpha_buf_off` | `0.1` | LIVE |  | bio_inspired_nanochat/cusp_certificate.py:116 |
| `alpha_prime` | `0.1` | LIVE |  | bio_inspired_nanochat/synaptic.py:1123 |
| `alpha_unprime` | `0.1` | LIVE |  | bio_inspired_nanochat/synaptic.py:1157 |
| `alpha_refill` | `0.1` | LIVE |  | bio_inspired_nanochat/synaptic.py:1124 |
| `energy_in` | `0.01` | LIVE |  | bio_inspired_nanochat/synaptic.py:1126 |
| `energy_cost_rel` | `0.015` | LIVE |  | bio_inspired_nanochat/synaptic.py:1160 |
| `energy_cost_pump` | `0.01` | LIVE |  | bio_inspired_nanochat/synaptic.py:1161 |
| `syt_fast_kd` | `0.4` | LIVE | ✓ | bio_inspired_nanochat/flex_synaptic.py:50 |
| `syt_slow_kd` | `1.0` | LIVE | ✓ | bio_inspired_nanochat/flex_synaptic.py:51 |
| `complexin_bias` | `0.0` | LIVE | ✓ | bio_inspired_nanochat/flex_synaptic.py:54 |
| `qmax` | `2.0` | LIVE |  | bio_inspired_nanochat/flex_synaptic.py:57 |
| `q_beta` | `1.0` | LIVE |  | bio_inspired_nanochat/flex_synaptic.py:57 |

### `postsynaptic` (17/17 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `post_fast_decay` | `0.95` | LIVE |  | bio_inspired_nanochat/synaptic.py:1388 |
| `post_fast_lr` | `0.0015` | LIVE |  | bio_inspired_nanochat/synaptic.py:1388 |
| `post_slow_lr` | `0.0005` | LIVE |  | bio_inspired_nanochat/synaptic.py:1376 |
| `post_trace_decay` | `0.96` | LIVE |  | bio_inspired_nanochat/synaptic.py:1508 |
| `fast_weight_normalized` | `True` | LIVE |  | bio_inspired_nanochat/synaptic.py:1542 |
| `fast_weight_eta` | `0.5` | LIVE |  | bio_inspired_nanochat/synaptic.py:1554 |
| `fast_weight_max_norm` | `1.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1556 |
| `camkii_up` | `0.05` | LIVE |  | bio_inspired_nanochat/synaptic.py:1288 |
| `pp1_tau` | `0.985` | LIVE |  | bio_inspired_nanochat/synaptic.py:1289 |
| `camkii_thr` | `1.0` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:206 |
| `pp1_thr` | `0.7` | LIVE |  | bio_inspired_nanochat/synaptic.py:1286 |
| `bdnf_tau` | `0.985` | LIVE |  | bio_inspired_nanochat/synaptic.py:1303 |
| `bdnf_scale` | `1.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1368 |
| `bdnf_gamma` | `0.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1368 |
| `bdnf_hebb_accumulate` | `True` | LIVE |  | bio_inspired_nanochat/synaptic.py:1300 |
| `bdnf_max` | `10.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1309 |
| `plasticity_during_training` | `True` | LIVE |  | bio_inspired_nanochat/synaptic.py:1671 |

### `latch` (14/14 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `bistable_latch` | `False` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:199 |
| `latch_ltd_thr` | `0.5` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:206 |
| `latch_input_gain` | `12.0` | LIVE |  | bio_inspired_nanochat/cusp_certificate.py:58 |
| `latch_alpha_ca` | `0.6` | LIVE |  | bio_inspired_nanochat/cusp_certificate.py:50 |
| `latch_beta_pp1` | `1.0` | LIVE |  | bio_inspired_nanochat/cusp_certificate.py:51 |
| `latch_gamma_auto` | `0.45` | LIVE |  | bio_inspired_nanochat/cusp_certificate.py:52 |
| `latch_hill_n` | `6.0` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:200 |
| `latch_hill_k` | `0.6` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:202 |
| `latch_alpha_pp1` | `0.5` | LIVE |  | bio_inspired_nanochat/cusp_certificate.py:274 |
| `latch_beta_camkii` | `0.3` | LIVE |  | bio_inspired_nanochat/cusp_certificate.py:275 |
| `latch_pp1_basal` | `0.3` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:204 |
| `latch_gate_beta` | `6.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1358 |
| `cusp_latch` | `False` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:211 |
| `cusp_eps_max` | `0.98` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:211 |

### `structural` (7/7 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `structural_interval` | `50000` | LIVE |  | bio_inspired_nanochat/synaptic.py:2432 |
| `structural_tau_util` | `0.2` | LIVE |  | bio_inspired_nanochat/synaptic.py:2421 |
| `structural_age_bias` | `1.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:2431 |
| `router_embed_dim` | `24` | LIVE |  | bio_inspired_nanochat/synaptic.py:2229 |
| `router_contrastive_lr` | `0.0001` | LIVE |  | bio_inspired_nanochat/synaptic.py:2389 |
| `router_contrastive_push` | `0.1` | LIVE |  | bio_inspired_nanochat/synaptic.py:2389 |
| `topological_nas` | `False` | LIVE |  | bio_inspired_nanochat/synaptic_splitmerge.py:997 |

### `genetics` (1/1 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `xi_dim` | `4` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:189 |

### `toggle` (5/5 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `enable_presyn` | `True` | LIVE |  | bio_inspired_nanochat/synaptic.py:844 |
| `enable_hebbian` | `True` | LIVE |  | bio_inspired_nanochat/synaptic.py:1445 |
| `enable_metabolism` | `True` | LIVE |  | bio_inspired_nanochat/synaptic.py:2308 |
| `use_flex_attention` | `False` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:227 |
| `tropical_skeleton` | `False` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:191 |

### `native_toggle` (1/1 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `native_genetics` | `False` | LIVE |  | bio_inspired_nanochat/synaptic.py:2319 |

---
*Status = LIVE when read by a runtime module (`bio_inspired_nanochat/**` or the Rust kernel), DEAD otherwise. “Read at” shows the first runtime/Rust read site; full evidence is in the JSON.*
