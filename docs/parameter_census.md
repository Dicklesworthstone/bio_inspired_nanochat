# Parameter Census — `SynapticConfig`

> **Generated** by `scripts/param_census.py` (bead `bio_inspired_nanochat-8j9.6`). Do not hand-edit; re-run `uv run python -m scripts.param_census`. Machine-readable companion: [`parameter_census.json`](./parameter_census.json).

`SynapticConfig` has **93 fields** — **93 LIVE** (read by runtime code) and **0 DEAD** (declared, read by nothing). This is the ground truth behind the README's *“48-parameter genome”* framing, which conflated three different counts.

## What the counts actually are

- **The learned genome is 4-D, not 48.** The biological 'genome' is the learned per-expert Xi vector (xi_dim=4), expanded by one shared learned decoder to six bounded phenotype kinetics (fatigue/recovery rates, CaMKII/PP1 gains, calcium retention/influx). Set xi_dim=0 for learned shared kinetics without per-expert Xi. The genome is NOT the SynapticConfig hyperparameters; every field here is a fixed hyperparameter, not a learned weight.

- **The wired search space is 10 params**, not 48. CMA-ES Phase 1 (`TOP10_PARAM_SPECS` in `scripts/tune_bio_params.py`) tunes: `alpha_ca`, `complexin_bias`, `doc2_gain`, `lambda_loge`, `nsf_recover`, `prime_rate`, `syt_fast_kd`, `syt_slow_kd`, `tau_c`, `unprime_per_release`. The 48-/82-parameter figures are the *aspirational* two-phase plan, not shipping code.

- **The config surface is 93 hyperparameters**, every one of which is read by runtime code — `8j9.5` pruned the last dead fields (`enabled`, `camkii_down`, `router_sim_threshold`, `native_presyn`, `native_metrics`, `native_plasticity`).


## Dead fields (read by nothing)

None — every `SynapticConfig` field is read on some runtime path (invariant enforced by `tests/test_param_census.py`).


## Full census by subsystem


### `general` (6/6 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `rank_eligibility` | `8` | LIVE |  | bio_inspired_nanochat/synaptic.py:1071 |
| `attn_topk` | `32` | LIVE |  | bio_inspired_nanochat/synaptic.py:1798 |
| `stochastic_train_frac` | `0.12` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:180 |
| `stochastic_mode` | `normal_reparam` | LIVE |  | bio_inspired_nanochat/synaptic.py:806 |
| `stochastic_tau` | `1.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:805 |
| `stochastic_count_cap` | `8` | LIVE |  | bio_inspired_nanochat/synaptic.py:802 |

### `presynaptic` (12/12 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `tau_c` | `6.0` | LIVE | ✓ | bio_inspired_nanochat/ablation_registry.py:186 |
| `learnable_kinetics` | `False` | LIVE |  | bio_inspired_nanochat/synaptic.py:669 |
| `differentiable_recurrence` | `False` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:207 |
| `recurrence_block_size` | `64` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:208 |
| `recurrence_chunk_len` | `0` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:212 |
| `doc2_gain` | `0.08` | LIVE | ✓ | bio_inspired_nanochat/flex_synaptic.py:52 |
| `prime_rate` | `0.075` | LIVE | ✓ | bio_inspired_nanochat/synaptic.py:885 |
| `unprime_per_release` | `0.05` | LIVE | ✓ | bio_inspired_nanochat/synaptic.py:890 |
| `nsf_recover` | `0.08` | LIVE | ✓ | bio_inspired_nanochat/synaptic.py:891 |
| `rec_rate` | `0.06` | LIVE |  | bio_inspired_nanochat/synaptic.py:880 |
| `endo_delay` | `3` | LIVE |  | bio_inspired_nanochat/synaptic.py:878 |
| `metriplectic_integrator` | `False` | LIVE |  | bio_inspired_nanochat/synaptic.py:677 |

### `initial_state` (6/6 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `init_rrp` | `6.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1582 |
| `init_reserve` | `18.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1583 |
| `init_snare` | `0.7` | LIVE |  | bio_inspired_nanochat/synaptic.py:1584 |
| `init_clamp` | `0.6` | LIVE |  | bio_inspired_nanochat/synaptic.py:1585 |
| `init_amp` | `1.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1588 |
| `init_energy` | `0.85` | LIVE |  | bio_inspired_nanochat/synaptic.py:1586 |

### `energy` (3/3 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `energy_fill` | `0.02` | LIVE |  | bio_inspired_nanochat/synaptic.py:899 |
| `energy_use` | `0.02` | LIVE |  | bio_inspired_nanochat/synaptic.py:900 |
| `energy_max` | `1.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:899 |

### `attention` (4/4 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `lambda_loge` | `1.0` | LIVE | ✓ | bio_inspired_nanochat/synaptic.py:1843 |
| `barrier_strength` | `0.1` | LIVE |  | bio_inspired_nanochat/synaptic.py:827 |
| `epsilon` | `1e-06` | LIVE |  | bio_inspired_nanochat/synaptic.py:1014 |
| `loge_bias_clamp` | `10.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1846 |

### `kernel_compat` (18/18 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `tau_buf` | `4.0` | LIVE |  | bio_inspired_nanochat/cusp_certificate.py:114 |
| `tau_prime` | `5.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:950 |
| `tau_rrp` | `40.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:951 |
| `tau_energy` | `50.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:952 |
| `alpha_ca` | `0.55` | LIVE | ✓ | bio_inspired_nanochat/ablation_registry.py:188 |
| `alpha_buf_on` | `0.1` | LIVE |  | bio_inspired_nanochat/cusp_certificate.py:115 |
| `alpha_buf_off` | `0.1` | LIVE |  | bio_inspired_nanochat/cusp_certificate.py:116 |
| `alpha_prime` | `0.1` | LIVE |  | bio_inspired_nanochat/synaptic.py:989 |
| `alpha_unprime` | `0.1` | LIVE |  | bio_inspired_nanochat/synaptic.py:1023 |
| `alpha_refill` | `0.1` | LIVE |  | bio_inspired_nanochat/synaptic.py:990 |
| `energy_in` | `0.01` | LIVE |  | bio_inspired_nanochat/synaptic.py:992 |
| `energy_cost_rel` | `0.015` | LIVE |  | bio_inspired_nanochat/synaptic.py:1026 |
| `energy_cost_pump` | `0.01` | LIVE |  | bio_inspired_nanochat/synaptic.py:1027 |
| `syt_fast_kd` | `0.4` | LIVE | ✓ | bio_inspired_nanochat/flex_synaptic.py:50 |
| `syt_slow_kd` | `1.0` | LIVE | ✓ | bio_inspired_nanochat/flex_synaptic.py:51 |
| `complexin_bias` | `0.0` | LIVE | ✓ | bio_inspired_nanochat/flex_synaptic.py:54 |
| `qmax` | `2.0` | LIVE |  | bio_inspired_nanochat/flex_synaptic.py:57 |
| `q_beta` | `1.0` | LIVE |  | bio_inspired_nanochat/flex_synaptic.py:57 |

### `postsynaptic` (17/17 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `post_fast_decay` | `0.95` | LIVE |  | bio_inspired_nanochat/synaptic.py:1254 |
| `post_fast_lr` | `0.0015` | LIVE |  | bio_inspired_nanochat/synaptic.py:1254 |
| `post_slow_lr` | `0.0005` | LIVE |  | bio_inspired_nanochat/synaptic.py:1242 |
| `post_trace_decay` | `0.96` | LIVE |  | bio_inspired_nanochat/synaptic.py:1374 |
| `fast_weight_normalized` | `False` | LIVE |  | bio_inspired_nanochat/synaptic.py:1408 |
| `fast_weight_eta` | `0.5` | LIVE |  | bio_inspired_nanochat/synaptic.py:1420 |
| `fast_weight_max_norm` | `1.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1422 |
| `camkii_up` | `0.05` | LIVE |  | bio_inspired_nanochat/synaptic.py:1154 |
| `pp1_tau` | `0.985` | LIVE |  | bio_inspired_nanochat/synaptic.py:1155 |
| `camkii_thr` | `1.0` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:197 |
| `pp1_thr` | `0.7` | LIVE |  | bio_inspired_nanochat/synaptic.py:1152 |
| `bdnf_tau` | `0.985` | LIVE |  | bio_inspired_nanochat/synaptic.py:1169 |
| `bdnf_scale` | `1.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1234 |
| `bdnf_gamma` | `0.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1234 |
| `bdnf_hebb_accumulate` | `True` | LIVE |  | bio_inspired_nanochat/synaptic.py:1166 |
| `bdnf_max` | `10.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1175 |
| `plasticity_during_training` | `True` | LIVE |  | bio_inspired_nanochat/synaptic.py:1537 |

### `latch` (14/14 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `bistable_latch` | `False` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:190 |
| `latch_ltd_thr` | `0.5` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:197 |
| `latch_input_gain` | `12.0` | LIVE |  | bio_inspired_nanochat/cusp_certificate.py:58 |
| `latch_alpha_ca` | `0.6` | LIVE |  | bio_inspired_nanochat/cusp_certificate.py:50 |
| `latch_beta_pp1` | `1.0` | LIVE |  | bio_inspired_nanochat/cusp_certificate.py:51 |
| `latch_gamma_auto` | `0.45` | LIVE |  | bio_inspired_nanochat/cusp_certificate.py:52 |
| `latch_hill_n` | `6.0` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:191 |
| `latch_hill_k` | `0.6` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:193 |
| `latch_alpha_pp1` | `0.5` | LIVE |  | bio_inspired_nanochat/cusp_certificate.py:274 |
| `latch_beta_camkii` | `0.3` | LIVE |  | bio_inspired_nanochat/cusp_certificate.py:275 |
| `latch_pp1_basal` | `0.3` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:195 |
| `latch_gate_beta` | `6.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1224 |
| `cusp_latch` | `False` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:202 |
| `cusp_eps_max` | `0.98` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:202 |

### `structural` (7/7 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `structural_interval` | `50000` | LIVE |  | bio_inspired_nanochat/synaptic.py:2287 |
| `structural_tau_util` | `0.2` | LIVE |  | bio_inspired_nanochat/synaptic.py:2276 |
| `structural_age_bias` | `1.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:2286 |
| `router_embed_dim` | `24` | LIVE |  | bio_inspired_nanochat/synaptic.py:2084 |
| `router_contrastive_lr` | `0.0001` | LIVE |  | bio_inspired_nanochat/synaptic.py:2244 |
| `router_contrastive_push` | `0.1` | LIVE |  | bio_inspired_nanochat/synaptic.py:2244 |
| `topological_nas` | `False` | LIVE |  | bio_inspired_nanochat/synaptic_splitmerge.py:997 |

### `genetics` (1/1 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `xi_dim` | `4` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:184 |

### `toggle` (4/4 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `enable_presyn` | `True` | LIVE |  | bio_inspired_nanochat/synaptic.py:740 |
| `enable_hebbian` | `True` | LIVE |  | bio_inspired_nanochat/synaptic.py:1311 |
| `enable_metabolism` | `True` | LIVE |  | bio_inspired_nanochat/synaptic.py:2163 |
| `use_flex_attention` | `False` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:218 |

### `native_toggle` (1/1 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `native_genetics` | `False` | LIVE |  | bio_inspired_nanochat/synaptic.py:2174 |

---
*Status = LIVE when read by a runtime module (`bio_inspired_nanochat/**` or the Rust kernel), DEAD otherwise. “Read at” shows the first runtime/Rust read site; full evidence is in the JSON.*
