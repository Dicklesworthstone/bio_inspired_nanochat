# Parameter Census — `SynapticConfig`

> **Generated** by `scripts/param_census.py` (bead `bio_inspired_nanochat-8j9.6`). Do not hand-edit; re-run `uv run python -m scripts.param_census`. Machine-readable companion: [`parameter_census.json`](./parameter_census.json).

`SynapticConfig` has **82 fields** — **82 LIVE** (read by runtime code) and **0 DEAD** (declared, read by nothing). This is the ground truth behind the README's *“48-parameter genome”* framing, which conflated three different counts.

## What the counts actually are

- **The learned genome is 4-D, not 48.** The biological 'genome' is the learned per-expert Xi vector (xi_dim=4: [alpha_fatigue, alpha_energy, camkii_gain, pp1_gain]), a torch parameter decoded to phenotype kinetics -- NOT the SynapticConfig hyperparameters. Every field here is a fixed hyperparameter, not a learned weight.

- **The wired search space is 10 params**, not 48. CMA-ES Phase 1 (`TOP10_PARAM_SPECS` in `scripts/tune_bio_params.py`) tunes: `alpha_ca`, `complexin_bias`, `doc2_gain`, `lambda_loge`, `nsf_recover`, `prime_rate`, `syt_fast_kd`, `syt_slow_kd`, `tau_c`, `unprime_per_release`. The 48-/82-parameter figures are the *aspirational* two-phase plan, not shipping code.

- **The config surface is 82 hyperparameters**, every one of which is read by runtime code — `8j9.5` pruned the last dead fields (`enabled`, `camkii_down`, `router_sim_threshold`, `native_presyn`, `native_metrics`, `native_plasticity`).


## Dead fields (read by nothing)

None — every `SynapticConfig` field is read on some runtime path (invariant enforced by `tests/test_param_census.py`).


## Full census by subsystem


### `general` (6/6 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `rank_eligibility` | `8` | LIVE |  | bio_inspired_nanochat/synaptic.py:661 |
| `attn_topk` | `32` | LIVE |  | bio_inspired_nanochat/synaptic.py:1284 |
| `stochastic_train_frac` | `0.12` | LIVE |  | bio_inspired_nanochat/synaptic.py:393 |
| `stochastic_mode` | `normal_reparam` | LIVE |  | bio_inspired_nanochat/synaptic.py:407 |
| `stochastic_tau` | `1.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:406 |
| `stochastic_count_cap` | `8` | LIVE |  | bio_inspired_nanochat/synaptic.py:403 |

### `presynaptic` (7/7 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `tau_c` | `6.0` | LIVE | ✓ | bio_inspired_nanochat/synaptic.py:372 |
| `doc2_gain` | `0.08` | LIVE | ✓ | bio_inspired_nanochat/flex_synaptic.py:52 |
| `prime_rate` | `0.075` | LIVE | ✓ | bio_inspired_nanochat/synaptic.py:479 |
| `unprime_per_release` | `0.05` | LIVE | ✓ | bio_inspired_nanochat/synaptic.py:484 |
| `nsf_recover` | `0.08` | LIVE | ✓ | bio_inspired_nanochat/synaptic.py:485 |
| `rec_rate` | `0.06` | LIVE |  | bio_inspired_nanochat/synaptic.py:474 |
| `endo_delay` | `3` | LIVE |  | bio_inspired_nanochat/synaptic.py:472 |

### `initial_state` (6/6 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `init_rrp` | `6.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1105 |
| `init_reserve` | `18.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1106 |
| `init_snare` | `0.7` | LIVE |  | bio_inspired_nanochat/synaptic.py:1107 |
| `init_clamp` | `0.6` | LIVE |  | bio_inspired_nanochat/synaptic.py:1108 |
| `init_amp` | `1.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1111 |
| `init_energy` | `0.85` | LIVE |  | bio_inspired_nanochat/synaptic.py:1109 |

### `energy` (3/3 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `energy_fill` | `0.02` | LIVE |  | bio_inspired_nanochat/synaptic.py:493 |
| `energy_use` | `0.02` | LIVE |  | bio_inspired_nanochat/synaptic.py:494 |
| `energy_max` | `1.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:493 |

### `attention` (4/4 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `lambda_loge` | `1.0` | LIVE | ✓ | bio_inspired_nanochat/synaptic.py:1322 |
| `barrier_strength` | `0.1` | LIVE |  | bio_inspired_nanochat/synaptic.py:428 |
| `epsilon` | `1e-06` | LIVE |  | bio_inspired_nanochat/synaptic.py:604 |
| `loge_bias_clamp` | `10.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1325 |

### `kernel_compat` (18/18 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `tau_buf` | `4.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:373 |
| `tau_prime` | `5.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:540 |
| `tau_rrp` | `40.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:541 |
| `tau_energy` | `50.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:542 |
| `alpha_ca` | `0.55` | LIVE | ✓ | bio_inspired_nanochat/synaptic.py:384 |
| `alpha_buf_on` | `0.1` | LIVE |  | bio_inspired_nanochat/synaptic.py:387 |
| `alpha_buf_off` | `0.1` | LIVE |  | bio_inspired_nanochat/synaptic.py:388 |
| `alpha_prime` | `0.1` | LIVE |  | bio_inspired_nanochat/synaptic.py:579 |
| `alpha_unprime` | `0.1` | LIVE |  | bio_inspired_nanochat/synaptic.py:613 |
| `alpha_refill` | `0.1` | LIVE |  | bio_inspired_nanochat/synaptic.py:580 |
| `energy_in` | `0.01` | LIVE |  | bio_inspired_nanochat/synaptic.py:582 |
| `energy_cost_rel` | `0.015` | LIVE |  | bio_inspired_nanochat/synaptic.py:616 |
| `energy_cost_pump` | `0.01` | LIVE |  | bio_inspired_nanochat/synaptic.py:617 |
| `syt_fast_kd` | `0.4` | LIVE | ✓ | bio_inspired_nanochat/flex_synaptic.py:50 |
| `syt_slow_kd` | `1.0` | LIVE | ✓ | bio_inspired_nanochat/flex_synaptic.py:51 |
| `complexin_bias` | `0.0` | LIVE | ✓ | bio_inspired_nanochat/flex_synaptic.py:54 |
| `qmax` | `2.0` | LIVE |  | bio_inspired_nanochat/flex_synaptic.py:57 |
| `q_beta` | `1.0` | LIVE |  | bio_inspired_nanochat/flex_synaptic.py:57 |

### `postsynaptic` (14/14 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `post_fast_decay` | `0.95` | LIVE |  | bio_inspired_nanochat/synaptic.py:821 |
| `post_fast_lr` | `0.0015` | LIVE |  | bio_inspired_nanochat/synaptic.py:821 |
| `post_slow_lr` | `0.0005` | LIVE |  | bio_inspired_nanochat/synaptic.py:809 |
| `post_trace_decay` | `0.96` | LIVE |  | bio_inspired_nanochat/synaptic.py:941 |
| `camkii_up` | `0.05` | LIVE |  | bio_inspired_nanochat/synaptic.py:721 |
| `pp1_tau` | `0.985` | LIVE |  | bio_inspired_nanochat/synaptic.py:722 |
| `camkii_thr` | `1.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:699 |
| `pp1_thr` | `0.7` | LIVE |  | bio_inspired_nanochat/synaptic.py:719 |
| `bdnf_tau` | `0.985` | LIVE |  | bio_inspired_nanochat/synaptic.py:736 |
| `bdnf_scale` | `1.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:801 |
| `bdnf_gamma` | `0.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:801 |
| `bdnf_hebb_accumulate` | `True` | LIVE |  | bio_inspired_nanochat/synaptic.py:733 |
| `bdnf_max` | `10.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:742 |
| `plasticity_during_training` | `True` | LIVE |  | bio_inspired_nanochat/synaptic.py:1067 |

### `latch` (12/12 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `bistable_latch` | `False` | LIVE |  | bio_inspired_nanochat/synaptic.py:694 |
| `latch_ltd_thr` | `0.5` | LIVE |  | bio_inspired_nanochat/synaptic.py:700 |
| `latch_input_gain` | `12.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:698 |
| `latch_alpha_ca` | `0.6` | LIVE |  | bio_inspired_nanochat/synaptic.py:710 |
| `latch_beta_pp1` | `1.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:702 |
| `latch_gamma_auto` | `0.45` | LIVE |  | bio_inspired_nanochat/synaptic.py:701 |
| `latch_hill_n` | `6.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:707 |
| `latch_hill_k` | `0.6` | LIVE |  | bio_inspired_nanochat/synaptic.py:707 |
| `latch_alpha_pp1` | `0.5` | LIVE |  | bio_inspired_nanochat/synaptic.py:713 |
| `latch_beta_camkii` | `0.3` | LIVE |  | bio_inspired_nanochat/synaptic.py:713 |
| `latch_pp1_basal` | `0.3` | LIVE |  | bio_inspired_nanochat/synaptic.py:714 |
| `latch_gate_beta` | `6.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:791 |

### `structural` (6/6 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `structural_interval` | `50000` | LIVE |  | bio_inspired_nanochat/synaptic.py:1624 |
| `structural_tau_util` | `0.2` | LIVE |  | bio_inspired_nanochat/synaptic.py:1613 |
| `structural_age_bias` | `1.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1623 |
| `router_embed_dim` | `24` | LIVE |  | bio_inspired_nanochat/gpt_synaptic.py:480 |
| `router_contrastive_lr` | `0.0001` | LIVE |  | bio_inspired_nanochat/synaptic.py:1581 |
| `router_contrastive_push` | `0.1` | LIVE |  | bio_inspired_nanochat/synaptic.py:1581 |

### `genetics` (1/1 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `xi_dim` | `4` | LIVE |  | bio_inspired_nanochat/synaptic.py:1467 |

### `toggle` (4/4 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `enable_presyn` | `True` | LIVE |  | bio_inspired_nanochat/synaptic.py:360 |
| `enable_hebbian` | `True` | LIVE |  | bio_inspired_nanochat/synaptic.py:878 |
| `enable_metabolism` | `True` | LIVE |  | bio_inspired_nanochat/synaptic.py:1505 |
| `use_flex_attention` | `False` | LIVE |  | bio_inspired_nanochat/synaptic.py:1153 |

### `native_toggle` (1/1 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `native_genetics` | `False` | LIVE |  | bio_inspired_nanochat/synaptic.py:1516 |

---
*Status = LIVE when read by a runtime module (`bio_inspired_nanochat/**` or the Rust kernel), DEAD otherwise. “Read at” shows the first runtime/Rust read site; full evidence is in the JSON.*
