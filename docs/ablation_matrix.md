# Bio-vs-Vanilla Ablation Matrix — Pre-Registered Experiment Spec (bead `hwxb.5.1`)

This document and its machine-readable twin, `bio_inspired_nanochat/ablation_matrix.py`, define the
**headline experiment** of the scale-up epic: *does the biology help, and which mechanism?* It is
**pre-registered** — the configs, seeds, equal-compute budget, metrics, and the decision rule are
fixed here **before** any 4090 time is spent, so the result cannot be rationalized after the fact.

It builds on, and does not duplicate, the general
[`docs/eval_benchmark_matrix.md`](eval_benchmark_matrix.md) (metric definitions, preset dimensions,
the `eval_matrix` harness contract). What this spec adds is the **experimental design** the headline
verdict needs: the architecture-vs-mechanism anchor, both ablation directions, param-matching,
staged compute with a GPU-hour gate, and the pinned statistical decision rule.

> The mechanism set, the leave-one-out columns, and the add-one-in columns are **derived from
> `ablation_registry.MECHANISMS`** in code, so this spec cannot silently drift from what the model
> actually treats as ablatable. The narrative below mirrors that code; the code is authoritative.

---

## 1) The confound, and the three anchors

`GPTSynaptic` with *every* bio flag off is **still a different architecture** than vanilla `GPT`:
it carries the presynaptic attention augmentation, the router probe, the genome-decoder scaffold,
and the MoE structure. So a naïve `vanilla` vs `bio_all` contrast **confounds the architecture with the
mechanisms** — a positive result could be the scaffolding, not the biology.

We therefore run **three anchors** and read the experiment as a decomposition:

| Anchor | What it is | Isolates |
|---|---|---|
| `vanilla` | standard `GPT`, **param-matched** to `GPTSynaptic` | the silicon baseline |
| `synaptic_off` | `GPTSynaptic` architecture, **every mechanism off** (byte-identical-default per the unit tests) | the architecture, with no biology |
| `bio_all` | `GPTSynaptic` with the default synaptic stack on | the full biology |

```
(synaptic_off − vanilla)   = effect of the synaptic ARCHITECTURE alone
(mechanism   − synaptic_off) = the CLEAN isolated effect of a mechanism (same arch, one flag flipped)
(bio_all     − vanilla)    = the TOTAL bio effect
```

**Param-matching.** The synaptic stack adds parameters, so `vanilla` is matched to `GPTSynaptic` by
adjusting depth/width (`n_layer`/`n_embd`); the runner records **both** param counts in every summary
row so the comparison is honest. Without this, a win could be a free-parameter artifact.

---

## 2) The two ablation directions

Derived from the registry, excluding the two **infrastructure** mechanisms (`flex_attention` is
prefill-only and incompatible with KV-cache decode eval; `native_genetics` needs CUDA) — these are
performance toggles, not biology.

### Leave-one-out — marginal contribution (primary)
`bio_all` minus each **default-on** mechanism. Answers "what do we lose by removing X, given the rest?"

`bio_no_presyn`, `bio_no_hebbian`, `bio_no_metabolism`, `bio_no_stochastic_release`, `bio_no_doc2`,
`bio_no_septin_barrier`, `bio_no_bdnf`, `bio_no_genome` (sets `xi_dim=0`, retaining one learned
shared phenotype while removing per-expert kinetic specialization).

### Add-one-in — standalone effect (secondary)
`synaptic_off` plus each **opt-in** mechanism (with its prerequisites turned back on). Answers "what
does X buy on its own, on the clean architecture anchor?" Because some mechanisms require others, the
column turns on the whole prerequisite chain (e.g. `add_differentiable_recurrence` also enables
`learnable_kinetics` and `enable_presyn`); the isolated effect is then read against the matching
prerequisite-only baseline.

Derived from the registry at import time — today nine columns: `add_glial_homeostasis`,
`add_bistable_latch` (needs `enable_hebbian`), `add_stdp` (needs `enable_hebbian`),
`add_native_presyn` (needs `enable_presyn`), `add_learnable_kinetics` (needs `enable_presyn`),
`add_differentiable_recurrence` (needs `learnable_kinetics`, `enable_presyn`), `add_cusp_latch`
(needs `bistable_latch`, `enable_hebbian`), `add_metriplectic_integrator` (needs
`enable_presyn`), and `add_neuromod` (needs `enable_presyn`, `enable_hebbian`; the harness
instantiates the DA/ACh/NE bus for it). `python -c "from bio_inspired_nanochat import
ablation_matrix as am; print([c.config_id for c in am.add_one_in()])"` prints the live list.

Add-one-in is more interpretable for "which mechanism helps"; leave-one-out catches interactions.
We run both where compute allows; the staging below keeps the cost bounded.

**Total screening columns:** 3 anchors + 8 leave-one-out + 9 add-one-in = **20** (locked by
`tests/test_scaleup_ablation_e2e.py::test_module_enumerates_the_full_matrix`).

---

## 3) Seeds, equal-compute budget, metrics

- **Seeds.** Screening `{1337, 1338}`; confirmation `{1337, 1338, 1339}` (≥3 for significance).
- **Equal compute.** Every cell trains the **same token budget** (equal-compute, not equal-steps —
  configs differ in throughput). Screening `10M` tokens; confirmation `100M`. The final budget is
  pinned by the Phase-0 decision rule and the **feasible set from `hwxb.4.5`** (the mechanisms that
  fit the dual-4090 memory/throughput envelope); this spec is parametric in that set.
- **Metrics** (per cell; defined in `eval_benchmark_matrix.md`): `val_bpb` (**primary**), `niah_acc`
  (long-context), `working_memory` (associative recall; honest, may be null), `moe_gini` and
  `dead_expert_frac` (routing health), plus `tok_per_sec` and `peak_mem_gb` for the equal-compute
  accounting and feasibility.

---

**Throughput budget (proposed 2026-09-03, bead `74f.10.1`; owner to confirm):** `tok_per_sec` and
`peak_mem_gb` are not only recorded — at D1 scale `bio_all` must reach ≥ 1/2.0 of the param-matched
vanilla's training tokens/s and ≤ 1.5× its decode latency on the same host/batch/sequence, or the
quality effect is judged at equal compute and the mechanism is a candidate for pruning (`hwxb.6.1`).

## 4) Staged compute + the go/no-go gate

Running all 13 columns × 3 seeds at full budget is wasteful if half the mechanisms do nothing.

1. **Screening pass** — all 13 columns × 2 seeds × 10M tokens. Cheap; drops mechanisms that clearly
   do not move the primary metric.
2. **Go/no-go gate** (`ablation_matrix.go_no_go`) — before the expensive pass, require:
   - ≥1 mechanism **survived** screening (else report the null result — the experiment is allowed to
     say "biology did not help at this scale"), **and**
   - the estimated GPU-hours of the confirmation pass fit the cap (`DEFAULT_GPU_HOUR_CAP = 72` GPU-h;
     tune to the allocation). The estimate is `runs × tokens / tok_per_sec`, with `tok_per_sec` from
     the `hwxb.2.2` measurement (or the planning rule-of-thumb until that exists).
3. **Confirmation pass** — anchors + survivors only, × 3 seeds × 100M tokens.

Commit the GPU-hour estimate and the gate decision to the run log before the confirmation pass; never
burn days of 4090 time blindly.

---

## 5) Pre-registered decision rule (consistent with Phase-0)

Evaluated by the stats layer (`bio_inspired_nanochat/eval_stats.py`: `aggregate` + `paired_t_test`
over per-seed deltas; bead `hwxb.5.3`). **Primary metric: `val_bpb`, lower is better**; all deltas are
direction-aware (improvement = bpb down).

- A **mechanism "helps"** iff, across the confirmation seeds, its clean isolated contrast
  (`synaptic_off − add_mechanism`, or `bio_no_X − bio_all` for leave-one-out) is an **improvement**
  with a paired 95% CI excluding 0, and the paired *t* and Wilcoxon signed-rank agree on direction.
  Because the contrast is taken against `synaptic_off` (not `vanilla`), a positive architecture effect
  cannot masquerade as a mechanism win.
- **"Bio helps" overall** iff `(bio_all − vanilla)` on `val_bpb` is an improvement with a 95% CI
  excluding 0; the per-mechanism rule then attributes that win.
- A **null or negative** result is reported honestly.

---

## 6) How to run it

Every column maps to an `eval_matrix` run and is accepted by `--preset` / `--presets`: the named
registry presets (`vanilla`, `bio_all`, `bio_no_*`) go through `ablation_registry.apply_preset`, and
the remaining columns (`synaptic_off`, the `add_*` set) are materialised by
`ablation_matrix.AblationConfig.build_syn_cfg()` (`eval_matrix.MATRIX_COLUMNS`), so the spec and the
runner cannot drift. The dry-run bead (`hwxb.7.4`) exercises the full screen→gate→confirm
orchestration on tiny models to validate the machinery before the real run (`hwxb.5.2`).

```python
from bio_inspired_nanochat import ablation_matrix as am
cols = am.screening_columns()                      # the 20 columns
hours = am.estimate_gpu_hours(cols, am.SCREENING_SEEDS, am.SCREENING_TOKENS, tok_per_sec=measured)
gate = am.go_no_go(survivors, tok_per_sec=measured)  # gate the confirmation pass
conf = am.confirmation_columns(survivors)            # anchors + survivors
```

**Producing the checkpoints.** `eval_matrix` scores finished `base_train` checkpoints; the
commands that produce them are derived from this spec, never written by hand:

```bash
# print the screening pass (nothing runs); add --execute to run it, --nproc 2 for torchrun
uv run --no-sync python -m scripts.matrix_launch --stage screening \
    --recipe="--depth=10 --tie_embeddings=1 --device_batch_size=32 --total_batch_size=524288 --num_iterations=950"
```

`ablation_matrix.base_train_argv(column, seed=…)` contributes `--synapses`, one
`--syn_cfg.<field>=<value>` per field the column sets, the seed as `--init_seed` (what
`eval_matrix` checks the checkpoint against) and `--model_tag=matrix_<column>_s<seed>`; the
launcher prints the matching `eval_matrix batch … --checkpoint-dir
"<base_dir>/base_checkpoints/matrix_{preset}_s{seed}"` command. `tests/test_matrix_launch.py`
round-trips every column through `base_train`'s own override parser, and
`tests/test_e2e_matrix_pipeline.py` runs the whole chain as subprocesses at toy scale — launcher
(`--columns vanilla,synaptic_off`, two seeds) → `base_train` checkpoints → `eval_matrix matrix
--checkpoint-dir` scoring from the checkpoints' own metadata → `eval_stats` pairing — so a broken
link fails there before it burns GPU hours (about four minutes on CPU; nightly).

**Structural arm (opt-in, not pre-registered).** The expert lifecycle is a training-loop knob,
so `structural_columns()` provides `moe_fixed` (bio_all on `SynapticMoE`, fixed experts) and
`moe_splitmerge` (the same plus `--splitmerge_every=100 --sm_health_mode=relative
--split_health_min=1.5 --merge_health_max=0.35`); `moe_splitmerge − moe_fixed` is the
lifecycle's effect. `--stage structural` launches the pair; the screening set stays at 20.
Caveat from the 2026-09-02 CPU pilot (`results/structural_pair_pilot_2026-09-02.json`): with the
default `moe_balance_loss=0.01` utilization stays within ±0.03 of the fair share and neither health
signal ever fires, so the arm as specified would measure a no-op. Switching the balance loss off
(`results/structural_pair_pilot_2026-09-02_balance0.json`) changed nothing: still zero events in
every finished arm, because the utilization EMA is slow and a fresh router stays near uniform.
Before D1 the arm needs a demand signal that is not a slow utilization average (loss- or
NeuroScore-based), or a much longer warm-up than the pilot could afford.

---

## 7) Known gaps (honest scope)

- **Global neuromodulation (`hy8.1`) is a registered mechanism since 2026-09-01** (`neuromod_enabled`,
  requires presyn + hebbian), so the matrix carries an `add_neuromod` column and `eval_matrix`
  instantiates the bus for it. **NeuroScore** is still a `SplitMergeConfig`-level knob, not a
  `SynapticConfig` mechanism, so it has no column yet.
- **Structural lifecycle** (split/merge) is toggled at the *training-script* level
  (`--splitmerge_every`), not via a `SynapticConfig` mechanism flag, so it is not one of the 20
  pre-registered columns; `enable_metabolism` covers the per-expert energy dynamics. Since
  2026-09-01 the opt-in `structural_columns()` pair (§6) is how the lifecycle gets evidence.
- **Roadmap features vs. columns** (2026-09-02, `74f.9`): README §Roadmap lists per feature whether a
  D1 column measures it (six do), whether it is unimplemented (Rab/SNARE), a recipe knob (CA init),
  or a research module off the live path (gauge, simplicial, ultrametric). Off-path modules cannot be
  ablated and get no column.
- The **feasible mechanism set** and the **final token budget** are pinned by `hwxb.4.5`; this spec is
  parametric in both.
