# D1 bio-vs-vanilla ablation protocol

**Prospective version `d1-paired-bpb-2026-11-22`** fixes the comparisons and selection rule before any
D1 outcome. The primary question is whether the default dense synaptic model lowers held-out FineWeb
validation bits per byte (BPB) relative to vanilla at the same training token budget. A second
comparison measures the synaptic architecture alone. Each mechanism has its own paired comparison.
The executable definitions are in [eval_stats.py](../bio_inspired_nanochat/eval_stats.py) and the
training columns are in [ablation_matrix.py](../bio_inspired_nanochat/ablation_matrix.py).

## Registered comparisons

Every difference below is treatment BPB minus control BPB on the same initialization seed. A
negative value favors treatment. The screen records both anchors. Only the thirteen mechanism or
stack comparisons can nominate a candidate. The MoE lifecycle belongs to a separate exploratory
family.

| Question | Treatment | Control |
|---|---|---|
| Total effect of the default configuration (primary) | `bio_all` | `vanilla` |
| Synaptic scaffold (architecture) | `synaptic_off` | `vanilla` |
| Presynaptic stack, including its dependent mechanisms | `bio_all` | `bio_no_presyn` |
| Marginal contribution of stochastic release, Doc2, or septin barrier | `bio_all` | Corresponding `bio_no_stochastic_release`, `bio_no_doc2`, or `bio_no_septin_barrier` |
| Hebbian plasticity alone | `add_hebbian` | `synaptic_off` |
| Incremental BDNF, bistable latch, or STDP effect | `add_bdnf`, `add_bistable_latch`, or `add_stdp` | `add_hebbian` |
| Cusp latch conditional on the bistable latch | `add_cusp_latch` | `add_bistable_latch` |
| Learnable kinetics or metriplectic integration conditional on presynaptic dynamics | `add_learnable_kinetics` or `add_metriplectic_integrator` | `control_presyn` |
| Differentiable recurrence given learnable kinetics | `add_differentiable_recurrence` | `add_learnable_kinetics` |
| Neuromodulation given presynaptic and Hebbian dynamics | `add_neuromod` | `control_presyn_hebbian` |

The October screening roster contained 17 columns. The prospective screen removes the
`add_native_presyn` acceleration comparison and adds two prerequisite-only controls, giving 18
columns. Both controls disable BDNF activity. The registered backend uses the matrix reference
settings `native_presyn=False` and `native_genetics=False` throughout. A comparison of fused
dispatch against the reference requires its own paired measurements. The historical roster and
CPU scores remain available under their original version.

## Screening, admission and confirmation

1. **Screen.** Run all 18 columns on seeds 1337 and 1338. A mechanism advances only when its
   specified treatment has strictly lower held-out BPB on **both** seeds. A tie or reversal on
   either seed rules out advancement. Screening uses no significance test. If no mechanism
   advances, record `no_go` and stop at 36 training cells. A favorable total or scaffold
   comparison alone cannot open confirmation. A missing, failed, duplicated, unexpected or
   mismatched cell makes the screen inconclusive.
2. **Size and cost the complete family.** If `k` mechanisms advance, test exactly `m = k + 2`
   hypotheses, including both anchors. Train every distinct treatment and control on fresh
   seeds starting at 2337. Use `n = ceil(log2(2m / 0.05))` paired seeds. With one survivor the
   family has three hypotheses and seven seeds. The smallest exact two-sided signed-rank
   p-value is then 1/64, and the smallest first Holm-adjusted value is 3/64. With 13 survivors
   the family has 15 hypotheses and ten seeds. This calculation establishes that significance
   is attainable with nonzero pairs. Actual power depends on effect size, seed variation and
   ties. Budget refusal yields no confirmatory test. Cost each arm at its measured aggregate
   training throughput, multiply wall hours by the training device count, and use the full set
   of control cells and fresh seeds.
3. **Confirm.** Pair by seed and compute the treatment-minus-control BPB vector for every
   registered hypothesis. Apply Holm across the whole family to the two-sided paired-t
   p-values, and separately to the exact two-sided Wilcoxon signed-rank p-values. At familywise
   alpha 0.05 a supported gain requires both adjusted values at or below alpha and a paired
   95% percentile-bootstrap interval wholly below zero (10,000 resamples, seed 0). The
   symmetric rule gives a supported regression. The training seed is the sampling unit.
   The signed-rank calculation drops zero differences. Means and intervals retain them.
   If ties make the adjusted signed-rank threshold unattainable given the other family
   results, report the contrast as inconclusive. Exact ties throughout yield a sample
   difference of zero and a null verdict. A null leaves equivalence unresolved.

The independent seeds address screening selection under the same fixed validation corpus.
Results describe variation across training seeds on that corpus. The protocol makes no
inference about variance across corpora, long-context memory, online adaptation, or performance
on downstream tasks.

## Training and evidence

Train dense depth-10, width-640, five-head models with tied embeddings and sequence length 1024 on two CUDA devices.
Each screening cell gets the largest complete training batch fitting 10 million tokens.
Confirmation uses 100 million, with the same rounding rule. Reject batches that would discard
over 5% of the stage budget. Score at least 1 million input tokens in complete held-out BPB
batches with the same tokenizer, byte mapping, split and evaluation geometry. Record the
actual number of scored input tokens and match the source revision, source content, ordered train/validation
shards, optimizer recipe per arm and training/evaluation geometry across stages. The scorer
checks the checkpoint against the training manifest before emitting its provenance. The
statistical reader compares the CSV with the scorer's sibling JSONL, rejects mixed stages or
campaigns, and verifies the screening selection digest on confirmation. Archive both scorer
files together. The shard fingerprints establish which local bytes the run used. Archive upstream
FineWeb acquisition checksums to establish corpus origin. The scored-token count includes
revisits if a small shard wraps. Keep the original scorer CSV, manifests and checkpoints with
any published result.

For a batch of 65,536 training tokens, 152 updates give 9,961,472 screening tokens per
cell. Before any training, freeze the resolved manifest on the worker hosting the shards.
Supply the slowest measured aggregate training rate and single-GPU scoring time from a
representative pilot of the frozen backend. The training launcher and the emitted scoring
command both enforce the remaining physical GPU-hour allocation. Confirmation charges the
screen's completed and failed attempts and uses its per-arm timings.

```bash
uv run python -m scripts.matrix_launch --stage screening --nproc 2 --plan \
  --recipe="--device_type=cuda --depth=10 --tie_embeddings=1 --max_seq_len=1024 --device_batch_size=8 --total_batch_size=65536 --num_iterations=152 --eval_tokens=1048576 --core_metric_every=-1 --sample_every=-1" \
  --tok-per-sec "$measured_aggregate_tokens_per_second" --score-seconds-per-cell "$measured_cell_score_seconds"
```

Use the same arguments with `--execute` to train. Run the scoring command printed on the
last line. Use `--execute --restart` to continue the same training manifest.

**Capacity and interpretation.** Shared width/depth and equal input tokens establish the
registered budget. Record the trainable parameter count, active parameter count, optimizer
names and recipe digest for each arm, including installed fast-weight matrices. A contrast
with different counts or optimization includes those differences in its estimated effect.
Only describe equal parameter capacity if both observed counts agree. Independent
throughput and GPU-hour measurements are required for a claim at equal compute.

The following analysis commands consume the CSVs produced by the checkpoint scorer. The
confirmation command derives its survivor list, family size and control columns from the exact
screening CSV admitted by the launcher.

```bash
uv run python -m bio_inspired_nanochat.eval_stats screen/summary.csv \
  --d1-stage screening --json-out screen/decision.json
uv run python -m bio_inspired_nanochat.eval_stats confirm/summary.csv \
  --d1-stage confirmation --d1-screen-csv screen/summary.csv --json-out confirm/decision.json
```

For exploratory structural MoE results, `structural_descriptive(data)` reports matched
per-seed BPB differences for lifecycle minus fixed, fixed minus metabolism-off, fixed minus
genome-off, and glial-on minus fixed. Report observed split/merge events alongside the
lifecycle contrast so that a dormant controller is visible. These comparisons never enter
the dense confirmation family.

## Historical registration and CPU experiments

The following archived text records the earlier design and its command examples. Its 17-column
roster, overlapping seeds, three-seed threshold and 498,073,600-token launch example were
superseded by the protocol above. Historical CPU evidence and the D1 acceptance gates are
summarized in the [bridge plan](bridge_plan.md).

<details>
<summary>Original registration, dated amendments and toy-scale measurements</summary>

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

**Param-matching requirement.** The registration calls for matching capacity by adjusting
depth/width and recording both parameter counts. At `16ce9fe` the launcher applies the shared
recipe to both models, and the checkpoint summary omits those counts (`hwxb.5.9`).

---

## 2) The two ablation directions

Derived from the registry, excluding the two **infrastructure** mechanisms (`flex_attention` is
prefill-only and incompatible with KV-cache decode eval; `native_genetics` needs CUDA) — these are
performance toggles, not biology.

### Leave-one-out — marginal contribution (primary)
`bio_all` minus each **default-on** mechanism. Answers "what do we lose by removing X, given the rest?"

`bio_no_presyn`, `bio_no_stochastic_release`, `bio_no_doc2`, `bio_no_septin_barrier`. Plan
change 2026-10-07 (MoE-only mechanisms): `bio_no_metabolism` and `bio_no_genome` moved to the MoE
stage (§6) as `moe_no_metabolism` and `moe_no_genome` (the latter sets `xi_dim=0`, retaining one
learned shared phenotype while removing per-expert kinetic specialization). Both mechanisms act only
inside `SynapticMoE` (`MechanismFlag.moe_only`), and the screening recipe uses dense MLP blocks: in
the 2026-10-07 toy screening both columns reproduced `bio_all`'s validation bpb to every printed
digit on every seed, so on the dense D1 recipe they would have spent GPU time measuring nothing.
`base_train` now warns when a MoE-only mechanism is on with `use_moe=0`. Plan change 2026-10-07: `bio_no_hebbian` and
`bio_no_bdnf` left this set because the pre-registered deciding experiment for online Hebbian
plasticity (bead `hwxb.9`, `results/hebbian_chunked_regime_2026-10-07_preregistered.json`) did not
demonstrate a benefit, and its decision rule moves `enable_hebbian` (with the BDNF gain that only
acts through it) to add-one-in.

### Add-one-in — standalone effect (secondary)
`synaptic_off` plus each **opt-in** mechanism (with its prerequisites turned back on). Answers "what
does X buy on its own, on the clean architecture anchor?" Because some mechanisms require others, the
column turns on the whole prerequisite chain (e.g. `add_differentiable_recurrence` also enables
`learnable_kinetics` and `enable_presyn`); the isolated effect is then read against the matching
prerequisite-only baseline. Each column switches on exactly that chain: an opt-in mechanism whose
default value is already its on-value (the BDNF gain `bdnf_scale=1.0`, which acts whenever Hebbian
plasticity is on) is set to its off-value in every other column. So `add_hebbian` is plain Hebbian
plasticity, `add_bdnf − add_hebbian` is BDNF's increment, and the latch/STDP/neuromod columns carry
no BDNF (plan change 2026-10-07: before it `add_bdnf` built the same config as `add_hebbian`, and
the toy screening scored the two identically on every seed; `tests/test_ablation_matrix.py` now
rejects any two columns that build the same model).

Derived from the registry at import time — today ten columns: `add_hebbian`, `add_bdnf` (needs
`enable_hebbian`), `add_bistable_latch` (needs `enable_hebbian`), `add_stdp` (needs `enable_hebbian`),
`add_native_presyn` (needs `enable_presyn`), `add_learnable_kinetics` (needs `enable_presyn`),
`add_differentiable_recurrence` (needs `learnable_kinetics`, `enable_presyn`), `add_cusp_latch`
(needs `bistable_latch`, `enable_hebbian`), `add_metriplectic_integrator` (needs
`enable_presyn`), and `add_neuromod` (needs `enable_presyn`, `enable_hebbian`; the harness
instantiates the DA/ACh/NE bus for it). `python -c "from bio_inspired_nanochat import
ablation_matrix as am; print([c.config_id for c in am.add_one_in()])"` prints the live list.
`add_glial_homeostasis` is MoE-only too and became the MoE stage's `moe_add_glial_homeostasis`.

Add-one-in is more interpretable for "which mechanism helps"; leave-one-out catches interactions.
We run both where compute allows; the staging below keeps the cost bounded.

**Total screening columns:** 3 anchors + 4 leave-one-out + 10 add-one-in = **17** (locked by
`tests/test_scaleup_ablation_e2e.py::test_module_enumerates_the_full_matrix`; 20 until the three
MoE-only columns moved to the MoE stage on 2026-10-07).

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

**Throughput budget (confirmed 2026-10-13, bead `74f.10.1`: `train_ratio_max=2.0`, `decode_ratio_max=1.5`):** `tok_per_sec` and
`peak_mem_gb` are not only recorded — at D1 scale `bio_all` must reach ≥ 1/2.0 of the param-matched
vanilla's training tokens/s and ≤ 1.5× its decode latency on the same host/batch/sequence (enforced
via `ThroughputBudget` in `bio_inspired_nanochat/perf_regression.py`), or the quality effect is
judged at equal compute and the mechanism is a candidate for pruning (`hwxb.6.1`).

## 4) Staged compute + the go/no-go gate

Running all 17 columns × 3 seeds at full budget is wasteful if half the mechanisms do nothing.

1. **Screening pass** — all 17 columns × 2 seeds × 10M tokens. Cheap; drops mechanisms that clearly
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

## 6) Historical launcher and CPU experiments

Every column maps to an `eval_matrix` run and is accepted by `--preset` / `--presets`: the named
registry presets (`vanilla`, `bio_all`, `bio_no_*`) go through `ablation_registry.apply_preset`, and
the remaining columns (`synaptic_off`, the `add_*` set) are materialised by
`ablation_matrix.AblationConfig.build_syn_cfg()` (`eval_matrix.MATRIX_COLUMNS`), so the spec and the
runner cannot drift. The dry-run bead (`hwxb.7.4`) exercises the full screen→gate→confirm
orchestration on tiny models to validate the machinery before the real run (`hwxb.5.2`).

```python
from bio_inspired_nanochat import ablation_matrix as am
cols = am.screening_columns()                      # the 17 dense columns
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
launcher at revision `320ffdc` printed `eval_matrix batch`, which argparse rejected. Its pipeline
test constructed a separate scoring command. The current launcher binds training and scoring
to one manifest. The end-to-end test executes the emitted scoring command verbatim.

**Toy-scale dry runs on real text (2026-10-07 and 2026-10-13).** The three anchors first ran through
this exact chain at 2L/128d, 600k WikiText-2 tokens, seeds 1337–1339
(`results/toy_screening_2026-10-07_verdict.md`): vanilla 1.958, synaptic_off 2.028, bio_all 2.038 val bpb.
Subsequent diagnosis (`results/scaffold_diagnosis_2026-10-07.json`, `03bb9bf` / `f96747e`) showed that the
`+0.069 bpb` substrate penalty was caused by a missing final RMSNorm before `lm_head` on `GPTSynaptic`:
the normalization experiment measured mean `1.9971` for vanilla and `1.9588` for normalized
`synaptic_off` on three paired seeds (difference `−0.0383`). It contains no `bio_all` arm. On
2026-10-13 (`hwxb.5.5`), after fixing two meta-device initialization bugs in `add_cusp_latch` and
`moe_add_glial_homeostasis`, all 17 dense columns × 2 seeds (`results/toy_screening_2026-10-13_verdict.md`)
and all 5 structural MoE columns × 2 seeds (`results/structural_smoke_2026-10-13.json`) were executed
end-to-end through `matrix_launch -> base_train -> eval_matrix -> eval_stats`. For the D1 GPU run set
`BIO_FUSED_PRESYN=1` after the GPU-marked kernel tests pass, so every synaptic column uses the one-launch
presyn scan.

**Structural arm (opt-in, not pre-registered).** The expert lifecycle is a training-loop knob,
so `structural_columns()` provides `moe_fixed` (bio_all on `SynapticMoE`, fixed experts) and
`moe_splitmerge` (the same plus `--splitmerge_every=100 --sm_health_mode=credit
--split_health_min=1.5 --merge_health_max=0.35`); `moe_splitmerge − moe_fixed` is the
lifecycle's effect. `--stage structural` launches the MoE stage: the pair plus the MoE-only
mechanisms' contrasts against `moe_fixed` (`moe_no_metabolism`, `moe_no_genome`,
`moe_add_glial_homeostasis`; §2; verified end-to-end across all 5 columns in
`results/structural_smoke_2026-10-13.json`). None of it is part of the dense screening set.
History of the demand signal: under utilization-based health (product or relative) the 2026-09-02
pilots fired zero events with and without the balance loss (`results/structural_pair_pilot_2026-09-02.json`,
`_balance0.json`), so the arm measured a no-op. The credit signal (NeuroScore gradient credit, bead
`uta.9`) first also fired nothing, but that was a defect: the credit EMA was seeded with the step-0
routing proxy and stayed frozen (fixed 2026-10-07, `aeaf162`). Re-run on three seeds
(`results/structural_pair_pilot_2026-10-07_credit.json`, 120 steps, controller every 10): 3–6
splits, 14–15 merges and 2–8 resets per run, and the final loss **higher** than the no-controller
arm on every seed (+0.030 to +0.050). The D1 pair therefore now measures a lifecycle that acts; at
toy scale it costs loss, mostly through frequent merges driven by noisy first-order credit.

---

## 7) Known gaps (honest scope)

- **Global neuromodulation (`hy8.1`) is a registered mechanism since 2026-09-01** (`neuromod_enabled`,
  requires presyn + hebbian), so the matrix carries an `add_neuromod` column and `eval_matrix`
  instantiates the bus for it. **NeuroScore** is still a `SplitMergeConfig`-level knob, not a
  `SynapticConfig` mechanism, so it has no column yet.
- **Structural lifecycle** (split/merge) is toggled at the *training-script* level
  (`--splitmerge_every`), not via a `SynapticConfig` mechanism flag, so it is not one of the
  pre-registered columns; `enable_metabolism` covers the per-expert energy dynamics (MoE stage).
  Since 2026-09-01 the opt-in `structural_columns()` pair (§6) is how the lifecycle gets evidence.
- **Roadmap features vs. columns** (2026-09-02, `74f.9`): README §Roadmap lists per feature whether a
  D1 column measures it (six do), whether it is unimplemented (Rab/SNARE), a recipe knob (CA init),
  or a research module off the live path (gauge, simplicial, ultrametric). Off-path modules cannot be
  ablated and get no column.
- The **feasible mechanism set** and the **final token budget** are pinned by `hwxb.4.5`; this spec is
  parametric in both.

</details>
