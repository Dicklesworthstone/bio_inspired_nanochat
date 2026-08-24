# Bio vs Vanilla — Benchmark Matrix (Design)

This document defines the **evaluation matrix** for comparing:

- **Vanilla**: standard `GPT` (static weights)
- **Bio**: `GPTSynaptic` (synaptic dynamics + MoE lifecycle)

It is the single source of truth for what the standardized harness (bead `bio_inspired_nanochat-41s`) should run.

---

## 1) Metrics (what we measure)

### Quality

- **Val bpb** (bits/byte): computed in `bio_inspired_nanochat/loss_eval.py::evaluate_bpb`.
  - Report bpb directly (primary) and optionally convert to a token-level proxy if needed.
- **CORE metric** (Karpathy eval bundle): `scripts/base_eval.py::evaluate_model`.
- **NIAH long-context accuracy**: Needle-in-a-Haystack retrieval swept over length × needle depth — `synthetic_tasks.niah_accuracy_by_length`, emitted as both scalar `niah_acc` and the JSON `recall_by_length` curve. Sweep to 4k/8k for large models.
- **Calibration / OOD detection**: token-level `id_ece` plus sequence-level `ood_auroc`. The OOD arm applies a deterministic token/position hash that destroys sequence structure without advancing model or loader RNG; mean predictive entropy is the predeclared OOD score.
- **Continual forgetting**: `forgetting_rate` is the mean peak-to-final accuracy loss over previously acquired disjoint online-copy tasks. The full lower-triangular accuracy trace and per-task drops are retained in `capability_metrics.jsonl`.

### Bio / MoE health (for `GPTSynaptic`)

- **Routing distribution**: mean per-layer routing-count Gini (`moe_gini`) plus per-layer counts and shares.
- **Specialization**: mean specialization + histogram (`bio_inspired_nanochat/neuroscore.py`).
- **Efficiency**: loss contribution per unit energy (`bio_inspired_nanochat/neuroscore.py`).
- **Dead expert fraction**: `dead_expert_frac`, the fraction of experts whose observed routing share is below `--dead-expert-threshold` (default `0.01`). Vanilla/non-MoE runs emit `null` with status `not_applicable`, never a misleading zero.

### Capability ownership (`74f.7`)

| Owning evaluation task | Matrix evidence |
|---|---|
| `u2t.2` — calibration/selective prediction | `id_ece`, `ood_auroc` |
| `cel.4` — continual learning | `forgetting_rate`, `forgetting_by_task`, full accuracy matrix |
| `uta.7` — structural plasticity | `moe_gini`, `dead_expert_frac`, per-layer routing shares |
| `sax.4` — working memory | `niah_acc`, `recall_by_length` |

Every run writes the scalar columns to `summary.csv`/`summary.jsonl`, all applicable scalars to the
canonical results registry, and four structured records (uncertainty, continual, routing, memory) to
the run-local `capability_metrics.jsonl`. Missing capability evidence is explicit
(`not_applicable`/`error` plus a reason); optional probes are never converted to zero.

#### `uta.7` controlled equal-compute verdict

Protocol `a2307f21b18f` exercised the live variable-count UTA controller on eight held-out seeds.
Measured routing assignment shares were published as lifecycle health credit, driving a
`4 → 3 → 5 → 4` apoptosis/neurogenesis schedule. The fixed comparator stayed at four experts.
Equal phase lengths made both arms exactly match on 20 training forwards, six diagnostic forwards,
average expert count (`4.0`), top-k dispatches, cumulative router-width work, and the project's
dominant MoE matmul FLOP budget (`190,080` per seed). Controller/surgery overhead was not counted as
model FLOPs and is reported separately: mean CPU wall time was `114.4 ms` for NAS versus `102.2 ms`
for fixed in this tiny run.

| Outcome (NAS − fixed) | NAS mean | Fixed mean | Paired bootstrap 95% CI | Paired t / Wilcoxon |
|---|---:|---:|---:|---:|
| Final MSE | 0.15375 | 0.15032 | `[+0.00200, +0.00492]` | `p=0.0036` / `p=0.0078` |
| Dead-expert fraction | 0.00 | 0.25 | `[-0.25, -0.25]` | `p<machine precision` / `p=0.0078` |
| Maximum event-loss spike | 0.00201 | 0.00 placebo | `[+0.00152, +0.00285]` | `p=0.0016` / `p=0.0078` |

All exact compute, schedule, optimizer-synchronization, finiteness, and predeclared stability-gate
checks passed. The written verdict is nevertheless **regression** (`registry_verdict=invalidated`):
NAS eliminated the dormant expert on every seed and made routing more balanced (Gini `0.250` versus
`0.384`), but final loss was worse on every matched seed. The result is deliberately not rescued by
the favorable health metric. Full strict-JSON evidence is committed at
`results/structural_nas_evaluation_a2307f21b18f.json`; registry rows are in
`results/registry.jsonl` with `eligible_for_best=false`.

This CPU synthetic task isolates lifecycle behavior; it is not language-model-scale evidence. Any
future scale run must be preregistered as a fresh hypothesis rather than reinterpreting this result.

### Performance

- **Training throughput**: `train/tok_per_sec` logged by `scripts/base_train.py`.
- **Peak VRAM**: `torch.cuda.max_memory_allocated()` logged by `scripts/base_train.py`.
- **Inference latency / throughput** (planned): prompt+decode latency and steady-state tok/s.

---

## 2) Preset dimensions (what we hold constant)

For each preset below, the harness should log:

- `run_id`, `preset_id`, `seed`
- `model_arch` (depth/width/heads/seq_len)
- `train_tokens`, `total_batch_size`, `device_batch_size`, `world_size`
- metrics above + walltime/tok/s + peak memory

### Seeds

- **CI / smoke**: 2 seeds → `{1337, 1338}`
- **Research estimation floor**: 3 seeds → `{1337, 1338, 1339}`
- **Supported directional claim floor**: at least 6 matched, non-zero pairs. An exact two-sided
  Wilcoxon test cannot attain `p < 0.05` with 5 or fewer all-favorable pairs (its best possible
  value at `n=5` is `0.0625`). Two-seed smoke results are descriptive, and three-seed confidence
  intervals are preliminary; neither count should be advertised as statistically supported merely
  because the paired t-test passes.

### Statistical decision rule (`rwg` / `74f.3`)

For each metric, `bio_inspired_nanochat.eval_stats.compare_matrix` aggregates every preset with a
Student-t 95% CI and compares treatments with the baseline only on matched seeds. Across all
treatments in that metric family, paired-t and exact/approximated Wilcoxon p-values are corrected
separately with Holm's step-down procedure. A result is:

- `supported_gain` only when the mean delta is favorable, its paired-bootstrap 95% CI excludes
  zero favorably, and **both** Holm-adjusted paired tests are at or below alpha;
- `supported_regression` under the symmetric adverse rule;
- `null` when enough matched seeds exist but neither directional rule passes; or
- `insufficient_evidence` when fewer than `--min-pairs` matched seeds exist.

`null` is not an equivalence claim. Missing baselines fail closed; the CLI no longer silently
substitutes another preset. Emit durable, strict-JSON and Markdown summaries alongside a completed
matrix with:

```bash
uv run python -m bio_inspired_nanochat.eval_stats runs/eval_matrix/<batch>/summary.csv \
  --metric val_bpb --baseline vanilla --min-pairs 3 \
  --json-out runs/eval_matrix/<batch>/val_bpb.stats.json \
  --markdown-out runs/eval_matrix/<batch>/val_bpb.stats.md
```

The committed controlled structural and uncertainty studies already use matched multi-seed paired
statistics. The full 10M FineWeb Matrix A run remains pending (`4fw`), so this methodology must not
be presented as a completed language-model-scale bio-vs-vanilla result.

### Token budgets

Three budgets (to keep iteration fast, then do real comparisons):

- **Smoke**: `10M` tokens (fast sanity + regression)
- **Short**: `100M` tokens (usable signal, still cheap)
- **Medium**: `500M` tokens (research-grade)

### Sequence lengths

- **Train**: `2048` (default), optionally `1024` for smoke.
- **Long-context (NIAH)**: length × depth sweep, configurable via `--niah-lengths "16,64,128"` (v7c; default `16/64/up-to-seq_len`, clamped to the model context; use `4096`/`8192` for large models). Pass a fixed `--seed` for reproducible needle placement.

---

## 3) Config presets (what we vary)

Notation:

- `synapses=0` → standard `GPT`
- `synapses=1` → `GPTSynaptic`
- `splitmerge_every=0` disables lifecycle
- `SynapticConfig` field names refer to `bio_inspired_nanochat/synaptic.py::SynapticConfig`

Important: `scripts/base_train.py` currently does **not** expose most `SynapticConfig` fields via CLI.
The standardized harness (`bio_inspired_nanochat-41s`) should own preset → config wiring.

### Baselines

| preset_id | model | synapses | lifecycle | notes |
|---|---:|---:|---:|---|
| `vanilla` | `GPT` | 0 | N/A | Reference baseline |
| `bio_all` | `GPTSynaptic` | 1 | on | Default synaptic stack |
| `bio_all_no_lifecycle` | `GPTSynaptic` | 1 | off | `splitmerge_every=0` |

### Ablations (currently implementable via `SynapticConfig` toggles)

| preset_id | change vs `bio_all` | intended effect |
|---|---|---|
| `bio_no_presyn` | `enable_presyn=False` | remove presynaptic fatigue/vesicles |
| `bio_no_hebbian` | `enable_hebbian=False` | remove postsynaptic fast weights |
| `bio_no_metabolism` | `enable_metabolism=False` | remove expert energy dynamics |

### Parameter ablations (planned, by setting scalars to 0)

| preset_id | change vs `bio_all` | note |
|---|---|---|
| `bio_no_stochastic_release` | `stochastic_train_frac=0.0` | objective stability vs realism |
| `bio_no_doc2` | `doc2_gain=0.0` | disable slow calcium sensor path |
| `bio_no_bdnf` | `bdnf_scale=0.0` | disable metaplasticity modulation |
| `bio_no_septin_barrier` | `barrier_strength=0.0` | remove distance barrier on logits |
| `bio_no_genome` | `xi_dim=0` | shared learned kinetics; removes per-expert Xi specialization |

---

## 4) Benchmark matrix (run IDs + budgets)

The harness should implement **matrix presets** by cross-producting:

- `{preset_id} × {budget} × {seed}`

and emitting one summary row per run.

### Matrix A — Quality (train slice + eval)

| run_id template | presets | train_tokens | seq_len | seeds | required outputs |
|---|---|---:|---:|---|---|
| `Q-{preset_id}-{tokens}M-s{seed}` | `vanilla`, `bio_all`, `bio_no_presyn`, `bio_no_hebbian`, `bio_no_metabolism` | 10 / 100 / 500 | 1024 (10M) / 2048 (100M+) | 1337, 1338, 1339 | val bpb, CORE metric, tok/s, peak mem |

### Matrix B — Performance (steady-state throughput)

| run_id template | presets | steps | seq_len | seeds | required outputs |
|---|---|---:|---:|---|---|
| `P-{preset_id}-s{seed}` | `vanilla`, `bio_all` | 200 warm + 500 measure | 2048 | 1337, 1338 | avg tok/s (last 200), peak mem, mfu |

---

## 5) Runtime / cost estimates (how to predict)

Because throughput depends heavily on hardware, the harness should compute and log:

- `tok_per_sec_measured`
- `walltime_seconds`
- `tokens_processed`

Then derive the estimate:

```
estimated_walltime_seconds ≈ train_tokens / tok_per_sec_measured
```

Rule-of-thumb ranges (for planning; update with real measurements once Matrix B exists):

- **Smoke 10M** tokens: target **≤ 30 minutes** per seed on a single high-end GPU.
- **Short 100M** tokens: target **≤ 4 hours** per seed on dual RTX 4090.
- **Medium 500M** tokens: target **overnight** per seed on dual RTX 4090.

---

## 6) Recipe-faithful checkpoint evaluation (`74f.4`)

Scientific matrix rows evaluate models trained by `scripts.base_train`; `eval_matrix` does not
retrain them. This removes the former recipe mismatch (the old inline loop lacked the real LR
schedule, Muon momentum ramp, gradient clipping, bf16/DDP behavior, and crash-resumable state).
Architecture, bio configuration, learned tensors, training token counts, wall time, and final
training loss are read from the checkpoint. A preset/model-family, seed, or canonical ablation-flag
mismatch fails closed rather than producing a mislabeled row.

Train each matrix cell with a deterministic tag, then evaluate the completed checkpoint. Example:

```bash
uv run python -m scripts.base_train --model_tag=vanilla_s1337 \
  --synapses=0 --init_seed=1337 --depth=10 --max_seq_len=2048 \
  --num_iterations=954 --warmup_ratio=0.1 --total_batch_size=524288

uv run python -m scripts.eval_matrix run --preset vanilla --seed 1337 \
  --checkpoint-dir /absolute/NANOCHAT_BASE_DIR/base_checkpoints/vanilla_s1337 \
  --checkpoint-step=-1 --device-type=cuda --data=fineweb \
  --eval-bpb --core-eval --niah-lengths=512,1024,2048
```

For a preset × seed matrix, `{preset}` and `{seed}` are the only supported template fields:

```bash
uv run python -m scripts.eval_matrix matrix \
  --presets=vanilla,bio_all --seeds=1337,1338,1339 \
  --checkpoint-dir='/absolute/NANOCHAT_BASE_DIR/base_checkpoints/{preset}_s{seed}' \
  --checkpoint-step=-1 --device-type=cuda --data=fineweb --eval-bpb --core-eval
```

The same command is multi-process capable. Every rank loads the identical checkpoint, the
validation loader partitions loss/bpb/ECE work and reduces those metrics across ranks; rank 0 runs
the non-distributed CORE/NIAH probes and writes the schema-stable summary artifacts:

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_LEVEL=PXB OMP_NUM_THREADS=8 \
  uv run torchrun --standalone --nproc_per_node=2 -m scripts.eval_matrix matrix \
  --presets=vanilla,bio_all --seeds=1337,1338,1339 \
  --checkpoint-dir='/absolute/NANOCHAT_BASE_DIR/base_checkpoints/{preset}_s{seed}' \
  --device-type=cuda --data=fineweb --eval-bpb --core-eval
```

`--inline-smoke-training` keeps the old tiny loop available for fast synthetic CI plumbing tests.
It emits `recipe_source=inline_smoke_noncanonical` and must not be used for scientific comparisons:

```bash
uv run python -m scripts.eval_matrix matrix --inline-smoke-training --data=synthetic \
  --presets=vanilla,bio_all --seeds=1337,1338 --train-tokens=2048
```
