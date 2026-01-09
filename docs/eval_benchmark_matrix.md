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
- **NIAH long-context accuracy** (planned): Needle-in-a-Haystack at 4k/8k context.
- **Calibration (ECE)** (planned): Expected Calibration Error on a held-out slice.

### Bio / MoE health (for `GPTSynaptic`)

- **Routing distribution**: Gini and frequency histogram (`bio_inspired_nanochat/neuroscore.py`).
- **Specialization**: mean specialization + histogram (`bio_inspired_nanochat/neuroscore.py`).
- **Efficiency**: loss contribution per unit energy (`bio_inspired_nanochat/neuroscore.py`).
- **Dead expert fraction** (planned): fraction of experts with routing freq < ε over a window.

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
- **Research**: 3 seeds → `{1337, 1338, 1339}`

### Token budgets

Three budgets (to keep iteration fast, then do real comparisons):

- **Smoke**: `10M` tokens (fast sanity + regression)
- **Short**: `100M` tokens (usable signal, still cheap)
- **Medium**: `500M` tokens (research-grade)

### Sequence lengths

- **Train**: `2048` (default), optionally `1024` for smoke.
- **Long-context (NIAH)**: `4096` and `8192` (planned).

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
| `bio_no_genome` | `xi_dim=0` (or bypass decoder) | may require code path changes |

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

## 6) Next implementation steps (unblocked work)

This design unblocks:

- `bio_inspired_nanochat-41s`: implement a CLI harness that maps `preset_id → config`,
  runs training slices, runs evals, and writes `runs/eval_matrix/*.jsonl|*.csv`.
- `bio_inspired_nanochat-1nr`: run full ablation sweep after harness exists.

Current harness implementation (initial):

- Single run:
  - `python -m scripts.eval_matrix run --preset vanilla --train-tokens 10000000 --seed 1337`
- Batch run (creates `runs/eval_matrix/matrix_<timestamp>/` unless `--batch-id` is provided):
  - `python -m scripts.eval_matrix matrix --presets vanilla,bio_all --seeds 1337,1338 --train-tokens 10000000`
- Standard ablation sweep (creates `runs/eval_matrix/ablation_<timestamp>/`):
  - `python -m scripts.eval_matrix ablation --seeds 1337,1338 --train-tokens 10000000`
