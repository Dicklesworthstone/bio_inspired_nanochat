# Dual-4090 DDP Training Path — Design & Launch Recipe

> **Bead:** `bio_inspired_nanochat-hwxb.2.1`. This documents how data-parallel training
> actually works in this repo (it is **not** stock `DistributedDataParallel`), what is and
> isn't synchronized across ranks, the correctness guarantees and their CPU tests, and the
> exact launch recipe for 2× RTX 4090 (PCIe, no NVLink).

## TL;DR

- There is **no `torch.nn.parallel.DistributedDataParallel` wrapper.** Gradient
  synchronization happens **inside the distributed optimizers** (`DistAdamW`, `DistMuon`),
  ZeRO-2 style: gradients are averaged across ranks and optimizer state is sharded.
- Each rank reads a **different data shard** (the dataloader strides row-groups by rank), so
  the effective batch is `device_batch_size × seq_len × world_size × grad_accum`.
- **Per-rank/per-batch synaptic state** (calcium, RRP, energy, eligibility traces) is **not**
  all-reduced — by design. Trained `Parameter`s (slow + fast weights, kinetics) **are** kept
  consistent across ranks (via the optimizer's all-gather and, for structural events, an
  explicit rank-0 broadcast).
- The optimizers are now correct for **every** parameter shape, including the synaptic
  model's 0-D scalar kinetics and odd-length vectors (previously a hard crash under DDP).

---

## 1. Why no DDP wrapper?

The model trains a 2-D matrix subset with **Muon** (Newton–Schulz orthogonalized SGD) and the
rest (embeddings, `lm_head`, 1-D/0-D params) with **AdamW**. Muon's update is not a simple
elementwise function of the gradient, so the standard "all-reduce grads then step locally"
DDP contract is folded directly into the optimizer instead:

| Optimizer | File | Cross-rank sync | State |
|-----------|------|-----------------|-------|
| `DistAdamW` | `bio_inspired_nanochat/adamw.py` | `reduce_scatter`(AVG) grads → update shard → `all_gather` param | sharded (ZeRO-2) |
| `DistMuon`  | `bio_inspired_nanochat/muon.py`  | `reduce_scatter`(AVG) grads → owner rank orthogonalizes → `all_gather` param | per-owner momentum |

`setup_optimizers` (`gpt.py` and `gpt_synaptic.py`) selects the `Dist*` variants when
`get_dist_info()` reports `ddp=True`, else the plain single-process `AdamW`/`Muon`. The
training loop (`scripts/base_train.py`) just calls `opt.step()` — the sync is invisible to it.

**Consequence:** the loop's `loss.backward()` produces **per-rank** gradients; they are only
averaged when `opt.step()` runs. So gradient accumulation across `grad_accum_steps` is
per-rank-local (correct — each rank accumulates its own micro-batches, then the optimizer
averages the accumulated grads once at step time).

## 2. The shardable / replicated split (the hardening fix)

`DistAdamW` shards each parameter along dim 0, which requires `param.shape[0] % world_size == 0`.
The synaptic model violates this for **small learnable Parameters**:

- **0-D scalars** — `learnable_kinetics=True` (bead `yw9.3`, scale-hardened in `hwxb.4.1`) adds
  per-layer `theta_rho_c`, `theta_alpha_ca`, … scalars. A 0-D tensor has no `shape[0]`, so the
  old code raised `IndexError` on `grad.shape[0]` the moment you enabled it under DDP.
- **Odd-length 1-D vectors** — would make `reduce_scatter_tensor` raise a size-mismatch.

`DistAdamW` now classifies each param:

- **shardable** (`ndim ≥ 1` and `shape[0] % world_size == 0`): ZeRO-2 path — `reduce_scatter`
  the grad, update this rank's slice, `all_gather` the param back.
- **replicated** (0-D, or `shape[0]` not divisible): `all_reduce(AVG)` the grad and apply the
  **full** AdamW update on **every** rank. No scatter/gather; all ranks stay bit-identical
  because they start identical (same init seed) and apply the same update to the same averaged
  grad.

The per-element AdamW math is a single shared helper (`_adamw_update_`), so the two paths are
provably equivalent. `DistMuon` already tolerates arbitrary param counts (it block-cyclically
assigns an owner rank per param and pads short groups with a zero buffer) and requires all
params be 2-D, so it needs no analogous change.

## 3. What is / isn't synchronized

| State | Synced across ranks? | Mechanism |
|-------|----------------------|-----------|
| Model `Parameter` gradients | **Yes** (averaged) | `DistAdamW` / `DistMuon` |
| Model `Parameter` values | **Yes** (kept identical) | identical init seed + optimizer all-gather/replicated update |
| Presyn per-key state (calcium, RRP, energy, buffer, priming) | **No** (per-rank, per-batch) | rebuilt each forward; never registered for reduction |
| Hebbian eligibility traces (`u_buf`, `v_buf`) | **No** (per-rank) | per-sequence buffers, reset at boundaries |
| Fast weights `w_fast` (a trained `Parameter`) | **Yes** | flows through the optimizer like any param |
| MoE metabolism EMAs (energy/fatigue) | drift per-rank between events; **re-synced** on structural events | `synaptic_splitmerge.py` `_broadcast_module_params` (rank-0 → broadcast) when `ddp_broadcast=True` |
| Split/merge decisions | **rank-0 decides**, then broadcasts new weights | `synaptic_splitmerge.py` guards `dist.get_rank()==0` + `dist.barrier()` |

The per-rank presyn/eligibility state is **correct to keep per-rank**: it is a function of each
rank's own data batch, analogous to per-rank activations. Averaging it would be meaningless. The
only thing that must stay globally consistent is the set of trained `Parameter`s and the
discrete structural topology — both handled above.

## 4. Numerics under bf16

Training uses `torch.amp.autocast(dtype=bfloat16)` on CUDA. The synaptic kinetics assume a
reasonable numeric range (calcium/RRP are O(1), Hill terms are bounded). bf16 has ~3 decimal
digits of mantissa, which is fine for the gated/bounded dynamics but means: keep the
`logit_softcap` on for the synaptic head (it bounds the `log(ε+release)` bias injected into
attention logits), and watch `spectral_radius` telemetry (bead `yw9.7`) — a value ≥ 1 signals a
kinetics instability that bf16 will amplify. Muon's Newton–Schulz already runs in bf16 by design.

## 5. NCCL for PCIe (no NVLink)

Two 4090s on a desktop talk over PCIe, not NVLink, so all-reduce bandwidth is the bottleneck.
Recommended environment for the launch:

```bash
export NCCL_P2P_LEVEL=PHB        # trj GPU0<->GPU1 topology is PHB (same NUMA node 0 PCIe Host Bridge)
export NCCL_IGNORE_DISABLED_P2P=1 # consumer RTX 4090 driver disables BAR1 P2P; suppress warning & use 4-ch SHM ring
export NCCL_SHM_DISABLE=0        # keep pinned SHM/direct/direct ring ON (+27.4% faster step than socket copy)
export NCCL_DEBUG=WARN           # surface NCCL issues without spamming
export OMP_NUM_THREADS=8         # avoid CPU oversubscription with 2 ranks
export CUDA_VISIBLE_DEVICES=0,1
```

On `trj` (`nvidia-smi topo -m`: `GPU0 <-> GPU1 = PHB`, NUMA node 0, CPUs `0-127`), the NVIDIA driver (`595.99.02`)
disables direct PCIe BAR1 peer-to-peer DMA between consumer GeForce RTX 4090 cards (`Check P2P Type
isAllDirectP2p 0 directMode 0`), and NCCL automatically establishes a 4-channel pinned shared-memory ring
(`Channel 00..03 : 0[0] <-> 1[1] via SHM/direct/direct`). Keeping `NCCL_SHM_DISABLE=0` (default) achieves
`340.69 ms/step` (`96,180.8 tok/s`, `81.75 ms` collective sync overhead) on `D1` `vanilla` (`batch=16, seq=1024`)
versus `434.04 ms/step` (`75,495.7 tok/s`, `175.10 ms` sync overhead) with `NCCL_SHM_DISABLE=1` — a **2.14×
reduction in collective communication time (+27.4% end-to-end step throughput)** for `vanilla` and a **2.05×
reduction in collective communication time (+12.4% end-to-end step throughput: `37,428.7` vs `33,291.3 tok/s`)**
for `bio` (`results/scale_memory_2x4090_{2026-10-14,noshm_2026-10-14}.jsonl`).

## 6. Launch recipe

Single GPU (smoke / debug):

```bash
.venv/bin/python -m scripts.base_train --depth=8 --max_seq_len=1024 \
  --device_batch_size=16 --total_batch_size=131072 --num_iterations=50
```

Dual 4090 (the real path) — `torchrun` sets `RANK`/`LOCAL_RANK`/`WORLD_SIZE`, which
`common.get_dist_info()` reads:

```bash
export NCCL_P2P_LEVEL=PHB NCCL_IGNORE_DISABLED_P2P=1 OMP_NUM_THREADS=8
.venv/bin/torchrun --standalone --nproc_per_node=2 -m scripts.base_train \
  --depth=10 --max_seq_len=1024 \
  --device_batch_size=16 --total_batch_size=262144   # = 16*1024*2*8 grad-accum
```

`total_batch_size` must be divisible by `device_batch_size × seq_len × world_size`
(`base_train.py` asserts this and derives `grad_accum_steps`).

## 7. Correctness guarantees & tests

**CPU-runnable (CI, no GPU)** — `tests/test_scaleup_ddp.py` spawns a real 2-process **gloo**
group and proves:

- `DistAdamW(world_size=2)` on split gradients == single-process AdamW on the averaged gradient,
  **including a 0-D scalar and an odd-length vector** (the replicated path), to fp32 precision.
- `DistMuon(world_size=2)` == single-process `Muon` on averaged gradients (bf16 NS tolerance).

These validate the parts that are hardware-independent: gradient averaging, optimizer-state
sharding + re-replication, and the non-shardable fallback. (To make them run on CPU, both
optimizers now `.wait()` on the collective `Work` handle instead of `.get_future()`, which gloo
does not implement; NCCL behavior is unchanged.)

**GPU-gated (manual, requires the 2×4090 box)** — the remaining acceptance from the bead: a
2-GPU run (vanilla **and** synaptic) whose loss trajectory matches a single-GPU run at matched
effective batch, with no NCCL hangs and finite loss. Run the launch recipe above on the
hardware and diff the `train/loss` curves; they should agree within bf16 tolerance. This is the
one piece that cannot be exercised without the GPUs and is tracked as the residual manual check.

## 8. Reproducible profiling matrix (`j9i`)

The memory helper is also the canonical fixed-seed train/inference performance harness. It sweeps
vanilla vs bio, batch size, and sequence length; synchronizes every CUDA timing window; records
aggregate tok/s, slowest-rank latency, rank-0 peak allocated/reserved VRAM, NVML GPU
utilization, and optional per-rank Chrome traces as schema-stable JSONL.

Single 4090:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m scripts.scale_memory --matrix \
  --device cuda --depth 10 --tie-embeddings \
  --variants vanilla,bio --modes train,infer \
  --batches 8,16 --seqs 512,1024 \
  --warmup 2 --steps 5 --seed 1337 \
  --output-jsonl results/scale_memory_1x4090_2026-10-14.jsonl
```

Dual 4090 uses the same matrix under `torchrun`; each JSONL row records `WORLD_SIZE`, and tok/s is
reported as aggregate device throughput. Keep the PCIe environment from §5:

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_LEVEL=PHB NCCL_IGNORE_DISABLED_P2P=1 OMP_NUM_THREADS=8 \
  uv run --no-sync torchrun --standalone --nproc_per_node=2 -m scripts.scale_memory --matrix \
  --device cuda --depth 10 --tie-embeddings \
  --variants vanilla,bio --modes train,infer \
  --batches 8,16 --seqs 512,1024 \
  --warmup 2 --steps 5 --seed 1337 \
  --output-jsonl results/scale_memory_2x4090_2026-10-14.jsonl
```

Committed RTX 4090 baseline artifacts (`2026-10-14`, `trj`):
- `results/scale_memory_1x4090_2026-10-14.jsonl` — single-GPU `D1` (`depth=10, d=640, tie=True`) matrix
  (`vanilla` train `63,272.4 tok/s`, `7.71 GB`; `bio` train `21,230.8 tok/s`, `14.84 GB` at `batch=16, seq=1024`).
- `results/scale_memory_2x4090_2026-10-14.jsonl` — dual-GPU `D1` matrix (`vanilla` train `96,180.8 tok/s`,
  `7.55 GB/rank`, `100.0%` GPU util; `bio` train `37,428.7 tok/s`, `14.69 GB/rank`, `1.763×` scaling).
- `results/scale_memory_2x4090_noshm_2026-10-14.jsonl` — `NCCL_SHM_DISABLE=1` comparison (`75,495.7 tok/s`
  `vanilla`, `33,291.3 tok/s` `bio`).
- `results/profiles/d1_4090_profile_2026-10-14.json` — `torch.profiler` CUDA kernel breakdown at `D1`
  (`batch=16, seq=1024`) comparing unfused TorchScript scan (`1,061,484` kernel launches, `74,958.6 ms/step`
  presyn / `159,242.7 ms/step` bio_all) against fused Triton `presyn_detached_scan_kernel` (`158` launches =
  `93.36 ms` CUDA time, `1,280.84 ms/step` bio_all = `124.3×` speedup, `1.870×` vs `vanilla`).
