# Checkpoint / Resume — Persistence Contract

> **Bead:** `bio_inspired_nanochat-hwxb.2.6`. Makes long (multi-hour) 2×4090 runs crash-safe and
> resumable **bit-comparably**. This is the contract for what a checkpoint must contain, what is
> safely *rebuilt* instead of saved, and the guarantees the loader provides.

## Why this matters

A 6-hour run that crashes at hour 5 with no resumable checkpoint wastes a day of 4090 time. Two
failure modes to defend against: (1) a crash *mid-write* leaving a corrupt file a resume then
loads; (2) a resume that silently *diverges* from the uninterrupted run because some state wasn't
restored.

## Atomicity

`save_checkpoint` writes every artifact to `<path>.tmp` and `os.replace`s it into place — atomic on
POSIX. A reader therefore sees either the previous complete file or the new complete file, never a
half-written one. A stray `*.tmp` from a crash is ignored: the loaders open the exact final names
(`model_NNNNNN.pt`, `meta_NNNNNN.json`, `optim_NNNNNN_rankR.pt`, `train_NNNNNN_rankR.pt`).

## What is persisted (the resume must restore all of it)

| State | Where | Why |
|-------|-------|-----|
| Model params + buffers (incl. **fast weights** `w_fast`, eligibility traces) | `model_*.pt` (rank 0) | the trained model; fast weights are `Parameter`s, not transient |
| Both optimizers (AdamW + Muon), **per-rank** | `optim_*_rankR.pt` | ZeRO-style optimizer state is sharded across ranks; each rank saves its own |
| Step, loop state (min_val_bpb, smoothed loss, total time), dataloader position, model config, full `SynapticConfig` + provenance | `meta_*.json` (rank 0) | resume the loop where it left off; rebuild the *exact* bio kinetics (vg9.6) |
| **RNG state** (torch + CUDA + python + numpy), **per-rank** | `train_*_rankR.pt` | **the synaptic forward is stochastic during training** (stochastic vesicle release draws from the global RNG); without restoring RNG a resume diverges — proven in `tests/test_scaleup_checkpoint.py::test_resume_is_bit_comparable` |

RNG is **per-rank** because each rank draws independently; restoring rank 0's RNG onto all ranks
would collapse their stochasticity. `capture_rng_state()` / `restore_rng_state()` handle this.

### Stateful controllers (persist when enabled)

These live in their own objects and expose their state to the `train_state` blob when active:

- **Split/merge controller** (`synaptic_splitmerge.py`): `_last_step` (so the lifecycle cadence
  resumes in phase) and the per-layer `router_logit_bias` (a `Parameter`, so already in `model_*.pt`).
- **Neuromodulatory bus** (`neuromod.py`): the DA/ACh/NE EMA levels (so gains resume smoothly).
- **Divergence guard** (`divergence_guard.py`): the last-good snapshot reference (so rollback still works).

The `train_state` dict is the extension point for these (`{"rng": ..., "step": ..., "controllers": {...}}`);
base_train currently persists RNG + step, and the controllers expose `state_dict`-style hooks as
they are scale-hardened (hwxb.4.x).

## What is NOT persisted (safely rebuilt / reset)

- **Presynaptic per-key state** (calcium, RRP, energy, buffer, priming): rebuilt fresh on every
  forward (it lives in the KV cache / is recomputed), so it never needs to be in the checkpoint.
- **Per-sequence transient adaptation** (the online fast-weight/eligibility *deltas* within a
  sequence): reset at sequence boundaries (`reset_sequence_state`). A checkpoint is taken at a
  boundary, so this is empty/irrelevant by construction.

This is why the checkpoint round-trip is verified in **eval mode after a reset** (deterministic),
while the *training* resume is verified bit-comparably *with* RNG restored.

## Rotation (keep last-K + best)

`prune_checkpoints(checkpoint_dir, keep_last=K, best_step=S)` keeps the `K` most recent steps plus
the best-by-`val_bpb` step and deletes the superseded artifacts — so a multi-day run does not fill
the disk. It is **opt-in** (the caller passes an explicit `keep_last`), only ever removes files
matching the strict checkpoint name pattern in the given dir, and logs every deletion.

## Verification (tests)

`tests/test_scaleup_checkpoint.py` (CPU, fast):

- `test_atomic_write_leaves_no_tmp_and_load_roundtrips` — no partial files; stray `*.tmp` ignored.
- `test_rng_capture_restore_is_reproducible` / `test_train_state_roundtrips_through_disk` — RNG
  restored in-memory and from disk reproduces draws exactly.
- `test_prune_keeps_last_k_and_best` — rotation keeps the right steps, deletes only superseded ones.
- `test_resume_is_bit_comparable` — **the headline**: a synaptic run resumed from a checkpoint
  (model + optimizer + RNG) continues the *bit-identical* loss trajectory of the uninterrupted run.

Wired into `scripts/base_train.py`: `save_checkpoint(..., train_state={"rng": capture_rng_state(), "step": step})`
on save; `restore_rng_state(train_state["rng"])` on `--resume_from_step`.
