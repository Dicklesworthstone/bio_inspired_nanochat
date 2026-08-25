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

Artifact-level atomicity is not enough for a multi-file checkpoint. Before replacing any artifact,
rank 0 durably activates the directory's marker regime and publishes an **in-progress**
`commit_NNNNNN.json`. It changes that marker to complete only after every rank has crossed the save
barrier. A leading barrier also prevents any rank from replacing a same-step shard until rank 0 has
durably invalidated the previous completion marker. Discovery and explicit
`load_checkpoint(step=...)` both reject an in-progress, malformed,
or shard-incomplete step. This also makes retrying the same step safe: the old complete marker is
invalidated before a mixed old/new artifact generation can become visible. When marker mode is
first activated in an older directory, the already-complete model+metadata steps are recorded in
the regime declaration; a crash during the first new save cannot misclassify its debris as legacy.

## What is persisted (the resume must restore all of it)

| State | Where | Why |
|-------|-------|-----|
| Model params + buffers (incl. **fast weights** `w_fast`, eligibility traces) | `model_*.pt` (rank 0) | the trained model; fast weights are `Parameter`s, not transient |
| Both optimizers (AdamW + Muon), **per-rank** | `optim_*_rankR.pt` | ZeRO-style optimizer state is sharded across ranks; each rank saves its own |
| Full model params + buffers, **per-rank for distributed runs** | `train_*_rankR.pt` | DDP synchronizes gradients, not rank-local forward mutations; biological buffers and online-updated weights can diverge between ranks and must resume from that rank's state |
| Step, loop state (min_val_bpb, smoothed loss, total time), collated dataloader position, model config, trajectory-defining training config, full `SynapticConfig` + provenance | `meta_*.json` (rank 0) | resume the loop where it left off; rebuild the *exact* bio kinetics and schedule |
| **RNG state** (torch + CUDA + python + numpy), **per-rank** | `train_*_rankR.pt` | **the synaptic forward is stochastic during training** (stochastic vesicle release draws from the global RNG); without restoring RNG a resume diverges |
| Exact next untrained (`x`, `y`) batch + its rank-local post-yield dataloader cursor | `train_*_rankR.pt` | `base_train` prefetches before the checkpoint boundary; saving only the post-yield cursor silently skipped that prefetched batch on resume |

RNG blobs are **per-rank** so each rank's exact draw stream survives a save/resume boundary;
restoring rank 0's stream onto all ranks would desynchronize them from their pre-save
futures. `capture_rng_state()` / `restore_rng_state()` handle this. Note that ranks are
*seeded identically* at run start (`compute_init`, for weight-init parity), so the streams
start correlated — per-rank blobs preserve each stream exactly; they do not create
independence between ranks (jgkf).
NumPy is a required dependency, so failure to capture its global RNG aborts the save instead of
silently emitting a checkpoint that cannot honor the exact-resume contract.

The rank-0 model artifact remains the canonical portable model checkpoint. During an exact
distributed training resume, each worker then overlays the full model state saved in its own
`train_*_rankR.pt` shard. This preserves rank-local biological state without changing ordinary
single-rank/inference loading. The extra model copy is intentionally omitted for single-rank runs.

### Stateful controllers (persist when enabled)

These live in their own objects and expose their state to the `train_state` blob when active:

- **Split/merge controller** (`synaptic_splitmerge.py`): `_last_step` (so the lifecycle cadence
  resumes in phase) and the per-layer `router_logit_bias` (a `Parameter`, so already in `model_*.pt`).
- **Neuromodulatory bus** (`neuromod.py`): the DA/ACh/NE EMA levels (so gains resume smoothly).
- **Divergence guard** (`divergence_guard.py`): the last-good snapshot reference (so rollback still works).

`base_train` now persists and restores all three controller states. Restoring the neuromodulator
also rebroadcasts its gains onto the model's non-persistent runtime attributes. The divergence
guard round-trips its loss EMA, saved policy, snapshot step, and optional model/optimizer rollback
snapshot under safe `weights_only=True` loading.

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
- production-order dataloader tests — the prefetched batch is trained first after resume and the
  following loader advance matches the uninterrupted stream with no skipped/repeated batch.
- distributed state tests — rank-local model state overlays the shared rank-0 artifact without
  replacing the materialized parameters; a distributed resume missing that shard fails closed.
- commit-marker tests — explicit loads reject uncommitted steps, a failed same-step overwrite
  invalidates the old marker, the first marker-era failure preserves only prior legacy steps, and
  a declared missing rank shard makes the step incomplete.
- `test_prune_keeps_last_k_and_best` — rotation keeps the right steps, deletes only superseded ones.
- `test_resume_is_bit_comparable` — **the headline**: a synaptic run resumed from a checkpoint
  (model + optimizer + RNG) continues the *bit-identical* loss trajectory of the uninterrupted run.

At the restored checkpoint boundary, `base_train` does not repeat evaluation, sampling, or saving:
those actions happened before the checkpoint was committed and can mutate stochastic or synaptic
state. It restores the saved validation metric, controller state, exact prefetched batch, optimizer
count, training schedule, and RNG before continuing the next untrained step. Metadata and the
rank-local training shard must both declare the requested step. RNG restoration happens only after
compile wrapping and runtime/controller reconstruction, so setup work cannot consume draws from
the checkpoint's saved future.

## Caveats

- **RNG blobs load on CPU.** `load_checkpoint` loads the `train_state` RNG blob with
  `map_location="cpu"` regardless of the compute device — torch's RNG `ByteTensor`s are CPU
  tensors and `torch.set_rng_state` rejects a moved/retyped copy, so loading them onto CUDA
  would crash a GPU resume. `restore_rng_state` routes the per-GPU CUDA RNG sub-state to the
  device itself via `torch.cuda.set_rng_state_all`. A CUDA RNG checkpoint requested on a host with
  no CUDA now fails before changing any CPU RNG stream; silently performing a partial restore is
  not exact.
- **Pre-prefetch checkpoints are not exact-resumable.** A checkpoint without the saved next
  untrained batch is rejected by `base_train`; advancing its post-yield cursor would silently lose
  data, and the consumed tokens cannot be reconstructed from that cursor.
- **`torch.compile` is not covered by the bit-comparable test.** `base_train` `torch.compile`s
  the model; the resume reproducibility test (`test_resume_is_bit_comparable`) uses an
  *uncompiled* model. `torch.compile` functionalizes RNG and can change the RNG semantics of
  stochastic ops, so a compiled resumed run may differ from a compiled uninterrupted run even
  with RNG restored. The mechanism (model + optimizer + RNG capture/restore) is proven
  bit-comparable uncompiled; compiled resume is "best effort" until a compiled e2e test exists.
