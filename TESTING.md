# Testing Guide

### Reproduce the CUDA timeline

On a qualified RTX 4090 worker with the project GPU environment and Nsight Systems on PATH, run

```bash
DSR_REPOS_FILE="$PWD/dsr/quality.yaml" dsr --json quality --tool bio_inspired_nanochat-cuda-atlas --work-dir "$PWD" --check-timeout 1800
```

Open `runs/dsr/$DSR_RUN_ID/atlas/atlas.svg` and compare its four named windows to the
[recorded timeline](results/profiles/rtx4090_cuda_atlas_2026-11-22.svg).
The [timing data](results/profiles/rtx4090_cuda_atlas_2026-11-22.json) record kernel counts,
the union of their active intervals and time in CUDA runtime calls. The cached synaptic
model launched 124 kernels per token versus 75 for vanilla. Kernels occupied 40.1 ms
of its 5,259 ms profiled window versus 27.9 ms of 2,380 ms for vanilla. Host scheduling
and tracing overhead enter both wall times. Use the ordinary throughput gate for speed claims.

### Check checkpoint arithmetic on an RTX 4090

Give a DSR GPU worker this payload with `NANOCHAT_BASE_DIR` pointing to the
original tokenizer and train/validation shards. Set `CHECKPOINT` to the directory
containing the model and metadata, and `STEP` to its saved optimizer step.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  uv run --no-sync python -m scripts.e2e.precision_sweep \
  --checkpoint-dir "$CHECKPOINT" --step "$STEP" --out "runs/dsr/$DSR_RUN_ID/precision-$STEP.json"
```

The first sweep used the inference head at `6434469`. Its reference uses FP32 matrix weights and PyTorch's `highest` matmul setting.
TF32 allows `high` precision matmuls. BF16 and FP16 use CUDA autocast with
`high`. Every arm retains the model's rotary tables and stored biological
state. Each of four held-out blocks supplies 32 prompt tokens and eight cached
next-token predictions. Each corresponding training block supplies 32 distinct
training positions for one cold, unscaled AdamW update from the same checkpoint.
The report includes the individual times, paired intervals, input hashes and
excluded warmup. Speed ratios above one favor the indicated mode.

| Checkpoint | Mode | Greedy matches with FP32 | Mean absolute block loss difference (nats) | Median training speed ratio | Median cached speed ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| Vanilla, step 24 | TF32 | 32/32 | 0.000027 | 1.196 | 1.251 |
| Vanilla, step 24 | BF16 | 0/32 | 0.004556 | 1.150 | 0.898 |
| Vanilla, step 24 | FP16 | 0/32 | 0.000373 | 0.943 | 0.954 |
| Synaptic, step 2 | TF32 | 32/32 | 0.001819 | 1.065 | 0.661 |
| Synaptic, step 2 | BF16 | 0/32 | 0.009144 | 1.175 | 0.362 |
| Synaptic, step 2 | FP16 | 2/32 | 0.003224 | 1.213 | 0.385 |

All updates and cached logits were finite. None of the six comparisons met the
joint rule for accuracy and a reproducible speed gain in either workload. Use
the FP32 reference for reproducing these greedy decisions. The existing CUDA
trainer uses BF16 autocast. The public chat CLI accepts `--dtype float32` or
`--dtype bfloat16` and its shared CUDA initializer enables `high` matmul
precision. Use the sweep for an explicit `highest` reference. These four
single-device blocks do not qualify a change to training or serving defaults.
[Vanilla measurements](results/profiles/precision_vanilla_2026-11-23.json) ·
[Synaptic measurements](results/profiles/precision_bio_2026-11-23.json)

An independent replay reproduced all 128 original vanilla choices. On each
of the 32 BF16 predictions the bounded logits gave 289–468 tokens the same
maximum score. FP16 gave 72–90 tokens the same maximum. Keeping the raw
lower-precision matrix product and evaluating just the final tanh in FP32
restored the FP32 choice on all 32 positions in both modes. Even the FP32
reference had a two-way tie twice. [Token-by-token replay with source](results/profiles/vanilla_softcap_replay_2026-11-23.json)

The corrected inference head computes the softcap in FP32 when the matrix
product returns BF16 or FP16. On the original blocks the two checkpoints
matched all 64 FP32 greedy choices in each format. On another four blocks
per checkpoint, selected with `--skip-blocks 5`, each format again matched
64/64. On the original blocks all 40 training losses, including warmups, were
identical to their pre-fix values. None of the follow-up cohorts cleared the
registered speed rule. These are teacher-forced next-token decisions from two
early checkpoints. [Vanilla replay](results/profiles/precision_repaired_vanilla_2026-11-23.json) ·
[Synaptic replay](results/profiles/precision_repaired_bio_2026-11-23.json) ·
[Later vanilla blocks](results/profiles/precision_repaired_holdout_vanilla_2026-11-23.json) ·
[Later synaptic blocks](results/profiles/precision_repaired_holdout_bio_2026-11-23.json)

During generation the widened softcap reuses its FP32 output buffer. At a
1,024-token vanilla prefill, peak CUDA allocation above inputs and weights
was 531,784,704 bytes, the same as the original BF16 head. The first FP32
implementation used 934,437,888 bytes. All 67,108,864 returned FP32 scores
had the same byte digest across the two FP32 implementations. At 32 tokens
all 2,097,152 scores also matched. [Paired allocator readings and tensor hashes](results/profiles/prefill_projection_2026-11-23.json)

The public `Engine.generate` also ran freely for 16 tokens on two later prompts
per checkpoint. The old BF16 and FP16 paths each matched 0/64 aligned FP32
tokens. The corrected paths each matched 64/64, including all four complete
continuations. Every FP32 continuation repeated one token throughout. This
comparison establishes reproducibility of greedy generation on these pilots.
[Prompt IDs and complete generated sequences](results/profiles/greedy_generation_precision_2026-11-23.json)

> The test discipline for `bio_inspired_nanochat`. Established by bead **eqyk.1**
> (*Unit-test framework, conventions, fixtures & coverage gate*) — the foundation
> every later "unit tests + e2e + detailed logging" bead builds on.

Run project checks through DSR. The command examples below describe the payloads for its
workers. The repository policy covers CPU, GPU, scheduled checks and release automation.
The standing DSR profiles are `bio_inspired_nanochat` for main and `bio_inspired_nanochat-nightly`
for the two nightly suites. Both run the repository's gate definitions with local `act` and retain
artifacts under `runs/dsr/$DSR_RUN_ID/`. Use a private, pinned checkout on the Linux worker,
with Docker, `act`, a static Linux `minisign` on PATH and the
`catthehacker/ubuntu:full-22.04` image. The local runner copies the checkout and uses
the Linux worker tool-cache UID (1000).
The main quality job installs the signed UBS installer and executable from the
same v5.5.26 release into its own cache. Mixing the development installer with
that released runner failed after checksum verification because their cache
publication protocols differed.
The project registry can also be used on a fresh dispatcher:

```bash
DSR_REPOS_FILE="$PWD/dsr/quality.yaml" dsr --json quality --tool bio_inspired_nanochat --work-dir "$PWD" --check-timeout 14400
DSR_REPOS_FILE="$PWD/dsr/quality.yaml" dsr --json quality --tool bio_inspired_nanochat-nightly --work-dir "$PWD" --check-timeout 14400
```

Beads `u0fl.1` and `u0fl.2` track terminal main, nightly and hardware qualification.

On November 22 the two-layer CUDA regression scenarios ran on one RTX 4090
with driver 595.99.02 and one host thread. The first acquisition and an
independent replay used synchronized CUDA timing and peak allocated memory.
All four absolute comparisons stayed within the 45% tolerance. On replay,
vanilla and synaptic training reached 476 and 347 tokens/s (1.37× slowdown).
Cached single-token decoding reached 124 and 51 tokens/s (2.43× slowdown),
exceeding the 1.50× budget. The relative gate stays red. These measurements
cover the small single-card test geometry. [First CUDA baseline](results/perf_baselines_rtx4090.json)

On November 24 the same registered CUDA gate measured 315.6 vanilla and 105.2
synaptic cached tokens/s on the same RTX 4090, a 3.00× slowdown. Nsight counted
75 vanilla launches and 124 eager synaptic launches per measured token. A
single fused state append reduced the synaptic count to 90. The first build
changed a newly initialized negative zero to positive zero and increased
median paired decode latency by 7.23%. Copying and filling the raw FP32 bits
then matched 20 state fixtures and 66 actual cached decode steps byte for byte.
After warming the complete 200-token window on both paths, the bit-preserving
append reduced median paired decode latency by 24.3% and won seven of eight
pairs. Median slowdown against live vanilla was 1.63× versus 2.33× for eager
synaptic append. The 1.50× ceiling still applies.

One run through the normal gate with the production change measured 372.0
vanilla and 321.5 synaptic cached tokens/s, passing the decode gate at 1.16×.
The same invocation failed the relative training gate at 2.26× against 2.00×.
The physical CUDA suite passed all 11 selected tests, and the production
checkout reproduced the 20 direct fixtures and 66 cached steps.

The fixed-shape CUDA graph experiment reused each graph on four fresh
prefixes with refreshed KV and recurrent state. Across the vanilla, synaptic
and packed synaptic arms, all 2,784 logit vectors matched their eager oracles,
as did the final caches. The packed arm accounted for 928 vectors. The
packed replay, including state refresh, took 1.21× vanilla graph time for the
200-token forced window and 1.00× for the 32-token seeded greedy window. The
same two ratios including fresh prefill and 20 eager warmup forwards were
3.95× and 3.34×, respectively. The fixed-window benchmark measures capture
separately.
[Decode observations and receipts](results/profiles/rtx4090_cached_decode_2026-11-24.json)

The single-prefill follow-up compared one causal forward to a 16-token
prefill followed by one-token forwards over identical prefixes. All 18 cases
at 36, 96 and 137 tokens met `rtol=atol=1e-4` for the prefix, forced suffix,
KV cache and recurrent state. The seeded greedy continuations agreed on all
384 positions, including 360 model-selected tokens.
At 36 tokens, median synaptic prefix preparation measured 14.5 ms in one
call versus 256.6 ms on the staged route. That timing retains and concatenates
the staged prefix logits.

With one 36-token prefill outside the graph, the packed request including the
prefill measured 2.56× vanilla for 200 forced tokens and 1.81× for 32 seeded
greedy tokens. Capturing the prefill as well matched the eager output and
cache in all 36 sampled requests. These fixed-shape measurements begin with
token IDs on the GPU and account for capture time separately.

On the interleaved holdout, the median whole-request graph ratios against
live vanilla graphs were 1.17× for 200 forced tokens after a 36-token prefix,
1.32× for 32 greedy tokens after a 36-token prefix and 1.32× for 32 greedy
tokens after a 128-token prefix. Twenty-two of the 24 pairs met 1.50×.
The holdout matched 7,424 logit vectors and the final caches against eager.

The two-graph experiment delivered the first token to the host before
replaying the remaining 31. Its first graph ran one cached decode forward
after prefill and before the host read. At 36 and 128 prompt tokens, median
first-token ratios against vanilla were 1.44× and 2.27×. Complete-request
ratios were 1.34× and 1.32×. All 2,736 checked logit vectors, emitted IDs and
final caches matched eager.

Nsight counted 165 synaptic launches per 36-token prefill and 193 per
128-token prefill, versus 86 for vanilla at both lengths. At 128 tokens,
the presynaptic scan accounted for 5.82 of 8.56 ms of synaptic kernel-busy
time across eight requests. Tracing the first-token graph directly gave
5.76 ms in the scan and 9.58 ms of total synaptic kernel-busy time across
eight requests, versus 2.61 ms total for vanilla. The unprofiled replay
supplies the latency comparisons above.
[Single-prefill measurements and receipts](results/profiles/rtx4090_single_prefill_2026-11-24.json)

A follow-up moved the first host read to immediately after the prefill
argmax. The experimental scan assigns disjoint sets of keys to CUDA
programs, so each program advances its own recurrent state. In 30 paired
128-token trials, it reduced median paired first-token latency by 46%
relative to the existing synaptic scan. The ratio against live vanilla was
1.94×, with nine pairs within 1.50×. At 36 tokens it was 1.41× vanilla, with
17 of 30 pairs within 1.50×. The one-warp and two-warp variants of the
existing scan increased median paired first-token latency at both lengths.
All compared tokens agreed, and the largest absolute logit difference was
2.39e-7 across the two prompt lengths and both graph boundaries. The
128-token two-graph full-request median ratio was 1.24× vanilla.

The initial prototype exceeded logit tolerance on all 30 tested 512-token
prompts. At that length, both warp-count controls matched the existing
scan exactly. Replaying eight captured blocks against the TorchScript
recurrence placed the first release divergence in the prototype at the
193-key frontier, with identical drive values and selected keys. The
incumbent passed the reference tolerance on all eight blocks.

Keeping each key's recycling queue in registers restored parity on the
eight captured blocks and six direct cases. Across 180 paired graph
requests at 36, 128 and 512 tokens, all emitted IDs matched and logits,
KV and recurrent states stayed within the existing tolerance. The
register variant reduced median paired first-token latency by 27%, 47%
and 74%, respectively, relative to the existing synaptic scan. The
same-invocation ratios against vanilla were 1.47×, 1.85× and 3.71×.
At 512 tokens the complete 32-token graph request reached 0.97× vanilla
by paired median, with 29 of 30 within 1.50×.

The width-640, 10-layer synthetic model exceeded logit tolerance on one
of 24 independently replayed 128-token prompts. All 24 first-token IDs
agreed. The first attention key-set change and the first out-of-tolerance
logits both appeared at zero-based position 118. Eager and captured runs
both reached an absolute logit error of 0.00267 on that prompt.

Replaying two frozen 64-query blocks from that prompt found 61 and 54
bitwise release differences with the default Triton builds. The two
kernels matched bit for bit with multiply-add contraction disabled on
both. Explicitly matching the incumbent's fused rounding for the calcium,
buffer and priming updates restored bitwise parity against the normal
incumbent on both blocks. In a replay of the original 24-prompt panel,
all 3,072 logit vectors, emitted first tokens, KV and recurrent state
matched bit for bit. Median paired first-token latency fell 44% relative
to the live synaptic incumbent. The repaired prototype took 2.09× live
vanilla, with zero of 24 pairs within 1.50×. The experiment uses the same
synthetic model and prompt seeds as the counterexample.
[Rounding diagnosis, repair and DSR receipts](results/profiles/rtx4090_scan_rounding_2026-11-24.json)

The live `Engine.generate` now copies the decode cache only when generation
needs a second token. The previous and revised engines agreed on 2,688
sampled token and mask positions, plus 210 positions with metrics enabled.
With four samples and a 128-token prompt, the revised synaptic engine
reached the first token 18% faster by median paired latency and won 18 of
24 comparisons. The single-sample measurements ranged from 7.5% faster to
6.6% slower, and the four-sample case was still 2.42× the live vanilla
engine. The four production regression cases exercise one-token and
continuing streams with both single and multiple samples.
The focused Linux DSR run passed 70 tests with one CUDA skip. Ruff and ty
passed. The full CPU suite reached its 15-minute deadline after reporting
failures.
[Streaming and scan experiments](results/profiles/rtx4090_first_token_2026-11-24.json)

The 2,103-case CPU run recorded 2,088 passes, 14 skips and one stale formal
runtime hash. Refreshing the binding passed all 17 formal-feedback tests and
the mapped runtime suite's 288 passing cases with three skips. The binding
uses the same Lean source digest as the archived compiled axiom audit.

The `bio_inspired_nanochat-gpu` profile uses a private checkout and a dedicated uv
environment on the two-4090 worker. It builds the source-matched native extension,
runs the CUDA-marked and canonical/Rust parity cases, then executes the registered
same-invocation presynaptic and throughput comparisons. The run directory contains
JUnit results and raw timing JSON. Invoke the GPU profile explicitly when the
cards are available. Absolute throughput comparisons require an independently
measured `results/perf_baselines_rtx4090.json`; the committed CPU timings
belong to a different device. Relative train/decode comparisons always run
both model variants in the same invocation.

The audit exercised two-rank vanilla checkpointing, greedy and selective web generation,
the default bio Quick Start and the printed matrix scoring command. The research host's
installed native CPU extension exposed the scan symbol with an older return signature.
The default bio trainer failed during evaluation. Explicitly selecting the reference backend
completed two training steps with the same inputs. That wheel's installed Linux CPython 3.14
shared library had SHA-256 `1ff8e09978dc7abded9932afc33af753d68bc20f1677711e948f6a8d25061c27`.
CPU training now checks the advertised presynaptic ABI before allocating a model. ABI 1
returns the release tensor and gradient factor from the detached scan. An unversioned or
incompatible native kernel raises an error that identifies the loaded package and explains
how to rebuild it. A missing extension or tokenizer-only install uses the reference recurrence.
In a Python 3.14 environment without `rustbpe`, both settings of `native_presyn` produced
bit-identical releases, drive gradients and synaptic state for the same seeded stochastic scan.
[Original probe results](results/project_reality_check_2026-11-22.json)

### Reproducing the CPU presynaptic checks

Use a fresh Python 3.14 environment on a Linux DSR worker. Run the build and tests
in the same checkout. The following are DSR check payloads.

```bash
uv sync --frozen --extra cpu --dev
uv run --no-sync maturin build --release --locked
uv run --no-sync maturin develop --release --locked --uv
uv run --no-sync python -m pytest tests/test_e2e_quick_start.py -v
uv run --no-sync python -m pytest tests/test_presyn_scan_kernel.py tests/test_presyn_rust_dispatch.py -v -m 'not gpu'
```

The Quick Start tests train a tokenizer, train a CPU bio model, reload its checkpoint
and exercise chat on a small local corpus. The recurrence tests compare native and
reference outputs, gradients and mutable state on identical inputs.

On November 22 DSR built `bio_inspired_nanochat-0.1.0-cp314-cp314-manylinux_2_34_x86_64.whl`
from the ABI change on top of `320ffdc8`. Its SHA-256 was
`d043271764179c0f6fc5b46805efc0c34ce8b29942060d2be57f3f10cd7d9b14`.
The worker used Python 3.14.2, installed the wheel into an isolated uv environment and
reused existing Torch dependencies. The packaged and loaded shared libraries both hashed
to `be4c6deb888e23e9e4934b9d3addd8445ed2266c8213523eb9ff2d59b3562854`.
The original wheel failed before model allocation with the new ABI diagnostic.
The unmodified nine-case CPU Quick Start suite passed on the rebuilt wheel in 963.47 seconds.

Both the default native trainer and `--syn_cfg.native_presyn=0` wrote complete two-step
checkpoints on identical local parquet shards. Validation BPB was 1.4349256529350143
and 1.434935665032893, respectively. Reloading the checkpoints and scoring “The synapse”
gave a maximum absolute logit difference of 0.0007157624. Of 904 logits, 874 exceeded
the predeclared `rtol=1e-5, atol=1e-6` comparison. Both training paths run, and this
strict end-to-end parity check failed.
Repeating each backend on its own reproduced all 195 model-state tensors and the prompt
logits bit for bit. Across backends, 109 of 195 tensors already differed after step one.
The maximum logit differences on that prompt were 0.0003334433 at step one and
0.0007157624 at step two. All 10 selected CPU scan and decode tests passed. The hardware
GPU case was deselected for that check.

Tracing the first forward showed that the first layer's first valid edge already differed
with identical inputs and CPU autocast off. Training release was 1.4492371082 in Rust
and 1.4492373466 in the reference. Before the first Muon step all weights were identical.
The first layer's query-projection gradients differed by at most 9.90e-10,
enough for two entries to round to different BF16 values. After Muon's Nesterov
interpolation, 14 entries in that matrix rounded differently before its Newton–Schulz
iterations. The updated weights differed by up to 0.0003124997. Both instrumented runs reproduced their
195 final checkpoint tensors exactly.

At the first edge, the scalar Rust sigmoid of the captured drive returned 0.2750027180,
while the Torch tensor sigmoid returned 0.2750027478. Using Torch to precompute sigmoid
and softplus for the real Rust scan made that edge match exactly in evaluation and training.
The resulting model still missed the original tolerance on 889 of 904 logits. Changing
only the normalization to direct division missed on 885. Both experiments stayed out of
the production path. [Measured trajectories and DSR receipts](results/cpu_native_trajectory_2026-11-22.json)

Replaying the first Muon step reproduced all 48 archived post-step matrices
exactly. The native/reference L2 weight gap was 0.021105 with BF16 and
0.0000068874 with FP32 orthogonalization on the same gradients.

The four-arm, two-step CPU training probe used the original parquet and tokenizer
inputs. At step two, 892 of 904 logits missed tolerance in the BF16 pair. All
904 passed with FP32 orthogonalization, with a maximum absolute difference of
`4.77 × 10⁻⁷` and validation BPB 1.434953 on both backends. The precision change
also changes the CPU training trajectory within each backend. The reference
backend's FP32 and BF16 logits differed by up to 0.00404 on this prompt.

The production CPU Muon path now uses FP32. DSR rebuilt the locked CPython
3.14 Linux wheel with SHA-256
`56baec1f242c4a3ff48728e6413c0c254326462ed9d0c240539ce4468f45025a`.
Its installed extension matched the previously qualified ELF byte for byte.
Two new runs through `scripts.base_train` and the production checkpoint loader
reproduced zero of 904 tolerance violations at both steps. The fresh-wheel
Quick Start, scan and decode run passed 20 tests and skipped one CUDA-only
test. The optimizer and native kernel subset passed 14 tests with the same
CUDA-only skip. [CPU Muon evidence and DSR receipts](results/cpu_native_muon_2026-11-24.json)

The unchanged live CPU throughput test uses AdamW and failed on both the
pre-change and FP32 Muon revisions on the same DSR worker. The measured
synaptic training slowdowns were 4.54× and 3.20× against a 2.00× ceiling.
In a separate diagnostic profile, the native detached-scan wrapper accounted
for the most self time among the measured synaptic training frames. Performance
follow-up is tracked in `bio_inspired_nanochat-r2d`.
[Paired throughput receipts](results/cpu_native_muon_2026-11-24.json)

The follow-up on `csd` measured 30 warmed scan calls. Median wrapper time
was 1.955 ms, including 0.914 ms in the native call, 0.546 ms in preparation,
0.044 ms at the Python-to-Rust boundary and 0.432 ms after the call. RNG
accounted for 0.389 ms within preparation. Across the 30 calls, the native
call accounted for 53.9% of measured wrapper time.

Two isolated Rust builds preserved the sampled release, gradient, state and
delay bytes. Skipping masked-edge arithmetic improved median paired training
slowdown by 1.67% over 12 pairs. Reusing per-field state views improved it by
2.61% over a separate 12 pairs. Both missed the 5% intervention threshold and
their median slowdowns, 2.07× and 2.24×, exceeded the existing 2.00× gate. Each
pair used one live vanilla measurement in the same process. Both candidates
were left out of the production kernel.

Preallocating the TorchScript random-draw buffers reproduced the draws and
generator states byte for byte in 192 cases. The candidate increased median
paired training slowdown by 5.78% across 12 fresh pairs. Its median measured
32-query RNG time rose from 0.275 ms to 0.309 ms. The experiment stayed out of
the trainer.
[Scan timings and DSR receipts](results/profiles/cpu_presyn_scan_2026-11-24.json)

A later single-thread profile measured a median slowdown of 2.14× against
live vanilla training. Zero of 20 pairs met the 2.00× ceiling. Within 42
traced scan calls, the median time was 0.92 ms for native dispatch, 0.43 ms
for stochastic input preparation, and 0.10 ms for the scalar EMA scan.

A Rust trial reused calcium from the release calculation for keys with one
finite selected drive. Keys selected multiple times kept the ordered sum.
All 144 native fixtures and three AdamW updates matched the incumbent bit
for bit. The isolated scan took 12–14% less time across three shapes. In
12 rotated live training pairs, the candidate had a median slowdown of
2.03× and three pairs met the ceiling. Its median paired training time fell
2.2% versus the installed wheel and 0.8% versus the rebuilt parent. The
trial missed the fixed training gate and the production kernel stayed on
the incumbent.
[Profile, candidate patch, and paired DSR receipts](results/profiles/cpu_presyn_postprocess_2026-11-24.json)

The two-process Gloo tests passed all six cases on both revisions on `csd`
with PyTorch `2.9.1+cpu`. On `trj` with CUDA hidden and PyTorch `2.9.1+cu128`,
both revisions reproduced the broad run's five failures. Four
optimizer cases exceeded the 30-second worker deadline and the deliberate
crash case exceeded its five-second deadline. The timeout cleanup case
passed. The existing deadlines were used on both hosts.
[Distributed optimizer receipts](results/cpu_native_muon_2026-11-24.json)

The single-worker CPU suite reached 1,578 of 2,100 collected cases before
DSR's 90-minute deadline. The first 1,485 cases produced 1,473 passes,
11 skips and the live throughput failure. The remaining 615 cases all
passed in three DSR shards on `csd`, using the qualified native extension
and the CPU PyTorch build. Standing CI qualification remains open under
`bio_inspired_nanochat-u0fl.1`.
[CPU suite split and receipts](results/cpu_native_muon_2026-11-24.json)

To identify the backend in an affected environment, start a fresh process.

```bash
uv run --no-sync python -c 'from bio_inspired_nanochat.synaptic import rust_presyn_cpu_backend; m = rust_presyn_cpu_backend(); print((m.__file__, m.PRESYN_CPU_ABI) if m else "reference backend")'
```

---

## TL;DR — running tests

```bash
# One-time: install deps AND build the Rust extension (rustbpe). Name the torch flavour
# explicitly; this is what CI does.
uv sync --extra cpu --dev                        # or --extra gpu on a CUDA box
# Shortcut without Rust (rustbpe-dependent tests skip):
uv sync --no-install-project --extra cpu --dev

# DSR change-validation payload:
uv run --no-sync python -m pytest -m "not slow" -q

# The fast unit subset:
uv run --no-sync python -m pytest -m "unit" -q

# With the opt-in coverage gate (fail_under in pyproject.toml):
uv run --no-sync python -m pytest -m "not slow" --cov

# A single file, verbose:
uv run --no-sync python -m pytest tests/test_framework_smoke.py -v
```

`pyproject.toml` sets `pythonpath = ["."]`, so `import bio_inspired_nanochat …`
resolves **without** an installed wheel; the maturin build only adds `rustbpe`.

> **Always pass `--no-sync` (or the same `--extra`) to `uv run`.** A bare `uv run`
> re-syncs the environment to the default extras, which silently swaps a
> `--extra cpu` torch install for the CUDA build.
>
> **The suite cannot pollute the results registry.** `tests/conftest.py` sets
> `BIO_RESULTS_REGISTRY` to a throwaway path before anything imports
> `results_registry`, so no test — in-process or subprocess — can append to the
> committed `results/registry.jsonl`. (41 rows of pytest temp-dir pollution were
> purged from it in 2026-09.)

---

## Markers (test taxonomy)

Registered in `pyproject.toml`; `--strict-markers` makes a typo'd marker an error.

| Marker     | Meaning                                                        | Intended DSR lane                      |
|------------|----------------------------------------------------------------|----------------------------------------|
| `unit`     | Fast, deterministic, no GPU/network/data downloads             | every push (`-m "not slow"`)           |
| `e2e`      | End-to-end run exercising a whole flow                         | every push + nightly `validate_all`    |
| `slow`     | Long-running (`-m "not slow"` to skip)                         | nightly only                           |
| `gpu`      | Requires CUDA; **auto-skipped** on CPU-only hosts              | trusted GPU worker                     |
| `golden`   | Compares against committed golden artifacts (`tests/golden/`)  | every push                             |
| `smoke`    | In-process smoke of the flagship harnesses (`pytest -m smoke`) | every push                             |

`gpu`-marked tests are skipped automatically when `torch.cuda.is_available()` is
False (see `tests/conftest.py::pytest_collection_modifyitems`), so the same suite
is green on a laptop and on GPU CI.

---

## The test kit — `tests/_bio_testkit.py`

Dependency-light helpers (torch + stdlib only). Import directly, or use the
`conftest.py` fixtures that wrap them.

| Helper | Purpose |
|--------|---------|
| `set_seed(seed=0)` | Seed python/numpy/torch (+CUDA), return a seeded `torch.Generator`. |
| `tensor_stats(t)` → `TensorStats` | Never-raises summary: shape/dtype/mean/std/min/max/‖·‖/NaN/Inf. `str(...)` is one tidy log line. |
| `summarize(name, t)` | `"logits: (2,16,97) float32 mean=… ⚠ NaN=…"` — the detailed-logging primitive. |
| `assert_finite(t, name)` | Assert no NaN/Inf, with a diagnostic on failure. |
| `make_tiny_synaptic(seed, **cfg)` | Tiny CPU `GPTSynaptic` (sub-second forward). |
| `make_tiny_vanilla(seed, **cfg)` | Tiny CPU vanilla `GPT` (bio-vs-vanilla baseline). |
| `random_tokens(B, T, vocab, seed)` | Deterministic token ids. |
| `count_params(model)` | Parameter count. |
| `cuda_available()` / `rustbpe_available()` | Capability probes. |
| `assert_golden(name, t, atol, rtol)` | Compare against `tests/golden/<name>.npy`; bootstraps the golden on first run. |

### Golden artifacts
`assert_golden` locks numerical semantics (used by parity/property beads, e.g.
`eqyk.13`, `eqyk.14`, and the theory-thrust certificates). On first run — or with
`BIO_UPDATE_GOLDEN=1` after an **intentional** semantics change — it (re)writes the
golden and skips the comparison. Otherwise values must match within tolerance.
Commit the resulting `tests/golden/*.npy` files.

---

## Fixtures — `tests/conftest.py`

| Fixture | Gives you |
|---------|-----------|
| *(autouse)* `_per_test_determinism` | Re-seeds RNGs **before every test** so randomness can't leak between tests. |
| `seed` / `rng` | The canonical seed / a freshly-seeded `torch.Generator`. |
| `device` / `cuda_device` | CPU-or-CUDA / CUDA (skips if absent). |
| `tmp_run_dir` | A throwaway per-test run directory (checkpoints/logs/artifacts). |
| `tiny_synaptic_model` / `tiny_vanilla_model` | Ready-built tiny models. |
| `tiny_synaptic_model_factory` | `make_tiny_synaptic` for building several / patching config. |

---

## Conventions for new tests

1. **Name & locate:** `tests/test_<area>.py`, functions `test_*`. Put a module
   docstring naming the bead it serves (e.g. `(bead vg9.1)`), mirroring existing tests.
2. **Mark it:** at least `@pytest.mark.unit` (or `e2e`/`slow`/`gpu`/`golden`).
3. **Be deterministic:** use the `seed`/`rng` fixtures or `set_seed`; never depend
   on global RNG state from another test.
4. **Be diagnostic:** assert with `assert_finite` / include `tensor_stats` in
   messages so failures are self-explaining. Per the Definition of Done, a feature
   isn't done until tests pass **and** logs demonstrate correct behavior — the
   structured run-logging infra is bead `eqyk.2`; this kit is its test-side analog.
5. **Keep `unit` fast:** sub-second on CPU. Heavy/long runs are `slow`/`e2e`.
6. **Optional deps:** guard with `pytest.importorskip("…")` (see `test_rustbpe.py`),
   never a bare top-level import that breaks collection.

---

## Coverage gate

`[tool.coverage]` in `pyproject.toml` measures branch coverage of
`bio_inspired_nanochat` and enforces `fail_under` **only when `--cov` is passed**
(so the default fast `pytest` stays lightweight). The floor is **25%** (the suite
measured ≈34% when the floor was set in 2026-06) and should be **ratcheted upward,
never lowered**. The DSR integration job must pass `--cov` to enforce the floor.

---

## Required DSR gates

The YAML files in `.github/workflows/` provide the gate inventory for direct DSR execution.
The times below are the schedules specified in those files. A run is qualified when DSR
records the source revision, exit status, denominators and retained artifacts.

- **`ci.yml` — every push to `main` and every PR:** `cargo fmt --check`, `cargo clippy -D warnings`,
  `cargo test` (native Rust unit tests in `rust_src/src/{presyn,moe}.rs`, complemented by the 27
  Python/PyO3 kernel parity tests in `tests/test_rust_kernels.py`); the Python quality gate (`scripts/quality_gate.py`: `ruff --fix --unsafe-fixes`
  on changed files and fails if it had to rewrite them, `ty check`, UBS resource-lifecycle scan), then
  `uv run --no-sync ty check` on the whole tree (zero diagnostics since 2026-09-02, bead `fkyw`);
  the Lean formal-feedback audit (`scripts/formal_feedback.py`); a `maturin build` wheel; then
  `maturin develop` + `pytest -m "not slow" --cov` + `validate_all --suite fast`. The wheel and test
  jobs depend on the Rust format job, so one unformatted hunk skips the whole suite.
  `tests/test_e2e_quick_start.py` is part of that run: it executes the README Quick Start as
  subprocesses (tokenizer, `base_train --synapses=1 --syn_cfg.…`, `chat_cli -i base`, and an
  unknown `--syn_cfg` field as the planted negative), so a script that cannot start fails CI.
- **`nightly-validation.yml` — 04:30 UTC:** `validate_all --suite all --timeout-scale 4`, the
  perf-regression gate (`scripts/perf_regression_gate.py`, which checks both stored-baseline drift
  and, under `--check-budget`, the `ThroughputBudget(train_ratio_max=2.0, decode_ratio_max=1.5)`
  synaptic-vs-standard relative slowdown budget in the same run), and `tests/test_e2e_matrix_pipeline.py`
  (the D1 chain launcher → `base_train` → `eval_matrix` → `eval_stats` at toy scale, e2e+slow),
  uploading `reports/` and the registry as artifacts.
- **`nightly-uncertainty.yml` — 05:17 UTC:** `tests/test_e2e_uncertainty.py` with its evidence
  directory uploaded.
