# Testing Guide

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
