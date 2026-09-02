# Testing Guide

> The test discipline for `bio_inspired_nanochat`. Established by bead **eqyk.1**
> (*Unit-test framework, conventions, fixtures & coverage gate*) — the foundation
> every later "unit tests + e2e + detailed logging" bead builds on.

The goal: **green = working.** Tests are deterministic, fast on CPU, portable to
GPU CI, and emit enough diagnostics that a failure tells you *why*.

---

## TL;DR — running tests

```bash
# One-time: install deps AND build the Rust extension (rustbpe). Name the torch flavour
# explicitly; this is what CI does.
uv sync --extra cpu --dev                        # or --extra gpu on a CUDA box
# Shortcut without Rust (rustbpe-dependent tests skip):
uv sync --no-install-project --extra cpu --dev

# What CI runs on every push / PR:
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

| Marker     | Meaning                                                        | Where it runs                          |
|------------|----------------------------------------------------------------|----------------------------------------|
| `unit`     | Fast, deterministic, no GPU/network/data downloads             | every push (`-m "not slow"`)           |
| `e2e`      | End-to-end run exercising a whole flow                         | every push + nightly `validate_all`    |
| `slow`     | Long-running (`-m "not slow"` to skip)                         | nightly only                           |
| `gpu`      | Requires CUDA; **auto-skipped** on CPU-only hosts              | a GPU host (none in CI today)          |
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
never lowered**. CI passes `--cov` on every push (the integration-tests job), so the floor is enforced there.

---

## What CI actually runs (`.github/workflows/`)

- **`ci.yml` — every push to `main` and every PR:** `cargo fmt --check`, `cargo clippy -D warnings`,
  `cargo test`; the Python quality gate (`scripts/quality_gate.py`: `ruff --fix --unsafe-fixes`
  on changed files and fails if it had to rewrite them, `ty check`, UBS resource-lifecycle scan), then
  `uv run ty check` on the whole tree (zero diagnostics since 2026-09-02, bead `fkyw`);
  the Lean formal-feedback audit (`scripts/formal_feedback.py`); a `maturin build` wheel; then
  `maturin develop` + `pytest -m "not slow" --cov` + `validate_all --suite fast`. The wheel and test
  jobs depend on the Rust format job, so one unformatted hunk skips the whole suite.
  `tests/test_e2e_quick_start.py` is part of that run: it executes the README Quick Start as
  subprocesses (tokenizer, `base_train --synapses=1 --syn_cfg.…`, `chat_cli -i base`, and an
  unknown `--syn_cfg` field as the planted negative), so a script that cannot start fails CI.
- **`nightly-validation.yml` — 04:30 UTC:** `validate_all --suite all --timeout-scale 4` and the
  perf-regression gate, uploading `reports/` and the registry as artifacts.
- **`nightly-uncertainty.yml` — 05:17 UTC:** `tests/test_e2e_uncertainty.py` with its evidence
  directory uploaded.
