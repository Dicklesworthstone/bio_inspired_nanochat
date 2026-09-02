# Claims Audit — bio_inspired_nanochat

**Bead:** `bio_inspired_nanochat-8j9.1` · **First audit:** TurquoiseFinch, 2026-06-10 · **Re-audited:** 2026-09-01 (reality check, commit `db544b2` + this session's changes)

Maps every public claim (README, planning docs) to a verified implementation status with evidence and the bead that closes each gap. Line numbers drift; module and function names are the stable references.

**Status legend**
- **SOLID** — implemented and on the live model path.
- **PARTIAL** — exists but limited, not on the live path, or test-only.
- **UNPROVEN** — implemented, but the *effect* the claim describes has never been measured at a scale where it could show.
- **ASPIRATIONAL** — claimed but absent, or only a roadmap item.

> **The one finding that dominates everything below (2026-09-01):** every mechanism is real and unit-tested, and no model larger than 2 layers × 64 dims has ever been trained. `results/registry.jsonl` is 100% `cpu:x86_64` and has no `val_bpb` row. The project has no evidence yet that biology helps a language model; the two directional toy-scale findings are null (online fast-weights ON = OFF to six decimals) and negative (split/merge NAS regressed loss on 8/8 seeds). The synaptic path is 4–18× slower than vanilla on CPU. The pre-registered experiment that produces the first signal is `docs/ablation_matrix.md` (beads `hwxb.3`–`hwxb.6`), gated on the dual-RTX-4090 host.

---

## Presynaptic biophysics

| Claim | Status | Evidence | Closing bead |
|---|---|---|---|
| Hill-equation release `Syt(C)=C/(C+Kd)`, Syt1/Syt7 mix (0.7/0.3) + Doc2 | **SOLID** | `synaptic.py` `_faithful_release_prob`; verified numerically to 6 digits; golden-locked (`tests/test_presyn_golden.py`) | — |
| The **live** attention path runs the faithful release (not the legacy sigmoid) | **SOLID** | standard path calls `release_canonical` for every query; the legacy `release()` was deleted, so no fallback remains | — |
| Calcium integrator with buffer ODE; `tau_c` decay | **SOLID** | `release_canonical` (BUF ODE active); `tau_c` unified as `exp(-1/τ)` | — |
| RRP depletion / "fatigue & boredom" frequency penalty | **SOLID** (mechanism) · **UNPROVEN** (effect) | scatter updates RRP; priming, endocytosis `DELAY` queue present. No generation-level measurement of repetition exists | `hwxb.5` |
| Stochastic vesicle release (3 STE modes) | **SOLID** | `_sample_binomial_counts`; on by default (`stochastic_train_frac=0.12`) | — |
| Septin-like distance barrier | **PARTIAL** | applied at the attention-logit level (global `q_pos−k_pos`), not a local windowed inhibition | — |

## Postsynaptic / Hebbian learning

| Claim | Status | Evidence | Closing bead |
|---|---|---|---|
| Online Hebbian learning during training | **SOLID** | `plasticity_during_training=True`; deferred autograd-safe Parameter writes | — |
| Low-rank eligibility traces (rank-R U/V) | **SOLID** | fixed random `proj_in/proj_out` + per-mode einsum accumulation (`vg9.9` fixed the mean-broadcast degeneracy) | — |
| Fast weights give "infinite local context" | **UNPROVEN → null at toy scale** | `hwxb.4.4` pilot: ON vs OFF held-out loss identical to 6 decimals over 400 steps, Δw_fast ≈ 1e-7; `docs/online_learning_status.md` (sax.1) reached the same conclusion | `sx1m` |
| CaMKII/PP1 gate; bistable latch (opt-in) | **SOLID** | default gate `σ(CaMKII−0.5)−0.3`; `bistable_latch` switches to the Lisman switch with PP1 in the gate (`tests/test_bistable_latch.py`) | — |
| BDNF metaplasticity "toggleable via `bdnf_gamma`" | **SOLID**, wording fixed | gain is `1 + γ·BDNF` with `γ = bdnf_gamma if > 0 else bdnf_scale` (default 1.0), so BDNF is ON by default; ablate with `bdnf_scale=0`. README now says so | — |

## Structural plasticity / MoE

| Claim | Status | Evidence | Closing bead |
|---|---|---|---|
| MoE split/merge/reset lifecycle | **SOLID** (mechanism) | `SplitMergeController`, optimizer-moment resets, DDP broadcast; wired in `base_train` behind `--splitmerge_every` | — |
| Function-preserving split/merge | **SOLID** | `hwxb.4.3` CPU pilot confirmed dense-regime output preservation end to end | — |
| Homeostasis guards (`uta.6`) | **SOLID**, now reachable | were unreachable from the CLI until 2026-09-01; `--sm_homeostasis_guards`, `--sm_gate_ramp_forwards`, `--sm_energy_floor` added to `base_train` | — |
| "Grows capacity exactly where the data complexity demands it" | **UNPROVEN → negative at toy scale; cause measured** | default thresholds fired 0 events in 270 checks (`hwxb.4.3` pilot); NAS-vs-fixed evaluation regressed final loss on 8/8 seeds, p=0.0036 (`results/structural_nas_evaluation_a2307f21b18f.json`). Measured 2026-09-01: health = util × energy with util = routed fraction ≈ `top_k/E`, so at E=4 no threshold is ever crossed and at E=8 (the `base_train` default) all experts are merge candidates; energy relaxes to `1−util`, so health ≈ `u(1−u)` peaks at u=0.5 and the 0.80 split threshold is unreachable. Details on the bead | `sx1m`, `hwxb.5` |
| NeuroScore credit assignment drives the lifecycle | **SOLID** (opt-in) · **UNPROVEN** | `use_neuroscore` blends fitness into health; no run has measured its effect | `uta.2` |
| Per-expert genome `Xi` → phenotype decoder | **SOLID** (needs MoE) | `SynapticGenomeDecoder`; only present when `use_moe=1` or `splitmerge_every>0` | — |
| `structural_every` hook | **removed 2026-09-01** | was a per-layer `pass` block behind a config field carried through checkpoints and the hybrid optimizer's genome; deleted everywhere | — |

## Neuromodulation and uncertainty

| Claim | Status | Evidence | Closing bead |
|---|---|---|---|
| Neuromodulatory bus DA/ACh/NE (opt-in) | **SOLID** (mechanism) · **UNPROVEN** | `neuromod.py`; consumers use `getattr(..., 1.0)` so off = neutral. The "1.72× sample efficiency, p=0.0012" previously in `docs/theory/neuromodulated_rl_study.md` had no artifact behind it and was removed | `hy8.3` (reopened) |
| Calibrated uncertainty via MC vesicle sampling; selective decoding | **PARTIAL** | `mc_predict`, `quality_guarded_predict` exist but are unreachable from `chat_web`, `chat_cli`, `engine`, or the serving engine. The cited numbers come from a 1-layer, 32-dim, 24-step toy; synaptic MC = softmax entropy (ΔECE 4e-6), artifact verdict `null` | — |

## Kernels & performance

| Claim | Status | Evidence | Closing bead |
|---|---|---|---|
| Triton decode kernel live on decode | **PARTIAL** | real `@triton.jit` kernel behind `native_presyn` (env `BIO_FUSED_PRESYN`), FP32 one-query no-grad CUDA only; never fires in training; **never executed on any GPU** | `3bnd` |
| Rust CPU kernel dispatched live | **SOLID** (opt-in, since 2026-09-01) · **not a general speedup** | `release_canonical` dispatches `rustbpe.presyn_release_canonical_cpu` for eval-mode one-query CPU decode when `native_presyn=1`; parity incl. DELAY queue locked in `tests/test_presyn_rust_dispatch.py`. Measured single-thread: 1.98× at 512 keys, 0.97× at 2,048, 0.75× at 4,096 | `ylo2` |
| Rust MoE kernels (`moe.rs`) live | **PARTIAL** | test-exercised only; `kernels/dispatcher.py` is imported by nothing | — |
| "90%+ GPU utilization (dual 4090)" | **ASPIRATIONAL** | zero GPU measurements exist in the repo | `j9i`, `6pj` |
| FlexAttention O(N) memory | **PARTIAL** | prefill-only; KV-cache decode raises `NotImplementedError` | `zsi` |
| Throughput cost of the synaptic path | **measured, unfavourable** | CPU toy: 8.5× slower training, 3.6× slower decode (`results/perf_baselines.json`); exact causal recurrence 12× slower after `08hm` (`l7c9`) | `l7c9` |

## Config, tuning & measurement

| Claim | Status | Evidence | Closing bead |
|---|---|---|---|
| `--syn_cfg.<field>=<value>` training overrides (documented since 2025-11) | **SOLID** (since 2026-09-01) | `cmaes_params.extract_syn_cfg_cli_overrides` / `apply_syn_cfg_overrides`; typed from the dataclass, validated by `synaptic_config_schema_errors` + `ablation_registry.validate_config`; refused on resume and for vanilla runs (`tests/test_syn_cfg_cli_overrides.py`). Before this the README's Quick Start failed on unknown keys | — |
| CMA-ES optimization of bio params | **SOLID** (tooling) · **null result** | `tune_bio_params` CLI matches README. The only Phase-1 run (2025-12-18, 2-layer CPU proxy, synthetic copy task) gained 1.46e-4 vs seed noise 2.1e-3 and is recorded as No-go; Phase 2 descoped | `idh4` |
| "48-parameter genome" / "96 hyperparameters" | **fixed** | `SynapticConfig` has **109** fields, all live (`docs/parameter_census.md`, `scripts/param_census.py`); the learned genome is the 4-D `Xi`; 10 fields are CMA-ES-tuned. README updated from 96 to 109 (108 + `neuromod_enabled`) | — |
| Feature toggles for clean ablation | **SOLID** | `ablation_registry.MECHANISMS` + `validate_config`; presets applied by `eval_matrix` | — |
| Results registry accumulates real runs | **PARTIAL** | schema-validated corpus works; it held 43 rows of pytest temp-dir pollution (purged 2026-09-01) and no LM metric. Tests now redirect the default via `BIO_RESULTS_REGISTRY` (`tests/conftest.py`) | `hwxb.3` |
| Rigorous bio-vs-vanilla matrix run with statistics | **ASPIRATIONAL** (run) · **SOLID** (harness) | `eval_matrix`, `eval_stats`, `ablation_matrix` exist; the run never happened. Bead `4fw` ("run Matrix A on FineWeb") had been closed on harness verification and was reopened | `hwxb.5.2` |
| Model zoo with 124M checkpoints (`docs/model_zoo.md`) | **was fabricated → rewritten** | no checkpoint, run, or registry row existed for the table; the page now states what exists and how an entry gets created | `vap.6` (reopened) |
| CA-init decision numbers (`docs/ca_init_decision.md`) | **unverifiable, flagged** | artifacts were under gitignored `runs/` and are gone; decision (default-off) stands | — |

---

## Process findings (2026-09-01)

- 244 of 431 closures happened on 2026-08-24. Five GPU-acceptance beads (`4fw`, `hwxb.2.5`, `hwxb.2.10`, `vap.6`, `hy8.3`) were closed on CPU proxies or documents and have been reopened with the reason recorded on each.
- 13 beads sat in `blocked` with no open blocker in the graph and were reset to `open`.
- CI had been red since 2026-08-25 on one `rustfmt` hunk, which skipped the wheel build and the 1,789-test suite; both nightlies failed 8/8 (a missing gitignored directory; per-check timeouts tuned for a fast dev box). All fixed. The Python quality gate is a no-op on pushes to `main` (empty `origin/main...HEAD` diff) — it only bites on PRs.
- Docs with numbers must cite a committed artifact under `results/` or a registry row. Three docs violated this and were corrected.

## Most significant remaining gaps (priority order)

1. **No GPU run, no baseline, no ablation.** Everything else is secondary until `hwxb.3` → `hwxb.5` run on real hardware.
2. **Inert defaults** (`sx1m`): the Hebbian write and the lifecycle thresholds must engage before GPU-hours are spent on them.
3. **Throughput** (`l7c9`, `ylo2`, `6pj`): the synaptic path's 4–18× CPU slowdown and the 12× recurrence regression are the practical wall at scale.
4. **Kernels on real GPUs** (`3bnd`, `jyb.2/3`).
