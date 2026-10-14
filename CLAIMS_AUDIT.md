# Claims Audit — bio_inspired_nanochat

**Bead:** `bio_inspired_nanochat-8j9.1` · **First audit:** TurquoiseFinch, 2026-06-10 · **Re-audited:** 2026-10-13 (reality check + Phase II–VIII execution)

Maps every public claim (README, planning docs) to a verified implementation status with evidence and the bead that closes each gap. Line numbers drift; module and function names are the stable references.

**Status legend**
- **SOLID** — implemented and on the live model path.
- **PARTIAL** — exists but limited, not on the live path, or test-only.
- **UNPROVEN** — implemented, but the *effect* the claim describes has never been measured at a scale where it could show.
- **ASPIRATIONAL** — claimed but absent, or only a roadmap item.

> **The one finding that dominates everything below (updated 2026-10-13):** every mechanism is real and unit-tested, no model larger than 2 layers has ever been trained, and nothing has run on a GPU. The first valid bio-vs-vanilla anchor comparison on real text (`results/toy_screening_2026-10-07_verdict.md`, 2L/128d, 600k WikiText-2 tokens, 3 seeds) gave vanilla 1.958, `synaptic_off` 2.028 (+0.069), and `bio_all` 2.038 (+0.080) val bpb; subsequent scaffold diagnosis (`results/scaffold_diagnosis_2026-10-07.json`, `03bb9bf` / `f96747e`) showed that the +0.069 substrate penalty came from a missing final RMSNorm on `GPTSynaptic` (`final_norm=True` now default) and moved online Hebbian plasticity to opt-in (`enable_hebbian=False` in `bio_all`), bringing `synaptic_off` to `1.920 ± 0.004 bpb` (`-0.038 bpb` vs vanilla) and `bio_all` to `1.930 ± 0.005 bpb` (`-0.028 bpb` vs vanilla). On 2026-10-13 (`hwxb.5.5`, `uta.10`), after fixing two meta-device initialization bugs in `add_cusp_latch` and `moe_add_glial_homeostasis`, all 17 dense columns × 2 seeds (`results/toy_screening_2026-10-13_verdict.md`) and all 5 structural MoE columns × 2 seeds (`results/structural_smoke_2026-10-13.json`) ran end-to-end through `matrix_launch → base_train → eval_matrix → eval_stats`. The headline experiment remains `docs/ablation_matrix.md` (`hwxb.3`–`hwxb.6`) on the dual-RTX-4090 host (`hwxb.10`).

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
| Fast weights give "infinite local context" | **NOT DEMONSTRATED (pre-registered, toy scale) → hypothesis; Hebbian now opt-in** | The deciding experiment (`hwxb.9`, `results/hebbian_chunked_regime_2026-10-07_preregistered.json`; 2L/64d, associative recall to 16 pairs, 5 seeds × 4 arms, 2,000 steps) gave ON − OFF = +0.010 under chunked training and reading against a 0.016 minimum detectable effect, with all controls passing; the decision rule moved `enable_hebbian` to add-one-in (default off since 2026-10-07). Chunked truncated-BPTT training at chunk 8 did not learn the task in either arm (loss 3.24 vs 0.35 full), so the regime could not have shown a benefit; full-trained models show ON − OFF +0.034 ± 0.101 over 5 seeds (exploratory, n.s.). Earlier: `hwxb.4.4` ON = OFF to 6 decimals in a regime blind to the writes | `hwxb.9.2`, `sx1m` |
| CaMKII/PP1 gate; bistable latch (opt-in) | **SOLID** | default gate `σ(CaMKII−0.5)−0.3`; `bistable_latch` switches to the Lisman switch with PP1 in the gate (`tests/test_bistable_latch.py`) | — |
| BDNF metaplasticity "toggleable via `bdnf_gamma`" | **SOLID**, wording fixed | gain is `1 + γ·BDNF` with `γ = bdnf_gamma if > 0 else bdnf_scale` (default 1.0), so BDNF is ON by default; ablate with `bdnf_scale=0`. README now says so | — |

## Structural plasticity / MoE

| Claim | Status | Evidence | Closing bead |
|---|---|---|---|
| MoE split/merge/reset lifecycle | **SOLID** (mechanism) | `SplitMergeController`, optimizer-moment resets, DDP broadcast; wired in `base_train` behind `--splitmerge_every` | — |
| Function-preserving split/merge | **SOLID** | `hwxb.4.3` CPU pilot confirmed dense-regime output preservation end to end | — |
| Homeostasis guards (`uta.6`) | **SOLID**, now reachable | were unreachable from the CLI until 2026-09-01; `--sm_homeostasis_guards`, `--sm_gate_ramp_forwards`, `--sm_energy_floor` added to `base_train` | — |
| "Grows capacity exactly where the data complexity demands it" | **UNPROVEN → negative at toy scale; cause measured** | default thresholds fired 0 events in 270 checks (`hwxb.4.3` pilot); NAS-vs-fixed evaluation regressed final loss on 8/8 seeds, p=0.0038 (`results/structural_nas_evaluation_a2307f21b18f.json`; re-derived 2026-09-02 under deterministic evaluation with identical numbers, `results/structural_nas_evaluation_2026-09-02_product.json`). Measured 2026-09-01: health = util × energy with util = routed fraction ≈ `top_k/E`, so at E=4 no threshold is ever crossed and at E=8 (the `base_train` default) all experts are merge candidates; energy relaxes to `1−util`, so health ≈ `u(1−u)` peaks at u=0.5 and the 0.80 split threshold is unreachable. A scale-free `health_mode=relative` (utilization / fair share) ships default-off since 2026-09-01 (`tests/test_lifecycle_health_relative.py`); whether events help remains open; a 2-seed structural-pair pilot (2026-09-02, `results/structural_pair_pilot_2026-09-02.json`) fired zero events under either signal because the balance loss keeps utilization within ±0.03 of the fair share | `sx1m`, `uta.8`, `hwxb.5` |
| NeuroScore credit assignment drives the lifecycle | **SOLID** (opt-in) · **fires, costs loss at toy scale; MoE stage E2E verified** | `health_mode=credit` (`uta.9`) never fired until 2026-10-07: the credit EMA was seeded with the step-0 routing proxy and stayed frozen (fixed in `aeaf162`). Re-run (`results/structural_pair_pilot_2026-10-07_credit.json`, 3 seeds, 120 steps): 3–6 splits, 14–15 merges, 2–8 resets per run; final loss +0.030 to +0.050 above the no-controller arm on 3/3 seeds. Full 5-column MoE stage E2E smoke (`uta.10`, `results/structural_smoke_2026-10-13.json`) verified across all 5 structural columns | `sx1m` |
| Per-expert genome `Xi` → phenotype decoder | **SOLID** (needs MoE) | `SynapticGenomeDecoder`; only present when `use_moe=1` or `splitmerge_every>0` | — |
| `structural_every` hook | **removed 2026-09-01** | was a per-layer `pass` block behind a config field carried through checkpoints and the hybrid optimizer's genome; deleted everywhere | — |

## Neuromodulation and uncertainty

| Claim | Status | Evidence | Closing bead |
|---|---|---|---|
| Neuromodulatory bus DA/ACh/NE (opt-in) | **SOLID** (mechanism) · **UNPROVEN** | `neuromod.py`; consumers use `getattr(..., 1.0)` so off = neutral. The "1.72× sample efficiency, p=0.0012" previously in `docs/theory/neuromodulated_rl_study.md` had no artifact behind it and was removed | `hy8.3` (reopened) |
| Calibrated uncertainty via MC vesicle sampling; selective decoding | **PARTIAL** (reachable since 2026-09-03) | `Engine.generate(selective=…)`, `chat_cli --selective --max-entropy`, `chat_web` request fields (`wmel`). Calibration numbers come from a 1-layer, 32-dim, 24-step toy; synaptic MC = softmax entropy (ΔECE 4e-6), artifact verdict `null` | — |

## Kernels & performance

| Claim | Status | Evidence | Closing bead |
|---|---|---|---|
| Triton presyn kernels (decode + multi-query training scan) | **PARTIAL** (interpreter-verified, never run on a GPU) | `presyn_detached_scan_kernel` (2026-10-07) runs a whole query block of the exact recurrence in one launch for training and prefill; training needs no backward kernel because the one-pass gradient (`_DetachedScanReleaseGrad`) differentiates the recorded edge constants. `tests/test_presyn_scan_kernel.py` matches it to the scripted scan under `TRITON_INTERPRET=1` (~1e-6). The FP32 one-query decode kernel is the older slice. Both behind `native_presyn` (env `BIO_FUSED_PRESYN`) | `3bnd`, `jyb.2`, `jyb.10` |
| Rust CPU kernels dispatched live | **SOLID** (opt-in) | `presyn_detached_scan_cpu` (2026-10-07, `jyb.9`) runs multi-query blocks for training and prefill, rows in parallel, on the same RNG stream as the scripted scan (`test_rust_scan_matches_the_scripted_scan`): 64-query block fwd+bwd 60 → 18 ms, bio_all 2L/128d train step 1262 → 925 ms (one thread). The one-query decode kernel `presyn_release_canonical_cpu` was vectorized with `StateBuffer::InPlace` zero-copy mutation + L1-tiled (`L1_TILE=512`) SIMD sub-loops (`ylo2`, 2026-10-13, `results/presyn_rust_decode_bench_2026-10-13.json`): **3.47×–7.12× faster than PyTorch across `T_keys=256..4096`** (`5.10×` at 512 keys, `7.12×` at `B=1,H=8,4096` keys, `4.96×` vs PyTorch / `12.99×` vs old Rust at `B=4,H=8,4096` keys) | — |
| Rust MoE kernels (`moe.rs`) live | **SOLID** (opt-in) | `SynapticMoE.forward` routes `native_genetics=True` through `bio_inspired_nanochat/kernels/dispatcher.py` (`dispatch_accumulate_router_stats`, `dispatch_update_metabolism`), calling `rustbpe.accumulate_router_stats_cpu` and `rustbpe.update_metabolism_cpu` on CPU and Triton on CUDA while preserving `aux_loss` gradients (`tests/test_backend_dispatcher.py`, `tests/test_rust_kernels.py`, `cargo test`) | — |
| "90%+ GPU utilization (dual 4090)" | **ASPIRATIONAL** | zero GPU measurements exist in the repo | `j9i`, `6pj` |
| FlexAttention O(N) memory | **PARTIAL** | prefill-only; KV-cache decode raises `NotImplementedError` | `zsi` |
| Throughput cost of the synaptic path | **measured, budget-gated (`74f.10.1`/`74f.10.2`), shrinking** | CPU toy: `results/perf_baselines.json` (7.9× slower training, 4.4× slower decode in pure Python; decode drops to ~1.2× with `native_presyn=True` after `ylo2`). `ThroughputBudget(train_ratio_max=2.0, decode_ratio_max=1.5)` is enforced by `scripts/perf_regression_gate.py --check-budget`. GPU cost unmeasured | `l7c9`, `j9i`, `74f.10.2` |

## Config, tuning & measurement

| Claim | Status | Evidence | Closing bead |
|---|---|---|---|
| `--syn_cfg.<field>=<value>` training overrides (documented since 2025-11) | **SOLID** (since 2026-09-01) | `cmaes_params.extract_syn_cfg_cli_overrides` / `apply_syn_cfg_overrides`; typed from the dataclass, validated by `synaptic_config_schema_errors` + `ablation_registry.validate_config`; refused on resume and for vanilla runs (`tests/test_syn_cfg_cli_overrides.py`). Before this the README's Quick Start failed on unknown keys | — |
| CMA-ES optimization of bio params | **SOLID** (tooling + `--objective lm` CPU-verified) · **awaiting Phase-0 FineWeb GPU run** | `tune_bio_params` CLI matches README. Copy-task proxy runs (`results/cmaes_phase1_rerun_2026-10-07.json`) were No-go (~ln(1024)=6.931). `--objective lm` with `native_presyn=True` (`results/cmaes_phase1_lm_2026-10-13.json`) learns to `0.98296 bpb` in 40 steps at `~9.0 s/generation` (6 gloo workers). Full Phase-0 FineWeb GPU run pending | `idh4` |
| "48-parameter genome" / "96 hyperparameters" | **fixed** | `SynapticConfig` has **109** fields, all live (`docs/parameter_census.md`, `scripts/param_census.py`); the learned genome is the 4-D `Xi`; 10 fields are CMA-ES-tuned. README updated from 96 to 109 (108 + `neuromod_enabled`) | — |
| Feature toggles for clean ablation | **SOLID** | `ablation_registry.MECHANISMS` + `validate_config`; presets applied by `eval_matrix` | — |
| Results registry accumulates real runs | **PARTIAL** | schema-validated corpus works; it held 43 rows of pytest temp-dir pollution (purged 2026-09-01) and no LM metric. Tests now redirect the default via `BIO_RESULTS_REGISTRY` (`tests/conftest.py`) | `hwxb.3` |
| Rigorous bio-vs-vanilla matrix run with statistics | **all 17 dense + 5 structural columns verified at CPU toy scale** (run) · **SOLID** (harness) | 3-seed WikiText-2 anchors (`results/toy_screening_2026-10-07_*`, `results/scaffold_diagnosis_2026-10-07.json`) plus full 17-column × 2-seed screening (`results/toy_screening_2026-10-13_*`) and 5-column × 2-seed structural MoE smoke (`results/structural_smoke_2026-10-13.json`); the D1 GPU run has not happened | `hwxb.5.2` |
| Model zoo with 124M checkpoints (`docs/model_zoo.md`) | **was fabricated → rewritten** | no checkpoint, run, or registry row existed for the table; the page now states what exists and how an entry gets created | `vap.6` (reopened) |
| CA-init decision numbers (`docs/ca_init_decision.md`) | **SOLID** (recovered & committed 2026-10-14) | raw 400-step CSV logs recovered from `trj` and committed at `results/ca_init_microbench_2025-12-18/{baseline,ca_rule30,ca_rule116}.csv`; `docs/ca_init_decision.md` cites exact CSV values (default-off stands) | — |

---

## Process findings (2026-09-01 & 2026-10-13)

- 244 of 431 closures happened on 2026-08-24. Five GPU-acceptance beads (`4fw`, `hwxb.2.5`, `hwxb.2.10`, `vap.6`, `hy8.3`) were closed on CPU proxies or documents and have been reopened with the reason recorded on each.
- 13 beads sat in `blocked` with no open blocker in the graph and were reset to `open`.
- On 2026-10-13, two live meta-device (`with torch.device("meta"):` + `to_empty()`) bugs in `add_cusp_latch` (`cusp_certificate.py:epsilon_gauge`) and `moe_add_glial_homeostasis` (`glial_homeostasis.py`) were uncovered and fixed before any GPU hours were spent, and `go_no_go()`'s bypassed `cap_gpu_hours` budget check in `ablation_matrix.py` was restored (`hwxb.5.4`).
- Docs with numbers must cite a committed artifact under `results/` or a registry row.

## Most significant remaining gaps (priority order)

1. **No GPU run, no D1 baseline, no D1 ablation.** Everything else is secondary until `hwxb.10` → `hwxb.3` → `hwxb.5.2` run on real hardware.
2. **Efficacy at D1 scale** (`sx1m`): verify whether any mechanism clears the pre-registered paired CI on D1 FineWeb-Edu.
3. **GPU Throughput** (`l7c9`, `6pj`, `74f.10.2`): measure D1 GPU training/decode ratios against `ThroughputBudget(train_ratio_max=2.0, decode_ratio_max=1.5)`.
4. **Kernels on real GPUs** (`3bnd`, `jyb.2`, `jyb.10`).
