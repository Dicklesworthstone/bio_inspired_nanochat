# Claims Audit — bio_inspired_nanochat

First audit on 2026-06-10 under `bio_inspired_nanochat-8j9.1`. Evidence reviewed on 2026-11-22 against committed code and results.

The November 22–23 follow-up measured three claims directly. On held-out command
composition, diffusion solved 52/576 examples versus 87/576 for plain attention,
and 32 wrong outputs had zero cycle residual. In the repaired continual-learning
experiment all three arms retained 0/768 first-task targets after the third task.
A two-GPU vanilla pilot reached 3.19468 validation bits per byte on 6.29M training
tokens. Its measured cost exceeded the registered allowance for a full neural
reference. [Command composition](results/sheaf_compositional_2026-11-22.json) ·
[Continual learning](results/continual_controls_bounded_hebbian_2026-11-22.json) ·
[Pilot and admission](results/vanilla_pilot_2026-11-22.json)

**Status legend**
- **SOLID** — implemented and on the live model path.
- **PARTIAL** — exists but limited, not on the live path, or test-only.
- **UNPROVEN** — implemented, but the *effect* the claim describes has never been measured at a scale where it could show.
- **ASPIRATIONAL** — claimed but absent, or only a roadmap item.

> **The scientific question remains open.** GPU training, real-text evaluation and Triton acceleration have recorded results. The registry includes six CUDA training records (including repeated/resumed runs) and one CUDA evaluation. The 150-step vanilla smoke reached 2.4480 validation bpb. A comparison of matched, trained D1 models at the declared budget is outstanding. [Registry](results/registry.jsonl) · [GPU runbook](docs/epic_runbooks.md)

On CPU, an initial three-seed WikiText-2 anchor comparison before the final-normalization fix measured 1.958, 2.028 and 2.038 bpb for vanilla, `synaptic_off` and `bio_all`. A separate 614,400-token diagnostic measured 1.9971 for vanilla and 1.9588 for the corrected mechanisms-off scaffold (paired Δ −0.0383). Only the latter two arms appear in that diagnostic. In the subsequent two-seed dense smoke, all 16 contrasts with vanilla returned `null`. [Initial comparison](results/toy_screening_2026-10-07_verdict.md) · [Scaffold diagnostic](results/scaffold_diagnosis_2026-10-07.json) · [Dense smoke](results/toy_screening_2026-10-13_verdict.md)

---

## Presynaptic biophysics

| Claim | Status | Evidence | Closing bead |
|---|---|---|---|
| Hill-equation release `Syt(C)=C/(C+Kd)`, Syt1/Syt7 mix (0.7/0.3) + Doc2 | **SOLID** | `synaptic.py` `_faithful_release_prob`; verified numerically to 6 digits; golden-locked (`tests/test_presyn_golden.py`) | — |
| The **live** attention path runs the faithful release (not the legacy sigmoid) | **SOLID** | standard path calls `release_canonical` for every query; the legacy `release()` was deleted, so no fallback remains | — |
| Calcium integrator with buffer ODE; `tau_c` decay | **SOLID** | `release_canonical` (BUF ODE active); `tau_c` unified as `exp(-1/τ)` | — |
| RRP depletion and generated repetition | **SOLID** mechanism · **MEASURED** on one smoke checkpoint | Scatter updates the readily releasable pool. The same-weight step-50 inference ablation below measured repetition and authentic continuation likelihood. Independent training-seed replication remains open. [Raw generations and scores](results/presyn_repetition_smoke_2026-11-22.json) | `74f.13` |
| Stochastic vesicle release (3 STE modes) | **SOLID** | `_sample_binomial_counts`; on by default (`stochastic_train_frac=0.12`) | — |
| Septin-like distance barrier | **PARTIAL** | applied at the attention-logit level (global `q_pos−k_pos`), not a local windowed inhibition | — |

On 16 held-out document prefixes, greedy generation repeated 242 of 352 possible trigram occurrences with release enabled and 256 of 352 with release disabled. The paired difference was −3.98 percentage points (95% document-bootstrap interval −8.24 to −0.28). Both greedy arms emitted all 384 tokens. Ordinary seeded sampling produced zero repeated trigrams in both arms. One disabled-release continuation stopped at 12 tokens, giving 372 emitted tokens compared with 384 when enabled. Teacher-forced mean negative log likelihood over the 384 authentic next tokens was 6.5586 nats with release enabled and 6.5750 with it disabled (paired difference −0.0164, interval −0.0393 to +0.0070). Four separate calibration prompts selected a frequency penalty of λ=0, so the penalty arms duplicated the unpenalized arms. The intervals describe prompt variation for this checkpoint and sampling seed. [Study and provenance](results/presyn_repetition_smoke_2026-11-22.json)

## Postsynaptic / Hebbian learning

| Claim | Status | Evidence | Closing bead |
|---|---|---|---|
| Online Hebbian updates | **SOLID** on opt-in training and probe paths | `SynapticLinear` applies deferred updates when enabled. Standard `Engine` sets `train_mode=False` and `GPTSynaptic` defaults `update_mem` to that value. | `hwxb.4.4` |
| Low-rank eligibility traces (rank-R U/V) | **SOLID** | fixed random `proj_in/proj_out` + per-mode einsum accumulation (`vg9.9` fixed the mean-broadcast degeneracy) | — |
| Fast weights give "infinite local context" | **NOT DEMONSTRATED (pre-registered, toy scale) → hypothesis; Hebbian now opt-in** | The deciding experiment (`hwxb.9`, `results/hebbian_chunked_regime_2026-10-07_preregistered.json`; 2L/64d, associative recall to 16 pairs, 5 seeds × 4 arms, 2,000 steps) gave ON − OFF = +0.010 under chunked training and reading against a 0.016 minimum detectable effect, with all controls passing; the decision rule moved `enable_hebbian` to add-one-in (default off since 2026-10-07). Chunked truncated-BPTT training at chunk 8 did not learn the task in either arm (loss 3.24 vs 0.35 full), so the regime could not have shown a benefit; full-trained models show ON − OFF +0.034 ± 0.101 over 5 seeds (exploratory, n.s.). Earlier: `hwxb.4.4` ON = OFF to 6 decimals in a regime blind to the writes | `hwxb.9.2`, `sx1m` |
| CaMKII/PP1 gate; bistable latch (opt-in) | **SOLID** | default gate `σ(CaMKII−0.5)−0.3`; `bistable_latch` switches to the Lisman switch with PP1 in the gate (`tests/test_bistable_latch.py`) | — |
| BDNF metaplasticity | **SOLID**, conditional on Hebbian updates | When Hebbian writes are active the default scale is 1.0. `bdnf_gamma > 0` overrides it, and setting both to zero removes the gain. | — |

## Structural plasticity / MoE

| Claim | Status | Evidence | Closing bead |
|---|---|---|---|
| MoE split/merge/reset lifecycle | **SOLID** (mechanism) | `SplitMergeController`, optimizer-moment resets, DDP broadcast; wired in `base_train` behind `--splitmerge_every` | — |
| Function-preserving split/merge | **PARTIAL** | Dense unit tests verify a split into a negligible slot or a merge of identical experts at zero divergence noise, within 10⁻⁵ relative L2 error. Default noise and sparse top-k allow output changes. See `tests/test_function_preserving_splitmerge.py`. | `hwxb.4.3` |
| Homeostasis guards (`uta.6`) | **SOLID**, now reachable | were unreachable from the CLI until 2026-09-01; `--sm_homeostasis_guards`, `--sm_gate_ramp_forwards`, `--sm_energy_floor` added to `base_train` | — |
| Demand-directed capacity improves results | **UNPROVEN** at LM scale, **regression** in the controlled toy task | In the eight-seed equal-FLOP study, variable-count NAS eliminated dead experts but increased final MSE from 0.15032 to 0.15375. The paired bootstrap 95% CI for the increase was [0.00200, 0.00492]. See `results/structural_nas_evaluation_a2307f21b18f.json` and `docs/eval_benchmark_matrix.md`. | `hwxb.4.3`, `hwxb.5` |
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
| Triton presynaptic decode and training scan | **MEASURED** on RTX 4090 | `results/presyn_live_rtx4090.json` records passing hardware, correctness and timing gates at clean revision `0965db1`. Release at 2,048 keys sped up 6.84× and whole-model cached synaptic decode sped up 1.88× over eager execution. The earlier `presyn_triton_4090_bench_2026-10-14.json` had a dirty tracked tree and `acceptance_eligible=false`. `results/profiles/d1_4090_profile_2026-10-14.json` records a separate multi-query training profile. | `jyb.3` for differentiable backward |
| Rust CPU kernels dispatched live | **SOLID** (opt-in) | `presyn_detached_scan_cpu` (2026-10-07, `jyb.9`) runs multi-query blocks for training and prefill, rows in parallel, on the same RNG stream as the scripted scan (`test_rust_scan_matches_the_scripted_scan`): 64-query block fwd+bwd 60 → 18 ms, bio_all 2L/128d train step 1262 → 925 ms (one thread). The one-query decode kernel `presyn_release_canonical_cpu` was vectorized with `StateBuffer::InPlace` zero-copy mutation + L1-tiled (`L1_TILE=512`) SIMD sub-loops (`ylo2`, 2026-10-13, `results/presyn_rust_decode_bench_2026-10-13.json`): **3.47×–7.12× faster than PyTorch across `T_keys=256..4096`** (`5.10×` at 512 keys, `7.12×` at `B=1,H=8,4096` keys, `4.96×` vs PyTorch / `12.99×` vs old Rust at `B=4,H=8,4096` keys) | — |
| Rust MoE kernels (`moe.rs`) live | **SOLID** (opt-in) | `SynapticMoE.forward` routes `native_genetics=True` through `bio_inspired_nanochat/kernels/dispatcher.py` (`dispatch_accumulate_router_stats`, `dispatch_update_metabolism`), calling `rustbpe.accumulate_router_stats_cpu` and `rustbpe.update_metabolism_cpu` on CPU and Triton on CUDA while preserving `aux_loss` gradients (`tests/test_backend_dispatcher.py`, `tests/test_rust_kernels.py`, `cargo test`) | — |
| At least 90% GPU utilization on dual 4090s | **PARTIAL** | `results/scale_memory_2x4090_2026-10-14.jsonl`, batch 16/rank, sequence 1,024, records 66.4% utilization for bio training and 99.4% for bio full-sequence inference. These are five-step timing windows after two warmups. | `6pj` |
| FlexAttention O(N) memory | **PARTIAL** | prefill-only; KV-cache decode raises `NotImplementedError` | `zsi` |
| Throughput cost of the synaptic path | **MEASURED** at matched shapes | The dual-4090 D1 training sweep at batch 16/rank and length 1,024 recorded 96,181 vanilla vs 37,429 bio tok/s (2.57× slower), and 7.55 vs 14.69 GiB peak allocated per rank. This point exceeds the 2.0× target. Full-sequence inference and fused-vs-eager decode timings cannot establish the separate 1.5× vanilla cached-decode target. See `results/scale_memory_2x4090_2026-10-14.jsonl`. | `6pj`, `r2d` |

## Config, tuning & measurement

| Claim | Status | Evidence | Closing bead |
|---|---|---|---|
| `--syn_cfg.<field>=<value>` training overrides (documented since 2025-11) | **SOLID** (since 2026-09-01) | `cmaes_params.extract_syn_cfg_cli_overrides` / `apply_syn_cfg_overrides`; typed from the dataclass, validated by `synaptic_config_schema_errors` + `ablation_registry.validate_config`; refused on resume and for vanilla runs (`tests/test_syn_cfg_cli_overrides.py`). Before this the README's Quick Start failed on unknown keys | — |
| CMA-ES optimization of bio params | **PARTIAL** | The copy-task proxy recorded a no-go. The six-rank CPU LM pilot exercised four generations. The scheduler assigns distinct candidates per rank while the LM optimizer factory enables cross-rank updates. A serial-versus-distributed replay must establish independent candidate fitness before the search can supply a reproduced optimum. [LM pilot](results/cmaes_phase1_lm_2026-10-13.json) · [Audit](docs/bridge_plan.md) | `0xd.1`, `idh4`, `yw9.6` |
| Search-space size and implementation | **PARTIAL** | The configuration census counts 109 fields. `scripts/tune_bio_params.py` declares 10 presynaptic, nine postsynaptic and four structural entries. Its default candidate is dense with Hebbian writes off, so listing a dimension does not establish an active control. The historical 48-dimensional study was conditional and descoped (`fg7`, `scq`). | `idh4` |
| Feature toggles for clean ablation | **SOLID** | `ablation_registry.MECHANISMS` + `validate_config`; presets applied by `eval_matrix` | — |
| Results registry accumulates real runs | **PARTIAL** | The tracked `results/registry.jsonl` at `16ce9fe` contains 215 rows, including ten training rows with `val_bpb`. GPU records include smoke runs and restart records. Run counts alone cannot establish independent replicates. | `hwxb.3` |
| Rigorous bio-vs-vanilla matrix | **PARTIAL** | The 34 CPU dense cells used 5,120 train tokens and 512 eval tokens per cell. The ten structural smoke cells used 256 train tokens and 128 eval tokens per cell. All 16 dense comparisons returned `null`. These runs exercise checkpoint scoring at toy scale. D1 baseline matching, eligible contrasts, full screening and confirmation remain outstanding. | `hwxb.5.2`, `hwxb.5.3` |
| Model zoo with 124M checkpoints (`docs/model_zoo.md`) | **was fabricated → rewritten** | no checkpoint, run, or registry row existed for the table; the page now states what exists and how an entry gets created | `vap.6` (reopened) |
| CA-init decision numbers (`docs/ca_init_decision.md`) | **SOLID** (recovered & committed 2026-10-14) | raw 400-step CSV logs recovered from `trj` and committed at `results/ca_init_microbench_2025-12-18/{baseline,ca_rule30,ca_rule116}.csv`; `docs/ca_init_decision.md` cites exact CSV values (default-off stands) | — |

---

## Process findings (2026-09-01 & 2026-10-13)

- 244 of 431 closures happened on 2026-08-24. Five GPU-acceptance beads (`4fw`, `hwxb.2.5`, `hwxb.2.10`, `vap.6`, `hy8.3`) were closed on CPU proxies or documents and have been reopened with the reason recorded on each.
- 13 beads sat in `blocked` with no open blocker in the graph and were reset to `open`.
- On 2026-10-13, two live meta-device (`with torch.device("meta"):` + `to_empty()`) bugs in `add_cusp_latch` (`cusp_certificate.py:epsilon_gauge`) and `moe_add_glial_homeostasis` (`glial_homeostasis.py`) were uncovered and fixed before any GPU hours were spent, and `go_no_go()`'s bypassed `cap_gpu_hours` budget check in `ablation_matrix.py` was restored (`hwxb.5.4`).
- Docs with numbers must cite a committed artifact under `results/` or a registry row.

## Most significant remaining gaps (priority order)

1. **Qualify the experiment before training.** Verify scored command lines, token budgets, parameter counts, survivor selection, prerequisite-matched controls and a feasible confirmation family. Use actual device-hours for the 72 GPU-hour cap. See `scripts/matrix_launch.py`, `bio_inspired_nanochat/ablation_matrix.py` and `docs/bridge_plan.md`.
2. **Run the D1 FineWeb baseline and comparisons.** Archive per-seed checkpoints, provenance, held-out scores and paired quality/cost contrasts. Preserve the outcome when the result is negative or inconclusive (`hwxb.3`, `hwxb.5`).
3. **Qualify performance at the operating point.** Obtain representative training and cached decode measurements against vanilla (`6pj`, `r2d`).
4. **Publish a reproducible checkpoint and sample.** A fresh checkout must retrieve the actual weights and reproduce the documented output (`hwxb.6.2.1`, `vap.6`).
