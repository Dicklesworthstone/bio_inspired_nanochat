# Changelog

All notable changes to [Bio-Inspired Nanochat](https://github.com/Dicklesworthstone/bio_inspired_nanochat) are documented here.

This project is a research fork of [Nanochat](https://github.com/karpathy/nanochat) that replaces static transformer weights with computational analogs of synaptic proteins. There are no formal release tags; the changelog is organized by development phase with commits linked for traceability.

**Repository**: <https://github.com/Dicklesworthstone/bio_inspired_nanochat>
**~650 commits** | **2025-11-18 to 2026-09-01** | **No tagged releases**

---

## 2026-09-01 -- Reality Check: Truth Pass, CLI Wiring, Rust Dispatch

A full reality check against the README and plan documents (assessment published separately) found that every mechanism is implemented and unit-tested at toy scale, that no model larger than 2 layers × 64 dims has ever been trained, and that several documents and closed beads claimed more than the artifacts support. This wave closes the gaps that did not need a GPU.

### Training CLI
- `--syn_cfg.<field>=<value>` overrides for every `SynapticConfig` field, typed from the dataclass and validated (schema + mechanism prerequisites), refused on resume and for vanilla runs (`bio_inspired_nanochat/cmaes_params.py`, `scripts/base_train.py`, `tests/test_syn_cfg_cli_overrides.py`). The README had documented this syntax since 2025-11; the script rejected it as an unknown key.
- `--sm_homeostasis_guards`, `--sm_gate_ramp_forwards`, `--sm_energy_floor` expose the `uta.6` guards, which were unreachable from the CLI.
- Removed the `structural_every` config knob: its only consumer was an empty `pass` block per layer (also dropped from checkpoint metadata, `eval_matrix`, and the hybrid optimizer's genome).

### Kernels
- `release_canonical` dispatches the Rust decode kernel (`rustbpe.presyn_release_canonical_cpu`) for eval-mode one-query CPU decode when `native_presyn=1`; release, advanced state, and DELAY-queue parity plus planted negatives locked in `tests/test_presyn_rust_dispatch.py`. Measured single-thread: 1.98× faster than PyTorch at 512 keys, 0.97× at 2,048, 0.75× at 4,096, so the toggle stays off by default (follow-up bead `ylo2`).

### The headline-experiment harness
- `SynapticConfig.neuromod_enabled` (default off) is registered in `ablation_registry` as the `neuromod` mechanism (requires presyn + hebbian), so the pre-registered matrix carries an `add_neuromod` column (20 screening columns: 3 anchors, 8 leave-one-out, 9 add-one-in). `neuromod.bus_for_config` is the single constructor both harnesses use.
- `eval_matrix` accepts every matrix column, not just the ten named presets: the `synaptic_off` anchor and all `add_*` columns are materialised through `ablation_matrix.AblationConfig.build_syn_cfg()` (`eval_matrix.MATRIX_COLUMNS`), and the inline training loop instantiates the neuromodulatory bus when the flag is on, logging its levels and gains per step (`tests/test_neuromod_mechanism.py`).
- Measured why the expert lifecycle is inert at 4 experts and merge-prone at 8 (utilization ≈ `top_k/E` against absolute thresholds; steady-state health ≈ `u(1−u)`), recorded in README, CLAIMS_AUDIT, and bead `sx1m`.

### Structural lifecycle
- `SplitMergeConfig.health_mode="relative"` (default-off; `--sm_health_mode` in `base_train`) scores each expert by utilization relative to the fair share `top_k / num_experts`, so 1.0 means uniform at any expert count, >1 overworked, <1 underused; thresholds are then in fair-share units and the config refuses the product-mode defaults. Motivated by the measurement above; whether events help under either signal is still open (`sx1m`). `tests/test_lifecycle_health_relative.py`.

### Registry, CI, and gates
- The test suite redirects the default results registry via `BIO_RESULTS_REGISTRY` (`tests/conftest.py`), and 43 rows whose artifacts were pytest temp directories were purged from `results/registry.jsonl`.
- CI: `cargo fmt` on `rust_src/src/presyn.rs` (red since 2026-08-25, which skipped the wheel build and the 1,789-test suite); nightly validation gets `validate_all --timeout-scale 4`; nightly uncertainty creates its gitignored evidence directory; `release.yml` uses the real `dtolnay/rust-toolchain` action; duplicate `master` push trigger dropped; the quality gate's inner `uv run` is `--no-sync` so it cannot re-sync the environment to the CUDA torch build.
- `SynapticConfig` schema validation moved from `certificate_bundle` to `ablation_registry`, so the training script no longer imports the certificate machinery through a two-function dependency.
- `[tool.ruff.lint] select` pinned to the pre-0.16 policy (E4/E7/E9/F): the 2026-08-27 lock bump broadened ruff's defaults and left ~1,400 findings that made the changed-files quality gate a guaranteed failure; the six findings the pinned set still flagged are fixed and the repo is lint-clean under it. Five pre-existing `ty` errors in `eval_matrix` (a legacy `TypeVar` mixed with a PEP 695 type parameter) fixed.
- Every workflow `uv run` passes `--no-sync` (a bare `uv run` re-resolved the environment to the CUDA torch build mid-job); the integration job enforces the coverage floor with `--cov`; `docs/epic_runbooks.md` commands corrected to the real CLIs.

### Documentation truth pass
- README: Quick Start rewritten with the real flags and the data/tokenizer prerequisites; counts fixed (476 issues / 33 epics, 108 config fields, `rank_eligibility=8`); an "Evidence status" paragraph; the roadmap replaced with the actual open epics and a done/not-done table; calibration numbers labelled with their toy scale; BDNF toggle wording corrected.
- `docs/model_zoo.md` (a checkpoint table with no checkpoints behind it), `docs/theory/neuromodulated_rl_study.md` (a 1.72× gain that appears in no code or artifact), and `docs/ca_init_decision.md` (numbers from a discarded run) rewritten or flagged; `CLAIMS_AUDIT.md` re-audited; `TESTING.md` now describes what CI actually runs; `docs/theory/README.md` reflects that thrusts D and G have test-only reference modules.
- `speedrun.sh` / `run1000.sh` point at `bio_inspired_nanochat.*` and `rust_src/Cargo.toml` (they referenced the upstream `nanochat` package).

### Tracker
- 13 beads stuck in `blocked` with no open blocker reset to `open`; five beads closed on CPU proxies or on documents reopened with the reason recorded (`4fw`, `hwxb.2.5`, `hwxb.2.10`, `vap.6`, `hy8.3`); new beads for the inert-default efficacy work (`sx1m`), CMA-ES on a real objective (`idh4`), 4090 acceptance of the Triton decode kernel (`3bnd`), and row-parallel Rust decode (`ylo2`).

---

## 2026-08-21 to 2026-08-27 -- Second Swarm Wave (≈300 commits)

Multi-agent wave that closed 244 beads on 2026-08-24 alone. Landed, all at toy scale on CPU:

- **Theory program (`0642.*`)**: runtime certificates for the tropical skeleton, thermodynamic UQ multi-seed verdicts, metriplectic paired-stability evidence, topology-aware expert lifecycle (`topological_nas`), p-adic ultrametric retrieval, the `certificate_bundle` fail-closed model card, and the Lean-to-Python proof feedback gate in CI.
- **Structural evolution (`uta.*`)**: gradient-based marginal expert credit, distributed-safe deterministic lifecycle surgery, homeostatic stability guards after lifecycle events, expert-count growth/shrink under a budget.
- **Capability frontier (`r00r.*`, `re4e.*`)**: sheaf obstruction detector with abstain/repair actions, theory-discovery engine, self-correcting generation loop, efficiency-Pareto evaluation (documented null).
- **Scale-up infrastructure (`hwxb.*`)**: exact-resume checkpointing with multi-file commit markers and per-rank state, dataloader v2 exact token-buffer resume, probing/representation analysis; CPU pilots recorded for learnable kinetics, function-preserving split/merge, and online Hebbian fast-weights (the last showing ON = OFF to six decimals).
- **Hardening**: cross-codebase defect hunts, type-baseline fixes, Rust MoE boundary validation, deterministic CPU evaluation under non-zero `stochastic_train_frac`, dependency upgrades clearing reported vulnerabilities ([`38d421f`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/38d421f), pyo3 0.29 [`7e0671c`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/7e0671c)).

---

## 2026-06-09 to 2026-06-15 -- First Swarm Wave (≈200 commits)

- **Truth & integrity (`8j9.*`)**: `CLAIMS_AUDIT.md`; the live attention path unified on the faithful `release_canonical` (the legacy sigmoid path deleted); dead config fields pruned; machine-verified parameter census.
- **Numerical robustness (`vg9.*`)**: logit softcap parity, optimizer-moment reset across all optimizers on split/merge, per-sequence reset, full `SynapticConfig` checkpoint round-trip, genuine rank-R eligibility traces.
- **Differentiable synaptic dynamics (`yw9.*`)**: SGD-learnable stability-preserving calcium/buffer kinetics, differentiable vesicle conservation, spectral-radius stability monitor, chunked-TBPTT recurrence.
- **Online learning (`sax.*`)**: normalized fast-weight substrate and its honest characterization (unsupervised Hebbian adaptation does not improve next-token prediction); the bistable CaMKII/PP1 latch.
- **Neuromodulation (`hy8.*`)**: DA/ACh/NE bus with three-factor gating; **structural (`uta.3`)**: function-preserving Net2Net split/merge.
- **Platform (`eqyk.*`, `hm4.*`)**: test kit and marker taxonomy, structured run logging, results registry, flagship-harness smoke tests, the ablation registry as single source of truth for toggles, and the theory-program overview (`docs/theory/README.md`).
- **Scale-up plan (`hwxb.*`)**: Phase-0 decisions, DDP hardening, memory-budget estimator, pre-registered bio-vs-vanilla ablation matrix with the synaptic-off anchor.

---

## 2026-03-11 -- Project Maintenance

Housekeeping pass on issue tracker infrastructure.

- Consolidate Beads issue tracker database and remove legacy base files ([`f9c67d6`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/f9c67d6cc06ba74158e5b563436baa273f63e4b2))

---

## 2026-02-21 to 2026-02-25 -- Licensing and Branding

### Licensing
- Adopt MIT License with OpenAI/Anthropic Rider ([`bee049f`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/bee049f05f964768fd5444776367fa8741db4616))
- Update README license references to match new terms ([`9717389`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/971738919d4ed07cf1a30802ab2c817a2e6e380e))

### Project Identity
- Add GitHub social preview image (1280x640) ([`eefc0b1`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/eefc0b152f76c8be7ab8bc457e7a7ce102038356))

### Developer Tooling
- Add CASS (Cross-Agent Session Search) tool reference to AGENTS.md ([`85d1cc2`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/85d1cc28307897d73e250efd67ff2a71e0e1c1bc))

---

## 2026-01-17 to 2026-01-25 -- CI/CD Pipeline, Dependency Refresh, and Stabilization

Established the project's first proper CI/CD infrastructure, updated every dependency, and hardened the split-merge algorithm.

### Continuous Integration (New Capability)
- Add comprehensive CI/CD workflows (`ci.yml` + `release.yml`) with Rust compilation checks, Python linting, type-checking, and integration tests ([`7147f92`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/7147f9286a69759cba7ae2696a3139f8010f8cab))
- Fix `dtolnay/rust-action` to `dtolnay/rust-toolchain` ([`7da0bcf`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/7da0bcf8163c38b2a08b4f1f15dcf7a481b04bf0))
- Install Python 3.14 with uv and repair UBS installer URL ([`5908183`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/5908183254b631d7726af442920909e10e1ee3da))
- Install ast-grep and apply rustfmt ([`2cb76b7`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/2cb76b761db4a3f2404506c3a08c6a3e1ddb23e0))
- Set PYTHONPATH for integration tests ([`7460633`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/7460633acb08e8e2ddc0245fb7d3ca8e57795fdb))
- Gracefully handle missing `.env` file ([`afd52c3`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/afd52c3d5cb60831271467276b4bf2d59738ed21))
- Remove hardcoded python path from `ty.toml` ([`23c0e0a`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/23c0e0a8c2ec980efcd66b6d099676f4234e77e5))
- Resolve clippy warnings ([`cc2a2a5`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/cc2a2a506d22a9b84fa412ef7ccd04f40605eb01))
- Fix cargo fmt formatting ([`2e27b82`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/2e27b82c17c623136cdc7a377007bdab829d5491))

### Synaptic Stability
- Improve split-merge algorithm stability with edge case handling and numerical guards ([`425914f`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/425914f5b1d19fef2c669909f4882bd38b2b6337))

### Dependency Management
- Update Rust dependencies to latest stable (hashbrown 0.16.1, indexmap 2.13.0, syn 2.0.114, etc.) ([`cec6bbf`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/cec6bbfb8bfd969ea7d096924cb30d1509a13a68))
- Update Python dependencies to latest stable (fastapi 0.128.0, huggingface-hub 1.3.2, numpy 2.4.1, etc.) ([`fab8fec`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/fab8fecb2e8eaf040735cdb487de848c43d0af40))

### Legal
- Add MIT License file ([`dc8ca18`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/dc8ca18df7bfb46013f09150419bc0eb8b5ea636))

---

## 2026-01-09 -- Major Feature Expansion (Batch Landing)

Large coordinated update touching every subsystem. Introduced ultrametric attention, expanded the CMA-ES framework to production scale, added comprehensive evaluation infrastructure, and broadened test coverage across all bio mechanisms.

### Synaptic Mechanisms
- Enhance calcium/energy gating and improved Hebbian dynamics in `synaptic.py` ([`d6d610a`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/d6d610a1d9f1ccd081aa918c63f85869af2ec6fb))
- Add ultrametric attention and improve synaptic transformer integration in `gpt.py` / `gpt_synaptic.py` ([`3fc2a45`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/3fc2a45ef1c3b944f1c9e98c0667867e5207cf93))

### CMA-ES Optimization (Major Expansion)
- Expand CMA-ES hyperparameter optimization framework to 1751 lines covering all 48 bio parameters, with distributed evaluation, stagnation detection, TensorBoard logging, and two-phase search strategy ([`8592489`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/8592489713fe8bd5a70e388b701b681946781185))

### Evaluation and Scoring
- Enhance expert fitness metrics, visualization, and kernel infrastructure ([`af63dd4`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/af63dd482753b1468ed8f6e2a7a8fd4f5bdab515))
- Add AST-safe calculator evaluation and improve checkpoint management ([`21fd700`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/21fd70096ee848671be2cf2afdacda686b62b121))
- Enhance data pipeline with improved dataset handling and evaluation framework ([`fbebaad`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/fbebaad550d712697ba2cc127b7e3b1d9ca3ea18))
- Update task evaluators for comprehensive benchmark suite ([`2bb6e37`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/2bb6e37873fd567a739eef89a56f3cf585f1fd3e))

### FlexAttention and Kernel Work
- Comprehensive FlexAttention benchmarking and correctness verification ([`15ce78b`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/15ce78bf581405010c49ee8bd50305c211c521f5))
- Update Rust kernel dependencies and PyO3 bindings ([`a6c80c4`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/a6c80c479e24f315c45c44a16d06e350b80336d4))
- Update common utilities with improved CA initialization and tokenizer fixes ([`b70eda6`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/b70eda609a1564a693be200bbc79e1effd315c77))

### Quality and Tooling
- Add quality gate script (`scripts/quality_gate.py`) enforcing ruff, ty, and UBS checks ([`4e39670`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/4e39670999aa56ea2bf6c1c0057cf92e2af496bb))
- Add GitHub Actions quality workflow for automated code quality checks ([`2b88c5e`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/2b88c5e0f468977ff4c584427cce795d343cfc5d))

### User-Facing Interfaces
- Update chat interfaces and dashboard with bio-inspired model support ([`71f1f6d`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/71f1f6d3deffce735c305305da450a38f63426b3))
- Update training and evaluation scripts with improved bio integration ([`1d59c1d`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/1d59c1d2db3a0dac47f8730c4f671eff0829cba7))

### Testing
- Comprehensive test suite expansion for synaptic mechanisms and evaluation ([`7e15295`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/7e15295af7ad1980448d8449f51b53d7194a1a43))

### Documentation
- Comprehensive documentation update with README overhaul and new guides ([`f26414c`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/f26414c0ca509f269eb372dd40e5b12131eeefd5))

### Dependencies
- Update to Python 3.14 and refresh all dependencies ([`8c43575`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/8c435755206b40713fe4d24e84bf353a858a1bc2))

---

## 2025-12-18 -- BDNF Metaplasticity

Added activity-dependent learning rate modulation inspired by Brain-Derived Neurotrophic Factor. The BDNF signal scales the Hebbian learning rate based on recent postsynaptic activity, enabling the model to learn faster in active regions and conserve resources in quiet ones.

### Synaptic Mechanisms
- Implement BDNF metaplasticity state and slow learning-rate modulation in `synaptic.py` -- controlled by the `bdnf_gamma` parameter in `SynapticConfig` ([`6b26c92`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/6b26c92a4fce8f029826f3166d82267bd0661726), [`df63e7f`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/df63e7f31a018db315378fd3169ebab614cc5ba8))

---

## 2025-11-20 -- FlexAttention, Feature Toggles, and Research Planning

Introduced O(N)-memory FlexAttention as a production-ready attention backend, made every bio mechanism individually toggleable via `SynapticConfig` flags, relocated the Rust code to its final home in `rust_src/`, and published extensive research planning documents.

### Attention System (New Capability)
- Add PyTorch 2.5+ FlexAttention support for O(N) memory synaptic attention (`flex_synaptic.py`, `benchmark_flex.py`) ([`51ba960`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/51ba96081b7e4bdec45bb85498d091b2f1f785a4))
- Add FlexAttention correctness verification suite ([`f49733d`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/f49733d59d0e8167495c5809264a431a475536a8))
- Increase FlexAttention benchmark sequence length to 4096 tokens ([`e28a7ff`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/e28a7ff68511ad3325d5f8bd4ed890e5525d3a4a))

### Modularity and Configuration
- Add modular per-feature toggles to `SynapticConfig`, making each bio mechanism individually switchable for clean ablation studies ([`d1f6331`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/d1f6331bbba0876d4bc6dc72e4b142a3da572ae3))
- Add real-time metrics export and presynaptic state management to inference engine ([`dec9e70`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/dec9e7070fbcbd437166ffc0cd677c117a2bc21b))

### Codebase Reorganization
- Relocate Rust code from `rustbpe/` to `rust_src/` and add kernel compatibility parameters ([`24f12bd`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/24f12bd57d0fa7a0d97b08b90184127e551baceb), [`dad3242`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/dad32420d527ced525af47df6fc47d854ed797d3))
- Improve Rust kernel type safety and add Python reference implementation ([`a81c504`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/a81c5046c222dd4633bd7409641f38a71872894d))
- Migrate UI from `ui.html` to `bio_nc_ui.html` for clearer naming ([`1e04bdc`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/1e04bdccdf9fbe594868db2da9da7e4dbac59e54))
- Move `metrics_fused` to main package for broader accessibility ([`4ce9eeb`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/4ce9eebb55fff2c6daad8aea3b347ac4ced771bc))

### Code Quality and Security
- Replace assert statements with explicit exception raising for production robustness ([`d5d47d6`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/d5d47d6698b0edfb841cb5dcc52b0fcfe1594c20))
- Add nosec annotations and improve default web server binding ([`6d6d64b`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/6d6d64bf8fa15347f4189b2c4afcae50655c1aed))

### Bug Fixes
- Improve error handling in Rust presynaptic kernel ([`30c4c08`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/30c4c08f27de089ba74bdab8facb3d69ad856663))
- Remove duplicate docstring line in Muon optimizer ([`d7d907b`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/d7d907bc2b16986a8f258e5a82ba6bf6d5e3dcc2))

### Research Documentation
- Add comprehensive roadmap for 11 advanced bio-inspired features (`NEW_RADICALLY_NEW_BIO_INSPIRED_FEATURES_TO_ADD_IN_MODULAR_WAY.md`) ([`9b45362`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/9b45362ea8e9fc017e91d6cc65869e2620cbb602))
- Add Claude Sonnet 4.5 technical predictions on bio-inspired features with deep code analysis ([`1ed9c63`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/1ed9c633aa3ac0de84476fdd032a268aa96af598), [`a905941`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/a905941e1caf4390a415351ce85dd24956633078))
- Add comprehensive CMA-ES hyperparameter optimization plan (15,000-word document covering 48 parameters, two-phase strategy, budget tracking) ([`4a4992f`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/4a4992fc841cc5c67a25de6a062db0a3a40b2118))
- Add markdown transcriptions of neurological transformer research proposals ([`bf3ba6a`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/bf3ba6a2af021c3d5a7124c1cd2ec27bf1712e80))
- Comprehensive README expansion reflecting current project state ([`8dee497`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/8dee4974f178311e593901707440c1a4dc3906b1))

### Project Tracking
- Initialize Beads issue tracker with comprehensive project roadmap covering 69 tasks across 7 epics ([`1177203`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/11772039c22adff0af5e7b061ce769ed2e567cb6), [`4225b1f`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/4225b1fea5ecea5288d6e523097469e077c20021), [`76b09d4`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/76b09d439fdfddee668ffe6fd6b6008c813ef084), [`8e6ecf2`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/8e6ecf2b50fd7da75975c291948ee1e31010d50d))

---

## 2025-11-19 -- Kernel Backends, Biophysics Overhaul, and Dashboard

The most intensive single day of development. Introduced all three kernel backends (Triton GPU, Rust/PyO3 CPU, Python reference), completely rewrote the synaptic biophysics with literature-grounded parameters, built the Streamlit dashboard, renamed the package, and added the dual-4090 training launcher.

### Synaptic Biophysics (BREAKING CHANGE)
- Complete overhaul of synaptic biophysics with scientifically accurate parameters grounded in Sudhof (2012), Kaeser & Regehr (2014), and Bacaj et al. (2013) -- 794 lines rewritten in `synaptic.py` ([`3fd421d`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/3fd421d43f80cf869cfdacf1b26a40a7e2819e4b))

### Triton GPU Kernels (New Capability)
- Add fused Triton kernel for expert split/merge integrated into structural plasticity (`kernels/structural_fused.py`, 140 lines) ([`becbac1`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/becbac18fa7b33949e6f322acbcba68c1ddba9d1))

### Rust CPU Kernels (New Capability)
- Add Rust PyO3 module for CPU-side presynaptic dynamics computation (`presyn.rs`, 305 lines) ([`cce64f2`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/cce64f29a36f1811427b50cce5b1025754a54875))
- Wire Rust CPU kernel into `SynapticPresyn` with automatic Python fallback ([`1175f10`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/1175f1011fb126c109200a11c1d54d02f448160d))
- Add Rust CPU implementation for MoE router statistics and metabolism updates (`moe.rs`) ([`14d42c4`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/14d42c49176582616f6355f305becc9e68b8a3dc))
- Improve thread safety and type correctness in Rust presyn module ([`9ff9961`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/9ff9961f28b1f503790d8e3fabd9537370c4e4ad))
- Refactor Rust presyn kernel to use direct pointer slicing for performance ([`c4d4912`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/c4d4912787c097eb5ab6c80f7f65aabdca126917))
- Simplify Rust pointer handling by using `usize` instead of `UnsafePtr` wrapper ([`8ddc533`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/8ddc533c2e039104dd01373fdc6ac9d04ea4a6f6))

### Dashboard and Visualization (New Capability)
- Add Streamlit dashboard for real-time NeuroViz monitoring (`scripts/dashboard.py`, 373 lines) ([`2cdd030`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/2cdd0307e57f1b1d583bbfd1f75dfd1a5d06c5b1))
- Enhance NeuroViz with educational plots and integrate into training loop ([`debf23e`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/debf23ee7282f3c012fdf3ecad727087ae3d25c3))
- Add genetics, metabolism, and router decision visualizations ([`deab556`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/deab5564f022aa1f4fc5cc001d2735781de75214))
- Add Semantic Space page and error handling ([`c06720b`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/c06720b69c95ba08e05e7f1ff6dc9ab20fb8a10f))
- Add 3D semantic space, connectivity graph, and system vitals pages ([`4595753`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/459575376a66424c6728bf0521ef6117babc23c7))
- Add Gini coefficient metric for expert utilization analysis ([`0cffc5a`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/0cffc5a579339d6c0369b5bb2d23181168d855dc))
- Standardize metric naming from `util` to `utilization` in NeuroViz ([`86fad29`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/86fad290e53cee7e5b82536ff6a8b45d83a799c3), [`d38bb9a`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/d38bb9a2b0c85e6e60676c927db3385501211db9))

### Package Rename (BREAKING CHANGE)
- Rename package from `nanochat` to `bio_inspired_nanochat` and add Triton kernel infrastructure (26 files, 740 insertions) ([`0f8bd81`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/0f8bd81fd7b4d01a48ccd7b4933868e3e31ea499))
- Update all script, task, and test imports to use `bio_inspired_nanochat` package ([`160c51f`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/160c51f696be67e9d72007915972ef07860dedb0), [`b273d53`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/b273d533833e7f79b788eab4d6abf5b8842894ad), [`5381fe4`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/5381fe4bd436db666bd3fcf96e2e13e52ce13983))

### Training Infrastructure
- Add turnkey dual-4090 training launcher with NeuroViz pipeline (`scripts/run_bio_dual4090.sh`) ([`de147a4`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/de147a48646b29e00b8942de0238f4808e2b9aab))
- Enhance synaptic and bio-inspired core logic ([`4395bb0`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/4395bb0bef37a11219e8a522ea5961374c0f56f6))
- Update training infrastructure and utilities ([`d99a132`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/d99a1329f46b1b2c1cccb6a30c6d9912809e0b46))
- Update tokenizer and rust extension ([`9f346aa`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/9f346aa85963f04bd8410aa0c15062d2e46d6aaa))
- Update training scripts and tasks ([`65289a3`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/65289a3fcb7e9830534c2dd9b712aea068dc4d37))

### Build System
- Add Cargo workspace configuration for Rust components ([`45f99dd`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/45f99dd7a910b5b2122ef8182f8635ca8748b900))
- Add python-decouple and triton dependencies ([`f9bc6ac`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/f9bc6ac5854bc95f77a9054ec0be09ed50f1f7dc))
- Add streamlit dependency ([`a20dd6b`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/a20dd6bcf4bd6c39b32ce1a7e2c6a21ec046855b))

### Testing
- Add comprehensive pytest suite for Rust kernel numerical parity against Python reference ([`3035e10`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/3035e10c51483a9fa3be45b5d8d0a4a052b7c546))

### Bug Fixes
- Initialize Hebbian eligibility traces with small random values to prevent zero-gradient deadlock ([`d18e91c`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/d18e91c25ca234b5fd85a97e86e4e911f04a1112))
- Separate matrix and non-matrix parameters for correct Muon/AdamW optimizer split ([`ef5bd3b`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/ef5bd3b65b5ec706a6e770f3bde6ffa820c745c6))
- Improve error handling and multi-epoch support in dataloader and tokenizer ([`0cc7633`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/0cc7633e16dbeb4cb5a39706c19701f76f990fe6))
- Correct tensor gather operation for 4D index in `SynapticPresyn` ([`bb11eba`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/bb11eba7e4d5f1bfa6ea2f338cf0c137eb398ad6))
- Add dtype casting for `scatter_add` and fix AMP state gathering ([`3a99323`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/3a993238c0ba22506d4fada7ed066b5132cd9836))
- Add stride documentation and reduce batch size for memory efficiency ([`0256cd5`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/0256cd5ec053688fccb1862978f74f31af4d99de))

### Documentation
- Add comprehensive optimization roadmap for biological dynamics acceleration ([`a8170af`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/a8170afc0f9fe63a9b227bbe829434563216a00e))

---

## 2025-11-18 -- Project Genesis

Initial creation of the repository. Forked from [Nanochat](https://github.com/karpathy/nanochat) and transformed into a bio-inspired research platform in a single day: built the full synaptic mechanism stack, NeuroScore fitness system, CMA-ES tuner, visualization suite, evolution verification tooling, and upgraded to CUDA 12.8 / Python 3.14.

### Foundation (Initial Commit)
- Import initial project structure from Nanochat: 60 files, ~13,750 lines covering the core engine, GPT model, synaptic mechanisms, training scripts, tokenizer, Rust BPE, evaluation tasks, and tests ([`aea83da`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/aea83da50b930f8e10166cad0f6ee2e3771d4a0c))

### Synaptic Mechanisms (Core Capability)
- Add comprehensive visualization system for Synaptic-MoE training (`neuroviz.py`, 545 lines; enhanced `synaptic_splitmerge.py`) ([`440e775`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/440e775a2d9f2e8e0946f2655cc9bb1ba4fd3699))
- Implement NeuroScore expert fitness system: efficiency, specialization, and resilience metrics (`neuroscore.py`, 202 lines) ([`071273c`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/071273c6e40f84d309bd7a226471abf07ee0621d))
- Enhance router architecture with probe-based alignment and improve attention computation ([`0253bb1`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/0253bb11c7a598ac703a1c3bd321fcb0bdd63e00))

### CMA-ES Optimization (New Capability)
- Add CMA-ES biological hyperparameter tuner (`scripts/tune_bio_params.py`, 365 lines) ([`cb51cc4`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/cb51cc45430af4e4cb9dd577d52c4d29e7dcbddf))

### Verification and Testing
- Add evolution verification script for validating that bio mechanisms evolve correctly over time ([`1ca87c6`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/1ca87c691a710e2d43033a915dd732e2b3a16d4e))
- Improve evolution verification script robustness ([`f5f1cf7`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/f5f1cf7b75863d6ca1dae16fe2b8aef68ac62232))

### Bug Fixes
- Add GPTSynaptic compatibility across evaluation and inference paths ([`ff2c97f`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/ff2c97ff232a5ed57c82bd6d7f35f11062cb4c72))
- Resolve type errors and restore missing neuroviz logic ([`b696cd7`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/b696cd7ab67ea5a6372aca0bc1993ffa74a9b863))
- Resolve critical bugs in core synaptic modules ([`fadd877`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/fadd87712f3add2288eae150b359ec473680de70))

### Code Quality
- Improve type safety and PyTorch module attribute handling ([`72813ee`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/72813eeba8780fbdee34e39ec633f02f92bb5f7c))
- Apply consistent code formatting across core modules ([`2130f9f`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/2130f9fe001fd3e8429d9a4ede7fcef89d9a5d70))
- Improve code quality and consistency across codebase ([`837141f`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/837141f20655529305282cf7dddcc210d86960b6))

### Documentation
- Rewrite README for Bio-Inspired Nanochat ([`8db7601`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/8db7601b2059eba48bdbb0bfc316ea2d64b4214e))
- Add deep dive into synaptic mathematics ([`c5715ce`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/c5715ceef0446d14e9c7be281d99ae31338ade9a))
- Add Synaptic Genome Map reference table ([`aacd8d9`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/aacd8d9f51fb692c29e470aca4cf55e9503b3fbf))
- Add section explaining CMA-ES bio-parameter tuner ([`6306e3f`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/6306e3f837cbccd0647bfc95831e73fd080c1081))
- Add theoretical foundation papers ([`e75b850`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/e75b850e76ca19628b195d64856187851c55f1e0))

### Dependencies
- Upgrade environment for CUDA 12.8 and Python 3.14 ([`01cfdf4`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/01cfdf4a077bf1fd886111694d103fd721f96289))
- Add ruff linter to development dependencies ([`6b7bdf4`](https://github.com/Dicklesworthstone/bio_inspired_nanochat/commit/6b7bdf4050dd40d2b0f6c74eb15d6530ed8ebc47))

---

## Architecture Reference

The project implements three biological subsystems mapped to tensor operations, each with multiple kernel backends:

| Subsystem | Purpose | Primary Files | Kernel Backends |
|:---|:---|:---|:---|
| **Presynaptic** | Vesicle depletion, calcium dynamics, frequency penalty | `synaptic.py`, `kernels/presyn_fused.py` | Triton GPU, Rust CPU (`rust_src/src/presyn.rs`), Python ref |
| **Postsynaptic** | Hebbian fast-weights, BDNF metaplasticity, CaMKII/PP1 gating | `synaptic.py` | Python |
| **Structural** | MoE expert split/merge, metabolism, neural architecture search | `synaptic_splitmerge.py`, `kernels/structural_fused.py` | Triton GPU, Rust CPU (`rust_src/src/moe.rs`) |

**Key supporting modules**: `neuroscore.py` (expert fitness), `neuroviz.py` (training visualization), `flex_synaptic.py` (FlexAttention O(N) memory), `scripts/tune_bio_params.py` (CMA-ES optimization of 48 parameters), `scripts/dashboard.py` (Streamlit monitoring), `scripts/quality_gate.py` (ruff + ty + UBS enforcement).
