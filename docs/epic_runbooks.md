# Epic Operations & Development Runbooks (bead wbp)

> **Scope**: Standardized, one-page operational runbooks for all core epics across `bio_inspired_nanochat`.  
> **Governance**: All runs must adhere to Python 3.14 / uv conventions, determinism policies (aiq), typed metrics schemas (d8l), budget accounting (2a7), and CI quality gates (5yi).

---

## 1. Runbook Template Standard

Each epic section specifies:
- **Scope & Objective**
- **Prerequisites & Hardware**
- **Canonical Execution Commands**
- **Produced Artifacts & Registries**
- **Success Criteria & Acceptance Gates**
- **Failure Modes & Rollback Flags**

---

## 2. Epic 114: Bio-Inspired Modular Features

- **Scope**: Presynaptic vesicle depletion, CaMKII/PP1 consolidation, BDNF metaplasticity, neuromodulatory bus, and MoE structural split/merge.
- **Prerequisites**: Python 3.14, PyTorch, Triton kernels (optional GPU acceleration).
- **Execution Commands**:
  ```bash
  # Run bio mechanism unit test suite
  uv run --no-sync python -m pytest tests/test_presyn_canonical.py tests/test_bdnf_metaplasticity.py tests/test_scaleup_ablation_e2e.py -v
  # Run structural split/merge e2e test
  uv run --no-sync python -m pytest tests/test_function_preserving_splitmerge.py -v
  ```
- **Produced Artifacts**: Checkpoint weights with bio-state dictionaries, `docs/parameter_census.json`.
- **Success Criteria**: Contractive kinetics ($\rho < 1.0$), finite loss, exact vesicle conservation ($RRP + RES = \text{const}$).
- **Rollback Flags**: Disable via `--synapses false`, `--topological-nas false`, `--neuromod false`.

---

## 3. Epic y5r: CMA-ES Hyperparameter Optimization

- **Scope**: Automated evolutionary search over biological kinetic constants ($\tau_C, \tau_{\text{buf}}, \alpha_{\text{Ca}}, \lambda_{\text{loge}}$) on proxy validation tasks.
- **Prerequisites**: Fast synthetic or tokenized shards, `cmaes` optimizer module.
- **Execution Commands**:
  ```bash
  # Optimize bio parameters
  uv run --no-sync python -m scripts.tune_bio_params optimize --seed 42 --max-evals 50
  # Validate optimized candidate configuration
  uv run --no-sync python -m scripts.tune_bio_params validate --params runs/cmaes/best_params.json
  ```
- **Produced Artifacts**: `runs/cmaes/best_params.json`, `docs/cmaes_params.md`.
- **Success Criteria**: Monotonically non-increasing surrogate loss without NaN/Inf parameter exploration.
- **Rollback Flags**: Revert to human-tuned defaults in `bio_inspired_nanochat/synaptic.py`.

---

## 4. Epic 6pj: Dual-4090 GPU Performance Optimization

- **Scope**: High-throughput training on dual PCIe RTX 4090 GPUs (NCCL/P2P, Triton fused kernels, mixed precision).
- **Prerequisites**: CUDA 12.8+, `NCCL_P2P_LEVEL=NVL` or `LOC`, 24GB VRAM.
- **Execution Commands**:
  ```bash
  # Check VRAM headroom and memory budget
  uv run --no-sync python -m scripts.scale_memory --profile dev_tiny --synapses
  # Run multi-GPU performance benchmark
  uv run --no-sync python -m pytest tests/test_perf_regression.py tests/test_scaleup_ddp.py -v
  ```
- **Produced Artifacts**: `results/perf_baselines.json`, `runs/budget_accounting.jsonl`.
- **Success Criteria**: $\ge 18\text{GB}$ VRAM headroom, throughput within $5\%$ of vanilla transformer baseline.
- **Rollback Flags**: Fallback to eager PyTorch reference kernels via `export USE_TRITON_SYNAPSE=0`.

---

## 5. Epic gzm: Bio vs Vanilla Evaluation Rigor

- **Scope**: Rigorous, reproducible benchmarking across FineWeb 10M Matrix A, NIAH long-context, ECE calibration, and continual learning forgetting rate.
- **Prerequisites**: Pre-tokenized FineWeb shards, validation benchmarks.
- **Execution Commands**:
  ```bash
  # Run full evaluation matrix on checkpoint
  uv run --no-sync python -m scripts.eval_matrix --checkpoint runs/checkpoints/model.pt --output runs/eval_summary.jsonl
  # Compute statistical significance with paired tests and bootstrap CIs
  uv run --no-sync python -m bio_inspired_nanochat.eval_stats runs/eval_summary.jsonl --baseline vanilla_baseline --alpha 0.05
  ```
- **Produced Artifacts**: `runs/eval_summary.jsonl`, statistical markdown/JSON report.
- **Success Criteria**: Statistically significant delta ($p < 0.05$, Holm-Bonferroni corrected) across 3 seeds.
- **Rollback Flags**: N/A (read-only evaluation).

---

## 6. Epic 4x9: Training Visualization & Insights

- **Scope**: Live Rich terminal console, attention/energy spatial maps, and living pedagogical storybook.
- **Prerequisites**: Modern terminal with ANSI/TrueColor support, browser for storybook HTML.
- **Execution Commands**:
  ```bash
  # Launch pedagogical storybook and render interactive HTML
  uv run --no-sync python -m scripts.bio_storybook --html docs/bio_storybook.html
  # Verify visualization test suite
  uv run --no-sync python -m pytest tests/test_run_logging.py tests/test_neuroviz_v2.py tests/test_bio_storybook.py -v
  ```
- **Produced Artifacts**: `docs/bio_storybook.html`, `runs/neuroviz/`.
- **Success Criteria**: Self-contained single-file HTML interactive report, live ASCII telemetry in console.
- **Rollback Flags**: Disable visual logging via `NeuroVizConfig(write_tensorboard=False)`.

---

## 7. Epic acd: Cross-Pollination with Model-Guided Research

- **Scope**: Reversible/measure-preserving memory blocks, simplicial 2-hop graph diffusion, ultrametric $p$-adic tree kernels, CA morphogenetic initialization.
- **Prerequisites**: `docs/mgr_landscape_digest.md`, `docs/mgr_cross_pollination_playbook.md`.
- **Execution Commands**:
  ```bash
  # Run cross-pollination and guardrail tests
  uv run --no-sync python -m pytest tests/test_scaleup_ablation_e2e.py -v
  ```
- **Produced Artifacts**: `docs/mgr_cross_pollination_playbook.md`, `docs/ca_init_decision.md`, `docs/prototype_guardrails.md`.
- **Success Criteria**: Passes canary protocols in `docs/prototype_guardrails.md`.
- **Rollback Flags**: All experimental features default to `False`. CA-init permanently disabled.

---

## 8. Cross-Cutting Engineering Protocol

Before committing code or completing any task, execute the full quality gate:
```bash
# 1. Format & Linting Auto-fix
uv run --no-sync ruff check --fix <changed-files>

# 2. Strict Type Checking
uv run --no-sync ty check <changed-files>

# 3. Bug Scanning
ubs <changed-files>

# 4. Targeted Test Suite
uv run --no-sync python -m pytest <test-path> -v
```
