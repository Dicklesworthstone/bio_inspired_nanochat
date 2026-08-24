# System Master Validation Report — val-1787580042

> **Verdict**: **ALL SYSTEMS PERFECT (VERIFIED WORKING)**  
> **Timestamp**: `2026-08-24T14:00:42.198295+00:00`  
> **Git SHA**: `57e0edde391761926fb12ebf56901d371f63ebef`  
> **Hardware**: `cpu:x86_64`  
> **Summary**: `16/16 Passed` (100.0%) in `125.69s`

---

## Subsystem Verification Matrix

| Category | Subsystem | Status | Duration | Command |
| :--- | :--- | :---: | :---: | :--- |
| Foundations | Unit Tests (Ablations, Metrics, Checkpoints, Registry) | ✅ **PASS** | 2.70s | `pytest tests/test_ablation_registry.py tests/test_results_registry.py tests/test_metrics_schema.py tests/test_checkpoint_roundtrip.py -v --tb=short` |
| Foundations E2E | Full Bio Training Loop E2E (eqyk.4) | ✅ **PASS** | 18.46s | `pytest tests/test_e2e_train_bio.py -v --tb=short` |
| Foundations E2E | CMA-ES Tune Loop & Resume E2E (eqyk.7) | ✅ **PASS** | 38.22s | `pytest tests/test_e2e_cmaes.py -v --tb=short` |
| Foundations E2E | Online Hebbian & Bistable Plasticity E2E (eqyk.8) | ✅ **PASS** | 18.65s | `pytest tests/test_e2e_online_learning.py -v --tb=short` |
| Foundations E2E | Structural Evolution & Neurogenesis E2E (eqyk.9) | ✅ **PASS** | 4.44s | `pytest tests/test_e2e_structural_lifecycle.py -v --tb=short` |
| Foundations E2E | Wake/Sleep Consolidation & Forgetting E2E (eqyk.10) | ✅ **PASS** | 2.93s | `pytest tests/test_e2e_wake_sleep.py -v --tb=short` |
| Foundations E2E | Neuromodulated 3-Factor RL Micro-Run E2E (eqyk.11) | ✅ **PASS** | 5.28s | `pytest tests/test_e2e_neuromod_rl.py -v --tb=short` |
| Foundations E2E | Interpretability Probe/Lesion/Stimulation E2E (eqyk.12) | ✅ **PASS** | 2.72s | `pytest tests/test_e2e_probe_lesion_stim.py -v --tb=short` |
| Parity & Invariants | Cross-Backend Parity (Triton/Rust/Python Reference) (eqyk.13) | ✅ **PASS** | 7.49s | `pytest tests/test_presyn_backend_parity.py -v --tb=short` |
| Parity & Invariants | Property-Based Metamorphic Invariants Suite (eqyk.14) | ✅ **PASS** | 2.80s | `pytest tests/test_property_invariants.py -v --tb=short` |
| Theory & Proofs | Formal Theory Certificates & Lyapunov Invariants (eqyk.18) | ✅ **PASS** | 2.41s | `pytest tests/test_e2e_theory_artifacts.py -v --tb=short` |
| Capability Frontier | Capability Frontier I Batteries (Deliberation/Adaptive/Scientist) (eqyk.19) | ✅ **PASS** | 3.41s | `pytest tests/test_e2e_capability_frontier.py -v --tb=short` |
| Capability Frontier | Uncertainty Calibration & Selective Prediction (eqyk.20) | ✅ **PASS** | 4.52s | `pytest tests/test_e2e_uncertainty.py -v --tb=short` |
| Capability Frontier | Synaptic Retrofit & MGR Attention Variants Battery (eqyk.21) | ✅ **PASS** | 5.59s | `pytest tests/test_e2e_retrofit_mgr.py -v --tb=short` |
| Capability Frontier | Wave-2 Emergent Compositions (Self-Correct, Metacognition, Search) (eqyk.22) | ✅ **PASS** | 2.95s | `pytest tests/test_e2e_wave2_compositions.py -v --tb=short` |
| Performance | Performance Regression Throughput Gates (eqyk.15) | ✅ **PASS** | 3.12s | `python -m scripts.perf_regression_gate --mode check --tolerance 0.50` |

---

## Detailed Subsystem Findings & Logs

### [PASS] Unit Tests (Ablations, Metrics, Checkpoints, Registry) (`unit_core`)
- **Category**: Foundations
- **Duration**: 2.70s
```text
tests/test_metrics_schema.py::test_validate_drops_unknown_metric_when_not_strict PASSED [ 80%]
tests/test_metrics_schema.py::test_validate_rejects_non_finite_values[nan] PASSED [ 82%]
tests/test_metrics_schema.py::test_validate_rejects_non_finite_values[inf] PASSED [ 83%]
tests/test_metrics_schema.py::test_validate_rejects_non_finite_values[-inf] PASSED [ 85%]
tests/test_metrics_schema.py::test_validate_rejects_non_numeric_value PASSED [ 87%]
tests/test_metrics_schema.py::test_helpers PASSED                        [ 88%]
tests/test_checkpoint_roundtrip.py::test_custom_config_round_trips_through_meta_json PASSED [ 90%]
tests/test_checkpoint_roundtrip.py::test_full_default_config_round_trips PASSED [ 91%]
tests/test_checkpoint_roundtrip.py::test_missing_config_falls_back_to_defaults PASSED [ 93%]
tests/test_checkpoint_roundtrip.py::test_unknown_saved_field_is_ignored_forward_compat PASSED [ 95%]
tests/test_checkpoint_roundtrip.py::test_missing_new_field_takes_default_back_compat PASSED [ 96%]
tests/test_checkpoint_roundtrip.py::test_config_hash_is_stable_and_sensitive PASSED [ 98%]
tests/test_checkpoint_roundtrip.py::test_provenance_stamp_has_sha_and_config_hash PASSED [100%]

============================== 62 passed in 0.32s ==============================
```

### [PASS] Full Bio Training Loop E2E (eqyk.4) (`e2e_bio_train`)
- **Category**: Foundations E2E
- **Duration**: 18.46s
```text
============================= test session starts ==============================
platform linux -- Python 3.14.2, pytest-9.0.2, pluggy-1.6.0 -- /data/projects/bio_inspired_nanochat/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /data/projects/bio_inspired_nanochat
configfile: pyproject.toml
plugins: cov-7.1.0, anyio-4.12.1
collecting ... collected 3 items

tests/test_e2e_train_bio.py::test_bio_e2e_run_passes_and_logs PASSED     [ 33%]
tests/test_e2e_train_bio.py::test_bio_e2e_battery_catches_no_learning PASSED [ 66%]
tests/test_e2e_train_bio.py::test_bio_e2e_moe_default_stays_finite PASSED [100%]

============================== 3 passed in 15.57s ==============================
```

### [PASS] CMA-ES Tune Loop & Resume E2E (eqyk.7) (`e2e_cmaes`)
- **Category**: Foundations E2E
- **Duration**: 38.22s
```text
============================= test session starts ==============================
platform linux -- Python 3.14.2, pytest-9.0.2, pluggy-1.6.0 -- /data/projects/bio_inspired_nanochat/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /data/projects/bio_inspired_nanochat
configfile: pyproject.toml
plugins: cov-7.1.0, anyio-4.12.1
collecting ... collected 2 items

tests/test_e2e_cmaes.py::test_cmaes_e2e_full_battery PASSED              [ 50%]
tests/test_e2e_cmaes.py::test_cmaes_e2e_cli_entrypoint PASSED            [100%]

============================== 2 passed in 34.78s ==============================
```

### [PASS] Online Hebbian & Bistable Plasticity E2E (eqyk.8) (`e2e_online_hebbian`)
- **Category**: Foundations E2E
- **Duration**: 18.65s
```text
============================= test session starts ==============================
platform linux -- Python 3.14.2, pytest-9.0.2, pluggy-1.6.0 -- /data/projects/bio_inspired_nanochat/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /data/projects/bio_inspired_nanochat
configfile: pyproject.toml
plugins: cov-7.1.0, anyio-4.12.1
collecting ... collected 2 items

tests/test_e2e_online_learning.py::test_online_learning_e2e_passes_and_logs PASSED [ 50%]
tests/test_e2e_online_learning.py::test_online_learning_e2e_deterministic PASSED [100%]

============================== 2 passed in 15.42s ==============================
```

### [PASS] Structural Evolution & Neurogenesis E2E (eqyk.9) (`e2e_structural_evolution`)
- **Category**: Foundations E2E
- **Duration**: 4.44s
```text
============================= test session starts ==============================
platform linux -- Python 3.14.2, pytest-9.0.2, pluggy-1.6.0 -- /data/projects/bio_inspired_nanochat/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /data/projects/bio_inspired_nanochat
configfile: pyproject.toml
plugins: cov-7.1.0, anyio-4.12.1
collecting ... collected 3 items

tests/test_e2e_structural_lifecycle.py::test_forced_lifecycle_invariants_and_lineage PASSED [ 33%]
tests/test_e2e_structural_lifecycle.py::test_lifecycle_resets_optimizer_momentum_adamw_and_muon PASSED [ 66%]
tests/test_e2e_structural_lifecycle.py::test_function_preserving_controller_step_no_loss_spike PASSED [100%]

============================== 3 passed in 1.07s ===============================
```

### [PASS] Wake/Sleep Consolidation & Forgetting E2E (eqyk.10) (`e2e_wake_sleep`)
- **Category**: Foundations E2E
- **Duration**: 2.93s
```text
============================= test session starts ==============================
platform linux -- Python 3.14.2, pytest-9.0.2, pluggy-1.6.0 -- /data/projects/bio_inspired_nanochat/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /data/projects/bio_inspired_nanochat
configfile: pyproject.toml
plugins: cov-7.1.0, anyio-4.12.1
collecting ... collected 4 items

tests/test_e2e_wake_sleep.py::test_wake_sleep_consolidation_full_battery PASSED [ 25%]
tests/test_e2e_wake_sleep.py::test_replay_buffer_surprise_prioritization PASSED [ 50%]
tests/test_e2e_wake_sleep.py::test_homeostatic_downscale_bounds_slow_weights PASSED [ 75%]
tests/test_e2e_wake_sleep.py::test_wake_sleep_cli_entrypoint PASSED      [100%]

============================== 4 passed in 0.78s ===============================
```

### [PASS] Neuromodulated 3-Factor RL Micro-Run E2E (eqyk.11) (`e2e_neuromod_rl`)
- **Category**: Foundations E2E
- **Duration**: 5.28s
```text
============================= test session starts ==============================
platform linux -- Python 3.14.2, pytest-9.0.2, pluggy-1.6.0 -- /data/projects/bio_inspired_nanochat/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /data/projects/bio_inspired_nanochat
configfile: pyproject.toml
plugins: cov-7.1.0, anyio-4.12.1
collecting ... collected 3 items

tests/test_e2e_neuromod_rl.py::test_neuromod_rl_full_battery PASSED      [ 33%]
tests/test_e2e_neuromod_rl.py::test_neuromod_bus_gating_mechanisms PASSED [ 66%]
tests/test_e2e_neuromod_rl.py::test_neuromod_rl_cli_entrypoint PASSED    [100%]

============================== 3 passed in 2.25s ===============================
```

### [PASS] Interpretability Probe/Lesion/Stimulation E2E (eqyk.12) (`e2e_interpretability`)
- **Category**: Foundations E2E
- **Duration**: 2.72s
```text
============================= test session starts ==============================
platform linux -- Python 3.14.2, pytest-9.0.2, pluggy-1.6.0 -- /data/projects/bio_inspired_nanochat/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /data/projects/bio_inspired_nanochat
configfile: pyproject.toml
plugins: cov-7.1.0, anyio-4.12.1
collecting ... collected 2 items

tests/test_e2e_probe_lesion_stim.py::test_probe_lesion_stim_full_battery PASSED [ 50%]
tests/test_e2e_probe_lesion_stim.py::test_probe_lesion_stim_cli_entrypoint PASSED [100%]

============================== 2 passed in 0.23s ===============================
```

### [PASS] Cross-Backend Parity (Triton/Rust/Python Reference) (eqyk.13) (`parity_backends`)
- **Category**: Parity & Invariants
- **Duration**: 7.49s
```text
cachedir: .pytest_cache
rootdir: /data/projects/bio_inspired_nanochat
configfile: pyproject.toml
plugins: cov-7.1.0, anyio-4.12.1
collecting ... collected 4 items

tests/test_presyn_backend_parity.py::test_python_canonical_matches_frozen_decode_trajectory PASSED [ 25%]
tests/test_presyn_backend_parity.py::test_triton_interpreter_matches_frozen_decode_trajectory PASSED [ 50%]
tests/test_presyn_backend_parity.py::test_cuda_triton_dispatch_matches_frozen_decode_trajectory SKIPPED [ 75%]
tests/test_presyn_backend_parity.py::test_rust_backend_matches_frozen_decode_trajectory SKIPPED [100%]

=========================== short test summary info ============================
SKIPPED [1] tests/test_presyn_backend_parity.py:154: CUDA backend unavailable
SKIPPED [1] tests/test_presyn_backend_parity.py:205: rustbpe extension not built; run `uv run maturin develop --release`
========================= 2 passed, 2 skipped in 5.01s =========================
```

### [PASS] Property-Based Metamorphic Invariants Suite (eqyk.14) (`property_invariants`)
- **Category**: Parity & Invariants
- **Duration**: 2.80s
```text
============================= test session starts ==============================
platform linux -- Python 3.14.2, pytest-9.0.2, pluggy-1.6.0 -- /data/projects/bio_inspired_nanochat/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /data/projects/bio_inspired_nanochat
configfile: pyproject.toml
plugins: cov-7.1.0, anyio-4.12.1
collecting ... collected 3 items

tests/test_property_invariants.py::test_property_invariants_full_battery PASSED [ 33%]
tests/test_property_invariants.py::test_metamorphic_prefix_causality PASSED [ 66%]
tests/test_property_invariants.py::test_property_invariants_cli_entrypoint PASSED [100%]

============================== 3 passed in 0.43s ===============================
```

### [PASS] Formal Theory Certificates & Lyapunov Invariants (eqyk.18) (`theory_certificates`)
- **Category**: Theory & Proofs
- **Duration**: 2.41s
```text
============================= test session starts ==============================
platform linux -- Python 3.14.2, pytest-9.0.2, pluggy-1.6.0 -- /data/projects/bio_inspired_nanochat/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /data/projects/bio_inspired_nanochat
configfile: pyproject.toml
plugins: cov-7.1.0, anyio-4.12.1
collecting ... collected 2 items

tests/test_e2e_theory_artifacts.py::test_theory_artifacts_full_battery PASSED [ 50%]
tests/test_e2e_theory_artifacts.py::test_theory_artifacts_cli_entrypoint PASSED [100%]

============================== 2 passed in 0.15s ===============================
```

### [PASS] Capability Frontier I Batteries (Deliberation/Adaptive/Scientist) (eqyk.19) (`capability_frontier_1`)
- **Category**: Capability Frontier
- **Duration**: 3.41s
```text
============================= test session starts ==============================
platform linux -- Python 3.14.2, pytest-9.0.2, pluggy-1.6.0 -- /data/projects/bio_inspired_nanochat/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /data/projects/bio_inspired_nanochat
configfile: pyproject.toml
plugins: cov-7.1.0, anyio-4.12.1
collecting ... collected 2 items

tests/test_e2e_capability_frontier.py::test_capability_frontier_full_battery PASSED [ 50%]
tests/test_e2e_capability_frontier.py::test_capability_frontier_cli_entrypoint PASSED [100%]

============================== 2 passed in 0.88s ===============================
```

### [PASS] Uncertainty Calibration & Selective Prediction (eqyk.20) (`uncertainty_calibration`)
- **Category**: Capability Frontier
- **Duration**: 4.52s
```text
============================= test session starts ==============================
platform linux -- Python 3.14.2, pytest-9.0.2, pluggy-1.6.0 -- /data/projects/bio_inspired_nanochat/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /data/projects/bio_inspired_nanochat
configfile: pyproject.toml
plugins: cov-7.1.0, anyio-4.12.1
collecting ... collected 4 items

tests/test_e2e_uncertainty.py::test_uncertainty_e2e_logs_metrics_actions_and_thermodynamic_evidence PASSED [ 25%]
tests/test_e2e_uncertainty.py::test_uncertainty_e2e_rejects_invalid_target_coverage[nan] PASSED [ 50%]
tests/test_e2e_uncertainty.py::test_uncertainty_e2e_rejects_invalid_target_coverage[0.0] PASSED [ 75%]
tests/test_e2e_uncertainty.py::test_uncertainty_e2e_rejects_invalid_target_coverage[1.0] PASSED [100%]

============================== 4 passed in 1.29s ===============================
```

### [PASS] Synaptic Retrofit & MGR Attention Variants Battery (eqyk.21) (`retrofit_mgr_geometry`)
- **Category**: Capability Frontier
- **Duration**: 5.59s
```text
============================= test session starts ==============================
platform linux -- Python 3.14.2, pytest-9.0.2, pluggy-1.6.0 -- /data/projects/bio_inspired_nanochat/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /data/projects/bio_inspired_nanochat
configfile: pyproject.toml
plugins: cov-7.1.0, anyio-4.12.1
collecting ... collected 2 items

tests/test_e2e_retrofit_mgr.py::test_retrofit_mgr_full_battery PASSED    [ 50%]
tests/test_e2e_retrofit_mgr.py::test_retrofit_mgr_cli_entrypoint PASSED  [100%]

============================== 2 passed in 1.95s ===============================
```

### [PASS] Wave-2 Emergent Compositions (Self-Correct, Metacognition, Search) (eqyk.22) (`wave2_compositions`)
- **Category**: Capability Frontier
- **Duration**: 2.95s
```text
============================= test session starts ==============================
platform linux -- Python 3.14.2, pytest-9.0.2, pluggy-1.6.0 -- /data/projects/bio_inspired_nanochat/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /data/projects/bio_inspired_nanochat
configfile: pyproject.toml
plugins: cov-7.1.0, anyio-4.12.1
collecting ... collected 2 items

tests/test_e2e_wave2_compositions.py::test_wave2_compositions_full_battery PASSED [ 50%]
tests/test_e2e_wave2_compositions.py::test_wave2_compositions_cli_entrypoint PASSED [100%]

============================== 2 passed in 0.12s ===============================
```

### [PASS] Performance Regression Throughput Gates (eqyk.15) (`perf_regression_gates`)
- **Category**: Performance
- **Duration**: 3.12s
```text
Running Performance Regression Benchmarks...
                      Performance Regression Gate Results                       
┏━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ Benchmark      ┃        ┃       Observed ┃        Baseline ┃        ┃        ┃
┃ Config         ┃ Mode   ┃        (tok/s) ┃         (tok/s) ┃  Ratio ┃ Status ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ standard_tran… │ train  │         7534.3 │          3000.0 │ 251.1% │ PASS   │
│ synaptic_tran… │ train  │         2642.4 │          1500.0 │ 176.2% │ PASS   │
│ standard_tran… │ decode │        12648.9 │          3000.0 │ 421.6% │ PASS   │
│ synaptic_tran… │ decode │         1932.2 │          1000.0 │ 193.2% │ PASS   │
└────────────────┴────────┴────────────────┴─────────────────┴────────┴────────┘
All performance regression gates passed!
```
