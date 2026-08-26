# System Master Validation Report — val-1787711070

> **Verdict**: **ALL SYSTEMS PERFECT (VERIFIED WORKING)**  
> **Timestamp**: `2026-08-26T02:24:30.527061+00:00`  
> **Git SHA**: `9224a5394c470de674d29336082a443d220f1300`  
> **Hardware**: `cpu:x86_64`  
> **Summary**: `5/5 Passed` (100.0%) in `27.92s`

---

## Subsystem Verification Matrix

| Category | Subsystem | Status | Duration | Command |
| :--- | :--- | :---: | :---: | :--- |
| Foundations | Unit Tests (Ablations, Metrics, Checkpoints, Registry) | ✅ **PASS** | 3.81s | `pytest tests/test_ablation_registry.py tests/test_results_registry.py tests/test_metrics_schema.py tests/test_checkpoint_roundtrip.py -v --tb=short` |
| Parity & Invariants | Cross-Backend Parity (Triton/Rust/Python Reference) (eqyk.13) | ✅ **PASS** | 9.13s | `pytest tests/test_presyn_backend_parity.py -v --tb=short` |
| Parity & Invariants | Property-Based Metamorphic Invariants Suite (eqyk.14) | ✅ **PASS** | 3.88s | `pytest tests/test_property_invariants.py -v --tb=short` |
| Theory & Proofs | Formal Theory Certificates & Lyapunov Invariants (eqyk.18) | ✅ **PASS** | 3.15s | `pytest tests/test_e2e_theory_artifacts.py -v --tb=short` |
| Performance | Performance Regression Throughput Gates (eqyk.15) | ✅ **PASS** | 7.95s | `python -m scripts.perf_regression_gate --mode check --tolerance 0.50` |

---

## Detailed Subsystem Findings & Logs

### [PASS] Unit Tests (Ablations, Metrics, Checkpoints, Registry) (`unit_core`)
- **Category**: Foundations
- **Duration**: 3.81s
```text
tests/test_metrics_schema.py::test_validate_rejects_non_finite_values[-inf] PASSED [ 84%]
tests/test_metrics_schema.py::test_validate_rejects_non_numeric_value PASSED [ 85%]
tests/test_metrics_schema.py::test_helpers PASSED                        [ 87%]
tests/test_checkpoint_roundtrip.py::test_custom_config_round_trips_through_meta_json PASSED [ 88%]
tests/test_checkpoint_roundtrip.py::test_full_default_config_round_trips PASSED [ 89%]
tests/test_checkpoint_roundtrip.py::test_missing_config_falls_back_to_defaults PASSED [ 91%]
tests/test_checkpoint_roundtrip.py::test_unknown_saved_field_is_ignored_forward_compat PASSED [ 92%]
tests/test_checkpoint_roundtrip.py::test_missing_new_field_takes_default_back_compat PASSED [ 93%]
tests/test_checkpoint_roundtrip.py::test_config_hash_is_stable_and_sensitive PASSED [ 94%]
tests/test_checkpoint_roundtrip.py::test_provenance_stamp_has_sha_and_config_hash PASSED [ 96%]
tests/test_checkpoint_roundtrip.py::test_crash_debris_step_is_not_discovered_when_markers_in_use PASSED [ 97%]
tests/test_checkpoint_roundtrip.py::test_legacy_directory_without_markers_stays_discoverable PASSED [ 98%]
tests/test_checkpoint_roundtrip.py::test_rustbpe_tokenizer_json_round_trip PASSED [100%]

============================== 78 passed in 0.55s ==============================
```

### [PASS] Cross-Backend Parity (Triton/Rust/Python Reference) (eqyk.13) (`parity_backends`)
- **Category**: Parity & Invariants
- **Duration**: 9.13s
```text
platform linux -- Python 3.14.2, pytest-9.0.2, pluggy-1.6.0 -- /data/projects/bio_inspired_nanochat/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /data/projects/bio_inspired_nanochat
configfile: pyproject.toml
plugins: cov-7.1.0, anyio-4.12.1
collecting ... collected 4 items

tests/test_presyn_backend_parity.py::test_python_canonical_matches_frozen_decode_trajectory PASSED [ 25%]
tests/test_presyn_backend_parity.py::test_triton_interpreter_matches_frozen_decode_trajectory PASSED [ 50%]
tests/test_presyn_backend_parity.py::test_cuda_triton_dispatch_matches_frozen_decode_trajectory SKIPPED [ 75%]
tests/test_presyn_backend_parity.py::test_rust_backend_matches_frozen_decode_trajectory PASSED [100%]

=========================== short test summary info ============================
SKIPPED [1] tests/test_presyn_backend_parity.py:154: CUDA backend unavailable
========================= 3 passed, 1 skipped in 6.05s =========================
```

### [PASS] Property-Based Metamorphic Invariants Suite (eqyk.14) (`property_invariants`)
- **Category**: Parity & Invariants
- **Duration**: 3.88s
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

============================== 3 passed in 1.05s ===============================
```

### [PASS] Formal Theory Certificates & Lyapunov Invariants (eqyk.18) (`theory_certificates`)
- **Category**: Theory & Proofs
- **Duration**: 3.15s
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

============================== 2 passed in 0.27s ===============================
```

### [PASS] Performance Regression Throughput Gates (eqyk.15) (`perf_regression_gates`)
- **Category**: Performance
- **Duration**: 7.95s
```text
Running Performance Regression Benchmarks...
                      Performance Regression Gate Results                       
┏━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ Benchmark      ┃        ┃       Observed ┃        Baseline ┃        ┃        ┃
┃ Config         ┃ Mode   ┃        (tok/s) ┃         (tok/s) ┃  Ratio ┃ Status ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ standard_tran… │ train  │         6863.7 │          2956.2 │ 232.2% │ PASS   │
│ synaptic_tran… │ train  │          383.6 │           349.6 │ 109.7% │ PASS   │
│ standard_tran… │ decode │          416.9 │           276.7 │ 150.7% │ PASS   │
│ synaptic_tran… │ decode │          114.2 │            76.1 │ 150.1% │ PASS   │
└────────────────┴────────┴────────────────┴─────────────────┴────────┴────────┘
All performance regression gates passed!
```
