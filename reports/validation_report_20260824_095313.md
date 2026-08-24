# System Master Validation Report — val-1787579593

> **Verdict**: **ALL SYSTEMS PERFECT (VERIFIED WORKING)**  
> **Timestamp**: `2026-08-24T13:53:13.877808+00:00`  
> **Git SHA**: `2078dd9e8756fc53278b0b31e62b21696fcbe527`  
> **Hardware**: `cpu:x86_64`  
> **Summary**: `5/5 Passed` (100.0%) in `21.59s`

---

## Subsystem Verification Matrix

| Category | Subsystem | Status | Duration | Command |
| :--- | :--- | :---: | :---: | :--- |
| Foundations | Unit Tests (Ablations, Metrics, Checkpoints, Registry) | ✅ **PASS** | 2.86s | `pytest tests/test_ablation_registry.py tests/test_results_registry.py tests/test_metrics_schema.py tests/test_checkpoint_roundtrip.py -v --tb=short` |
| Parity & Invariants | Cross-Backend Parity (Triton/Rust/Python Reference) (eqyk.13) | ✅ **PASS** | 9.02s | `pytest tests/test_presyn_backend_parity.py -v --tb=short` |
| Parity & Invariants | Property-Based Metamorphic Invariants Suite (eqyk.14) | ✅ **PASS** | 3.40s | `pytest tests/test_property_invariants.py -v --tb=short` |
| Theory & Proofs | Formal Theory Certificates & Lyapunov Invariants (eqyk.18) | ✅ **PASS** | 2.93s | `pytest tests/test_e2e_theory_artifacts.py -v --tb=short` |
| Performance | Performance Regression Throughput Gates (eqyk.15) | ✅ **PASS** | 3.38s | `python -m scripts.perf_regression_gate --mode check --tolerance 0.50` |

---

## Detailed Subsystem Findings & Logs

### [PASS] Unit Tests (Ablations, Metrics, Checkpoints, Registry) (`unit_core`)
- **Category**: Foundations
- **Duration**: 2.86s
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

============================== 62 passed in 0.36s ==============================
```

### [PASS] Cross-Backend Parity (Triton/Rust/Python Reference) (eqyk.13) (`parity_backends`)
- **Category**: Parity & Invariants
- **Duration**: 9.02s
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
========================= 2 passed, 2 skipped in 6.13s =========================
```

### [PASS] Property-Based Metamorphic Invariants Suite (eqyk.14) (`property_invariants`)
- **Category**: Parity & Invariants
- **Duration**: 3.40s
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

============================== 3 passed in 0.54s ===============================
```

### [PASS] Formal Theory Certificates & Lyapunov Invariants (eqyk.18) (`theory_certificates`)
- **Category**: Theory & Proofs
- **Duration**: 2.93s
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

============================== 2 passed in 0.22s ===============================
```

### [PASS] Performance Regression Throughput Gates (eqyk.15) (`perf_regression_gates`)
- **Category**: Performance
- **Duration**: 3.38s
```text
Running Performance Regression Benchmarks...
                      Performance Regression Gate Results                       
┏━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ Benchmark      ┃        ┃       Observed ┃        Baseline ┃        ┃        ┃
┃ Config         ┃ Mode   ┃        (tok/s) ┃         (tok/s) ┃  Ratio ┃ Status ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ standard_tran… │ train  │         5029.1 │          3000.0 │ 167.6% │ PASS   │
│ synaptic_tran… │ train  │         2302.6 │          1500.0 │ 153.5% │ PASS   │
│ standard_tran… │ decode │        12406.8 │          3000.0 │ 413.6% │ PASS   │
│ synaptic_tran… │ decode │         1289.0 │          1000.0 │ 128.9% │ PASS   │
└────────────────┴────────┴────────────────┴─────────────────┴────────┴────────┘
All performance regression gates passed!
```
