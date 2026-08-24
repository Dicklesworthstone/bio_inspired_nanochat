# System Master Validation Report — val-1787579537

> **Verdict**: **DEGRADED (4 FAILURE(S))**  
> **Timestamp**: `2026-08-24T13:52:17.892320+00:00`  
> **Git SHA**: `2078dd9e8756fc53278b0b31e62b21696fcbe527`  
> **Hardware**: `cpu:x86_64`  
> **Summary**: `1/5 Passed` (20.0%) in `17.47s`

---

## Subsystem Verification Matrix

| Category | Subsystem | Status | Duration | Command |
| :--- | :--- | :---: | :---: | :--- |
| Foundations | Unit Tests (Ablations, Metrics, Checkpoints, Registry) | ❌ **FAIL** | 2.89s | `pytest tests/test_ablation_registry.py tests/test_results_registry.py tests/test_metrics_schema.py tests/test_checkpoint_manager.py -v --tb=short` |
| Parity & Invariants | Cross-Backend Parity (Triton/Rust/Python Reference) (eqyk.13) | ❌ **FAIL** | 2.93s | `pytest tests/test_backend_parity.py -v --tb=short` |
| Parity & Invariants | Property-Based Metamorphic Invariants Suite (eqyk.14) | ✅ **PASS** | 4.56s | `pytest tests/test_property_invariants.py -v --tb=short` |
| Theory & Proofs | Formal Theory Certificates & Lyapunov Invariants (eqyk.18) | ❌ **FAIL** | 2.96s | `pytest tests/test_e2e_theory.py -v --tb=short` |
| Performance | Performance Regression Throughput Gates (eqyk.15) | ❌ **FAIL** | 4.13s | `python -m scripts.perf_regression_gate --mode check --tolerance 0.40` |

---

## Detailed Subsystem Findings & Logs

### [FAIL] Unit Tests (Ablations, Metrics, Checkpoints, Registry) (`unit_core`)
- **Category**: Foundations
- **Duration**: 2.89s
- **Error**: `Command exited with code 4`
```text
============================= test session starts ==============================
platform linux -- Python 3.14.2, pytest-9.0.2, pluggy-1.6.0 -- /data/projects/bio_inspired_nanochat/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /data/projects/bio_inspired_nanochat
configfile: pyproject.toml
plugins: cov-7.1.0, anyio-4.12.1
collecting ... collected 0 items

============================ no tests ran in 0.00s =============================
```
**Stderr**:
```text
ERROR: file or directory not found: tests/test_checkpoint_manager.py
```

### [FAIL] Cross-Backend Parity (Triton/Rust/Python Reference) (eqyk.13) (`parity_backends`)
- **Category**: Parity & Invariants
- **Duration**: 2.93s
- **Error**: `Command exited with code 4`
```text
============================= test session starts ==============================
platform linux -- Python 3.14.2, pytest-9.0.2, pluggy-1.6.0 -- /data/projects/bio_inspired_nanochat/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /data/projects/bio_inspired_nanochat
configfile: pyproject.toml
plugins: cov-7.1.0, anyio-4.12.1
collecting ... collected 0 items

============================ no tests ran in 0.00s =============================
```
**Stderr**:
```text
ERROR: file or directory not found: tests/test_backend_parity.py
```

### [PASS] Property-Based Metamorphic Invariants Suite (eqyk.14) (`property_invariants`)
- **Category**: Parity & Invariants
- **Duration**: 4.56s
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

============================== 3 passed in 1.19s ===============================
```

### [FAIL] Formal Theory Certificates & Lyapunov Invariants (eqyk.18) (`theory_certificates`)
- **Category**: Theory & Proofs
- **Duration**: 2.96s
- **Error**: `Command exited with code 4`
```text
============================= test session starts ==============================
platform linux -- Python 3.14.2, pytest-9.0.2, pluggy-1.6.0 -- /data/projects/bio_inspired_nanochat/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /data/projects/bio_inspired_nanochat
configfile: pyproject.toml
plugins: cov-7.1.0, anyio-4.12.1
collecting ... collected 0 items

============================ no tests ran in 0.00s =============================
```
**Stderr**:
```text
ERROR: file or directory not found: tests/test_e2e_theory.py
```

### [FAIL] Performance Regression Throughput Gates (eqyk.15) (`perf_regression_gates`)
- **Category**: Performance
- **Duration**: 4.13s
- **Error**: `Command exited with code 1`
```text
Running Performance Regression Benchmarks...
                      Performance Regression Gate Results                       
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━┓
┃ Benchmark       ┃        ┃       Observed ┃        Baseline ┃       ┃        ┃
┃ Config          ┃ Mode   ┃        (tok/s) ┃         (tok/s) ┃ Ratio ┃ Status ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━┩
│ standard_trans… │ train  │         2748.2 │          8423.8 │ 32.6% │ FAIL   │
│ synaptic_trans… │ train  │         1470.4 │          1523.8 │ 96.5% │ PASS   │
│ standard_trans… │ decode │         2506.1 │         13122.5 │ 19.1% │ FAIL   │
│ synaptic_trans… │ decode │          987.4 │          1925.8 │ 51.3% │ FAIL   │
└─────────────────┴────────┴────────────────┴─────────────────┴───────┴────────┘
Performance regression gate failed!
```
