# Seed & Determinism Policy Framework (beads aiq, hm4.4)

> **Standard**: Every reproducible workload (training, eval matrix, HPO tuning, profiling) must configure deterministic execution and record seed provenance.

---

## 1. Core Principles

1. **Explicit Seeds Only**: Never allow unseeded random state. Every script accepts an explicit `--seed` parameter (default `42` or pre-registered test seeds `[1337, 1338, 1339]`).
2. **Environment Synchronization**: `configure_determinism(seed)` synchronizes:
   - Python `random`
   - PyTorch CPU `torch.manual_seed(seed)`
   - PyTorch CUDA `torch.cuda.manual_seed_all(seed)`
   - `torch.backends.cudnn.deterministic = True`
   - `CUBLAS_WORKSPACE_CONFIG=:4096:8`
   - `PYTHONHASHSEED=seed`
3. **Provenance Stamping**: Determinism configuration is stamped onto every `RunRecord` and logged in `events.jsonl` under `provenance.determinism`.

---

## 2. Usage Guide

```python
from bio_inspired_nanochat.determinism import configure_determinism, determinism_provenance_dict

# 1. Initialize deterministic execution at run startup
state = configure_determinism(seed=1337, deterministic=True)

# 2. Record provenance into telemetry / registry
meta = {
    "run_id": "run_12345",
    "determinism": determinism_provenance_dict(seed=1337),
}
```

---

## 3. Performance vs Determinism Tradeoff

| Mode | Throughput | Reproducibility | Recommended Use |
|:---|:---|:---|:---|
| **Strict Deterministic** (`deterministic=True, warn_only=True`) | ~95–98% peak tok/s | Bitwise / float-reproducible within fp32 tolerance | CI, test suites, evaluation matrix, HPO CMA-ES |
| **High Throughput** (`deterministic=False`) | 100% peak tok/s | Stochastic at kernel scheduling boundaries | Large-scale production pre-training runs |
