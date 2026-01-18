# Dependency Upgrade Log

**Date:** 2026-01-18  |  **Project:** bio_inspired_nanochat  |  **Language:** Rust + Python

## Summary
- **Updated:** Rust: 15, Python: 30+
- **Skipped:** 1 (ndarray 0.17 - numpy crate incompatibility)
- **Failed:** 0
- **Needs attention:** 0

## Rust Updates

| Package | Old | New |
|---------|-----|-----|
| hashbrown | 0.16.0 | 0.16.1 |
| indexmap | 2.12.0 | 2.13.0 |
| log | 0.4.28 | 0.4.29 |
| proc-macro2 | 1.0.103 | 1.0.105 |
| syn | 2.0.110 | 2.0.114 |
| zerocopy | 0.8.27 | 0.8.33 |

### Skipped: ndarray 0.16 → 0.17
- **Reason:** Breaking changes with numpy crate - dimension trait bounds not satisfied
- **Action:** Kept at 0.16.1 for numpy compatibility

## Python Updates (via uv sync --upgrade)

Major updates:
- fastapi 0.121.2 → 0.128.0
- huggingface-hub 0.36.0 → 1.3.2
- numpy 2.3.5 → 2.4.1
- pyarrow 22.0.0 → 23.0.0
- wandb 0.23.0 → 0.24.0
- uvicorn 0.38.0 → 0.40.0

## Verification

- `cargo build` - Passed (2 warnings)
- `uv run python -c "import fastapi; import torch"` - OK
