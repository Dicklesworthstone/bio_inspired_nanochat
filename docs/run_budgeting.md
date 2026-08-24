# Run Budgeting & Compute Accounting (bead 2a7)

> **Purpose**: Standardized tracking of GPU compute hours and dollar costs across benchmarks, training sweeps, visualization captures, and HPO searches.

---

## 1. Accounting Standard

- **Default Unit Cost**: $0.75 / GPU-hour (approximate dual RTX 4090 cloud equivalent rate).
- **Log Location**: `runs/budget_accounting.jsonl` (JSON Lines format).
- **Required Metadata**: `run_id`, `purpose`, `num_gpus`, `duration_seconds`, `gpu_hours`, `estimated_cost_usd`, `objective_type` (`proxy` vs `full`).

---

## 2. Usage Examples

### Using `RunBudgetTracker` as Context Manager / Timer:
```python
from bio_inspired_nanochat.budgeting import RunBudgetTracker

tracker = RunBudgetTracker(
    run_id="eval_matrix_matrix_a_seed42",
    purpose="Eval Matrix A 10M FineWeb",
    num_gpus=2,
    objective_type="full",
).start()

# ... run training / evaluation workload ...

entry = tracker.stop(extra={"dataset": "fineweb", "tokens": 10_000_000})
print(f"Run completed: {entry.gpu_hours:.3f} GPU-hrs (${entry.estimated_cost_usd:.2f})")
```

### Direct Logging Helper:
```python
from bio_inspired_nanochat.budgeting import log_run_cost

log_run_cost(
    run_id="cmaes_iter_10",
    purpose="Hyperparameter optimization proxy search",
    duration_seconds=124.5,
    num_gpus=1,
    objective_type="proxy",
)
```

### Summarize via CLI:
```bash
uv run python -m scripts.summarize_budget
```
