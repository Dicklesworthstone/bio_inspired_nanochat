"""E2E: adaptive-vs-fixed compute Pareto evaluation (bead `r00r.3.4`).

Drives ``scripts/e2e/pareto_efficiency.py`` (:func:`run_pareto_e2e`), which falsifies the
METABOLIC ADAPTIVE COMPUTE claim at toy scale: does ATP-budgeted allocation
(r00r.3.1 router/budget + r00r.3.2 dynamic levers + r00r.3.3 quality guard) spend fewer
compute units than the fixed-compute baseline at equal quality?

What this test locks:
  1. The fixed baseline costs exactly ``maximum_compute_units`` per token (by construction).
  2. Tighter budgets spend strictly less per token (the allocation policy responds to budget).
  3. ATP accounting is exact (spent + remaining == total) on every sequence account.
  4. A stats-backed verdict exists: either a Pareto improvement at matched quality or a
     DOCUMENTED NULL (current calibrated outcome at this scale — see the bead note).
  5. The run emits a machine-readable JSONL trace (train_step + per-arm pareto_sequence).

Run:  pytest tests/test_e2e_pareto_efficiency.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# scripts/ is a namespace package on sys.path (same import path README uses for `python -m scripts.*`).
from scripts.e2e.pareto_efficiency import ParetoE2EConfig, run_pareto_e2e

pytestmark = pytest.mark.e2e

INVARIANTS = (
    "fixed_baseline_uses_max_compute",
    "tighter_budget_spends_less",
    "atp_accounting_exact",
    "pareto_verdict_with_stats",
    "jsonl_trace_written",
)


def _read_events(run_dir: Path) -> list[dict]:
    path = run_dir / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_pareto_e2e_runs_and_logs(tmp_path):
    """The battery runs end-to-end, every invariant holds, and the trace is readable."""
    cfg = ParetoE2EConfig(steps=1000, seed=0)
    report = run_pareto_e2e(cfg, run_dir=tmp_path, verbose=False)

    report.assert_passed()
    names = {r.name: r for r in report.invariants}
    for inv in INVARIANTS:
        assert inv in names, f"missing invariant {inv}"
        assert names[inv].passed, f"{inv} failed: {names[inv].detail}"

    # The verdict is always one of the two honest outcomes, whichever the numbers support.
    assert report.summary["verdict"] in ("improvement", "null")

    # JSONL artifact: train_step events plus per-arm scoring events.
    events = _read_events(tmp_path)
    train_steps = [e for e in events if e["event"] == "train_step"]
    pareto_seqs = [e for e in events if e["event"] == "pareto_sequence"]
    assert len(train_steps) >= cfg.steps - 20, "training trace present (minus guard slack)"
    assert len(pareto_seqs) >= 2, "at least one scored adaptive arm must be traced"


def test_pareto_e2e_deterministic(tmp_path):
    """Fixed seed ⇒ identical invariant outcomes across two runs (single-threaded CPU)."""
    r1 = run_pareto_e2e(
        ParetoE2EConfig(steps=200, seed=7, max_eval_sequences=6),
        run_dir=tmp_path / "a", verbose=False,
    )
    r2 = run_pareto_e2e(
        ParetoE2EConfig(steps=200, seed=7, max_eval_sequences=6),
        run_dir=tmp_path / "b", verbose=False,
    )
    assert [(i.name, i.passed, i.observed) for i in r1.invariants] == [
        (i.name, i.passed, i.observed) for i in r2.invariants
    ]
