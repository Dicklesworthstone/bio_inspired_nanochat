"""Tests for CMA-ES tuning loop E2E harness (bead eqyk.7).

Verifies the full CMA-ES optimization lifecycle in CI time:
  1. 2-generation optimization on the synthetic associative recall task.
  2. Structured artifacts produced: progress.jsonl, best_params.json, inert JSON replay states.
  3. Resume contract: resuming from checkpoint continues without loss of state.
  4. Stagnation policy: triggers early-stopping or sigma-reset under stalled loss.
  5. Detailed per-generation logging in events.jsonl.

Run:
    pytest tests/test_e2e_cmaes.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.e2e.cmaes_tune import CmaesE2EConfig, main as e2e_main, run_cmaes_e2e

pytestmark = pytest.mark.e2e

EXPECTED_INVARIANTS = (
    "initial_optimization_exits_clean",
    "progress_jsonl_written",
    "best_params_json_written",
    "checkpoints_written",
    "checkpoint_resume_contract",
    "stagnation_policy_fires",
    "results_registry_appended",
)


def test_cmaes_e2e_full_battery(tmp_path: Path):
    """The entire CMA-ES battery runs end-to-end and every invariant holds."""
    cfg = CmaesE2EConfig(
        generations=2,
        popsize=4,
        steps=2,
        batch_size=4,
        seed=1337,
        device="cpu",
    )
    report = run_cmaes_e2e(cfg, run_dir=tmp_path, verbose=False)
    report.assert_passed()

    inv_map = {inv.name: inv for inv in report.invariants}
    for name in EXPECTED_INVARIANTS:
        assert name in inv_map, f"Missing invariant: {name}"
        assert inv_map[name].passed, f"Invariant {name} failed: {inv_map[name].detail}"

    # Verify events.jsonl was logged
    events_path = tmp_path / "events.jsonl"
    assert events_path.exists(), "events.jsonl trace must be created"
    events = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        line_s = line.strip()
        if line_s:
            try:
                events.append(json.loads(line_s))
            except json.JSONDecodeError:
                pass
    assert len(events) >= len(EXPECTED_INVARIANTS) + 1


def test_cmaes_e2e_cli_entrypoint(tmp_path: Path):
    """The main entrypoint can be executed directly with CLI arguments."""
    ret = e2e_main(["--run-dir", str(tmp_path), "--device", "cpu", "--seed", "42"])
    assert ret == 0, f"CLI entrypoint exited with non-zero code {ret}"
