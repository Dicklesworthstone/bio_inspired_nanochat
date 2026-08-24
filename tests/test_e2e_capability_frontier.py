"""Tests for the Capability Frontier end-to-end verification battery (beads r00r, eqyk.19)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.e2e.capability_frontier_suite import (
    CapabilityFrontierConfig,
    main as e2e_main,
    run_capability_frontier_e2e,
)

EXPECTED_CAPABILITY_INVARIANTS = (
    "deliberation_energy_monotonicity_and_halting",
    "adaptive_compute_atp_budget_and_routing",
    "automated_scientist_preregistration",
    "cross_architecture_bio_adapter_injection",
    "dream_sleep_consolidation_replay",
)


def test_capability_frontier_full_battery(tmp_path: Path):
    """The entire Capability-Frontier verification battery executes end-to-end and all invariants hold."""
    cfg = CapabilityFrontierConfig(
        deliberation_max_iters=25,
        deliberation_tol=1e-4,
        atp_initial_budget=100,
        expert_dim=32,
        vocab_size=64,
        seed=42,
    )
    report = run_capability_frontier_e2e(cfg, run_dir=tmp_path, verbose=False)
    report.assert_passed()

    inv_map = {inv.name: inv for inv in report.invariants}
    for name in EXPECTED_CAPABILITY_INVARIANTS:
        assert name in inv_map, f"Missing invariant: {name}"
        assert inv_map[name].passed, f"Invariant {name} failed: {inv_map[name].detail}"

    # Verify structured event stream was written and contains records
    events_path = tmp_path / "events.jsonl"
    assert events_path.exists(), "events.jsonl trace must be created"
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(events) >= len(EXPECTED_CAPABILITY_INVARIANTS) + 1


def test_capability_frontier_cli_entrypoint(tmp_path: Path):
    """The main entrypoint can be executed directly with CLI arguments."""
    ret = e2e_main([
        "--run-dir",
        str(tmp_path),
        "--delib-iters",
        "20",
        "--atp-budget",
        "80",
        "--seed",
        "42",
    ])
    assert ret == 0, f"CLI entrypoint exited with non-zero code {ret}"
