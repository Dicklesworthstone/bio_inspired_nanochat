"""Tests for the Theory Artifacts end-to-end verification battery (beads 0642, eqyk.18)."""

from __future__ import annotations

import json
from pathlib import Path


from scripts.e2e.theory_artifacts_suite import (
    TheoryArtifactsConfig,
    main as e2e_main,
    run_theory_artifacts_e2e,
)

EXPECTED_THEORY_INVARIANTS = (
    "metriplectic_energy_and_free_energy",
    "singular_perturbation_and_cusp_latch",
    "stochastic_thermo_and_tur_bounds",
    "structural_geometry_and_optimal_transport",
    "timescale_separation_coupling",
)


def test_theory_artifacts_full_battery(tmp_path: Path):
    """The entire Leapfrog-Theory verification battery executes end-to-end and all invariants hold."""
    cfg = TheoryArtifactsConfig(
        dt=0.05,
        metriplectic_steps=30,
        thermo_trajectories=1000,
        thermo_steps=2.0,
        expert_dim=24,
        seed=1337,
    )
    report = run_theory_artifacts_e2e(cfg, run_dir=tmp_path, verbose=False)
    report.assert_passed()

    inv_map = {inv.name: inv for inv in report.invariants}
    for name in EXPECTED_THEORY_INVARIANTS:
        assert name in inv_map, f"Missing invariant: {name}"
        assert inv_map[name].passed, f"Invariant {name} failed: {inv_map[name].detail}"

    # Verify structured event stream was written and contains records
    events_path = tmp_path / "events.jsonl"
    assert events_path.exists(), "events.jsonl trace must be created"
    events = []
    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    assert len(events) >= len(EXPECTED_THEORY_INVARIANTS) + 1


def test_theory_artifacts_cli_entrypoint(tmp_path: Path):
    """The main entrypoint can be executed directly with CLI arguments."""
    ret = e2e_main(["--run-dir", str(tmp_path), "--steps", "20", "--thermo-trajectories", "500", "--seed", "42"])
    assert ret == 0, f"CLI entrypoint exited with non-zero code {ret}"
