"""Tests for Probing / Lesion / Optogenetic Stimulation E2E harness (bead eqyk.12).

Verifies in-silico neuroscience patch-clamp probing, acute causal lesioning,
and optogenetic clamping on living bio-inspired transformer models.

Run:
    pytest tests/test_e2e_probe_lesion_stim.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.e2e.probe_lesion_stim import (
    ProbeLesionStimConfig,
    main as e2e_main,
    run_probe_lesion_stim_e2e,
)

pytestmark = pytest.mark.e2e

EXPECTED_INVARIANTS = (
    "probe_records_live_biostate_traces",
    "lesion_causes_measurable_causal_deficit_and_restores",
    "optogenetic_stimulation_modulates_dynamics_and_rescues",
)


def test_probe_lesion_stim_full_battery(tmp_path: Path):
    """The entire In-Silico Probing / Lesion / Stim battery executes end-to-end and all invariants hold."""
    cfg = ProbeLesionStimConfig(
        n_layer=2,
        n_head=4,
        n_embd=64,
        vocab_size=64,
        sequence_len=32,
        batch_size=4,
        num_pairs=2,
        seed=1337,
        device="cpu",
    )
    report = run_probe_lesion_stim_e2e(cfg, run_dir=tmp_path, verbose=False)
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


def test_probe_lesion_stim_cli_entrypoint(tmp_path: Path):
    """The main entrypoint can be executed directly with CLI arguments."""
    ret = e2e_main(["--run-dir", str(tmp_path), "--device", "cpu", "--seed", "42"])
    assert ret == 0, f"CLI entrypoint exited with non-zero code {ret}"
