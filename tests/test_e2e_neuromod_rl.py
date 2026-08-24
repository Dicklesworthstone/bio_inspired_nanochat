"""Tests for Neuromodulated Three-Factor / RL E2E harness (bead eqyk.11).

Verifies the global neuromodulatory bus (DA/ACh/NE), reward-gated three-factor learning,
and stable RL micro-training in CI time.

Run:
    pytest tests/test_e2e_neuromod_rl.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bio_inspired_nanochat.neuromod import NeuromodulatoryBus, NeuromodConfig
from scripts.e2e.neuromod_rl import (
    NeuromodRLE2EConfig,
    _build_model,
    main as e2e_main,
    run_neuromod_rl_e2e,
)

pytestmark = pytest.mark.e2e

EXPECTED_INVARIANTS = (
    "bus_broadcast_gates_plasticity_and_exploration",
    "three_factor_consolidates_rewarded_associations_only",
    "rl_microrun_improves_reward_and_stays_finite",
)


def test_neuromod_rl_full_battery(tmp_path: Path):
    """The entire Neuromod / Three-Factor / RL battery executes end-to-end and all invariants hold."""
    cfg = NeuromodRLE2EConfig(
        n_layer=2,
        n_head=4,
        n_embd=64,
        vocab_size=64,
        sequence_len=32,
        rl_steps=25,
        batch_size=8,
        context_len=4,
        seed=1337,
        device="cpu",
    )
    report = run_neuromod_rl_e2e(cfg, run_dir=tmp_path, verbose=False)
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


def test_neuromod_bus_gating_mechanisms():
    """Neuromodulatory bus accurately updates signals and gates model layers."""
    bus = NeuromodulatoryBus(NeuromodConfig(enabled=True))
    cfg = NeuromodRLE2EConfig()
    model = _build_model(cfg, seed_offset=1)

    # Signal injection
    bus.update(reward=2.0, entropy=1.8, loss=0.5)
    levels = bus.levels()
    gains = bus.gains()

    assert levels["da"] > 0
    assert levels["ach"] > 0
    assert gains["plasticity"] > 1.0

    num_broadcast = bus.broadcast(model)
    assert num_broadcast > 0


def test_neuromod_rl_cli_entrypoint(tmp_path: Path):
    """The main entrypoint can be executed directly with CLI arguments."""
    ret = e2e_main(["--run-dir", str(tmp_path), "--device", "cpu", "--seed", "42", "--steps", "10"])
    assert ret == 0, f"CLI entrypoint exited with non-zero code {ret}"
