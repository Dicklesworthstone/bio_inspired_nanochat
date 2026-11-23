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
import torch

from bio_inspired_nanochat.neuromod import NeuromodulatoryBus, NeuromodConfig
from bio_inspired_nanochat.synaptic import SynapticLinear
from bio_inspired_nanochat.synthetic_tasks import reward_task
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
    assert report.summary["reward_episode_sites"] == 4
    assert report.summary["reward_episode_commits"] == cfg.rl_steps * 4

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


def test_same_episode_feedback_commits_deferred_slow_write_exactly_once():
    cfg = NeuromodRLE2EConfig(seed=53)
    batch, _ = reward_task(batch=4, vocab_size=cfg.vocab_size, seed=66)
    drifts = []
    for dopamine_slope in (0.0, 1.0):
        model = _build_model(cfg).train()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        _, loss = model(batch.inputs, batch.targets, train_mode=True)
        loss.backward()
        optimizer.step()
        layers = [layer for layer in model.modules() if isinstance(layer, SynapticLinear)]
        assert len(layers) == 4 and all(layer._plasticity_pending for layer in layers)
        bus = NeuromodulatoryBus(NeuromodConfig(ema_tau=0.0, da_gain_slope=dopamine_slope))
        bus.update(reward=1.0)
        bus.broadcast(model)
        drift = []
        for layer in layers:
            before = layer.w_slow.detach().clone()
            assert layer.commit_pending_plasticity()
            after = layer.w_slow.detach().clone()
            drift.append(float((after - before).norm()))
            assert not layer.commit_pending_plasticity()
            assert not layer._plasticity_pending and layer._last_gate_scale is None
            torch.testing.assert_close(layer.w_slow, after, atol=0, rtol=0)
        drifts.append(drift)
    assert drifts[0] == pytest.approx([.03] * 4, abs=1e-7)
    assert drifts[1] == pytest.approx([.03 * (1 + float(torch.tanh(torch.tensor(2.0))))] * 4, abs=1e-7)
