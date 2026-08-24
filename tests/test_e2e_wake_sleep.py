"""Tests for Wake/Sleep consolidation E2E harness (bead eqyk.10).

Verifies the offline consolidation, fast->slow knowledge distillation,
homeostatic norm bounds (SHY), and catastrophic forgetting prevention in CI time.

Run:
    pytest tests/test_e2e_wake_sleep.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from bio_inspired_nanochat.sleep_consolidation import (
    ReplayBuffer,
    homeostatic_downscale,
)
from scripts.e2e.wake_sleep_consolidation import (
    WakeSleepE2EConfig,
    _build_model,
    _total_slow_norm,
    main as e2e_main,
    run_wake_sleep_e2e,
)

pytestmark = pytest.mark.e2e

EXPECTED_INVARIANTS = (
    "consolidation_moves_info_fast_to_slow",
    "homeostatic_downscaling_bounds_norms",
    "catastrophic_forgetting_reduced",
)


def test_wake_sleep_consolidation_full_battery(tmp_path: Path):
    """The entire Wake/Sleep battery executes end-to-end and all invariants hold."""
    cfg = WakeSleepE2EConfig(
        n_layer=2,
        n_head=4,
        n_embd=64,
        num_cycles=4,
        batch_size=8,
        num_pairs=3,
        seed=1337,
        device="cpu",
    )
    report = run_wake_sleep_e2e(cfg, run_dir=tmp_path, verbose=False)
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


def test_replay_buffer_surprise_prioritization():
    """ReplayBuffer stores items, evicts lowest-surprise on overflow, and samples properly."""
    buf = ReplayBuffer(max_capacity=3, alpha=1.0, seed=42)
    t = torch.zeros(2, 4)

    buf.add(t, t, loss=0.2, step=1)
    buf.add(t, t, loss=1.5, step=2)
    buf.add(t, t, loss=0.8, step=3)
    assert len(buf) == 3

    # Add high-surprise item -> should evict loss=0.2 item
    buf.add(t, t, loss=2.0, step=4)
    assert len(buf) == 3
    losses = {item.loss for item in buf.buffer}
    assert 0.2 not in losses
    assert 2.0 in losses

    # Sample batch
    sampled = buf.sample(2)
    assert len(sampled) == 2


def test_homeostatic_downscale_bounds_slow_weights():
    """Homeostatic downscale scales down weights when norm exceeds threshold."""
    cfg = WakeSleepE2EConfig()
    model = _build_model(cfg, seed_offset=5)

    # Artificially expand slow weights
    for lin in model.modules():
        if hasattr(lin, "w_slow") and lin.w_slow is not None:
            lin.w_slow.data.mul_(10.0)

    initial_norm = _total_slow_norm(model)
    stats = homeostatic_downscale(model, max_slow_norm=10.0, decay_factor=0.9)
    post_norm = _total_slow_norm(model)

    assert stats["scaling_factor"] < 1.0
    assert post_norm < initial_norm
    assert post_norm <= 10.0 + 1e-4


def test_wake_sleep_cli_entrypoint(tmp_path: Path):
    """The main entrypoint can be executed directly with CLI arguments."""
    ret = e2e_main(["--run-dir", str(tmp_path), "--device", "cpu", "--seed", "42", "--cycles", "2"])
    assert ret == 0, f"CLI entrypoint exited with non-zero code {ret}"
