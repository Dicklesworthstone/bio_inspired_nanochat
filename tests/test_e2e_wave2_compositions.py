"""Tests for the Wave-2 Capability-Frontier Compositions E2E battery (beads re4e, eqyk.22)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.e2e.wave2_compositions_suite import (
    Wave2CompositionsConfig,
    main as e2e_main,
    run_wave2_compositions_e2e,
)

EXPECTED_WAVE2_INVARIANTS = (
    "self_correcting_generation_loop",
    "metacognition_self_model",
    "energy_guided_search",
    "persistent_lifelong_memory",
    "synaptic_serving_engine_sla",
    "conformal_certified_abstention",
    "speculative_decode_cheap_path",
)


def test_wave2_compositions_full_battery(tmp_path: Path):
    """The entire Wave-2 Compositions verification battery executes end-to-end and all invariants hold."""
    cfg = Wave2CompositionsConfig(
        deliberation_max_iters=5,
        conformal_target_alpha=0.15,
        memory_dim=32,
        search_depth=3,
        seed=42,
    )
    report = run_wave2_compositions_e2e(cfg, run_dir=tmp_path, verbose=False)
    report.assert_passed()

    inv_map = {inv.name: inv for inv in report.invariants}
    for name in EXPECTED_WAVE2_INVARIANTS:
        assert name in inv_map, f"Missing invariant: {name}"
        assert inv_map[name].passed, f"Invariant {name} failed: {inv_map[name].detail}"

    # Verify structured event stream was written and contains records
    events_path = tmp_path / "events.jsonl"
    assert events_path.exists(), "events.jsonl trace must be created"
    events: list[dict] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    assert len(events) >= len(EXPECTED_WAVE2_INVARIANTS) + 1


def test_wave2_compositions_cli_entrypoint(tmp_path: Path):
    """The main entrypoint can be executed directly with CLI arguments."""
    ret = e2e_main([
        "--run-dir",
        str(tmp_path),
        "--delib-iters",
        "4",
        "--conformal-alpha",
        "0.20",
        "--seed",
        "42",
    ])
    assert ret == 0, f"CLI entrypoint exited with non-zero code {ret}"
