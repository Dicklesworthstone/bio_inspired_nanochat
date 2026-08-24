"""Tests for Property-Based, Metamorphic, and Universal Invariant harness (bead eqyk.14).

Verifies vesicle conservation, evaluation determinism, reset isolation,
causality, monotonic depletion, stochastic convergence, and NaN-freedom.

Run:
    pytest tests/test_property_invariants.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.e2e.property_invariants import (
    PropertyInvariantsConfig,
    _build_model,
    main as e2e_main,
    run_property_invariants_e2e,
)

pytestmark = pytest.mark.e2e

EXPECTED_INVARIANTS = (
    "prop_vesicle_conservation",
    "prop_eval_determinism",
    "prop_reset_isolation",
    "prop_monotonic_depletion",
    "prop_stochastic_expectation_convergence",
    "prop_extreme_input_robustness",
    "prop_causal_invariance",
)


def test_property_invariants_full_battery(tmp_path: Path):
    """The entire Property-Based / Metamorphic battery executes end-to-end and all invariants hold."""
    cfg = PropertyInvariantsConfig(
        n_layer=2,
        n_head=4,
        n_embd=64,
        vocab_size=64,
        sequence_len=32,
        mc_samples=40,
        seed=1337,
        device="cpu",
    )
    report = run_property_invariants_e2e(cfg, run_dir=tmp_path, verbose=False)
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


def test_metamorphic_prefix_causality():
    """Prefix outputs are invariant to future tokens for any sequence pair sharing a prefix."""
    cfg = PropertyInvariantsConfig()
    model = _build_model(cfg, seed_offset=5)

    T = 16
    cutoff = 8
    seq1 = torch.randint(0, cfg.vocab_size, (1, T))
    seq2 = seq1.clone()
    seq2[:, cutoff:] = torch.randint(0, cfg.vocab_size, (1, T - cutoff))

    model.reset_sequence_state(reset_fast_weights=True)
    with torch.no_grad():
        out1 = model(seq1, train_mode=False)[0]

    model.reset_sequence_state(reset_fast_weights=True)
    with torch.no_grad():
        out2 = model(seq2, train_mode=False)[0]

    # Past prefix must be identical
    assert torch.allclose(out1[:, :cutoff], out2[:, :cutoff], atol=1e-6)


def test_property_invariants_cli_entrypoint(tmp_path: Path):
    """The main entrypoint can be executed directly with CLI arguments."""
    ret = e2e_main(["--run-dir", str(tmp_path), "--device", "cpu", "--seed", "42", "--mc-samples", "20"])
    assert ret == 0, f"CLI entrypoint exited with non-zero code {ret}"
