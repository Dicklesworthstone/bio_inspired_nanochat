"""Tests for the Synaptic Retrofit & MGR Attention Variants E2E verification battery (beads vap, eqyk.21)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.e2e.retrofit_mgr_suite import (
    RetrofitMGRConfig,
    main as e2e_main,
    run_retrofit_mgr_e2e,
)

EXPECTED_RETROFIT_MGR_INVARIANTS = (
    "synaptic_checkpoint_retrofit",
    "hf_bio_adapter_injection",
    "mgr_attention_variants_forward_backward",
    "mgr_reversible_block_reconstruction",
    "mgr_ordinal_lr_scheduler",
)


def test_retrofit_mgr_full_battery(tmp_path: Path):
    """The entire Retrofit & MGR Attention verification battery executes end-to-end and all invariants hold."""
    cfg = RetrofitMGRConfig(
        vocab_size=64,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=32,
        sequence_len=16,
        finetune_steps=2,
        seed=42,
    )
    report = run_retrofit_mgr_e2e(cfg, run_dir=tmp_path, verbose=False)
    report.assert_passed()

    inv_map = {inv.name: inv for inv in report.invariants}
    for name in EXPECTED_RETROFIT_MGR_INVARIANTS:
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
    assert len(events) >= len(EXPECTED_RETROFIT_MGR_INVARIANTS) + 1


def test_retrofit_mgr_cli_entrypoint(tmp_path: Path):
    """The main entrypoint can be executed directly with CLI arguments."""
    ret = e2e_main([
        "--run-dir",
        str(tmp_path),
        "--finetune-steps",
        "2",
        "--seed",
        "42",
    ])
    assert ret == 0, f"CLI entrypoint exited with non-zero code {ret}"
