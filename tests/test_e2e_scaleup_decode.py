"""Tests for Scale-Up Inference & Autoregressive Decode Battery (bead `hwxb.6.4`).

Verifies the high-performance autoregressive generation path at scale on single-GPU/CPU:
1. Autoregressive decode with KV-cache and presynaptic biophysical state carried across steps.
2. Exact decode-vs-contiguous prefix parity across token boundaries.
3. Per-prompt scratchpad isolation: fast weights and volatile state reset cleanly between prompts.
4. Online fast-weight adaptation during decode ('learns context online').
5. Throughput and latency benchmarking with structured Rich console logs and JSONL traces.
6. Non-degenerate generation and diversity checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.e2e.scaleup_decode import (
    ScaleupDecodeConfig,
    main as e2e_main,
    run_scaleup_decode,
)


EXPECTED_INVARIANTS = (
    "decode_contiguous_parity",
    "presyn_state_carried_across_steps",
    "per_prompt_reset_isolation",
    "non_degenerate_generation_diversity",
    "decode_throughput_bounded",
)


@pytest.mark.unit
def test_scaleup_decode_battery_full_run(tmp_path: Path):
    """The entire Scale-Up Decode battery executes and all invariants hold."""
    cfg = ScaleupDecodeConfig(
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        vocab_size=64,
        sequence_len=64,
        prompt_len=8,
        decode_len=16,
        batch_size=2,
        seed=1337,
        device="cpu",
    )
    report = run_scaleup_decode(cfg, run_dir=tmp_path, verbose=False)
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


@pytest.mark.unit
def test_scaleup_decode_cli_entrypoint(tmp_path: Path):
    """The main entrypoint can be executed directly with CLI arguments."""
    ret = e2e_main([
        "--run-dir", str(tmp_path),
        "--device", "cpu",
        "--prompt-len", "8",
        "--decode-len", "12",
        "--seed", "42",
    ])
    assert ret == 0, f"CLI entrypoint exited with non-zero code {ret}"
