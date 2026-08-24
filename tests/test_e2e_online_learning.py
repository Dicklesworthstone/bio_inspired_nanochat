"""E2E: online-learning / working-memory traces with assertions + detailed logs (bead `eqyk.9`).

Drives the runnable script ``scripts/e2e/online_learning_traces.py`` (:func:`run_online_e2e`),
which proves the headline working-memory claim end-to-end on an associative-recall task:

  1. ``fast_weights_adapt_during_training`` — W_fast actually moves across grad-enabled training
     forwards at FULL-MODEL level: ``_plasticity_pending`` set (only possible inside the
     run_plasticity branch), deferred write lands on the next forward, state strictly accumulates.
     Guards the vg9.2 inert-gate regression; unit coverage lives in
     tests/test_hebbian_training_plasticity.py.
  2. ``recall_improves_with_online_memory`` — training on the binding task lifts live retrieval
     accuracy well above init, and the online-fast-weight model is not worse than a
     no-fast-weight control twin at equal compute (the bead's bio-vs-control comparison).
  3. ``scratchpad_state_written_and_read_back`` — double-presentation contrast: the frozen eval
     path leaves EXACTLY zero residue and is idempotent, while the plasticity-live path
     accumulates fast-weight state through real forwards and READS it back (pass-2 logits
     differ from pass-1); the accuracy delta is reported observationally.
  4. ``per_sequence_reset_is_exact`` — reset_sequence_state returns the EXACT factory fingerprint;
     an identical replay reproduces identical logits (guards vg9.4 cross-sequence leakage).
  5. ``bistable_latch_persists`` — sax.2 hysteresis exercised through the live GPTSynaptic stack.

What this test additionally locks:
  - The run emits a machine-readable JSONL trace (the eqyk.2 stream) with per-step train_step +
    bio_state events carrying fast-weight norms and CaMKII/PP1/latch telemetry for BOTH models.

Run:  pytest tests/test_e2e_online_learning.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# scripts/ is a namespace package on sys.path (same import path README uses for `python -m scripts.*`).
from scripts.e2e.online_learning_traces import OnlineLearningConfig, run_online_e2e

pytestmark = pytest.mark.e2e

INVARIANTS = (
    "fast_weights_adapt_during_training",
    "recall_improves_with_online_memory",
    "scratchpad_state_written_and_read_back",
    "per_sequence_reset_is_exact",
    "bistable_latch_persists",
    "jsonl_trace_written",
)


def _read_events(run_dir: Path) -> list[dict]:
    path = run_dir / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_online_learning_e2e_passes_and_logs(tmp_path):
    """A healthy tiny online-learning run passes every invariant and writes a readable trace."""
    cfg = OnlineLearningConfig(steps=250, seed=0)
    report = run_online_e2e(cfg, run_dir=tmp_path, verbose=False)

    # Whole battery green (raises a labelled AssertionError listing any failures).
    report.assert_passed()
    names = {r.name: r for r in report.invariants}
    for inv in INVARIANTS:
        assert inv in names, f"missing invariant {inv}"
        assert names[inv].passed, f"{inv} failed: {names[inv].detail}"

    # Non-vacuous: SGD genuinely lifted live retrieval accuracy over its init.
    assert report.summary["bio_acc_final"] > report.summary["bio_acc_init"]

    # JSONL artifact structure: per-step events for BOTH models with bio-state telemetry.
    events = _read_events(tmp_path)
    train_steps = [e for e in events if e["event"] == "train_step"]
    bio_states = [e for e in events if e["event"] == "bio_state"]
    assert len(train_steps) == 2 * cfg.steps, "one train_step per model per step"
    assert len(bio_states) >= 2 * cfg.steps, "one bio_state per model per step"
    tags = {e.get("model_tag") for e in train_steps}
    assert {"bio", "ctrl"} <= tags, "both the online model and the control must be traced"

    fw_fields = [k for k in bio_states[0]["tensors"] if k.startswith("bio_fw_norm")]
    latch_fields = [k for k in bio_states[0]["tensors"] if k.startswith("bio_camkii")]
    assert fw_fields and latch_fields, "fast-weight norms + latch state must be in the trace"


def test_online_learning_e2e_deterministic(tmp_path):
    """Fixed seed ⇒ identical invariant outcomes (name, passed, observed) across two runs."""
    r1 = run_online_e2e(OnlineLearningConfig(steps=30, seed=11), run_dir=tmp_path / "a", verbose=False)
    r2 = run_online_e2e(OnlineLearningConfig(steps=30, seed=11), run_dir=tmp_path / "b", verbose=False)
    assert [(i.name, i.passed, i.observed) for i in r1.invariants] == [
        (i.name, i.passed, i.observed) for i in r2.invariants
    ]
