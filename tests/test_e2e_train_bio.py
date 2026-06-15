"""E2E: full bio training run (tiny) with assertions + detailed logs (bead `eqyk.4`).

The single most important "is the whole thing wired correctly?" check, as a pytest. It drives the
runnable script ``scripts/e2e/train_bio.py`` (``run_bio_e2e``), which trains a tiny GPTSynaptic with
the per-synapse bio stack ON (presyn calcium/RRP/energy kinetics + postsynaptic Hebbian/CaMKII/BDNF
consolidation) on a small learnable synthetic task and asserts a battery of health + bio invariants.

What this test locks:
  1. A healthy run PASSES the whole battery (loss finite + trends down, grads bounded, params finite,
     checkpoint round-trips exactly, generation non-degenerate, Hebbian state engaged) AND the
     five bead-named bio buffers (calcium/RRP/energy/CaMKII/BDNF) stay in range and demonstrably move.
  2. The run emits a machine-readable JSONL trace with **per-step bio-state** (the eqyk.2 stream)
     plus the presynaptic calcium/RRP/energy curve — the artifact for human inspection.
  3. The battery is NOT vacuous: a no-learning run (a fully frozen model — lr=0 with plasticity off)
     is CAUGHT (loss does not decrease), so the PASS in (1) is meaningful.

Run:  pytest tests/test_e2e_train_bio.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# scripts/ is a namespace package on sys.path (same import path README uses for `python -m scripts.*`).
from scripts.e2e.train_bio import BioE2EConfig, run_bio_e2e

pytestmark = pytest.mark.e2e


def _read_events(run_dir: Path) -> list[dict]:
    path = run_dir / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_bio_e2e_run_passes_and_logs(tmp_path):
    """A healthy tiny all-(per-synapse-)bio run passes every invariant and writes a readable trace."""
    cfg = BioE2EConfig(steps=60, seed=1234)
    report = run_bio_e2e(cfg, run_dir=tmp_path, verbose=False)

    # Whole battery green (raises a labelled AssertionError listing any failures).
    report.assert_passed()
    assert report.passed
    names = {r.name: r for r in report.invariants}
    # The eqyk.4-specific invariants must be present and green (not skipped/absent).
    for inv in ("loss_decreases", "checkpoint_roundtrip", "mechanism_engaged",
                "bio_buffers_in_range", "bio_buffers_change"):
        assert inv in names, f"missing invariant {inv}"
        assert names[inv].passed, f"{inv} failed: {names[inv].detail}"

    # Loss genuinely trended down on the learnable task.
    assert report.summary["final_loss"] < report.summary["initial_loss"]

    # JSONL artifact: one train_step + one bio_state per training step, plus a presyn-curve event.
    events = _read_events(tmp_path)
    train_steps = [e for e in events if e["event"] == "train_step"]
    bio_states = [e for e in events if e["event"] == "bio_state"]
    assert len(train_steps) == cfg.steps
    # per-step postsynaptic bio-state (CaMKII/BDNF/PP1) + the final presyn curve = steps + 1
    assert len(bio_states) == cfg.steps + 1
    per_step = [e for e in bio_states if "step" in e and e["step"] < cfg.steps]
    assert {"camkii", "bdnf", "pp1"} <= set(per_step[0]["tensors"].keys())
    # the presynaptic calcium/RRP/energy curve is logged for inspection
    presyn = bio_states[-1]["tensors"]
    assert {"presyn_C", "presyn_RRP", "presyn_E"} <= set(presyn.keys())


def test_bio_e2e_battery_catches_no_learning(tmp_path):
    """Sanity that the battery is not vacuous: a no-learning run must FAIL, so the PASS above is a
    real signal.

    We freeze the model COMPLETELY and make the forward STATELESS — lr=0 (no gradient step) plus
    every stateful mechanism off (``enable_hebbian``: no fast-weight memorization that could drive
    loss DOWN even at lr=0; ``enable_presyn``: its ``ema_e`` normalizer drifts every forward and
    would otherwise leave a residual cross-step signal; ``enable_metabolism``). With nothing carrying
    across steps, the forward is a pure function of (params, batch), so the fixed pool of period
    ``pool_size`` makes the loss EXACTLY periodic and the first- and last-quartile means are equal —
    ``loss_decreases`` then deterministically fails. steps=32 spans whole pool cycles (pool_size=8)
    so the quartiles cover identical batches.
    """
    cfg = BioE2EConfig(
        steps=32, seed=1234, lr=0.0,
        syn_overrides={"enable_hebbian": False, "enable_metabolism": False, "enable_presyn": False},
    )
    report = run_bio_e2e(cfg, run_dir=tmp_path, verbose=False)

    assert not report.passed
    names = {r.name: r for r in report.invariants}
    assert not names["loss_decreases"].passed, "a fully-frozen model must not decrease the loss"
    with pytest.raises(AssertionError):
        report.assert_passed()
