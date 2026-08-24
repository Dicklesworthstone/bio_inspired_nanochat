"""E2E: bio-vs-vanilla scaling-law probe (bead `74f.6`).

Drives ``scripts/e2e/scaling_law_study.py`` (:func:`run_scaling_study`). The production
multi-scale study is correctly GPU-blocked (hwxb.*); what THIS test locks is the harness:

  1. The full grid trains without crashing for BOTH families (vanilla GPT + GPTSynaptic).
  2. Each family yields a log-log power-law fit (exponent b) from the FLOP-proxy vs
     held-out-NLL points.
  3. A verdict is always emitted, including the honest ``unclear_noisy_fits`` /
     ``insufficient_data`` outcomes — the machinery never fabricates decisiveness.
  4. Determinism: fixed seed ⇒ identical exponent fits across two runs.

Run:  pytest tests/test_e2e_scaling_law.py -v
"""

from __future__ import annotations

import pytest

from scripts.e2e.scaling_law_study import ScalingStudyConfig, run_scaling_study

pytestmark = pytest.mark.e2e


def _same(x, y) -> bool:
    return x == y or (isinstance(x, float) and isinstance(y, float) and x != x and y != y)


def test_scaling_study_runs_and_verdicts(tmp_path):
    """Micro-grid end-to-end: both families train, fit, and emit an honest verdict."""
    cfg = ScalingStudyConfig(depths=(1,), widths=(64, 128), seeds=(0,), base_steps=60)
    cfg.eval_batches = 2
    report = run_scaling_study(cfg, run_dir=tmp_path, verbose=False)

    report.assert_passed()
    names = {r.name: r for r in report.invariants}
    for inv in ("grid_completed", "both_families_produced_fits",
                "verdict_with_cis", "jsonl_trace_written"):
        assert inv in names, f"missing invariant {inv}"
        assert names[inv].passed, f"{inv} failed: {names[inv].detail}"
    assert report.summary["verdict"]


def test_scaling_study_deterministic(tmp_path):
    """Fixed seed ⇒ identical exponent fits across two runs."""

    def run(tag):
        cfg = ScalingStudyConfig(depths=(1,), widths=(64, 128), seeds=(0,), base_steps=40)
        cfg.eval_batches = 2
        return run_scaling_study(cfg, run_dir=tmp_path / tag, verbose=False)

    r1, r2 = run("a"), run("b")
    f1, f2 = r1.summary["fits"], r2.summary["fits"]
    for family in ("vanilla", "bio"):
        assert _same(f1[family]["exponent_mean"], f2[family]["exponent_mean"])
        assert len(f1[family]["per_seed"]) == len(f2[family]["per_seed"])
        for p, q in zip(f1[family]["per_seed"], f2[family]["per_seed"]):
            assert _same(p["r_squared"], q["r_squared"])
            assert _same(p["exponent_b"], q["exponent_b"])
