"""
At-scale e2e suite: ablation-harness dry-run + per-mechanism engagement (bead hwxb.7.4).

Two things this proves at reduced (CI-fast) scale, so the real 4090 runs only change the scale:

1. **Ablation-harness dry-run.** The full pre-registered matrix (hwxb.5.1) runs end-to-end through
   the reduced-training harness (hwxb.7.3) and the real stats/verdict pipeline (eval_stats): every
   cell runs, the summary rows are in the ``eval_matrix`` schema (and load back through
   ``load_matrix_csv``), every non-baseline config gets a paired test, the architecture-vs-mechanism
   decomposition contrasts compute, the go/no-go gate fires, and a verdict table renders. This is the
   "validate the machinery BEFORE burning real GPU time" guarantee.

2. **Per-mechanism engagement — differentiable kinetics.** In a real reduced training run, the
   learnable calcium-decay kinetic moves under SGD ONLY when the differentiable recurrence is wired
   on (hwxb.4.6), and the calcium↔buffer subsystem stays contractive (spectral radius < 1) the whole
   time. This is the at-scale counterpart of the unit-level wiring contract.

Run:  pytest tests/test_scaleup_ablation_e2e.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bio_testkit import make_tiny_synaptic, random_tokens  # noqa: E402

from bio_inspired_nanochat import ablation_matrix as am  # noqa: E402
from bio_inspired_nanochat.ablation_dryrun import (  # noqa: E402
    DryRunConfig,
    render_verdict_table,
    run_ablation_dryrun,
)
from bio_inspired_nanochat.eval_stats import load_matrix_csv  # noqa: E402
from bio_inspired_nanochat.synaptic import SynapticConfig  # noqa: E402

pytestmark = pytest.mark.unit


def _reduced_cfg(tmp_path, *, extra=("add_learnable_kinetics", "bio_no_presyn")) -> DryRunConfig:
    """A fast representative slice: the 3 anchors + a couple of mechanisms, 2 seeds, few steps."""
    pick = set(extra)
    cols = am.anchors() + [c for c in (am.add_one_in() + am.leave_one_out()) if c.config_id in pick]
    return DryRunConfig(columns=cols, seeds=(1337, 1338), e2e_steps=10, run_dir=Path(tmp_path))


# --------------------------------------------------------------------------- #
# 1. Ablation-harness dry-run
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def dryrun(tmp_path_factory):
    cfg = _reduced_cfg(tmp_path_factory.mktemp("ablation_dryrun"))
    return run_ablation_dryrun(cfg)


def test_every_cell_ran_without_crashing(dryrun):
    assert dryrun.n_cells == 5 * 2, "3 anchors + 2 mechanisms × 2 seeds"
    assert dryrun.n_ok == dryrun.n_cells, [r for r in dryrun.rows if r["status"] != "ok"]


def test_summary_rows_load_through_the_eval_matrix_reader(dryrun):
    # The dry-run writes the eval_matrix summary schema, so the real stats reader consumes it.
    data = load_matrix_csv(dryrun.csv_path, metric="val_bpb")
    assert "vanilla" in data and "bio_all" in data
    for preset, by_seed in data.items():
        assert set(by_seed) == {1337, 1338}, f"{preset} missing a seed"


def test_verdict_gives_every_non_baseline_a_paired_test(dryrun):
    presets = dryrun.verdict["presets"]
    assert dryrun.verdict["baseline"] == "vanilla"
    for name, entry in presets.items():
        if name == "vanilla":
            assert "paired_vs_baseline" not in entry
        else:
            assert "paired_vs_baseline" in entry and "better" in entry and "significant" in entry


def test_decomposition_contrasts_are_present_and_paired(dryrun):
    d = dryrun.decomposition
    assert any("architecture_effect" in k for k in d)
    assert any("total_bio_effect" in k for k in d)
    assert any("mechanism (" in k for k in d)
    for name, pc in d.items():
        assert pc is not None, f"{name} should have >=2 shared seeds"
        assert pc["n_pairs"] == 2


def test_go_no_go_gate_fires_with_an_estimate(dryrun):
    g = dryrun.gate
    assert g.estimated_gpu_hours > 0
    assert isinstance(g.proceed, bool)
    assert g.n_survivors == 2  # the two non-anchor columns in the reduced slice


def test_verdict_table_renders_all_presets(dryrun):
    table = render_verdict_table(dryrun)
    assert "ablation dry-run" in table and "go/no-go" in table
    for preset in dryrun.verdict["presets"]:
        assert preset in table


def test_module_enumerates_the_full_matrix():
    # The reduced test runs a slice; assert the module still defines the full 13-column matrix.
    assert len(am.screening_columns()) == 13


# --------------------------------------------------------------------------- #
# 2. Per-mechanism engagement: differentiable learnable kinetics (hwxb.4.6 at scale)
# --------------------------------------------------------------------------- #
def _train_reduced(*, differentiable_recurrence: bool, steps: int = 40, seq: int = 48):
    """A real reduced training loop; returns (model, calcium-decay movement, max spectral radius)."""
    syn = SynapticConfig(
        learnable_kinetics=True,
        differentiable_recurrence=differentiable_recurrence,
        recurrence_block_size=8,
    )
    model = make_tiny_synaptic(seed=0, train=True, sequence_len=seq, syn_cfg=syn)
    kin = model.h[0].attn.attn.pre.kinetics
    theta0 = float(kin.theta_rho_c.detach())
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    x = random_tokens(2, seq, 97, seed=1)
    y = random_tokens(2, seq, 97, seed=2)
    max_rho = 0.0
    for _ in range(steps):
        _, loss = model(x, y, train_mode=True)
        assert torch.isfinite(loss), "training loss must stay finite"
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        max_rho = max(max_rho, float(kin.spectral_radius().detach()))
    movement = abs(float(kin.theta_rho_c.detach()) - theta0)
    return model, movement, max_rho


def test_calcium_decay_moves_under_sgd_only_when_wired():
    _, moved_on, _ = _train_reduced(differentiable_recurrence=True)
    _, moved_off, _ = _train_reduced(differentiable_recurrence=False)
    assert moved_on > 1e-5, f"the calcium decay must learn when wired on (moved {moved_on:.2e})"
    # On the single-call snapshot path the decay gets ~no gradient, so it barely moves; the wired
    # run must move it at least an order of magnitude more.
    assert moved_on > 10 * moved_off + 1e-6, (
        f"wired-on movement ({moved_on:.2e}) must dominate off ({moved_off:.2e})"
    )


def test_kinetics_stay_contractive_throughout_training():
    _, _, max_rho = _train_reduced(differentiable_recurrence=True, steps=40)
    assert max_rho < 1.0, f"calcium↔buffer spectral radius must stay < 1, peaked at {max_rho:.4f}"


# --------------------------------------------------------------------------- #
# 3. Per-mechanism engagement: online Hebbian fast-weights (ON vs OFF)
# --------------------------------------------------------------------------- #
def test_online_fast_weights_adapt_and_stay_bounded_only_when_hebbian_on():
    from bio_inspired_nanochat.e2e_harness import E2EConfig, run_e2e

    on = run_e2e(E2EConfig(synapses=True, steps=40), verbose=False)
    off = run_e2e(E2EConfig(synapses=True, steps=40, syn_overrides={"enable_hebbian": False}),
                  verbose=False)

    # ON: the fast weights adapt online (nonzero Hebbian-state delta), the run is healthy, and the
    # weights stay bounded (no runaway).
    assert on.passed, on.failures()
    assert abs(on.summary["hebbian_delta"]) > 0.0, "fast-weights must adapt online when hebbian on"
    (stable,) = [r for r in on.invariants if r.name == "mechanism_stable"]
    assert stable.passed, "online fast-weights must stay bounded"

    # OFF: no online adaptation, and the ONLY thing the harness flags is "mechanism not engaged" —
    # every other health invariant still passes (the mechanism is cleanly off, not broken).
    assert off.summary["hebbian_delta"] == 0.0, "no online adaptation when hebbian is off"
    assert {r.name for r in off.failures()} == {"mechanism_engaged"}, off.failures()
